import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
import sys
from pathlib import Path
from code.common.fsf_wrapper import FSFEncoder
from code.common.utils import set_seed, validate_plan
from code.modeling.models import StateCentricLSTM, StateCentricLSTM_Delta
from code.tokenization.factory import create_tokenizer
from code.tokenization.multidomain import MultiDomainUnionTokenizer, load_tokenizer_from_manifest

import numpy as np
import torch
import torch.nn.functional as F
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm


DEFAULT_SPLITS = ["validation", "test-interpolation", "test-extrapolation"]


def progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device from CLI preference."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    return torch.device("cpu")


def normalize_encoding_name(name: str) -> str:
    """Normalize encoding aliases to canonical tokenizer names."""
    return "wl" if name == "graphs" else name


def score_distances(
    pred_batch: torch.Tensor,
    candidate_batch: torch.Tensor,
    *,
    score_metric: str,
    delta: bool,
) -> torch.Tensor:
    """
    Compute successor distances under the requested metric.

    `native` preserves the original behavior from the main study:
      - cosine distance for state prediction
      - L2 distance for delta prediction
    """
    metric = score_metric
    if metric == "native":
        metric = "l2" if delta else "cosine"

    if metric == "l2":
        return torch.norm(pred_batch - candidate_batch, p=2, dim=-1).reshape(-1)
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(pred_batch, candidate_batch, dim=-1).reshape(-1)
    raise ValueError(f"Unsupported score metric: {score_metric}")


def resolve_model_file_path(data_dir: str, filename: str) -> str:
    """
    Resolve files from data/encodings/models robustly.
    Supports callers passing either `data` or an encoding subdir.
    """
    p = Path(data_dir).resolve()
    search_dirs = [
        p / "encodings" / "models",
        p / "models",
        p.parent / "models",
        p.parent.parent / "models",
        Path("data") / "encodings" / "models",
    ]

    seen = set()
    for d in search_dirs:
        d_str = str(d)
        if d_str in seen:
            continue
        seen.add(d_str)
        candidate = d / filename
        if candidate.exists():
            return str(candidate)

    # Default fallback path for clear error messages.
    return str(search_dirs[0] / filename)


def resolve_vocab_path(data_dir: str, domain: str, raw_encoding: str) -> str | None:
    """Find the most likely vocabulary file for a tokenizer."""
    normalized = normalize_encoding_name(raw_encoding)
    names = [f"{domain}_{normalized}.json"]

    if normalized == "wl":
        names.extend(
            [
                f"{domain}_wl_tok.json",
                f"{domain}_wl.json",
                f"{domain}_graphs.json",
            ]
        )

    if raw_encoding != normalized:
        names.append(f"{domain}_{raw_encoding}.json")

    for name in names:
        path = resolve_model_file_path(data_dir, name)
        if os.path.exists(path):
            return path

    return None


def get_fsf_tensor(atoms_set, encoder, objects, obj_map, device):
    """Helper for FSF Inference embedding"""
    # Convert set of strings to list of tuples
    atom_tuples = []
    for a in atoms_set:
        content = a.replace("(", "").replace(")", "").lower()
        atom_tuples.append(tuple(content.split()))

    vec = encoder._state_to_vector(atom_tuples, objects, obj_map)
    # [1, 1, D]
    return torch.tensor(vec).float().to(device).unsqueeze(0).unsqueeze(0)


def transform_state_compat(tokenizer, state_atoms, goal_atoms, objects, problem_path):
    """
    Call tokenizer.transform_state with optional problem path when supported.
    WLTokenizer needs it; other tokenizers generally do not.
    """
    try:
        return tokenizer.transform_state(
            state_atoms,
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
        )
    except TypeError:
        return tokenizer.transform_state(state_atoms, goal_atoms, objects)


def transform_state_cached(
    tokenizer,
    state_atoms,
    goal_atoms,
    objects,
    problem_path,
    *,
    wl_prob=None,
):
    """Call tokenizer.transform_state while reusing a pre-parsed WL problem when available."""
    try:
        return tokenizer.transform_state(
            state_atoms,
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
            _wl_prob=wl_prob,
        )
    except TypeError:
        return transform_state_compat(tokenizer, state_atoms, goal_atoms, objects, problem_path)


def transform_goal_compat(tokenizer, goal_atoms, objects, problem_path):
    """
    Call tokenizer.transform_goal with optional problem path when supported.
    WLTokenizer needs it; other tokenizers generally do not.
    """
    try:
        return tokenizer.transform_goal(
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
        )
    except TypeError:
        return tokenizer.transform_goal(goal_atoms, objects)


def transform_goal_cached(tokenizer, goal_atoms, objects, problem_path, *, wl_prob=None):
    """Call tokenizer.transform_goal while reusing a pre-parsed WL problem when available."""
    try:
        return tokenizer.transform_goal(
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
            _wl_prob=wl_prob,
        )
    except TypeError:
        return transform_goal_compat(tokenizer, goal_atoms, objects, problem_path)


def get_generic_tensor(
    atoms_set,
    tokenizer,
    goal_atoms,
    objects,
    problem_path,
    device,
    *,
    wl_prob=None,
):
    """Helper for Generic Tokenizer Inference embedding"""
    state_atoms = list(atoms_set)
    vec = transform_state_cached(
        tokenizer,
        state_atoms,
        goal_atoms,
        objects,
        problem_path,
        wl_prob=wl_prob,
    )
    # [1, 1, D]
    return torch.tensor(vec).float().to(device).unsqueeze(0).unsqueeze(0)


def extract_objects_from_problem(prob, dom) -> list[str]:
    """Extract sorted object names from a parsed pyperplan problem/domain pair."""
    objs = set()

    prob_objects = getattr(prob, "objects", {})
    if isinstance(prob_objects, dict):
        objs.update(str(name) for name in prob_objects.keys())
    else:
        for obj in prob_objects:
            objs.add(obj.name if hasattr(obj, "name") else str(obj))

    dom_constants = getattr(dom, "constants", {})
    if isinstance(dom_constants, dict):
        objs.update(str(name) for name in dom_constants.keys())
    else:
        for obj in dom_constants:
            objs.add(obj.name if hasattr(obj, "name") else str(obj))

    return sorted(objs)


def solve_problem(
    args,
    split,
    prob_file,
    model,
    device,
    encoder_type,
    feature_encoder,
    objects=None,
    obj_map=None,
    collect_search_stats: bool = False,
):
    """Unified Solver for Generic Tokenizers and FSF"""
    search_start_time = None
    if collect_search_stats:
        import time

        search_start_time = time.perf_counter()
        search_stats = {
            "beam_expansions": 0,
            "model_calls": 0,
            "successor_evals": 0,
            "outer_steps": 0,
            "terminated_reason": "max_steps",
        }

    domain_path = os.path.join(args.pddl_dir, args.domain, "domain.pddl")
    prob_path = os.path.join(args.pddl_dir, args.domain, split, prob_file)

    # 1. Pyperplan Parsing (for successors)
    try:
        parser = Parser(domain_path, prob_path)
        dom = parser.parse_domain()
        prob = parser.parse_problem(dom)
        task = ground(prob)
    except Exception as e:
        print(f"Pyperplan Parsing Error on {prob_file}: {e}")
        raise e

    # Match upstream inference behavior: allow longer searches on larger problems.
    num_objects = len(prob.objects) + len(dom.constants)
    effective_max_steps = max(args.max_steps, args.steps_per_object * num_objects)

    if objects is None and encoder_type != "fsf":
        objects = extract_objects_from_problem(prob, dom)

    initial_atoms = task.initial_state
    goal_set = set(task.goals)
    goal_atoms_list = list(goal_set)
    state_cache = {}
    successor_cache = {}
    sorted_operators = sorted(task.operators, key=lambda op: op.name)
    wl_prob = None

    if encoder_type == "wl":
        from wlplan.planning import parse_problem as wl_parse_problem

        wl_prob = wl_parse_problem(domain_path, prob_path)

    # 2. Embedding Setup
    if encoder_type == "fsf":
        encoder = feature_encoder
        objects = encoder._get_sorted_objects(prob_path)
        obj_map = encoder._get_object_indices(objects)

        def get_cached_tensor(atoms):
            key = frozenset(atoms)
            cached = state_cache.get(key)
            if cached is None:
                cached = get_fsf_tensor(atoms, encoder, objects, obj_map, device)
                state_cache[key] = cached
            return cached

        # Embed Goal
        goal_vec = encoder.embed_goal(prob_path)
        goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0)  # [1, D]

        # Embed Init
        init_tensor = get_cached_tensor(initial_atoms)

    else:
        # Generic Tokenizer Logic
        tokenizer = feature_encoder

        def get_cached_tensor(atoms):
            key = frozenset(atoms)
            cached = state_cache.get(key)
            if cached is None:
                cached = get_generic_tensor(
                    atoms,
                    tokenizer,
                    goal_atoms_list,
                    objects,
                    prob_path,
                    device,
                    wl_prob=wl_prob,
                )
                state_cache[key] = cached
            return cached
        
        # Embed Goal
        goal_vec = transform_goal_cached(
            tokenizer,
            goal_atoms_list,
            objects,
            prob_path,
            wl_prob=wl_prob,
        )
        goal_tensor = torch.tensor(goal_vec).float().to(device).unsqueeze(0) # [1, D]

        # Embed Init
        init_tensor = get_cached_tensor(initial_atoms)

    # 3. Beam Search
    beam = [
        (0.0, None, init_tensor, initial_atoms, [], set())
    ]  # score, hidden, tensor, atoms, plan, visited

    def get_successors(atoms):
        state_hash = frozenset(atoms)
        cached = successor_cache.get(state_hash)
        if cached is None:
            cached = []
            for op in sorted_operators:
                if op.applicable(atoms):
                    next_atoms = op.apply(atoms)
                    cached.append((op.name, next_atoms, frozenset(next_atoms)))
            successor_cache[state_hash] = cached
        return cached

    for _ in range(effective_max_steps):
        if collect_search_stats:
            search_stats["outer_steps"] += 1
            search_stats["effective_max_steps"] = effective_max_steps
        candidates = []
        for score, hidden, last_tensor, current_atoms, plan, visited in beam:
            if collect_search_stats:
                search_stats["beam_expansions"] += 1
            # Check Goal (Internal Check)
            if goal_set.issubset(current_atoms):
                result = {
                    "problem": prob_file,
                    "search_solved": True,
                    "plan_len": len(plan),
                    "plan": plan,
                    "effective_max_steps": effective_max_steps,
                }
                if collect_search_stats:
                    search_stats["terminated_reason"] = "goal_reached"
                    search_stats["search_elapsed_sec"] = time.perf_counter() - search_start_time
                    result.update(search_stats)
                return result

            # Predict Next Latent State/Delta
            with torch.inference_mode():
                if collect_search_stats:
                    search_stats["model_calls"] += 1
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=getattr(args, "_use_amp", False),
                ):
                    # The model predicts the State directly
                    pred, next_hidden = model(last_tensor, goal_tensor, hidden=hidden)

                # reconstruct the next state (S_t + Delta) if delta
                # else, the model already predicts S_t+1 directly
                pred_next_emb = (last_tensor + pred) if args.delta else pred

            # Score Successors
            successor_items = []
            successor_tensors = []
            for op_name, next_atoms, next_hash in get_successors(current_atoms):
                if collect_search_stats:
                    search_stats["successor_evals"] += 1
                if next_hash in visited:
                    continue  # Skip cycles

                cand_tensor = get_cached_tensor(next_atoms)
                successor_items.append((op_name, next_atoms, next_hash, cand_tensor))
                successor_tensors.append(cand_tensor)

            if not successor_items:
                continue

            candidate_batch = torch.cat(successor_tensors, dim=0)
            pred_batch = pred_next_emb.expand(candidate_batch.shape[0], -1, -1)

            sims = score_distances(
                pred_batch,
                candidate_batch,
                score_metric=args.score_metric,
                delta=args.delta,
            )

            for (op_name, next_atoms, next_hash, cand_tensor), sim in zip(
                successor_items,
                sims.tolist(),
            ):

                # Update Score
                new_score = score + sim

                # Update Visited
                new_visited = visited.copy()
                new_visited.add(next_hash)

                # Append tuple
                candidates.append(
                    (
                        new_score,
                        next_hidden,
                        cand_tensor,
                        next_atoms,
                        plan + [op_name],
                        new_visited,
                    )
                )

            # Prune Beam
            if not candidates:
                if collect_search_stats:
                    search_stats["terminated_reason"] = "dead_end"
                break  # All paths led to dead ends

            # Stable Sort:
            candidates.sort(key=lambda x: (x[0], str(x[4])))

            beam = candidates[: args.beam_width]

    best = beam[0] if beam else (0, 0, 0, [], [], 0)

    result = {
        "problem": prob_file,
        "search_solved": False,
        "plan_len": len(best[4]),
        "plan": best[4],
        "effective_max_steps": effective_max_steps,
    }
    if collect_search_stats:
        if search_stats["terminated_reason"] == "max_steps":
            search_stats["terminated_reason"] = "max_steps"
        search_stats["search_elapsed_sec"] = time.perf_counter() - search_start_time
        result.update(search_stats)
    return result


def run_inference(args):
    set_seed(args.seed)

    device = resolve_device(args.device)
    args._use_amp = bool(args.amp and device.type == "cuda")
    if args.fast and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'enabled' if args._use_amp else 'disabled'}")
    
    # Common variables
    domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")

    # Load Encoder
    feature_encoder = None
    input_dim = 0
    
    normalized_encoding = normalize_encoding_name(args.encoding)

    if normalized_encoding == "fsf":
        # 1. Load Config
        config_path = resolve_model_file_path(
            args.data_dir, f"{args.domain}_fsf_config.json"
        )
        if not os.path.exists(config_path):
            print(f"Error: FSF Config not found at {config_path}")
            return

        with open(config_path, "r") as f:
            config = json.load(f)
            max_objects = config["max_objects"]

        # 2. Init Encoder
        feature_encoder = FSFEncoder(args.domain, domain_pddl, max_objects)

        # 3. Set Input Dim (Max Objects + 1 Global)
        input_dim = feature_encoder.vector_size
        print(f"FSF Input Dimension: {input_dim}")
    else:
        # Generic Tokenizer
        try:
            if args.tokenizer_manifest:
                tokenizer = load_tokenizer_from_manifest(args.tokenizer_manifest)
                print(f"Loaded tokenizer manifest from {args.tokenizer_manifest}")
            else:
                vocab_path = resolve_vocab_path(args.data_dir, args.domain, args.encoding)
                tokenizer = create_tokenizer(normalized_encoding)

                if vocab_path and os.path.exists(vocab_path):
                    tokenizer.load_vocabulary(vocab_path)
                    print(f"Loaded {normalized_encoding} vocabulary from {vocab_path}")
                else:
                    print(
                        f"Warning: Vocabulary file not found for '{args.encoding}'. Using default params."
                    )

            if isinstance(tokenizer, MultiDomainUnionTokenizer):
                tokenizer.set_active_domain(args.domain, domain_pddl)
            elif hasattr(tokenizer, "set_domain"):
                tokenizer.set_domain(domain_pddl)

            feature_encoder = tokenizer
            input_dim = tokenizer.get_embedding_dim()

        except Exception as e:
            print(f"Failed to initialize tokenizer '{args.encoding}': {e}")
            return

    # Load Model
    print(f"Loading LSTM from {args.checkpoint}...")

    # Determine projection usage
    use_projection = not args.no_projection

    if args.delta:
        model = StateCentricLSTM_Delta(
            input_dim, hidden_dim=args.hidden_dim, use_projection=use_projection
        ).to(device)
    else:
        model = StateCentricLSTM(
            input_dim, hidden_dim=args.hidden_dim, use_projection=use_projection
        ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_split_solved = []
    all_split_exec = []

    # 4. Run on Splits
    splits = args.splits
    try:
        validation_workers = max(1, args.validation_workers)
        for split in splits:
            print(f"\n*** Testing on {split} ***")
            split_dir = os.path.join(args.pddl_dir, args.domain, split)
            if not os.path.exists(split_dir):
                print(f"Skipping {split} (not found)")
                continue

            results = []
            solved_count = 0
            executable_count = 0
            prob_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pddl")])

            if args.problems:
                requested = set(args.problems)
                prob_files = [f for f in prob_files if f in requested]

            if args.max_problems is not None:
                prob_files = prob_files[: args.max_problems]

            print(f" Found {len(prob_files)} problems for {split}")

            pending_validations = []
            with ThreadPoolExecutor(max_workers=validation_workers) as validation_pool:
                for prob_file in tqdm(
                    prob_files,
                    desc=f"Solving {split}",
                    disable=(not progress_enabled()),
                ):
                    prob_path = os.path.join(split_dir, prob_file)

                    try:
                        # Generate Plan
                        res = solve_problem(
                            args,
                            split,
                            prob_file,
                            model,
                            device,
                            normalized_encoding,
                            feature_encoder,
                            objects=None,
                            collect_search_stats=args.collect_search_stats,
                        )

                        if args.skip_validation:
                            # Debug fallback: trust internal goal check when VAL is unavailable.
                            is_solved = bool(res.get("search_solved", False))
                            is_executable = is_solved
                            res["val_skipped"] = True
                            res["val_solved"] = is_solved
                            res["val_executable"] = is_executable
                            res["solved"] = is_solved
                            results.append(res)
                            if is_solved:
                                solved_count += 1
                            if is_executable:
                                executable_count += 1
                        else:
                            future = validation_pool.submit(
                                validate_plan,
                                domain_pddl,
                                prob_path,
                                res["plan"],
                                args.val_path,
                            )
                            pending_validations.append((prob_file, res, future))

                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        print(f"Error processing {prob_file}: {e}")
                        results.append({"problem": prob_file, "solved": False, "error": str(e)})

                if not args.skip_validation:
                    for prob_file, res, future in pending_validations:
                        try:
                            is_solved, is_executable = future.result()
                        except Exception as e:
                            print(f"Validation failed for {prob_file}: {e}")
                            is_solved, is_executable = False, False

                        res["val_solved"] = is_solved
                        res["val_executable"] = is_executable
                        res["solved"] = is_solved
                        results.append(res)

                        if is_solved:
                            solved_count += 1
                        if is_executable:
                            executable_count += 1

            # Report
            total = len(prob_files)
            accuracy = solved_count / total if total else 0
            exec_rate = executable_count / total if total else 0
            avg_plan_len = (
                sum(r.get("plan_len", 0) for r in results if "plan_len" in r) / total
                if total
                else 0.0
            )

            print(
                f"Result {split}: Solved {solved_count}/{total} ({accuracy:.2%}) | Executable {executable_count}/{total} ({exec_rate:.2%})"
            )

            # Save
            os.makedirs(args.results_dir, exist_ok=True)
            tag_suffix = f"_{args.tag}" if getattr(args, "tag", "") else ""
            out_file = os.path.join(
                args.results_dir,
                f"{args.domain}_{args.encoding}_{split}{tag_suffix}_results.json",
            )
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {out_file}")

            all_split_solved.append(accuracy)
            all_split_exec.append(exec_rate)
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint", required=True)
    # Replaced choices with free text to allow all tokenizers
    parser.add_argument("--encoding", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pddl_dir", default="data/pddl")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--tokenizer_manifest",
        default=None,
        help="Optional explicit tokenizer manifest for pooled/all-domain runs",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device selection policy",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument(
        "--steps_per_object",
        type=int,
        default=10,
        help="Minimum search budget scales to max(max_steps, steps_per_object * num_objects).",
    )
    parser.add_argument("--beam_width", type=int, default=3, help="Search beam width")
    parser.add_argument(
        "--score_metric",
        choices=["native", "cosine", "l2"],
        default="native",
        help="Successor scoring metric. 'native' reproduces the original study setting.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable automatic mixed precision for CUDA",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast CUDA settings (less deterministic, more throughput)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Splits to run (default: validation/test-interpolation/test-extrapolation)",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=None,
        help="Optional explicit problem file names to run (e.g., probBLOCKS-8-0.pddl)",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Optional cap on number of problems per split after filtering",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip VAL-based validation and use internal search goal check for solved status",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=8,
        help="Number of background VAL validation workers to overlap with search.",
    )
    parser.add_argument(
        "--collect_search_stats",
        action="store_true",
        help="Record search-effort and termination statistics in the output JSON.",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds. Def. is False",
    )
    parser.add_argument(
        "--tag",
        default="state",
        help="Optional tag to disambiguate results, e.g., 'state' or 'delta'",
    )
    parser.add_argument(
        "--no_projection",
        action="store_true",
        help="If set, disables the input projection layer (must match training)",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    HOME = os.path.expanduser("~")
    ROOT_DIR = f"{HOME}/planning/"
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/bin/Validate"),
        help="Path to VAL binary",
    )
    parser.set_defaults(amp=True)

    args = parser.parse_args()

    run_inference(args)


