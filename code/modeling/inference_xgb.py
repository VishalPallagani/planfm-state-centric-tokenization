import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
import time
import sys
from pathlib import Path
from code.common.fsf_wrapper import FSFEncoder
from code.common.utils import set_seed, validate_plan
from code.tokenization.factory import create_tokenizer
from code.tokenization.multidomain import MultiDomainUnionTokenizer, load_tokenizer_from_manifest

import numpy as np
import xgboost as xgb
from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from tqdm import tqdm


DEFAULT_SPLITS = ["validation", "test-interpolation", "test-extrapolation"]


def progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def xgb_cuda_supported() -> bool:
    """Best-effort check for CUDA support in installed XGBoost build."""
    try:
        info = xgb.build_info()
    except Exception:
        return False

    flag = info.get("USE_CUDA")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.lower() in {"1", "true", "on", "yes"}
    return False


def resolve_xgb_device(device_arg: str) -> str:
    """Resolve XGBoost device target from CLI preference."""
    if device_arg in {"cpu", "cuda"}:
        return device_arg
    return "cuda" if xgb_cuda_supported() else "cpu"


def score_distances(
    pred_next_emb: np.ndarray,
    cand_matrix: np.ndarray,
    *,
    score_metric: str,
    delta: bool,
) -> np.ndarray:
    """
    Compute successor distances under the requested metric.

    `native` preserves the original behavior from the main study:
      - cosine distance for state prediction
      - L2 distance for delta prediction
    """
    metric = score_metric
    if metric == "native":
        metric = "l2" if delta else "cosine"

    u = pred_next_emb.reshape(-1)
    if metric == "cosine":
        u_norm = np.linalg.norm(u)
        v_norms = np.linalg.norm(cand_matrix, axis=1)
        denom = u_norm * v_norms
        cos_sim = np.divide(
            cand_matrix @ u,
            denom,
            out=np.zeros_like(v_norms, dtype=np.float32),
            where=(denom != 0),
        )
        return np.where((u_norm == 0) | (v_norms == 0), 1.0, 1.0 - cos_sim)
    if metric == "l2":
        return np.linalg.norm(cand_matrix - u, axis=1)
    raise ValueError(f"Unsupported score metric: {score_metric}")


def normalize_encoding_name(name: str) -> str:
    """Normalize encoding aliases to canonical tokenizer names."""
    return "wl" if name == "graphs" else name


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


def embed_state_generic(atoms_set, tokenizer, goal_atoms, objects, problem_path):
    """
    Helper to convert a set of atoms (strings) into a Numpy Array [1, D] using generic tokenizer.
    """
    # 1. Convert set to list
    state_atoms = list(atoms_set)
    
    # 2. Transform
    # Passing problem_pddl_path as kwarg for WLTokenizer which needs it
    try:
        vec = tokenizer.transform_state(
            state_atoms,
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
        )
    except TypeError:
        vec = tokenizer.transform_state(state_atoms, goal_atoms, objects)

    # Return [1, D]
    return vec.reshape(1, -1)


def embed_state_generic_cached(
    atoms_set,
    tokenizer,
    goal_atoms,
    objects,
    problem_path,
    *,
    wl_prob=None,
):
    """Embed a state while reusing a pre-parsed WL problem when available."""
    try:
        vec = tokenizer.transform_state(
            list(atoms_set),
            goal_atoms,
            objects,
            problem_pddl_path=problem_path,
            _wl_prob=wl_prob,
        )
    except TypeError:
        vec = embed_state_generic(atoms_set, tokenizer, goal_atoms, objects, problem_path)
    return vec.reshape(1, -1)


def embed_state_fsf(atoms_set, encoder, objects, obj_map):
    """
    Helper to convert a set of atoms (strings) into a Numpy Array [1, D] using FSF.
    """
    # Convert set of strings to list of tuples: "(on a b)" -> ("on", "a", "b")
    atom_tuples = []
    for a in atoms_set:
        content = a.replace("(", "").replace(")", "").lower()
        parts = content.split()
        if parts:
            atom_tuples.append(tuple(parts))

    # Use the encoder's internal logic
    vec = encoder._state_to_vector(atom_tuples, objects, obj_map)

    # Return [1, D]
    return vec.reshape(1, -1)


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
    prob_file,
    domain_path,
    prob_path,
    model,
    max_steps,
    steps_per_object,
    delta,
    encoding_type,
    feature_encoder=None,  # Tokenizer or FSFEncoder
    objects=None, # List of strings (objects)
    obj_map=None, # For FSF
    beam_width=3,
    score_metric="native",
    collect_search_stats: bool = False,
):
    """
    Runs Latent Space Search using Beam Search (XGBoost version).
    Supports both Generic Tokenizers and FSF encodings.
    """
    # print(
    #     f"Inference using {'Delta Prediction' if delta else 'State Prediction'} for {prob_file}"
    # )

    search_start_time = None
    if collect_search_stats:
        search_start_time = time.perf_counter()
        search_stats = {
            "beam_expansions": 0,
            "model_calls": 0,
            "successor_evals": 0,
            "outer_steps": 0,
            "terminated_reason": "max_steps",
            "score_metric": score_metric,
            "beam_width": beam_width,
        }

    # 1. Pyperplan Parsing (Ground Truth Physics)
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
    effective_max_steps = max(max_steps, steps_per_object * num_objects)

    if objects is None and encoding_type != "fsf":
        objects = extract_objects_from_problem(prob, dom)

    initial_atoms = task.initial_state
    goal_set = set(task.goals)
    state_cache = {}
    successor_cache = {}
    sorted_operators = sorted(task.operators, key=lambda op: op.name)
    
    # Pre-compute goal atoms list for generic tokenizer
    goal_atoms_list = list(goal_set)
    wl_prob = None

    if encoding_type == "wl":
        from wlplan.planning import parse_problem as wl_parse_problem

        wl_prob = wl_parse_problem(domain_path, prob_path)

    # 2. Embedding Setup (Goal & Init)
    if encoding_type == "fsf":
        # FSF Setup
        encoder = feature_encoder
        # FSF requires problem-specific object mapping
        # objects and obj_map passed in are ignored/recomputed for FSF usually? 
        # The original code recomputed them per problem for FSF.
        fsf_objects = encoder._get_sorted_objects(prob_path)
        fsf_obj_map = encoder._get_object_indices(fsf_objects)

        # Embed Goal
        goal_vec_1d = encoder.embed_goal(prob_path)
        goal_vec = goal_vec_1d.reshape(1, -1)  # [1, D]

        # Embed Init
        init_vec = embed_state_fsf(initial_atoms, encoder, fsf_objects, fsf_obj_map)
        
        # Update these for loop usage
        objects = fsf_objects
        obj_map = fsf_obj_map

        def get_cached_vec(atoms):
            key = frozenset(atoms)
            cached = state_cache.get(key)
            if cached is None:
                cached = embed_state_fsf(atoms, encoder, objects, obj_map)
                state_cache[key] = cached
            return cached
    else:
        # Generic Tokenizer Setup
        tokenizer = feature_encoder

        def get_cached_vec(atoms):
            key = frozenset(atoms)
            cached = state_cache.get(key)
            if cached is None:
                cached = embed_state_generic_cached(
                    atoms,
                    tokenizer,
                    goal_atoms_list,
                    objects,
                    prob_path,
                    wl_prob=wl_prob,
                )
                state_cache[key] = cached
            return cached
        
        # Embed Goal
        try:
            goal_vec_1d = tokenizer.transform_goal(
                goal_atoms_list,
                objects,
                problem_pddl_path=prob_path,
                _wl_prob=wl_prob,
            )
        except TypeError:
            goal_vec_1d = tokenizer.transform_goal(goal_atoms_list, objects)
        goal_vec = goal_vec_1d.reshape(1, -1)

        # Embed Init
        init_vec = get_cached_vec(initial_atoms)

    # 3. Initialize Beam
    # Beam Element: (score, current_vec, atoms, plan, visited_hashes)
    # Note: XGBoost is stateless (no hidden state), unlike LSTM.
    initial_hash = frozenset(initial_atoms)
    
    # Explicitly initialize set to avoid dict confusion
    visited_set = set()
    visited_set.add(initial_hash)
    
    beam = [(0.0, init_vec, initial_atoms, [], visited_set)]

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

        for score, current_vec, current_atoms, plan, visited in beam:
            if collect_search_stats:
                search_stats["beam_expansions"] += 1
            # Check Goal
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

            # A. Predict Next Latent State/Delta
            # Input: Concat [State, Goal] -> [1, 2D]
            model_input = np.hstack([current_vec, goal_vec])

            # Predict
            if collect_search_stats:
                search_stats["model_calls"] += 1
            pred = model.predict(model_input)  # [1, D]

            # Reshape is crucial: XGBoost might return (D,) or (1, D)
            pred = pred.reshape(1, -1)

            if delta:
                pred_next_emb = current_vec + pred
            else:
                pred_next_emb = pred

            # B. Generate and score successors
            successor_items = []
            successor_vecs = []
            for op_name, next_atoms, next_hash in get_successors(current_atoms):
                if collect_search_stats:
                    search_stats["successor_evals"] += 1
                if next_hash in visited:
                    continue

                cand_vec = get_cached_vec(next_atoms)
                successor_items.append((op_name, next_atoms, next_hash, cand_vec))
                successor_vecs.append(cand_vec.reshape(-1))

            if not successor_items:
                continue

            cand_matrix = np.vstack(successor_vecs)
            dists = score_distances(
                pred_next_emb,
                cand_matrix,
                score_metric=score_metric,
                delta=delta,
            )

            for (op_name, next_atoms, next_hash, cand_vec), dist in zip(successor_items, dists.tolist()):
                new_score = score + dist
                new_visited = visited.copy()
                new_visited.add(next_hash)

                candidates.append(
                    (new_score, cand_vec, next_atoms, plan + [op_name], new_visited)
                )

            # D. Prune Beam
            if not candidates:
                if collect_search_stats:
                    search_stats["terminated_reason"] = "dead_end"
                break

            # Stable Sort:
            # Primary Key: Score (float)
            # Secondary Key: String representation of the plan (deterministic tie-breaker)
            candidates.sort(key=lambda x: (x[0], str(x[3])))
            beam = candidates[:beam_width]

    best_attempt = beam[0] if beam else (0, None, None, [], set())

    result = {
        "problem": prob_file,
        "search_solved": False,
        "plan_len": len(best_attempt[3]),
        "plan": best_attempt[3],
        "effective_max_steps": effective_max_steps,
    }
    if collect_search_stats:
        search_stats["search_elapsed_sec"] = time.perf_counter() - search_start_time
        result.update(search_stats)
    return result


def run_inference(args):
    set_seed(args.seed)

    device = resolve_xgb_device(args.device)
    print(f"Using device: {device}")

    # 0. Load Metadata to determine encoding
    model_name = args.model_name or args.domain
    meta_path = os.path.join(args.checkpoint_dir, f"{model_name}_xgb_meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: Metadata not found at {meta_path}. Cannot determine encoding.")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    # Preserve historical file naming while using canonical tokenizer ids internally.
    encoding_type = meta.get("encoding_alias", meta.get("encoding", "graphs"))
    tokenizer_name = meta.get("encoding", encoding_type)
    tokenizer_name = normalize_encoding_name(tokenizer_name)

    # Override delta with what the model was actually trained on
    trained_delta = meta.get("delta", args.delta)
    if trained_delta != args.delta:
        print(f"Warning: Argument --delta={args.delta} but model was trained with delta={trained_delta}. Using model setting.")
    
    print(
        f"Detected Encoding: {encoding_type} (tokenizer: {tokenizer_name}) | Delta Mode: {trained_delta}"
    )

    # 1. Load Encoders
    feature_encoder = None
    
    # Common variables
    domain_pddl = os.path.join(args.pddl_dir, args.domain, "domain.pddl")

    if tokenizer_name == "fsf":
        # Load FSF Config
        config_path = resolve_model_file_path(
            args.data_dir, f"{args.domain}_fsf_config.json"
        )
        if not os.path.exists(config_path):
            print(f"Error: FSF Config not found at {config_path}")
            return

        with open(config_path, "r") as f:
            config = json.load(f)
            max_objects = config["max_objects"]

        feature_encoder = FSFEncoder(args.domain, domain_pddl, max_objects)
        print(f"Initialized FSF Encoder with Max Objects: {max_objects}")

    else:
        # Generic Tokenizer Support (WL, SimHash, etc.)
        # The metadata keeps 'encoding' as the tokenizer name (e.g. 'simhash')
        # We need to recreate it.

        try:
            if args.tokenizer_manifest:
                tokenizer = load_tokenizer_from_manifest(args.tokenizer_manifest)
                print(f"Loaded tokenizer manifest from {args.tokenizer_manifest}")
            else:
                vocab_path = resolve_vocab_path(args.data_dir, args.domain, tokenizer_name)
                tokenizer = create_tokenizer(tokenizer_name)

                if vocab_path and os.path.exists(vocab_path):
                    tokenizer.load_vocabulary(vocab_path)
                    print(f"Loaded {tokenizer_name} vocabulary from {vocab_path}")
                else:
                    print(
                        f"Warning: Vocabulary file not found for '{tokenizer_name}'. Using default params."
                    )

            if isinstance(tokenizer, MultiDomainUnionTokenizer):
                tokenizer.set_active_domain(args.domain, domain_pddl)
            elif hasattr(tokenizer, "set_domain"):
                tokenizer.set_domain(domain_pddl)
                
            feature_encoder = tokenizer
            
        except Exception as e:
            print(f"Failed to initialize tokenizer '{tokenizer_name}': {e}")
            return

    # 2. Load XGBoost
    xgb_path = os.path.join(args.checkpoint_dir, f"{model_name}_xgb.json")
    print(f"Loading XGBoost from {xgb_path}...")

    model = xgb.XGBRegressor(device=device, n_jobs=args.n_jobs)
    model.load_model(xgb_path)

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
                        res = solve_problem(
                            prob_file=prob_file,
                            domain_path=domain_pddl,
                            prob_path=prob_path,
                            model=model,
                            max_steps=args.max_steps,
                            steps_per_object=args.steps_per_object,
                            delta=trained_delta,
                            encoding_type=tokenizer_name,
                            feature_encoder=feature_encoder,
                            objects=None,
                            beam_width=args.beam_width,
                            score_metric=args.score_metric,
                            collect_search_stats=args.collect_search_stats,
                        )

                        if args.skip_validation:
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
                f"{args.domain}_{encoding_type}_{split}{tag_suffix}_results.json",
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
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--model_name",
        default=None,
        help="Optional filename prefix for pooled/shared models",
    )
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
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="XGBoost device target",
    )
    parser.add_argument("--n_jobs", type=int, default=8)
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
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--tag", default="state")
    parser.add_argument("--seed", type=int, default=13)
    HOME = os.path.expanduser("~")
    ROOT_DIR = f"{HOME}/planning/"
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", f"{ROOT_DIR}VAL/bin/Validate"),
        help="Path to VAL binary",
    )

    args = parser.parse_args()
    run_inference(args)


