"""
Unified embedding generation for multiple tokenization strategies.

Generates .npy embedding files in the same format as generate_graph_embeddings.py,
allowing all downstream training and inference scripts to work unchanged.

Usage:
    uv run python -m code.encoding_generation.generate_multi_embeddings \
        --tokenizer simhash --domain blocks
    uv run python -m code.encoding_generation.generate_multi_embeddings \
        --tokenizer random --domain blocks

Output: data/encodings/<tokenizer>/<domain>/<split>/<problem>.npy
        data/encodings/<tokenizer>/<domain>/<split>/<problem>_goal.npy
"""

import argparse
import os
import re
import sys

import numpy as np
import pddl
import pddl.logic.predicates
from tqdm import tqdm

ALL_DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def _extract_goal_atoms(problem) -> list[str]:
    """Extract goal atoms from a pddl Problem object as strings."""
    goals = []

    def visit(node):
        if isinstance(node, pddl.logic.predicates.Predicate):
            args = [t.name if hasattr(t, "name") else str(t) for t in node.terms]
            goals.append(f"({node.name} {' '.join(args)})")
        elif hasattr(node, "operands"):
            for op in node.operands:
                visit(op)
        elif hasattr(node, "_operands"):
            for op in node._operands:
                visit(op)

    visit(problem.goal)
    return goals


def _get_objects(problem, domain) -> list[str]:
    """Get sorted object names from problem + domain constants."""
    objs = set()
    for o in problem.objects:
        objs.add(o.name)
    for o in domain.constants:
        objs.add(o.name)
    return sorted(objs)


def create_tokenizer(name: str, **kwargs):
    """Factory function to create a tokenizer by name."""
    if name == "wl":
        from code.tokenization.wl import WLTokenizer

        return WLTokenizer(iterations=kwargs.get("iterations", 2))
    elif name == "simhash":
        from code.tokenization.simhash import SimHashTokenizer

        return SimHashTokenizer(
            hash_dim=kwargs.get("hash_dim", 128),
            seed=kwargs.get("seed", 42),
        )
    elif name == "shortest_path":
        from code.tokenization.shortest_path import ShortestPathTokenizer

        return ShortestPathTokenizer(
            max_path_length=kwargs.get("max_path_length", 5),
        )
    elif name == "graphbpe":
        from code.tokenization.graphbpe import GraphBPETokenizer

        return GraphBPETokenizer(
            vocab_size=kwargs.get("vocab_size", 1000),
            num_iterations=kwargs.get("num_iterations", 100),
        )
    elif name == "random":
        from code.tokenization.random import RandomTokenizer

        return RandomTokenizer(
            random_dim=kwargs.get("random_dim", 128),
            seed=kwargs.get("seed", 42),
            normalize=kwargs.get("normalize", True),
        )
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings using multiple tokenization strategies."
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        choices=["wl", "simhash", "shortest_path", "graphbpe", "random"],
        help="Tokenization strategy to use.",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--model_dir", default=None, help="Override model save dir")
    parser.add_argument("--domain", type=str, default=None, help="Specific domain")

    # Tokenizer-specific params
    parser.add_argument("--iterations", type=int, default=2, help="WL iterations")
    parser.add_argument("--hash_dim", type=int, default=128, help="SimHash dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SimHash")
    parser.add_argument(
        "--max_path_length", type=int, default=5, help="ShortestPath max length"
    )
    parser.add_argument("--vocab_size", type=int, default=1000, help="GraphBPE vocab")
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="GraphBPE merge iterations"
    )
    parser.add_argument("--random_dim", type=int, default=128, help="Random baseline dimension")
    parser.add_argument(
        "--no_random_normalize",
        action="store_true",
        help="Disable unit-normalization for random embeddings",
    )
    args = parser.parse_args()

    # Set output directories
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "encodings", args.tokenizer)
    if args.model_dir is None:
        args.model_dir = os.path.join(args.data_dir, "encodings", "models")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    domains_to_run = [args.domain] if args.domain else ALL_DOMAINS

    for domain_name in domains_to_run:
        print(f"\n{'='*60}")
        print(f"Domain: {domain_name} | Tokenizer: {args.tokenizer}")
        print(f"{'='*60}")

        domain_pddl = os.path.join(args.data_dir, "pddl", domain_name, "domain.pddl")
        train_states_dir = os.path.join(args.data_dir, "states", domain_name, "train")
        train_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, "train")

        if not os.path.exists(domain_pddl):
            print(f"  [Error] Domain PDDL not found: {domain_pddl}")
            continue

        if not os.path.exists(train_states_dir):
            print(f"  [Error] Training states not found: {train_states_dir}")
            continue

        # ---------- WL uses its own pipeline ----------
        if args.tokenizer == "wl":
            _run_wl_pipeline(args, domain_name, domain_pddl, train_states_dir, train_pddl_dir)
            continue

        # ---------- Generic tokenizer pipeline ----------
        # 1. Create and fit tokenizer
        tokenizer = create_tokenizer(
            args.tokenizer,
            iterations=args.iterations,
            hash_dim=args.hash_dim,
            seed=args.seed,
            max_path_length=args.max_path_length,
            vocab_size=args.vocab_size,
            num_iterations=args.num_iterations,
            random_dim=args.random_dim,
            normalize=(not args.no_random_normalize),
        )

        print(f"  Fitting {args.tokenizer} tokenizer...")
        tokenizer.fit(domain_pddl, train_states_dir, train_pddl_dir)
        print(f"  Embedding dimension: {tokenizer.get_embedding_dim()}")

        # Save vocabulary
        vocab_path = os.path.join(
            args.model_dir, f"{domain_name}_{args.tokenizer}.json"
        )
        tokenizer.save_vocabulary(vocab_path)
        print(f"  Saved vocabulary to {vocab_path}")

        # Parse domain for object/goal extraction
        domain = pddl.parse_domain(domain_pddl)

        # 2. Embed all splits
        for split in SPLITS:
            print(f"  Embedding split: {split}")
            split_state_dir = os.path.join(
                args.data_dir, "states", domain_name, split
            )
            split_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, split)
            split_out_dir = os.path.join(args.output_dir, domain_name, split)
            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                print(f"    Skipping {split} (states dir not found)")
                continue

            traj_files = sorted(
                [f for f in os.listdir(split_state_dir) if f.endswith(".traj")]
            )

            for t_file in tqdm(
                traj_files,
                desc=f"  Embedding {split}",
                disable=(not progress_enabled()),
            ):
                prob_name = t_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, t_file)
                out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    problem = pddl.parse_problem(prob_pddl)
                    objects = _get_objects(problem, domain)
                    goal_atoms = _extract_goal_atoms(problem)

                    # Read trajectory
                    with open(traj_path, "r") as f:
                        lines = f.readlines()

                    # Embed each state
                    state_embeddings = []
                    for line in lines:
                        state_atoms = _PREDICATE_REGEX.findall(line.strip())
                        # Wrap each match in parens to match expected format
                        state_atoms_str = [f"({a})" for a in state_atoms]

                        emb = tokenizer.transform_state(
                            state_atoms_str, goal_atoms, objects
                        )
                        state_embeddings.append(emb)

                    traj_matrix = np.array(state_embeddings, dtype=np.float32)

                    # Embed goal
                    goal_vec = tokenizer.transform_goal(goal_atoms, objects)
                    goal_vec = goal_vec.astype(np.float32)

                    # Save
                    np.save(out_traj_path, traj_matrix)
                    np.save(out_goal_path, goal_vec)

                except Exception as e:
                    print(f"    Error embedding {prob_name}: {e}")

    print("\nDone!")


def _run_wl_pipeline(args, domain_name, domain_pddl, train_states_dir, train_pddl_dir):
    """
    Run the WL pipeline using the WLTokenizer wrapper.

    This produces output identical to generate_graph_embeddings.py but
    going through the tokenizer abstraction.
    """
    from code.tokenization.wl import WLTokenizer

    tokenizer = WLTokenizer(iterations=args.iterations)
    tokenizer.fit(domain_pddl, train_states_dir, train_pddl_dir)
    print(f"  WL Embedding dimension: {tokenizer.get_embedding_dim()}")

    # Save vocabulary
    vocab_path = os.path.join(args.model_dir, f"{domain_name}_wl_tok.json")
    tokenizer.save_vocabulary(vocab_path)
    print(f"  Saved WL vocabulary to {vocab_path}")

    # Use wlplan directly for embedding (consistent with original pipeline)
    from wlplan.data import DomainDataset, ProblemDataset
    from wlplan.planning import Atom, State, parse_domain, parse_problem

    wl_domain = parse_domain(domain_pddl)
    pred_map = {p.name: p for p in wl_domain.predicates}

    def parse_line_to_state(line):
        line = line.strip()
        if not line:
            return State([])
        matches = re.findall(r"\(([\w-]+(?: [\w-]+)*)\)", line)
        atoms = []
        for m in matches:
            parts = m.split()
            if parts[0] in pred_map:
                atoms.append(Atom(pred_map[parts[0]], parts[1:]))
        return State(atoms)

    for split in SPLITS:
        print(f"  Embedding split: {split}")
        split_state_dir = os.path.join(args.data_dir, "states", domain_name, split)
        split_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, split)
        split_out_dir = os.path.join(args.output_dir, domain_name, split)
        os.makedirs(split_out_dir, exist_ok=True)

        if not os.path.exists(split_state_dir):
            continue

        traj_files = sorted(
            [f for f in os.listdir(split_state_dir) if f.endswith(".traj")]
        )

        for t_file in tqdm(
            traj_files,
            desc=f"  Embedding {split}",
            disable=(not progress_enabled()),
        ):
            prob_name = t_file.replace(".traj", "")
            prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
            traj_path = os.path.join(split_state_dir, t_file)
            out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
            out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

            if not os.path.exists(prob_pddl):
                continue

            try:
                wl_prob = parse_problem(domain_pddl, prob_pddl)

                with open(traj_path, "r") as f:
                    lines = f.readlines()

                states = [parse_line_to_state(l) for l in lines]

                # Embed trajectory via wlplan
                mini_ds = DomainDataset(
                    wl_domain, [ProblemDataset(wl_prob, states)]
                )
                embs = tokenizer._feature_gen.embed(mini_ds)
                traj_matrix = np.array(embs, dtype=np.float32)

                # Embed goal
                goal_atoms = list(wl_prob.positive_goals)
                goal_state = State(goal_atoms)
                goal_ds = DomainDataset(
                    wl_domain, [ProblemDataset(wl_prob, [goal_state])]
                )
                goal_embs = tokenizer._feature_gen.embed(goal_ds)
                goal_vec = np.array(goal_embs[0], dtype=np.float32)

                np.save(out_traj_path, traj_matrix)
                np.save(out_goal_path, goal_vec)

            except Exception as e:
                print(f"    Error embedding {prob_name}: {e}")


if __name__ == "__main__":
    main()
