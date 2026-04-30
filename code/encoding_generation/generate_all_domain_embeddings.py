"""
Generate embeddings for one all-domain tokenizer model.

This script creates one shared tokenizer model for the requested
domains, then embeds every split for every domain into a fixed space
compatible with a single pooled downstream model.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pddl
import pddl.logic.predicates
from tqdm import tqdm

from code.tokenization.multidomain import (
    MultiDomainUnionTokenizer,
    build_all_domain_tokenizer,
    build_domain_specs,
    save_tokenizer_manifest,
)

SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]
_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def _extract_goal_atoms(problem) -> list[str]:
    goals: list[str] = []

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
    objs = {o.name for o in problem.objects}
    objs.update(o.name for o in domain.constants)
    return sorted(objs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings with one all-domain tokenizer model."
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        choices=["wl", "simhash", "shortest_path", "graphbpe", "random"],
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="Domains included in the shared tokenizer fit.",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--strategy",
        choices=["auto", "pooled", "union"],
        default="auto",
        help="All-domain tokenizer fit strategy.",
    )

    parser.add_argument("--iterations", type=int, default=2, help="WL iterations")
    parser.add_argument("--hash_dim", type=int, default=128, help="SimHash dimension")
    parser.add_argument("--seed", type=int, default=42, help="Tokenizer seed")
    parser.add_argument("--max_path_length", type=int, default=5)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--random_dim", type=int, default=128)
    parser.add_argument(
        "--no_random_normalize",
        action="store_true",
        help="Disable unit normalization for the random baseline",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    tokenizer_kwargs = {
        "iterations": args.iterations,
        "hash_dim": args.hash_dim,
        "seed": args.seed,
        "max_path_length": args.max_path_length,
        "vocab_size": args.vocab_size,
        "num_iterations": args.num_iterations,
        "random_dim": args.random_dim,
        "normalize": (not args.no_random_normalize),
    }

    domain_specs = build_domain_specs(args.data_dir, args.domains)
    tokenizer, resolved_strategy = build_all_domain_tokenizer(
        args.tokenizer,
        domain_specs,
        strategy=args.strategy,
        **tokenizer_kwargs,
    )

    manifest_path = os.path.join(
        args.model_dir,
        f"all_domains_{args.tokenizer}.json",
    )
    save_tokenizer_manifest(
        tokenizer=tokenizer,
        manifest_path=manifest_path,
        tokenizer_name=args.tokenizer,
        domains=args.domains,
        fit_strategy=resolved_strategy,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    print(f"Saved all-domain tokenizer manifest to {manifest_path}")
    print(
        f"Resolved strategy: {resolved_strategy} | "
        f"Embedding dimension: {tokenizer.get_embedding_dim()}"
    )

    for domain_name in args.domains:
        domain_pddl = os.path.join(args.data_dir, "pddl", domain_name, "domain.pddl")
        domain = pddl.parse_domain(domain_pddl)

        if isinstance(tokenizer, MultiDomainUnionTokenizer):
            tokenizer.set_active_domain(domain_name, domain_pddl)

        print(f"\n{'=' * 60}")
        print(f"Embedding domain {domain_name} with shared tokenizer [{args.tokenizer}]")
        print(f"{'=' * 60}")

        for split in SPLITS:
            split_state_dir = os.path.join(args.data_dir, "states", domain_name, split)
            split_pddl_dir = os.path.join(args.data_dir, "pddl", domain_name, split)
            split_out_dir = os.path.join(args.output_dir, domain_name, split)
            os.makedirs(split_out_dir, exist_ok=True)

            if not os.path.exists(split_state_dir):
                print(f"Skipping {domain_name}/{split}: state dir not found")
                continue

            traj_files = sorted(
                file_name
                for file_name in os.listdir(split_state_dir)
                if file_name.endswith(".traj")
            )

            for traj_file in tqdm(
                traj_files,
                desc=f"{domain_name}/{split}",
                disable=(not progress_enabled()),
            ):
                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(split_pddl_dir, f"{prob_name}.pddl")
                traj_path = os.path.join(split_state_dir, traj_file)
                out_traj_path = os.path.join(split_out_dir, f"{prob_name}.npy")
                out_goal_path = os.path.join(split_out_dir, f"{prob_name}_goal.npy")

                if not os.path.exists(prob_pddl):
                    continue

                try:
                    problem = pddl.parse_problem(prob_pddl)
                    objects = _get_objects(problem, domain)
                    goal_atoms = _extract_goal_atoms(problem)

                    with open(traj_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    state_embeddings = []
                    for line in lines:
                        state_atoms = [f"({a})" for a in _PREDICATE_REGEX.findall(line.strip())]
                        try:
                            emb = tokenizer.transform_state(
                                state_atoms,
                                goal_atoms,
                                objects,
                                problem_pddl_path=prob_pddl,
                            )
                        except TypeError:
                            emb = tokenizer.transform_state(state_atoms, goal_atoms, objects)
                        state_embeddings.append(emb)

                    traj_matrix = np.array(state_embeddings, dtype=np.float32)

                    try:
                        goal_vec = tokenizer.transform_goal(
                            goal_atoms,
                            objects,
                            problem_pddl_path=prob_pddl,
                        )
                    except TypeError:
                        goal_vec = tokenizer.transform_goal(goal_atoms, objects)
                    goal_vec = goal_vec.astype(np.float32)

                    np.save(out_traj_path, traj_matrix)
                    np.save(out_goal_path, goal_vec)
                except Exception as exc:
                    print(f"Error embedding {domain_name}/{split}/{prob_name}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()

