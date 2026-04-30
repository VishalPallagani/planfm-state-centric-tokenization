from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean


DOMAIN_ORDER = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLIT_ORDER = ["test-interpolation", "test-extrapolation"]

GOAL_REACHED_RE = re.compile(r"Goal reached", re.IGNORECASE)
NODES_RE = re.compile(r"(\d+)\s+Nodes expanded", re.IGNORECASE)
PLAN_LEN_RE = re.compile(r"Plan length:\s*([0-9]+)", re.IGNORECASE)
SEARCH_TIME_RE = re.compile(r"Search time:\s*([0-9.]+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual classical-planning baselines with pyperplan.")
    parser.add_argument("--pddl_root", default="data/pddl")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["gbf:hff", "astar:lmcut"],
        help="Baseline specs in search:heuristic form.",
    )
    parser.add_argument("--timeout_sec", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_problems_per_split", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/baselines",
    )
    return parser.parse_args()


def parse_output(stdout: str) -> dict[str, object]:
    solved = bool(GOAL_REACHED_RE.search(stdout))
    nodes_match = NODES_RE.search(stdout)
    plan_match = PLAN_LEN_RE.search(stdout)
    time_match = SEARCH_TIME_RE.search(stdout)
    return {
        "solved": solved,
        "nodes_expanded": int(nodes_match.group(1)) if nodes_match else None,
        "plan_len": int(plan_match.group(1)) if plan_match else None,
        "search_time_sec": float(time_match.group(1)) if time_match else None,
    }


def run_one(
    *,
    python_exe: str,
    domain: str,
    split: str,
    problem_name: str,
    search: str,
    heuristic: str,
    pddl_root: Path,
    timeout_sec: int,
) -> dict[str, object]:
    domain_pddl = pddl_root / domain / "domain.pddl"
    problem_pddl = pddl_root / domain / split / problem_name
    cmd = [
        python_exe,
        "-m",
        "pyperplan",
        "-H",
        heuristic,
        "-s",
        search,
        str(domain_pddl),
        str(problem_pddl),
    ]
    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        parsed = parse_output(result.stdout + "\n" + result.stderr)
        status = "solved" if parsed["solved"] else ("failed" if result.returncode == 0 else "error")
        return {
            "baseline": f"{search}:{heuristic}",
            "domain": domain,
            "split": split,
            "problem": problem_name,
            "status": status,
            "returncode": result.returncode,
            "timeout_sec": timeout_sec,
            **parsed,
        }
    except subprocess.TimeoutExpired:
        return {
            "baseline": f"{search}:{heuristic}",
            "domain": domain,
            "split": split,
            "problem": problem_name,
            "status": "timeout",
            "returncode": None,
            "timeout_sec": timeout_sec,
            "solved": False,
            "nodes_expanded": None,
            "plan_len": None,
            "search_time_sec": None,
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["baseline"]), str(row["domain"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (baseline, domain, split), members in sorted(grouped.items()):
        solved_rows = [row for row in members if row["status"] == "solved"]
        out.append(
            {
                "baseline": baseline,
                "domain": domain,
                "split": split,
                "num_problems": len(members),
                "solved_rate": sum(bool(row["solved"]) for row in members) / len(members),
                "timeout_rate": sum(row["status"] == "timeout" for row in members) / len(members),
                "mean_plan_len_solved": mean([row["plan_len"] for row in solved_rows]) if solved_rows else None,
                "mean_nodes_expanded_solved": mean([row["nodes_expanded"] for row in solved_rows if row["nodes_expanded"] is not None])
                if solved_rows
                else None,
                "mean_search_time_solved": mean([row["search_time_sec"] for row in solved_rows if row["search_time_sec"] is not None])
                if solved_rows
                else None,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    pddl_root = Path(args.pddl_root)
    output_dir = Path(args.output_dir)
    baselines: list[tuple[str, str]] = [tuple(spec.split(":", 1)) for spec in args.baselines]

    cases: list[dict[str, object]] = []
    for domain in DOMAIN_ORDER:
        for split in SPLIT_ORDER:
            problem_dir = pddl_root / domain / split
            problems = sorted(path.name for path in problem_dir.glob("*.pddl"))
            if args.max_problems_per_split is not None:
                problems = problems[: args.max_problems_per_split]
            for search, heuristic in baselines:
                for problem_name in problems:
                    cases.append(
                        {
                            "python_exe": sys.executable,
                            "domain": domain,
                            "split": split,
                            "problem_name": problem_name,
                            "search": search,
                            "heuristic": heuristic,
                            "pddl_root": pddl_root,
                            "timeout_sec": args.timeout_sec,
                        }
                    )

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, **case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: (row["baseline"], row["domain"], row["split"], row["problem"]))
    summary_rows = aggregate(rows)
    write_csv(output_dir / "classical_baseline_raw.csv", rows)
    write_csv(output_dir / "classical_baseline_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
