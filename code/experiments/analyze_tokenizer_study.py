from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from statistics import mean, pstdev

import numpy as np

from code.tokenization.factory import create_tokenizer
from code.tokenization.multidomain import load_tokenizer_from_manifest

SPLITS = ["validation", "test-interpolation", "test-extrapolation"]
REGIMES = ["domain_dependent", "all_domains"]


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv_rows(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_tokenizer_name(name: str) -> str:
    return "wl" if name == "graphs" else name


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int = 20000) -> float:
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, values.size))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_perm + 1))


def exact_mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n01 = int(np.sum((a == 1) & (b == 0)))
    n10 = int(np.sum((a == 0) & (b == 1)))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return n01, n10, min(1.0, 2.0 * tail)


def holm_adjust_by_group(rows: list[dict], p_key: str = "p_value", group_key: str = "family") -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row[group_key])].append(idx)

    for group, indices in groups.items():
        ordered = sorted(indices, key=lambda idx: float(rows[idx][p_key]))
        m = len(ordered)
        running = 0.0
        for rank, idx in enumerate(ordered, start=1):
            raw = float(rows[idx][p_key])
            adj = (m - rank + 1) * raw
            running = max(running, adj)
            rows[idx]["holm_p_value"] = min(1.0, running)
            rows[idx]["significant_0_05"] = rows[idx]["holm_p_value"] < 0.05
            rows[idx]["holm_family"] = group


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_mean_std_pct(mean_value: float, std_value: float) -> str:
    return f"{100.0 * mean_value:.2f}% +- {100.0 * std_value:.2f}%"


def resolve_domain_vocab_path(model_dir: str, domain: str, tokenizer: str) -> str | None:
    names = [f"{domain}_{tokenizer}.json"]
    if tokenizer == "wl":
        names.extend(
            [
                f"{domain}_wl_tok.json",
                f"{domain}_wl.json",
                f"{domain}_graphs.json",
            ]
        )
    for name in names:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_embedding_stats(run_root: str, manifest: dict) -> tuple[dict, dict]:
    dd_model_dir = os.path.join(run_root, "domain_dependent", "data", "encodings", "models")
    dd_stats = {}
    for tokenizer in manifest["tokenizers"]:
        dd_stats[tokenizer] = {}
        for domain in manifest["domains"]:
            vocab_path = resolve_domain_vocab_path(dd_model_dir, domain, tokenizer)
            if vocab_path is None:
                dd_stats[tokenizer][domain] = {"embedding_dim": None, "fit_strategy": "domain_specific"}
                continue
            tok = create_tokenizer(tokenizer)
            tok.load_vocabulary(vocab_path)
            dd_stats[tokenizer][domain] = {
                "embedding_dim": int(tok.get_embedding_dim()),
                "fit_strategy": "domain_specific",
                "vocab_path": vocab_path,
            }

    ad_stats = {}
    for tokenizer in manifest["tokenizers"]:
        manifest_path = os.path.join(
            run_root,
            "all_domains",
            "tokenizers",
            tokenizer,
            f"all_domains_{tokenizer}.json",
        )
        tok = load_tokenizer_from_manifest(manifest_path)
        tokenizer_manifest_data = read_json(manifest_path)
        ad_stats[tokenizer] = {
            "embedding_dim": int(tok.get_embedding_dim()),
            "fit_strategy": tokenizer_manifest_data.get("fit_strategy", tokenizer_manifest_data.get("representation_type")),
            "manifest_path": manifest_path,
            "representation_type": tokenizer_manifest_data.get("representation_type"),
        }
    return dd_stats, ad_stats


def find_result_file(base_dir: str, domain: str, split: str, mode: str) -> str:
    pattern = os.path.join(base_dir, f"{domain}_*_{split}_{mode}_results.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result file matched: {pattern}")
    return matches[-1]


def summarize_result_file(path: str) -> dict:
    rows = read_json(path)
    total = len(rows)
    solved = sum(1 for row in rows if bool(row.get("solved", False)))
    executable = sum(1 for row in rows if bool(row.get("val_executable", False)))
    lengths = [float(row["plan_len"]) for row in rows if row.get("plan_len") is not None]
    return {
        "num_problems": total,
        "solved_rate": (solved / total) if total else 0.0,
        "exec_rate": (executable / total) if total else 0.0,
        "avg_plan_len": safe_mean(lengths),
    }


def collect_split_rows(run_root: str, manifest: dict, dd_stats: dict, ad_stats: dict) -> list[dict]:
    rows = []
    for regime in REGIMES:
        for seed in manifest["seeds"]:
            for tokenizer in manifest["tokenizers"]:
                for model in manifest["models"]:
                    for mode in manifest["modes"]:
                        base_dir = os.path.join(
                            run_root,
                            regime,
                            "results",
                            f"seed_{seed}",
                            tokenizer,
                            f"{model}_{mode}",
                        )
                        for domain in manifest["domains"]:
                            stats = dd_stats[tokenizer][domain] if regime == "domain_dependent" else ad_stats[tokenizer]
                            for split in SPLITS:
                                result_file = find_result_file(base_dir, domain, split, mode)
                                summary = summarize_result_file(result_file)
                                rows.append(
                                    {
                                        "regime": regime,
                                        "seed": int(seed),
                                        "tokenizer": tokenizer,
                                        "model": model,
                                        "mode": mode,
                                        "domain": domain,
                                        "split": split,
                                        "embedding_dim": stats["embedding_dim"],
                                        "fit_strategy": stats["fit_strategy"],
                                        "result_file": result_file,
                                        **summary,
                                    }
                                )
    return rows


def build_seeded_comparison_rows(split_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for row in split_rows:
        key = (
            row["regime"],
            row["seed"],
            row["domain"],
            row["tokenizer"],
            row["model"],
            row["mode"],
        )
        entry = grouped.setdefault(
            key,
            {
                "Regime": row["regime"],
                "Seed": row["seed"],
                "Domain": row["domain"],
                "Tokenizer": row["tokenizer"],
                "Model": row["model"],
                "Mode": row["mode"],
                "Embedding Dim": row["embedding_dim"],
                "Fit Strategy": row["fit_strategy"],
            },
        )
        prefix = {
            "validation": "Val",
            "test-interpolation": "Interp",
            "test-extrapolation": "Extrap",
        }[row["split"]]
        entry[f"{prefix} Solved"] = format_pct(row["solved_rate"])
        entry[f"{prefix} Exec"] = format_pct(row["exec_rate"])
        entry[f"{prefix} Avg Plan Len"] = f"{row['avg_plan_len']:.2f}"
        entry[f"{prefix} Num Problems"] = row["num_problems"]
    return list(grouped.values())


def build_aggregate_split_rows(split_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in split_rows:
        key = (
            row["regime"],
            row["tokenizer"],
            row["model"],
            row["mode"],
            row["domain"],
            row["split"],
        )
        grouped[key].append(row)

    rows = []
    for key, members in grouped.items():
        rows.append(
            {
                "regime": key[0],
                "tokenizer": key[1],
                "model": key[2],
                "mode": key[3],
                "domain": key[4],
                "split": key[5],
                "embedding_dim": members[0]["embedding_dim"],
                "fit_strategy": members[0]["fit_strategy"],
                "num_seeds": len(members),
                "num_problems": members[0]["num_problems"],
                "solved_rate_mean": safe_mean([m["solved_rate"] for m in members]),
                "solved_rate_std": safe_std([m["solved_rate"] for m in members]),
                "exec_rate_mean": safe_mean([m["exec_rate"] for m in members]),
                "exec_rate_std": safe_std([m["exec_rate"] for m in members]),
                "avg_plan_len_mean": safe_mean([m["avg_plan_len"] for m in members]),
                "avg_plan_len_std": safe_std([m["avg_plan_len"] for m in members]),
            }
        )
    return rows


def build_aggregate_comparison_rows(aggregate_split_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for row in aggregate_split_rows:
        key = (
            row["regime"],
            row["domain"],
            row["tokenizer"],
            row["model"],
            row["mode"],
        )
        entry = grouped.setdefault(
            key,
            {
                "Regime": row["regime"],
                "Domain": row["domain"],
                "Tokenizer": row["tokenizer"],
                "Model": row["model"],
                "Mode": row["mode"],
                "Embedding Dim": row["embedding_dim"],
                "Fit Strategy": row["fit_strategy"],
                "Num Seeds": row["num_seeds"],
            },
        )
        prefix = {
            "validation": "Val",
            "test-interpolation": "Interp",
            "test-extrapolation": "Extrap",
        }[row["split"]]
        entry[f"{prefix} Solved Mean"] = format_pct(row["solved_rate_mean"])
        entry[f"{prefix} Solved Std"] = format_pct(row["solved_rate_std"])
        entry[f"{prefix} Exec Mean"] = format_pct(row["exec_rate_mean"])
        entry[f"{prefix} Exec Std"] = format_pct(row["exec_rate_std"])
        entry[f"{prefix} Avg Plan Len Mean"] = f"{row['avg_plan_len_mean']:.2f}"
        entry[f"{prefix} Avg Plan Len Std"] = f"{row['avg_plan_len_std']:.2f}"
    return list(grouped.values())


def collect_problem_matrix(
    run_root: str,
    regime: str,
    tokenizer: str,
    model: str,
    mode: str,
    seeds: list[int],
    domains: list[str],
    splits: list[str],
    outcome_key: str = "solved",
) -> tuple[list[str], np.ndarray]:
    rows_by_seed = []
    for seed in seeds:
        seed_rows = {}
        base_dir = os.path.join(
            run_root,
            regime,
            "results",
            f"seed_{seed}",
            tokenizer,
            f"{model}_{mode}",
        )
        for domain in domains:
            for split in splits:
                path = find_result_file(base_dir, domain, split, mode)
                for row in read_json(path):
                    problem_id = f"{domain}::{split}::{row['problem']}"
                    seed_rows[problem_id] = float(bool(row.get(outcome_key, False)))
        rows_by_seed.append(seed_rows)

    common_ids = sorted(set.intersection(*(set(seed_rows.keys()) for seed_rows in rows_by_seed)))
    matrix = np.array(
        [[seed_rows[problem_id] for problem_id in common_ids] for seed_rows in rows_by_seed],
        dtype=float,
    )
    return common_ids, matrix


def compare_problem_matrices(
    ids: list[str],
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    mean_a = matrix_a.mean(axis=0)
    mean_b = matrix_b.mean(axis=0)
    diff = mean_b - mean_a
    ci_low, ci_high = bootstrap_mean_ci(diff, rng)

    if matrix_a.shape[0] == 1 and matrix_b.shape[0] == 1:
        n01, n10, p_value = exact_mcnemar_p(matrix_a[0].astype(int), matrix_b[0].astype(int))
        test_name = "Exact McNemar"
    else:
        n01 = int(np.sum(diff < 0))
        n10 = int(np.sum(diff > 0))
        p_value = sign_flip_pvalue(diff, rng)
        test_name = "Paired sign-flip on per-problem seed-mean solved outcomes"

    return {
        "num_problems": len(ids),
        "method_a_mean": float(mean_a.mean()) if mean_a.size else 0.0,
        "method_b_mean": float(mean_b.mean()) if mean_b.size else 0.0,
        "mean_diff_b_minus_a": float(diff.mean()) if diff.size else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "a_better_problem_count": n01,
        "b_better_problem_count": n10,
        "p_value": p_value,
        "test": test_name,
    }


def build_pairwise_tokenizer_rows(run_root: str, manifest: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    overall_rows = []
    by_split_rows = []
    tokenizers = manifest["tokenizers"]

    for regime in REGIMES:
        for model in manifest["models"]:
            for mode in manifest["modes"]:
                family = f"pairwise::{regime}::{model}::{mode}"
                for tok_a, tok_b in combinations(tokenizers, 2):
                    ids, mat_a = collect_problem_matrix(
                        run_root, regime, tok_a, model, mode, manifest["seeds"], manifest["domains"], SPLITS
                    )
                    ids_b, mat_b = collect_problem_matrix(
                        run_root, regime, tok_b, model, mode, manifest["seeds"], manifest["domains"], SPLITS
                    )
                    if ids != ids_b:
                        raise RuntimeError(f"Problem alignment mismatch for {regime}/{model}/{mode}/{tok_a}/{tok_b}")
                    comp = compare_problem_matrices(ids, mat_a, mat_b, rng)
                    overall_rows.append(
                        {
                            "family": family,
                            "regime": regime,
                            "model": model,
                            "mode": mode,
                            "tokenizer_a": tok_a,
                            "tokenizer_b": tok_b,
                            **comp,
                        }
                    )

                    for domain in manifest["domains"]:
                        for split in SPLITS:
                            ids, mat_a = collect_problem_matrix(
                                run_root, regime, tok_a, model, mode, manifest["seeds"], [domain], [split]
                            )
                            ids_b, mat_b = collect_problem_matrix(
                                run_root, regime, tok_b, model, mode, manifest["seeds"], [domain], [split]
                            )
                            if ids != ids_b:
                                raise RuntimeError(
                                    f"Problem alignment mismatch for {regime}/{model}/{mode}/{domain}/{split}/{tok_a}/{tok_b}"
                                )
                            comp = compare_problem_matrices(ids, mat_a, mat_b, rng)
                            by_split_rows.append(
                                {
                                    "family": family,
                                    "regime": regime,
                                    "model": model,
                                    "mode": mode,
                                    "domain": domain,
                                    "split": split,
                                    "tokenizer_a": tok_a,
                                    "tokenizer_b": tok_b,
                                    **comp,
                                }
                            )

    holm_adjust_by_group(overall_rows)
    holm_adjust_by_group(by_split_rows)
    return overall_rows, by_split_rows


def build_regime_comparison_rows(run_root: str, manifest: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    overall_rows = []
    by_split_rows = []
    for tokenizer in manifest["tokenizers"]:
        for model in manifest["models"]:
            for mode in manifest["modes"]:
                family = f"regime::{tokenizer}::{model}::{mode}"
                ids, mat_dd = collect_problem_matrix(
                    run_root, "domain_dependent", tokenizer, model, mode, manifest["seeds"], manifest["domains"], SPLITS
                )
                ids_ad, mat_ad = collect_problem_matrix(
                    run_root, "all_domains", tokenizer, model, mode, manifest["seeds"], manifest["domains"], SPLITS
                )
                if ids != ids_ad:
                    raise RuntimeError(f"Problem alignment mismatch for regime comparison on {tokenizer}/{model}/{mode}")
                comp = compare_problem_matrices(ids, mat_dd, mat_ad, rng)
                overall_rows.append(
                    {
                        "family": family,
                        "tokenizer": tokenizer,
                        "model": model,
                        "mode": mode,
                        "regime_a": "domain_dependent",
                        "regime_b": "all_domains",
                        **comp,
                    }
                )

                for domain in manifest["domains"]:
                    for split in SPLITS:
                        ids, mat_dd = collect_problem_matrix(
                            run_root, "domain_dependent", tokenizer, model, mode, manifest["seeds"], [domain], [split]
                        )
                        ids_ad, mat_ad = collect_problem_matrix(
                            run_root, "all_domains", tokenizer, model, mode, manifest["seeds"], [domain], [split]
                        )
                        if ids != ids_ad:
                            raise RuntimeError(
                                f"Problem alignment mismatch for regime comparison on {tokenizer}/{model}/{mode}/{domain}/{split}"
                            )
                        comp = compare_problem_matrices(ids, mat_dd, mat_ad, rng)
                        by_split_rows.append(
                            {
                                "family": family,
                                "tokenizer": tokenizer,
                                "model": model,
                                "mode": mode,
                                "domain": domain,
                                "split": split,
                                "regime_a": "domain_dependent",
                                "regime_b": "all_domains",
                                **comp,
                            }
                        )

    holm_adjust_by_group(overall_rows)
    holm_adjust_by_group(by_split_rows)
    return overall_rows, by_split_rows


def build_overall_rows(split_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in split_rows:
        key = (row["regime"], row["tokenizer"], row["model"], row["mode"], row["seed"])
        grouped[key].append(row)

    per_seed = []
    for key, members in grouped.items():
        weighted_total = sum(m["num_problems"] for m in members)
        solved_num = sum(m["solved_rate"] * m["num_problems"] for m in members)
        exec_num = sum(m["exec_rate"] * m["num_problems"] for m in members)
        per_seed.append(
            {
                "regime": key[0],
                "tokenizer": key[1],
                "model": key[2],
                "mode": key[3],
                "seed": key[4],
                "weighted_solved_rate": solved_num / weighted_total if weighted_total else 0.0,
                "weighted_exec_rate": exec_num / weighted_total if weighted_total else 0.0,
                "mean_split_solved_rate": safe_mean([m["solved_rate"] for m in members]),
                "mean_split_exec_rate": safe_mean([m["exec_rate"] for m in members]),
            }
        )

    grouped_final: dict[tuple, list[dict]] = defaultdict(list)
    for row in per_seed:
        key = (row["regime"], row["tokenizer"], row["model"], row["mode"])
        grouped_final[key].append(row)

    final = []
    for key, members in grouped_final.items():
        final.append(
            {
                "regime": key[0],
                "tokenizer": key[1],
                "model": key[2],
                "mode": key[3],
                "num_seeds": len(members),
                "weighted_solved_rate_mean": safe_mean([m["weighted_solved_rate"] for m in members]),
                "weighted_solved_rate_std": safe_std([m["weighted_solved_rate"] for m in members]),
                "weighted_exec_rate_mean": safe_mean([m["weighted_exec_rate"] for m in members]),
                "weighted_exec_rate_std": safe_std([m["weighted_exec_rate"] for m in members]),
                "mean_split_solved_rate_mean": safe_mean([m["mean_split_solved_rate"] for m in members]),
                "mean_split_solved_rate_std": safe_std([m["mean_split_solved_rate"] for m in members]),
                "mean_split_exec_rate_mean": safe_mean([m["mean_split_exec_rate"] for m in members]),
                "mean_split_exec_rate_std": safe_std([m["mean_split_exec_rate"] for m in members]),
            }
        )
    return final


def build_best_configuration_rows(overall_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in overall_rows:
        grouped[(row["regime"], row["model"], row["mode"])].append(row)

    best_rows = []
    for key, members in sorted(grouped.items()):
        best = max(
            members,
            key=lambda row: (row["weighted_solved_rate_mean"], row["weighted_exec_rate_mean"]),
        )
        best_rows.append(
            {
                "regime": key[0],
                "model": key[1],
                "mode": key[2],
                "tokenizer": best["tokenizer"],
                "num_seeds": best["num_seeds"],
                "weighted_solved_rate_mean": best["weighted_solved_rate_mean"],
                "weighted_solved_rate_std": best["weighted_solved_rate_std"],
                "weighted_exec_rate_mean": best["weighted_exec_rate_mean"],
                "weighted_exec_rate_std": best["weighted_exec_rate_std"],
            }
        )
    return best_rows


def build_markdown_report(
    run_root: str,
    manifest: dict,
    overall_rows: list[dict],
    regime_rows: list[dict],
    best_rows: list[dict],
) -> None:
    report_path = os.path.join(run_root, "analysis", "study_report.md")
    overall_sorted = sorted(
        overall_rows,
        key=lambda row: row["weighted_solved_rate_mean"],
        reverse=True,
    )

    lines = [
        "# Tokenizer Study Report",
        "",
        "## Scope",
        "",
        f"- Domains: {', '.join(manifest['domains'])}",
        f"- Tokenizers: {', '.join(manifest['tokenizers'])}",
        f"- Models: {', '.join(manifest['models'])}",
        f"- Modes: {', '.join(manifest['modes'])}",
        f"- Seeds: {', '.join(str(seed) for seed in manifest['seeds'])}",
        f"- Device request: {manifest['device']}",
        f"- All-domain tokenizer strategy: {manifest['all_domain_strategy']}",
        f"- Skip VAL validation: {manifest.get('skip_validation', False)}",
        "",
        "## Statistical Protocol",
        "",
        "- Aggregate metrics are reported as means and population standard deviations across seeds.",
        "- Pairwise tokenizer and regime tests use exact McNemar for single-seed comparisons.",
        "- Multi-seed comparisons use a paired sign-flip randomization test on per-problem seed-mean solved outcomes.",
        "- Mean-difference confidence intervals are nonparametric bootstrap 95% intervals over per-problem differences.",
        "- Holm correction is applied within each significance family.",
        "",
        "## Top Configurations",
        "",
        "| Rank | Regime | Tokenizer | Model | Mode | Weighted Solved Mean | Weighted Solved Std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for rank, row in enumerate(overall_sorted[:12], start=1):
        lines.append(
            f"| {rank} | {row['regime']} | {row['tokenizer']} | {row['model']} | {row['mode']} | "
            f"{format_pct(row['weighted_solved_rate_mean'])} | {format_pct(row['weighted_solved_rate_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Tokenizer Per Regime/Model/Mode",
            "",
            "| Regime | Model | Mode | Best Tokenizer | Weighted Solved Mean | Weighted Exec Mean |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in best_rows:
        lines.append(
            f"| {row['regime']} | {row['model']} | {row['mode']} | {row['tokenizer']} | "
            f"{format_pct(row['weighted_solved_rate_mean'])} | {format_pct(row['weighted_exec_rate_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Regime Comparisons",
            "",
            "| Tokenizer | Model | Mode | Domain-Dependent | All-Domains | Diff (all-domains minus domain-dependent) | Raw p | Holm p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in regime_rows:
        lines.append(
            f"| {row['tokenizer']} | {row['model']} | {row['mode']} | "
            f"{format_pct(row['method_a_mean'])} | {format_pct(row['method_b_mean'])} | "
            f"{format_pct(row['mean_diff_b_minus_a'])} | {row['p_value']:.4g} | {row['holm_p_value']:.4g} |"
        )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a fresh tokenizer study run.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--seed", type=int, default=13, help="Analysis RNG seed")
    args = parser.parse_args()

    run_root = args.run_root
    manifest = read_json(os.path.join(run_root, "manifest.json"))
    rng = np.random.default_rng(args.seed)

    dd_stats, ad_stats = load_embedding_stats(run_root, manifest)

    split_rows = collect_split_rows(run_root, manifest, dd_stats, ad_stats)
    aggregate_split_rows = build_aggregate_split_rows(split_rows)
    seeded_comparison_rows = build_seeded_comparison_rows(split_rows)
    aggregate_comparison_rows = build_aggregate_comparison_rows(aggregate_split_rows)
    overall_rows = build_overall_rows(split_rows)
    best_rows = build_best_configuration_rows(overall_rows)
    pairwise_overall_rows, pairwise_by_split_rows = build_pairwise_tokenizer_rows(
        run_root, manifest, rng
    )
    regime_overall_rows, regime_by_split_rows = build_regime_comparison_rows(
        run_root, manifest, rng
    )

    analysis_dir = os.path.join(run_root, "analysis")
    write_csv_rows(os.path.join(analysis_dir, "split_summary.csv"), split_rows)
    write_csv_rows(os.path.join(analysis_dir, "aggregate_split_summary.csv"), aggregate_split_rows)
    write_csv_rows(os.path.join(analysis_dir, "tokenizer_comparison_seeded.csv"), seeded_comparison_rows)
    write_csv_rows(os.path.join(analysis_dir, "tokenizer_comparison_aggregate.csv"), aggregate_comparison_rows)
    write_csv_rows(os.path.join(analysis_dir, "overall_summary.csv"), overall_rows)
    write_csv_rows(
        os.path.join(analysis_dir, "pairwise_tokenizer_significance_overall.csv"),
        pairwise_overall_rows,
    )
    write_csv_rows(
        os.path.join(analysis_dir, "pairwise_tokenizer_significance_by_domain_split.csv"),
        pairwise_by_split_rows,
    )
    write_csv_rows(
        os.path.join(analysis_dir, "regime_significance_overall.csv"),
        regime_overall_rows,
    )
    write_csv_rows(
        os.path.join(analysis_dir, "regime_significance_by_domain_split.csv"),
        regime_by_split_rows,
    )
    write_json(
        os.path.join(analysis_dir, "embedding_stats.json"),
        {"domain_dependent": dd_stats, "all_domains": ad_stats},
    )
    write_csv_rows(
        os.path.join(analysis_dir, "best_config_by_regime_model_mode.csv"),
        best_rows,
    )

    build_markdown_report(run_root, manifest, overall_rows, regime_overall_rows, best_rows)


if __name__ == "__main__":
    main()


