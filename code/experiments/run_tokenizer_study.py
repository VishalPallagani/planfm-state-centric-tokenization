"""
End-to-end tokenizer study runner.

This orchestrates:
- domain-dependent embeddings and transition models
- all-domain tokenizer files and pooled transition models
- seeded LSTM and XGBoost training
- per-problem inference with VAL validation when available
- analysis outputs in one run directory
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from code.experiments.config import DOMAINS, MODEL_CONFIGS, TOKENIZATION_CONFIGS

DEFAULT_MODELS = ["lstm", "xgboost"]
DEFAULT_MODES = ["state", "delta"]
DEFAULT_SEEDS = [13, 23, 37]


def canonical_tokenizer_name(name: str) -> str:
    return "wl" if name == "graphs" else name


def get_tokenizer_config(tokenizer: str) -> tuple[str, dict]:
    canonical = canonical_tokenizer_name(tokenizer)
    if canonical not in TOKENIZATION_CONFIGS:
        valid = sorted(set(TOKENIZATION_CONFIGS) | {"graphs"})
        raise ValueError(
            f"Unknown tokenizer '{tokenizer}'. Valid tokenizers: {', '.join(valid)}"
        )
    return canonical, TOKENIZATION_CONFIGS[canonical]


def save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_git_capture(repo_root: str, *git_args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *git_args],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _xgb_cuda_supported(build_info: dict | None) -> bool | None:
    if not build_info:
        return None

    flag = build_info.get("USE_CUDA")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.lower() in {"1", "true", "on", "yes"}
    return None


def capture_environment(repo_root: str) -> dict:
    packages = ["numpy", "pddl", "pyperplan", "scipy", "torch", "xgboost", "wlplan"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    torch_info = {
        "cuda_available": None,
        "cuda_device_count": None,
        "cuda_devices": [],
        "mps_available": None,
    }
    try:
        import torch

        torch_info = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": [
                torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
            ],
            "mps_available": bool(torch.backends.mps.is_available()),
        }
    except Exception as exc:
        torch_info["error"] = str(exc)

    xgb_build_info = None
    try:
        import xgboost as xgb

        xgb_build_info = xgb.build_info()
    except Exception as exc:
        xgb_build_info = {"error": str(exc)}

    return {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "repo_root": repo_root,
        "package_versions": versions,
        "torch_runtime": torch_info,
        "xgboost_runtime": {
            "build_info": xgb_build_info,
            "cuda_supported": _xgb_cuda_supported(xgb_build_info),
        },
        "git": {
            "head": _safe_git_capture(repo_root, "rev-parse", "HEAD"),
            "branch": _safe_git_capture(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "status_porcelain": _safe_git_capture(repo_root, "status", "--short"),
        },
    }


def validate_requested_runtime(device: str, models: list[str], environment: dict) -> None:
    if device != "cuda":
        return

    if "lstm" in models:
        torch_cuda = environment.get("torch_runtime", {}).get("cuda_available")
        if not torch_cuda:
            raise RuntimeError(
                "CUDA was requested for LSTM experiments, but PyTorch CUDA is not available."
            )

    if "xgboost" in models:
        xgb_cuda = environment.get("xgboost_runtime", {}).get("cuda_supported")
        if not xgb_cuda:
            raise RuntimeError(
                "CUDA was requested for XGBoost experiments, but the installed XGBoost "
                "build does not report CUDA support."
            )


def copy_reproducibility_files(repo_root: str, run_root: str) -> None:
    repro_root = os.path.join(run_root, "repro")
    relpaths = [
        "README.md",
        os.path.join("docs", "REPRODUCIBILITY.md"),
        "environment.yml",
        "pyproject.toml",
        "FILE_CATALOG.md",
        os.path.join("code", "README.md"),
        os.path.join("code", "experiments", "config.py"),
        os.path.join("code", "experiments", "run_tokenizer_study.py"),
        os.path.join("code", "experiments", "analyze_tokenizer_study.py"),
        os.path.join("code", "encoding_generation", "generate_multi_embeddings.py"),
        os.path.join("code", "encoding_generation", "generate_all_domain_embeddings.py"),
        os.path.join("code", "tokenization", "__init__.py"),
        os.path.join("code", "tokenization", "base.py"),
        os.path.join("code", "tokenization", "factory.py"),
        os.path.join("code", "tokenization", "graphbpe.py"),
        os.path.join("code", "tokenization", "multidomain.py"),
        os.path.join("code", "tokenization", "random.py"),
        os.path.join("code", "tokenization", "shortest_path.py"),
        os.path.join("code", "tokenization", "simhash.py"),
        os.path.join("code", "tokenization", "wl.py"),
        os.path.join("code", "modeling", "dataset.py"),
        os.path.join("code", "modeling", "train_lstm.py"),
        os.path.join("code", "modeling", "train_xgb.py"),
        os.path.join("code", "modeling", "inference_lstm.py"),
        os.path.join("code", "modeling", "inference_xgb.py"),
    ]

    for relpath in relpaths:
        src = os.path.join(repo_root, relpath)
        if not os.path.exists(src):
            continue
        dst = os.path.join(repro_root, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def run_command(cmd: list[str], desc: str, cwd: str, log_path: str) -> None:
    stamp = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] {desc}\n")
        log.write(f"CMD: {' '.join(cmd)}\n")

    print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}")
    with open(log_path, "a", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=log,
        )
        log.write(f"[exit_code] {result.returncode}\n")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def resolve_val_path(repo_root: str, user_val_path: str | None) -> str | None:
    if user_val_path:
        return user_val_path

    candidates = [
        os.path.join(repo_root, "VAL", "build", "bin", "Validate.exe"),
        os.path.join(repo_root, "VAL", "build", "bin", "Validate"),
        os.path.join(repo_root, "VAL", "bin", "Validate.exe"),
        os.path.join(repo_root, "VAL", "bin", "Validate"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def build_run_root(
    output_root: str,
    study_name: str | None,
    overwrite: bool,
    resume_existing: bool = False,
) -> str:
    if study_name:
        run_name = study_name
    else:
        run_name = f"tokenizer_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_root = os.path.join(output_root, run_name)
    if os.path.exists(run_root):
        if resume_existing and not overwrite:
            return run_root
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_root}. Use --overwrite to replace it."
            )
        resolved = str(Path(run_root).resolve())
        resolved_output_root = str(Path(output_root).resolve())
        expected_prefix = resolved_output_root + os.sep
        if resolved != resolved_output_root and not resolved.startswith(expected_prefix):
            raise RuntimeError(
                f"Refusing to delete run directory outside output_root: {resolved}"
            )
        shutil.rmtree(run_root)

    os.makedirs(run_root, exist_ok=True)
    return run_root


def tokenizer_cli_params(tokenizer: str) -> list[str]:
    _, config = get_tokenizer_config(tokenizer)
    params = []
    for key, value in config["params"].items():
        params.extend([f"--{key}", str(value)])
    return params


def build_lstm_train_cmd(
    domains: list[str],
    data_dir: str,
    save_dir: str,
    mode: str,
    seed: int,
    args,
    *,
    run_name: str | None,
    encoding: str,
) -> list[str]:
    cfg = MODEL_CONFIGS["lstm"][f"{mode}_mode"]
    epochs = args.lstm_epochs if args.lstm_epochs is not None else cfg["epochs"]
    cmd = [
        sys.executable,
        "-m",
        "code.modeling.train_lstm",
        "--data_dir",
        data_dir,
        "--save_dir",
        save_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(cfg["batch_size"]),
        "--hidden_dim",
        str(cfg["hidden_dim"]),
        "--lr",
        str(cfg["lr"]),
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--encoding",
        encoding,
    ]
    if len(domains) == 1:
        cmd.extend(["--domain", domains[0]])
    else:
        cmd.extend(["--domains", *domains])
    if run_name:
        cmd.extend(["--run_name", run_name])
    if mode == "delta":
        cmd.append("--delta")
    if cfg.get("no_projection"):
        cmd.append("--no_projection")
    if args.lstm_amp:
        cmd.append("--amp")
    else:
        cmd.append("--no_amp")
    if args.fast:
        cmd.append("--fast")
    return cmd


def build_xgb_train_cmd(
    domains: list[str],
    data_dir: str,
    save_dir: str,
    mode: str,
    seed: int,
    args,
    *,
    run_name: str | None,
    encoding: str,
) -> list[str]:
    cfg = MODEL_CONFIGS["xgboost"][f"{mode}_mode"]
    cmd = [
        sys.executable,
        "-m",
        "code.modeling.train_xgb",
        "--data_dir",
        data_dir,
        "--save_dir",
        save_dir,
        "--encoding",
        encoding,
        "--n_estimators",
        str(args.xgb_n_estimators if args.xgb_n_estimators is not None else cfg["n_estimators"]),
        "--max_depth",
        str(args.xgb_max_depth if args.xgb_max_depth is not None else cfg["max_depth"]),
        "--lr",
        str(args.xgb_lr if args.xgb_lr is not None else cfg["lr"]),
        "--early_stopping",
        str(args.xgb_early_stopping if args.xgb_early_stopping is not None else cfg["early_stopping"]),
        "--device",
        args.device if args.device in {"cpu", "cuda"} else "cuda",
        "--n_jobs",
        str(args.xgb_n_jobs),
        "--seed",
        str(seed),
    ]
    if len(domains) == 1:
        cmd.extend(["--domain", domains[0]])
    else:
        cmd.extend(["--domains", *domains])
    if run_name:
        cmd.extend(["--run_name", run_name])
    if mode == "delta":
        cmd.append("--delta")
    return cmd


def build_lstm_infer_cmd(
    domain: str,
    checkpoint: str,
    results_dir: str,
    encoding: str,
    data_dir: str,
    seed: int,
    args,
    *,
    tokenizer_manifest: str | None = None,
) -> list[str]:
    cfg = MODEL_CONFIGS["lstm"][f"{args.current_mode}_mode"]
    cmd = [
        sys.executable,
        "-m",
        "code.modeling.inference_lstm",
        "--domain",
        domain,
        "--checkpoint",
        checkpoint,
        "--data_dir",
        data_dir,
        "--results_dir",
        results_dir,
        "--encoding",
        encoding,
        "--pddl_dir",
        os.path.join(args.source_data_dir, "pddl"),
        "--device",
        args.device,
        "--hidden_dim",
        str(cfg["hidden_dim"]),
        "--tag",
        args.current_mode,
        "--seed",
        str(seed),
        "--steps_per_object",
        str(args.inference_steps_per_object),
        "--validation_workers",
        str(args.validation_workers),
    ]
    if args.current_mode == "delta":
        cmd.append("--delta")
    if cfg.get("no_projection"):
        cmd.append("--no_projection")
    if args.lstm_amp:
        cmd.append("--amp")
    else:
        cmd.append("--no_amp")
    if args.fast:
        cmd.append("--fast")
    if tokenizer_manifest:
        cmd.extend(["--tokenizer_manifest", tokenizer_manifest])
    if args.val_path:
        cmd.extend(["--val_path", args.val_path])
    if args.skip_validation:
        cmd.append("--skip_validation")
    if args.max_problems is not None:
        cmd.extend(["--max_problems", str(args.max_problems)])
    return cmd


def build_xgb_infer_cmd(
    domain: str,
    checkpoint_dir: str,
    results_dir: str,
    data_dir: str,
    seed: int,
    args,
    *,
    tokenizer_manifest: str | None = None,
    model_name: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "code.modeling.inference_xgb",
        "--domain",
        domain,
        "--checkpoint_dir",
        checkpoint_dir,
        "--data_dir",
        data_dir,
        "--results_dir",
        results_dir,
        "--pddl_dir",
        os.path.join(args.source_data_dir, "pddl"),
        "--device",
        args.device if args.device in {"cpu", "cuda"} else "cuda",
        "--n_jobs",
        str(args.xgb_n_jobs),
        "--tag",
        args.current_mode,
        "--seed",
        str(seed),
        "--steps_per_object",
        str(args.inference_steps_per_object),
        "--validation_workers",
        str(args.validation_workers),
    ]
    if args.current_mode == "delta":
        cmd.append("--delta")
    if tokenizer_manifest:
        cmd.extend(["--tokenizer_manifest", tokenizer_manifest])
    if model_name:
        cmd.extend(["--model_name", model_name])
    if args.val_path:
        cmd.extend(["--val_path", args.val_path])
    if args.skip_validation:
        cmd.append("--skip_validation")
    if args.max_problems is not None:
        cmd.extend(["--max_problems", str(args.max_problems)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tokenizer study.")
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Reuse an existing study directory instead of deleting it. Useful for resuming inference/analysis-only runs.",
    )
    parser.add_argument(
        "--source_data_dir",
        default=None,
        help="Optional source data root containing pddl/ and states/.",
    )
    parser.add_argument("--tokenizers", nargs="+", default=list(TOKENIZATION_CONFIGS.keys()))
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="cuda",
        help="Preferred training/inference device. Use 'cuda' for full GPU runs.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--xgb_n_jobs", type=int, default=8)
    parser.add_argument("--lstm_amp", dest="lstm_amp", action="store_true")
    parser.add_argument("--no_lstm_amp", dest="lstm_amp", action="store_false")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--lstm_epochs", type=int, default=None)
    parser.add_argument("--xgb_n_estimators", type=int, default=None)
    parser.add_argument("--xgb_max_depth", type=int, default=None)
    parser.add_argument("--xgb_lr", type=float, default=None)
    parser.add_argument("--xgb_early_stopping", type=int, default=None)
    parser.add_argument(
        "--all_domain_strategy",
        choices=["auto", "pooled", "union"],
        default="auto",
        help="Tokenizer fit strategy for the all-domain regime.",
    )
    parser.add_argument("--val_path", default=None)
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip VAL plan validation. Use only for local debugging when validator-approved solved rates are not needed.",
    )
    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--analysis_seed", type=int, default=13)
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Optional cap for bounded local debugging before a full run.",
    )
    parser.add_argument(
        "--inference_steps_per_object",
        type=int,
        default=10,
        help="Inference search budget scales to max(max_steps, inference_steps_per_object * num_objects).",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=8,
        help="Number of background VAL validation workers to overlap with search.",
    )
    parser.set_defaults(lstm_amp=True)
    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parents[2])
    args.output_root = (
        args.output_root
        if os.path.isabs(args.output_root)
        else os.path.join(repo_root, args.output_root)
    )
    args.source_data_dir = (
        str(Path(args.source_data_dir).resolve())
        if args.source_data_dir
        else os.path.join(repo_root, "data")
    )
    args.val_path = resolve_val_path(repo_root, args.val_path)
    environment = capture_environment(repo_root)
    validate_requested_runtime(args.device, args.models, environment)
    if not args.skip_validation and args.val_path is None:
        raise RuntimeError(
            "VAL was not found. Build/provide VAL for validator-backed runs, or use "
            "--skip_validation only for local debugging."
        )

    run_root = build_run_root(args.output_root, args.study_name, args.overwrite, args.resume_existing)
    copy_reproducibility_files(repo_root, run_root)
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    command_log = os.path.join(logs_dir, "commands.log")

    manifest = {
        "study_name": os.path.basename(run_root),
        "run_root": run_root,
        "repo_root": repo_root,
        "source_data_dir": args.source_data_dir,
        "resume_existing": args.resume_existing,
        "domains": args.domains,
        "tokenizers": [canonical_tokenizer_name(tok) for tok in args.tokenizers],
        "models": args.models,
        "modes": args.modes,
        "seeds": args.seeds,
        "device": args.device,
        "val_path": args.val_path,
        "skip_validation": args.skip_validation,
        "all_domain_strategy": args.all_domain_strategy,
        "analysis_seed": args.analysis_seed,
        "inference_steps_per_object": args.inference_steps_per_object,
        "validation_workers": args.validation_workers,
        "max_problems": args.max_problems,
        "argv": sys.argv,
        "environment": environment,
    }
    save_json(os.path.join(run_root, "manifest.json"), manifest)

    dd_root = os.path.join(run_root, "domain_dependent")
    ad_root = os.path.join(run_root, "all_domains")

    if not args.skip_embeddings:
        for tokenizer in args.tokenizers:
            canonical, tok_cfg = get_tokenizer_config(tokenizer)
            enc_dir = tok_cfg["encoding_dir"]
            for domain in args.domains:
                cmd = [
                    sys.executable,
                    "-m",
                    "code.encoding_generation.generate_multi_embeddings",
                    "--tokenizer",
                    canonical,
                    "--domain",
                    domain,
                    "--data_dir",
                    args.source_data_dir,
                    "--output_dir",
                    os.path.join(dd_root, "data", "encodings", enc_dir),
                    "--model_dir",
                    os.path.join(dd_root, "data", "encodings", "models"),
                ]
                cmd.extend(tokenizer_cli_params(canonical))
                run_command(
                    cmd,
                    f"Domain-dependent embeddings: tokenizer={canonical}, domain={domain}",
                    repo_root,
                    command_log,
                )

        for tokenizer in args.tokenizers:
            canonical, _ = get_tokenizer_config(tokenizer)
            cmd = [
                sys.executable,
                "-m",
                "code.encoding_generation.generate_all_domain_embeddings",
                "--tokenizer",
                canonical,
                "--domains",
                *args.domains,
                "--data_dir",
                args.source_data_dir,
                "--output_dir",
                os.path.join(ad_root, "data", "encodings", canonical),
                "--model_dir",
                os.path.join(ad_root, "tokenizers", canonical),
                "--strategy",
                args.all_domain_strategy,
            ]
            cmd.extend(tokenizer_cli_params(canonical))
            run_command(
                cmd,
                f"All-domain embeddings: tokenizer={canonical}, strategy={args.all_domain_strategy}",
                repo_root,
                command_log,
            )

    if not args.skip_training:
        for seed in args.seeds:
            for tokenizer in args.tokenizers:
                canonical, tok_cfg = get_tokenizer_config(tokenizer)
                enc_dir = tok_cfg["encoding_dir"]

                for domain in args.domains:
                    for model in args.models:
                        for mode in args.modes:
                            if model == "lstm":
                                cmd = build_lstm_train_cmd(
                                    domains=[domain],
                                    data_dir=os.path.join(dd_root, "data", "encodings", enc_dir),
                                    save_dir=os.path.join(
                                        dd_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                        domain,
                                    ),
                                    mode=mode,
                                    seed=seed,
                                    args=args,
                                    run_name=domain,
                                    encoding=canonical,
                                )
                            else:
                                cmd = build_xgb_train_cmd(
                                    domains=[domain],
                                    data_dir=os.path.join(dd_root, "data", "encodings", enc_dir),
                                    save_dir=os.path.join(
                                        dd_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                        domain,
                                    ),
                                    mode=mode,
                                    seed=seed,
                                    args=args,
                                    run_name=domain,
                                    encoding=enc_dir,
                                )
                            run_command(
                                cmd,
                                f"Train domain-dependent: seed={seed}, tokenizer={canonical}, domain={domain}, model={model}, mode={mode}",
                                repo_root,
                                command_log,
                            )

                for model in args.models:
                    for mode in args.modes:
                        if model == "lstm":
                            cmd = build_lstm_train_cmd(
                                domains=args.domains,
                                data_dir=os.path.join(ad_root, "data", "encodings", canonical),
                                save_dir=os.path.join(
                                    ad_root,
                                    "checkpoints",
                                    f"seed_{seed}",
                                    canonical,
                                    f"{model}_{mode}",
                                ),
                                mode=mode,
                                seed=seed,
                                args=args,
                                run_name="all_domains",
                                encoding=canonical,
                            )
                        else:
                            cmd = build_xgb_train_cmd(
                                domains=args.domains,
                                data_dir=os.path.join(ad_root, "data", "encodings", canonical),
                                save_dir=os.path.join(
                                    ad_root,
                                    "checkpoints",
                                    f"seed_{seed}",
                                    canonical,
                                    f"{model}_{mode}",
                                ),
                                mode=mode,
                                seed=seed,
                                args=args,
                                run_name="all_domains",
                                encoding=canonical,
                            )
                        run_command(
                            cmd,
                            f"Train all-domains: seed={seed}, tokenizer={canonical}, model={model}, mode={mode}",
                            repo_root,
                            command_log,
                        )

    if not args.skip_inference:
        for seed in args.seeds:
            for tokenizer in args.tokenizers:
                canonical, tok_cfg = get_tokenizer_config(tokenizer)
                enc_dir = tok_cfg["encoding_dir"]
                pooled_manifest = os.path.join(
                    ad_root,
                    "tokenizers",
                    canonical,
                    f"all_domains_{canonical}.json",
                )

                for model in args.models:
                    for mode in args.modes:
                        args.current_mode = mode
                        for domain in args.domains:
                            if model == "lstm":
                                cmd = build_lstm_infer_cmd(
                                    domain=domain,
                                    checkpoint=os.path.join(
                                        dd_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                        domain,
                                        f"{domain}_lstm_best.pt",
                                    ),
                                    results_dir=os.path.join(
                                        dd_root,
                                        "results",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                    ),
                                    encoding=enc_dir,
                                    data_dir=os.path.join(dd_root, "data", "encodings", enc_dir),
                                    seed=seed,
                                    args=args,
                                )
                            else:
                                cmd = build_xgb_infer_cmd(
                                    domain=domain,
                                    checkpoint_dir=os.path.join(
                                        dd_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                        domain,
                                    ),
                                    results_dir=os.path.join(
                                        dd_root,
                                        "results",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                    ),
                                    data_dir=os.path.join(dd_root, "data", "encodings", enc_dir),
                                    seed=seed,
                                    args=args,
                                )
                            run_command(
                                cmd,
                                f"Infer domain-dependent: seed={seed}, tokenizer={canonical}, domain={domain}, model={model}, mode={mode}",
                                repo_root,
                                command_log,
                            )

                            if model == "lstm":
                                cmd = build_lstm_infer_cmd(
                                    domain=domain,
                                    checkpoint=os.path.join(
                                        ad_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                        "all_domains_lstm_best.pt",
                                    ),
                                    results_dir=os.path.join(
                                        ad_root,
                                        "results",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                    ),
                                    encoding=canonical,
                                    data_dir=os.path.join(ad_root, "data", "encodings", canonical),
                                    seed=seed,
                                    args=args,
                                    tokenizer_manifest=pooled_manifest,
                                )
                            else:
                                cmd = build_xgb_infer_cmd(
                                    domain=domain,
                                    checkpoint_dir=os.path.join(
                                        ad_root,
                                        "checkpoints",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                    ),
                                    results_dir=os.path.join(
                                        ad_root,
                                        "results",
                                        f"seed_{seed}",
                                        canonical,
                                        f"{model}_{mode}",
                                    ),
                                    data_dir=os.path.join(ad_root, "data", "encodings", canonical),
                                    seed=seed,
                                    args=args,
                                    tokenizer_manifest=pooled_manifest,
                                    model_name="all_domains",
                                )
                            run_command(
                                cmd,
                                f"Infer all-domains: seed={seed}, tokenizer={canonical}, domain={domain}, model={model}, mode={mode}",
                                repo_root,
                                command_log,
                            )

    if not args.skip_analysis:
        cmd = [
            sys.executable,
            "-m",
            "code.experiments.analyze_tokenizer_study",
            "--run_root",
            run_root,
            "--seed",
            str(args.analysis_seed),
        ]
        run_command(cmd, "Tokenizer study analysis", repo_root, command_log)

    print(f"\nStudy complete. Outputs written to: {run_root}")


if __name__ == "__main__":
    main()

