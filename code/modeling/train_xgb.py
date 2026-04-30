import argparse
import json
import os
import pickle
import time
from code.common.utils import set_seed
from code.modeling.dataset import load_flat_dataset_for_xgboost

import xgboost as xgb


def canonical_tokenizer_name(name: str) -> str:
    """Normalize encoding aliases to canonical tokenizer names."""
    return "wl" if name == "graphs" else name


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


def train(args):
    set_seed(args.seed)
    domains = args.domains if args.domains else [args.domain]
    run_name = args.run_name or (args.domain if args.domain else "all_domains")
    print(
        f"Training XGBoost using {'Delta Prediction' if args.delta else 'State Prediction'} with [{args.encoding}] encoding."
    )
    print(f"Training domains: {', '.join(domains)}")
    print(f"Run name: {run_name}")

    # Adjust data directory based on encoding if the user relied on default 'graphs' path
    # If the user explicitly passed a path with 'fsf' in it (via SLURM), this block is skipped.
    if args.encoding == "fsf" and "graphs" in args.data_dir:
        print(f"Switching data_dir from {args.data_dir} to fsf path...")
        args.data_dir = args.data_dir.replace("graphs", "fsf")

    # Construct save directory structure: checkpoints/<encoding>/xgboost_<mode>/
    # (Or rely on the user providing the correct --save_dir from the SLURM script)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading datasets from {args.data_dir}...")

    # Debug: Check if path exists
    train_path_checks = [os.path.join(args.data_dir, domain_name, "train") for domain_name in domains]
    if not any(os.path.exists(path) for path in train_path_checks):
        print(f"CRITICAL ERROR: No training paths exist for domains: {', '.join(domains)}")
        return

    X_train, y_train = load_flat_dataset_for_xgboost(
        args.data_dir, domains, "train", delta=args.delta
    )
    X_val, y_val = load_flat_dataset_for_xgboost(
        args.data_dir, domains, "validation", delta=args.delta
    )

    if X_train is None:
        print(f"Error: No training data found for domains: {', '.join(domains)}.")
        return

    print(f"  Train Data: X={X_train.shape}, y={y_train.shape}")
    if X_val is not None:
        print(f"  Val Data:   X={X_val.shape}, y={y_val.shape}")
    else:
        print("  Val Data:   None (Validation skipped)")

    # 2. Configure XGBoost
    # Check for GPU
    device = resolve_xgb_device(args.device)
    print(f"Training on device: {device}")

    print("Configured hyperparameters:")
    print(f"  Boosting rounds (n_estimators): {args.n_estimators}")
    print(f"  Tree depth (max_depth): {args.max_depth}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early Stopping Rounds: {args.early_stopping}")

    # Determine if we can use early stopping (requires validation data)
    es_rounds = args.early_stopping if X_val is not None else None

    # XGBRegressor automatically handles Multi-Output regression
    # if y is 2D and objective is squarederror.
    # early_stopping_rounds: this ensures that if validation score doesn't improve for N rounds, training stops.
    # The model object will automatically keep the best iteration's weights.
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        tree_method="hist",  # Required for efficient training
        device=device,  # GPU support
        objective="reg:squarederror",
        n_jobs=args.n_jobs,
        random_state=args.seed,
        early_stopping_rounds=es_rounds,
        verbosity=0,
    )

    # 3. Train
    start_time = time.time()

    eval_set = []
    if X_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set if eval_set else None,
        verbose=False,
    )

    duration = time.time() - start_time
    print(f"Training finished in {duration:.2f} seconds.")

    # Check if early stopping was triggered
    if hasattr(model, "best_iteration"):
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best score: {model.best_score}")

    # PRINT PARAMS
    # Get the underlying booster
    booster = model.get_booster()

    # get_dump() returns a list of strings, where each string represents a tree
    # and contains lines representing nodes/leaves.
    trees_dump = booster.get_dump()
    n_trees = len(trees_dump)

    # Count total lines across all tree dumps to get total nodes
    total_nodes = sum(len(t.splitlines()) for t in trees_dump)

    print("-" * 30)
    print("Model Complexity:")
    print(f"  Total Trees: {n_trees}")
    print(f"  Total Nodes: {total_nodes}")
    # In a tree, every split has ~2 params (feature, threshold) and leaf has 1 (weight).
    # Total nodes is a fair approximation of 'trainable parameters'.
    print(f"  Approx. Parameters: {total_nodes}")
    print("-" * 30)

    # 4. Save
    # When early_stopping_rounds is used, save_model saves the trees up to the best iteration
    # (plus the patience window), but metadata marks the best iteration.
    model_path = os.path.join(args.save_dir, f"{run_name}_xgb.json")
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # 5. Save Metadata
    meta = {
        "run_name": run_name,
        "domains": domains,
        "input_dim": X_train.shape[1],
        "output_dim": y_train.shape[1],
        "delta": args.delta,
        # Keep output naming compatibility while storing canonical tokenizer id.
        "encoding": canonical_tokenizer_name(args.encoding),
        "encoding_alias": args.encoding,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "early_stopping": args.early_stopping,
        "seed": args.seed,
        "device": device,
        "best_iteration": getattr(model, "best_iteration", -1),
    }
    meta_pkl_path = os.path.join(args.save_dir, f"{run_name}_xgb_meta.pkl")
    with open(meta_pkl_path, "wb") as f:
        pickle.dump(meta, f)
    meta_json_path = os.path.join(args.save_dir, f"{run_name}_xgb_meta.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None, help="Single training domain")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional list of domains for pooled training",
    )
    parser.add_argument("--save_dir", required=True, help="Directory to save model")
    parser.add_argument("--data_dir", default="data/encodings/graphs")
    parser.add_argument(
        "--encoding",
        required=True,
        help="Encoding strategy used",
    )

    # XGB Hyperparams
    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="XGBoost device target",
    )
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Stop if val loss doesn't improve",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Filename prefix for saved model and metadata",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if not args.domain and not args.domains:
        parser.error("Provide either --domain or --domains.")
    if args.domain and args.domains:
        parser.error("Use either --domain or --domains, not both.")

    train(args)
