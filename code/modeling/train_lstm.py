import argparse
import os
import sys
import time
from code.common.utils import set_seed, worker_init_fn
from code.modeling.dataset import PlanningTrajectoryDataset, collate_trajectories
from code.modeling.models import StateCentricLSTM, StateCentricLSTM_Delta

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def evaluate(model, val_loader, device, delta, non_blocking=False, use_amp=False):
    """
    Computes Cosine loss on the validation set.
    The `delta` param
    """
    model.eval()
    total_loss = 0
    count = 0
    print(f"Evaluation using {'Delta MSE Loss' if delta else 'Cosine Loss'}")
    if delta:
        criterion = MSELoss(reduction="none")  # We will mask it manually

    with torch.no_grad():
        for states, goals, lengths in val_loader:
            # Need at least 2 states to predict next state
            valid_mask = lengths > 1
            if not valid_mask.any():
                continue

            states = states[valid_mask].to(device, non_blocking=non_blocking)
            goals = goals[valid_mask].to(device, non_blocking=non_blocking)
            lengths = lengths[valid_mask].to(device, non_blocking=non_blocking)

            # Input: S_0 ... S_{T-1}
            input_states = states[:, :-1, :]

            # Target State: S_1 ... S_T
            target_states = states[:, 1:, :]

            if delta:
                # Target Delta: (S_{t+1} - S_t)
                target_deltas = target_states - input_states

            input_lengths = lengths - 1

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                preds, _ = model(input_states, goals, input_lengths)

            # Create Boolean Mask [B, T-1]
            mask = (
                torch.arange(input_states.size(1), device=device)[None, :]
                < input_lengths[:, None]
            )

            # Flatten using the mask to get only valid steps
            # This avoids issues with CosineSimilarity on zero-padded vectors
            active_preds = preds[mask]

            if not delta:
                active_targets = target_states[mask]

                # Cosine Loss: 1 - CosineSimilarity
                loss = (
                    1.0
                    - F.cosine_similarity(active_preds, active_targets, dim=-1).mean()
                )
            else:
                active_targets = target_deltas[mask]

                # MSE Loss on Deltas
                loss = criterion(active_preds, active_targets).mean()

            total_loss += loss.item()
            count += 1

    if count == 0:
        print("No valid trajectories in validation set. Returning 0 loss.")
        return 0.0
    return total_loss / count


def train(args):
    set_seed(args.seed)
    use_proj_str = "Enabled" if not args.no_projection else "Disabled"
    print(f"Training using {'Delta Prediction' if args.delta else 'State Prediction'}")
    print(f"Projection Layer: {use_proj_str}")
    domains = args.domains if args.domains else [args.domain]
    run_name = args.run_name or (args.domain if args.domain else "all_domains")
    print(f"Training domains: {', '.join(domains)}")
    print(f"Run name: {run_name}")

    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    pin_memory = bool(args.pin_memory and device.type == "cuda")
    num_workers = max(0, args.num_workers)
    non_blocking = pin_memory

    if args.fast and device.type == "cuda":
        # Fast path: favor throughput over strict determinism.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"DataLoader workers: {num_workers} | pin_memory: {pin_memory}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    print("Loading datasets...")
    train_ds = PlanningTrajectoryDataset(args.data_dir, domains, "train")
    val_ds = PlanningTrajectoryDataset(args.data_dir, domains, "validation")
    print(f"  Train Trajectories: {len(train_ds)} | Val Trajectories: {len(val_ds)}")

    if len(train_ds) == 0:
        print(f"Error: No training data found for {', '.join(domains)}. Skipping.")
        return

    # Use worker_init_fn and a generator
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_trajectories,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # Determine input dimension safely
    input_dim = 0
    # Check first few items
    for i in range(min(10, len(train_ds))):
        sample_traj, _ = train_ds[i]
        if sample_traj.dim() > 1:
            input_dim = sample_traj.shape[1]
            break

    if input_dim == 0:
        # Fallback
        sample_traj, _ = train_ds[0]
        input_dim = sample_traj.shape[-1]

    print(f"Feature Dimension: {input_dim}")

    # 2. Model
    use_projection = not args.no_projection

    if args.delta:
        model = StateCentricLSTM_Delta(
            input_dim, hidden_dim=args.hidden_dim, use_projection=use_projection
        ).to(device)
    else:
        model = StateCentricLSTM(
            input_dim, hidden_dim=args.hidden_dim, use_projection=use_projection
        ).to(device)

    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if args.delta:
        criterion = MSELoss(reduction="none")

    # Logging
    log_file = os.path.join(args.save_dir, f"{run_name}_training_log.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.save_dir, f"{run_name}_lstm_best.pt")

    meta_path = os.path.join(args.save_dir, f"{run_name}_lstm_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "run_name": run_name,
                "domains": domains,
                "model": "lstm",
                "mode": "delta" if args.delta else "state",
                "encoding": args.encoding,
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "no_projection": args.no_projection,
                "amp": use_amp,
            },
            f,
            indent=2,
        )

    print(f"Starting training on {device}")
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0
            count = 0

            # Training Loop
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                leave=False,
                disable=(not progress_enabled()),
            )
            for states, goals, lengths in pbar:
                # Filter T=1
                valid_mask = lengths > 1
                if not valid_mask.any():
                    continue

                states = states[valid_mask].to(device, non_blocking=non_blocking)
                goals = goals[valid_mask].to(device, non_blocking=non_blocking)
                lengths = lengths[valid_mask].to(device, non_blocking=non_blocking)

                # Prepare Inputs and Targets
                # Input: S_0 ... S_{T-1}
                # Target: S_1 ... S_T

                # We need to slice the padded sequences based on lengths
                # But simpler: just slice everything and mask loss later

                # Input sequence: remove last step
                # Input: S_0 ... S_{T-1}
                input_states = states[:, :-1, :]
                target_states = states[:, 1:, :]

                if args.delta:
                    target_deltas = target_states - input_states

                # Adjust lengths for the sliced sequence
                input_lengths = lengths - 1

                # Forward
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    preds, _ = model(input_states, goals, input_lengths)

                    # Masking padding for Loss
                    # Create a mask [B, T-1, D]
                    mask = (
                        torch.arange(input_states.size(1), device=device)[None, :]
                        < input_lengths[:, None]
                    )

                    # Flatten for loss calculation
                    # preds: [B, T, D] -> [N, D]
                    # targets: [B, T, D] -> [N, D]
                    active_preds = preds[mask]

                    if not args.delta:
                        # We predict the State directly
                        active_targets = target_states[mask]

                        # Cosine Embedding Loss
                        # We want preds and targets to point in the same direction (target=1)
                        # Loss = 1 - cos_sim(x, y)
                        loss = (
                            1.0
                            - F.cosine_similarity(active_preds, active_targets, dim=-1).mean()
                        )

                    else:
                        active_targets = target_deltas[mask]

                        # Loss: MSE between Predicted Delta and Actual Delta
                        loss = criterion(active_preds, active_targets).mean()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                count += 1
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / count if count > 0 else 0

            # Validation Loop
            avg_val_loss = evaluate(
                model,
                val_loader,
                device,
                args.delta,
                non_blocking=non_blocking,
                use_amp=use_amp,
            )

            print(
                f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.6f} | Val Loss {avg_val_loss:.6f}"
            )

            # Log
            with open(log_file, "a") as f:
                f.write(f"{epoch + 1},{avg_train_loss},{avg_val_loss}\n")

            # Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    best_checkpoint_path,
                )
                print(
                    f"  -> Updated best checkpoint (overwrites same file): "
                    f"{best_checkpoint_path}"
                )

            # Save Last Model (Checkpoint)
            if (epoch + 1) % 10 == 0:
                if args.domain:
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.save_dir, f"{args.domain}_lstm_last.pt"),
                    )
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, f"{run_name}_lstm_last.pt"),
                )

    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default=None,
        help="Single training domain (legacy mode)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Optional list of domains for pooled training",
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing trajectory data"
    )
    parser.add_argument(
        "--save_dir", required=True, help="Directory to save models and logs"
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device selection policy",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Enable pinned host memory for faster CUDA transfers",
    )
    parser.add_argument(
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory",
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
        "--delta",
        action="store_true",
        help="Flag to whether perform delta-based preds. Def. is False",
    )
    parser.add_argument(
        "--no_projection",
        action="store_true",
        help="If set, disables the input projection layer (uses raw input dim)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Filename prefix for checkpoints/logs (default: domain or all_domains)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional tokenizer/encoding label for metadata",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.set_defaults(pin_memory=True, amp=True)
    args = parser.parse_args()

    if not args.domain and not args.domains:
        parser.error("Provide either --domain or --domains.")
    if args.domain and args.domains:
        parser.error("Use either --domain or --domains, not both.")

    train(args)
