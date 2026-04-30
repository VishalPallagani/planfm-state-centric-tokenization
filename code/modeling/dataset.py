import glob
import os
import sys
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def _normalize_domains(domain) -> list[str]:
    if isinstance(domain, str):
        return [domain]
    if isinstance(domain, Iterable):
        return [str(d) for d in domain]
    raise TypeError(f"Unsupported domain spec: {domain!r}")


class PlanningTrajectoryDataset(Dataset):
    def __init__(self, data_dir, domain, split="train"):
        """
        data_dir: root data dir (e.g., 'data/encodings/graphs')
        """
        self.files = []
        self.domains = _normalize_domains(domain)
        self.traj_files = []

        for domain_name in self.domains:
            target_dir = os.path.join(data_dir, domain_name, split)
            if not os.path.exists(target_dir):
                print(f"Warning: Directory {target_dir} does not exist.")
                continue

            all_files = glob.glob(os.path.join(target_dir, "*.npy"))
            filtered_files = [f for f in all_files if not f.endswith("_goal.npy")]
            self.traj_files.extend(sorted(filtered_files))

        self.traj_files = sorted(self.traj_files)

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, idx):
        traj_path = self.traj_files[idx]
        goal_path = traj_path.replace(".npy", "_goal.npy")

        # Load numpy arrays
        # Traj: [T, D]
        # Goal: [D]
        traj = np.load(traj_path).astype(np.float32)
        goal = np.load(goal_path).astype(np.float32)

        # Ensure Goal is at least 1D [D]
        if goal.ndim == 0:
            goal = goal.reshape(1)

        # Ensure Traj is 2D [T, D]
        if traj.ndim == 1:
            # Ambiguity: Is it [T] (D=1) or [D] (T=1)?
            # We use Goal dimension D to decide.
            D = goal.shape[0]

            if traj.shape[0] == D:
                # Likely T=1, D=D
                traj = traj.reshape(1, D)
            else:
                # Likely T=T, D=1
                traj = traj.reshape(-1, 1)

        return torch.from_numpy(traj), torch.from_numpy(goal)


def collate_trajectories(batch):
    """
    Custom collate function to handle variable length trajectories.
    Pads sequences to the longest in the batch.
    Returns:
        padded_trajs: [B, MaxT, D]
        goals: [B, D]
        lengths: [B]
    """
    trajs, goals = zip(*batch)

    # Lengths for packing
    lengths = torch.tensor([t.shape[0] for t in trajs])

    # Pad trajectories: [B, MaxT, D]
    padded_trajs = torch.nn.utils.rnn.pad_sequence(trajs, batch_first=True)

    # Stack goals: [B, D]
    goals = torch.stack(goals)

    return padded_trajs, goals, lengths


def load_flat_dataset_for_xgboost(data_dir, domain, split="train", delta=False):
    """
    Loads trajectories and flattens them into (X, y) pairs for XGBoost.
    X: Concatenation of [State_t, Goal]
    y: State_{t+1} (if delta=False) OR (State_{t+1} - State_t) (if delta=True)
    """
    domains = _normalize_domains(domain)
    traj_files = []
    for domain_name in domains:
        target_dir = os.path.join(data_dir, domain_name, split)
        if not os.path.exists(target_dir):
            print(f"Warning: Directory {target_dir} does not exist.")
            continue

        all_files = glob.glob(os.path.join(target_dir, "*.npy"))
        traj_files.extend(sorted([f for f in all_files if not f.endswith("_goal.npy")]))

    traj_files = sorted(traj_files)
    if not traj_files:
        return None, None

    X_list = []
    y_list = []

    print(
        f"Loading {len(traj_files)} trajectories for {split} "
        f"across domains: {', '.join(domains)}"
    )

    for traj_path in tqdm(
        traj_files,
        desc=f"Flattening {split}",
        disable=(not _progress_enabled()),
    ):
        goal_path = traj_path.replace(".npy", "_goal.npy")

        # Load raw numpy (skip torch conversion)
        traj = np.load(traj_path).astype(np.float32)  # [T, D]
        goal = np.load(goal_path).astype(np.float32)  # [D]

        # Fix dimensions
        if goal.ndim == 0:
            goal = goal.reshape(1)
        D = goal.shape[0]
        if traj.ndim == 1:
            if traj.shape[0] == D:
                traj = traj.reshape(1, D)
            else:
                traj = traj.reshape(-1, 1)

        # Need at least 2 steps to form a pair
        T = traj.shape[0]
        if T < 2:
            continue

        # Inputs: S_0 ... S_{T-1}
        states_in = traj[:-1, :]  # [T-1, D]

        # Targets: S_1 ... S_T
        states_out = traj[1:, :]  # [T-1, D]

        # Expand Goal: [T-1, D]
        goals_in = np.tile(goal, (T - 1, 1))

        # Create X: [S_t, G]
        # Shape: [T-1, 2D]
        x_chunk = np.hstack([states_in, goals_in])

        # Create y
        if delta:
            y_chunk = states_out - states_in
        else:
            y_chunk = states_out

        X_list.append(x_chunk)
        y_list.append(y_chunk)

    if not X_list:
        return None, None

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y
