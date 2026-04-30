import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enforce strict deterministic algorithms
    # Note: This might throw errors if an operation doesn't have a deterministic implementation,
    # but for LSTM/Linear it is supported.
    torch.use_deterministic_algorithms(True)

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set CUBLAS workspace config for deterministic LSTM on CUDA >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Global seed set to: {seed}")


def worker_init_fn(worker_id):
    """
    Function to ensure DataLoader workers are seeded deterministically.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_plan(domain_path, problem_path, plan_actions, val_path):
    """
    Writes the plan to a temp file and runs VAL.
    Returns:
      - is_solved (bool): Goal reached (VAL: "Plan valid")
      - is_executable (bool): All actions applied validly (VAL: "Plan executed successfully")
    """
    # 0. Pre-checks
    if not plan_actions:
        print("Empty plan provided for validation.")
        return False, False

    val_bin = Path(val_path)

    # Check existence and permissions of VAL binary
    if not val_bin.exists() or not os.access(val_bin, os.X_OK):
        print(f"VAL binary not found or not executable at: {val_path}")
        return False, False

    # Ensure domain/prob paths are absolute
    abs_domain = Path(domain_path).resolve()
    abs_problem = Path(problem_path).resolve()

    # 1. Write plan to temporary file
    # VAL expects actions on separate lines: (action arg1 arg2)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".plan") as tmp:
        for action in plan_actions:
            # Clean up action string. Pyperplan might give "(action a b)" or "action a b"
            act_str = str(action).strip()
            # Ensure lowercase
            act_str = act_str.lower()
            if not act_str.startswith("("):
                act_str = f"({act_str})"

            line = f"{act_str}\n"
            tmp.write(line)

        tmp_plan_path = Path(tmp.name).resolve()

    # 2. Run VAL
    # Command: Validate -v domain.pddl problem.pddl plan.plan
    cmd = [str(val_bin), "-v", str(abs_domain), str(abs_problem), str(tmp_plan_path)]

    try:
        # Capture both stdout and stderr
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout

        # 3. Parse Output
        # "Plan valid" implies both executable AND goal reached.
        # "Plan executed successfully" implies executable, but goal might not be reached.
        is_solved = "Plan valid" in output
        is_executable = is_solved or "Plan executed successfully" in output

    except Exception as e:
        print(f"Error running VAL: {e}")
        is_solved = False
        is_executable = False

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_plan_path):
            os.remove(tmp_plan_path)

    return is_solved, is_executable
