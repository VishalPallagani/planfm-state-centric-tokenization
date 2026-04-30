"""
Integration tests for the multi-tokenization pipeline.

Integration tests run a minimal end-to-end pipeline for each tokenizer
on a subset of Blocksworld data and verify output compatibility
with the downstream training pipeline.

Run with: uv run python -m pytest tests/test_integration.py -v
"""

import os
import tempfile

import numpy as np
import pytest

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOMAIN = "blocks"
DOMAIN_PDDL = os.path.join(DATA_DIR, "pddl", DOMAIN, "domain.pddl")
TRAIN_STATES = os.path.join(DATA_DIR, "states", DOMAIN, "train")
TRAIN_PDDL = os.path.join(DATA_DIR, "pddl", DOMAIN, "train")


@pytest.fixture(scope="module")
def has_data():
    """Check that training data exists; skip if not."""
    if not os.path.exists(DOMAIN_PDDL):
        pytest.skip(f"Data not found: {DOMAIN_PDDL}")
    if not os.path.exists(TRAIN_STATES):
        pytest.skip(f"Data not found: {TRAIN_STATES}")
    return True


def _get_sample_problem():
    """Get a single problem for minimal testing."""
    import re, pddl, pddl.logic.predicates

    traj_files = sorted([f for f in os.listdir(TRAIN_STATES) if f.endswith(".traj")])
    if not traj_files:
        pytest.skip("No .traj files found")

    t_file = traj_files[0]
    prob_name = t_file.replace(".traj", "")
    prob_pddl = os.path.join(TRAIN_PDDL, f"{prob_name}.pddl")
    traj_path = os.path.join(TRAIN_STATES, t_file)

    domain = pddl.parse_domain(DOMAIN_PDDL)
    problem = pddl.parse_problem(prob_pddl)
    objects = sorted({o.name for o in problem.objects} | {o.name for o in domain.constants})

    # Extract goal atoms
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

    # Read trajectory
    with open(traj_path, "r") as f:
        lines = f.readlines()

    predicate_regex = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")
    states = []
    for line in lines:
        matches = predicate_regex.findall(line.strip())
        states.append([f"({m})" for m in matches])

    return {
        "objects": objects,
        "goal_atoms": goals,
        "states": states,
        "prob_name": prob_name,
    }


class TestEndToEndPipeline:
    """Embed a trajectory and verify .npy compatibility."""

    @pytest.mark.parametrize("tokenizer_name,cls_name,module_name,kwargs", [
        ("random", "RandomTokenizer", "code.tokenization.random", {"random_dim": 64, "seed": 42}),
        ("simhash", "SimHashTokenizer", "code.tokenization.simhash", {"hash_dim": 64, "seed": 42}),
        ("shortest_path", "ShortestPathTokenizer", "code.tokenization.shortest_path", {"max_path_length": 3}),
        ("graphbpe", "GraphBPETokenizer", "code.tokenization.graphbpe", {"vocab_size": 200, "num_iterations": 10}),
    ])
    def test_embed_and_load_trajectory(
        self, has_data, tokenizer_name, cls_name, module_name, kwargs, tmp_path
    ):
        import importlib

        # 1. Create and fit tokenizer
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        tok = cls(**kwargs)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)

        # 2. Get sample problem
        sample = _get_sample_problem()

        # 3. Embed trajectory
        embeddings = []
        for state_atoms in sample["states"]:
            emb = tok.transform_state(state_atoms, sample["goal_atoms"], sample["objects"])
            embeddings.append(emb)

        traj_matrix = np.array(embeddings, dtype=np.float32)
        goal_vec = tok.transform_goal(sample["goal_atoms"], sample["objects"]).astype(np.float32)

        # 4. Save as .npy
        traj_path = os.path.join(str(tmp_path), f"{sample['prob_name']}.npy")
        goal_path = os.path.join(str(tmp_path), f"{sample['prob_name']}_goal.npy")
        np.save(traj_path, traj_matrix)
        np.save(goal_path, goal_vec)

        # 5. Load and verify shapes
        loaded_traj = np.load(traj_path)
        loaded_goal = np.load(goal_path)

        # Verify shapes
        assert loaded_traj.ndim == 2, f"Expected 2D, got {loaded_traj.ndim}D"
        assert loaded_traj.shape[0] == len(sample["states"])
        assert loaded_traj.shape[1] == tok.get_embedding_dim()
        assert loaded_goal.shape == (tok.get_embedding_dim(),)

        # Verify consistent dims
        assert loaded_traj.shape[1] == loaded_goal.shape[0], (
            "Trajectory and goal embedding dims must match"
        )

        # Verify dtype
        assert loaded_traj.dtype == np.float32
        assert loaded_goal.dtype == np.float32


class TestDatasetCompatibility:
    """Verify that generated .npy files work with PlanningTrajectoryDataset."""

    def test_simhash_dataset_loading(self, has_data, tmp_path):
        """SimHash embeddings load correctly via PlanningTrajectoryDataset."""
        from code.tokenization.simhash import SimHashTokenizer

        tok = SimHashTokenizer(hash_dim=64, seed=42)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)

        sample = _get_sample_problem()

        # Create directory structure matching PlanningTrajectoryDataset expectations
        ds_dir = os.path.join(str(tmp_path), "simhash", DOMAIN, "train")
        os.makedirs(ds_dir)

        for state_atoms in sample["states"]:
            emb = tok.transform_state(state_atoms, sample["goal_atoms"], sample["objects"])

        traj_matrix = np.array(
            [tok.transform_state(s, sample["goal_atoms"], sample["objects"])
             for s in sample["states"]],
            dtype=np.float32,
        )
        goal_vec = tok.transform_goal(sample["goal_atoms"], sample["objects"]).astype(np.float32)

        np.save(os.path.join(ds_dir, f"{sample['prob_name']}.npy"), traj_matrix)
        np.save(os.path.join(ds_dir, f"{sample['prob_name']}_goal.npy"), goal_vec)

        # Load with PlanningTrajectoryDataset (requires torch)
        try:
            from code.modeling.dataset import PlanningTrajectoryDataset
        except (ImportError, OSError) as e:
            pytest.skip(f"torch/dataset unavailable: {e}")

        dataset = PlanningTrajectoryDataset(
            os.path.join(str(tmp_path), "simhash"), DOMAIN, "train"
        )

        assert len(dataset) == 1
        traj_tensor, goal_tensor = dataset[0]
        assert traj_tensor.shape[0] == len(sample["states"])
        assert traj_tensor.shape[1] == 64
        assert goal_tensor.shape[0] == 64
