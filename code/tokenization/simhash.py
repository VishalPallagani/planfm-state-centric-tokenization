"""
SimHash tokenizer for Instance Learning Graphs.

Uses random projections (locality-sensitive hashing) to create
fixed-dimensional binary embeddings from ILG feature vectors.

Reference: Charikar, M. S. (2002). Similarity estimation techniques
from rounding algorithms.
"""

import hashlib
import json
import logging
import os
import re
import sys

import numpy as np
from tqdm import tqdm

from code.tokenization.base import TokenizationStrategy

logger = logging.getLogger(__name__)

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


class SimHashTokenizer(TokenizationStrategy):
    """
    SimHash-based graph tokenization using random projections.

    Algorithm:
    1. Extract a sparse feature dictionary from ILG structure
       (node attributes, edge labels, structural patterns).
    2. During fit(), collect all unique feature keys and create a
       random Gaussian projection matrix.
    3. During transform(), project features and apply sign() to
       produce a binary hash vector.

    The output is permutation-invariant (features use sorted, canonical names).
    """

    def __init__(self, hash_dim: int = 128, seed: int = 42):
        super().__init__(name="SimHash")
        self.hash_dim = hash_dim
        self.seed = seed
        self.embedding_dim = hash_dim

        # Learnable parameters
        self._feature_keys: list[str] | None = None
        self._feature_to_idx: dict[str, int] | None = None
        self._projection_matrix: np.ndarray | None = None

        # Domain info (cached from the PDDL)
        self._domain_info: dict[str, int] | None = None

    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        """
        Build feature vocabulary and random projection matrix.

        1. Parse domain to learn predicate arities.
        2. Scan all training (state, goal) pairs to collect unique feature keys.
        3. Create Gaussian projection matrix R ∈ ℝ^(n_features × hash_dim).
        """
        import pddl

        # 1. Parse domain
        domain = pddl.parse_domain(domain_pddl_path)
        self._domain_info = {p.name.lower(): p.arity for p in domain.predicates}

        # 2. Collect feature keys from training data
        all_feature_keys: set[str] = set()

        train_files = sorted(
            [f for f in os.listdir(train_states_dir) if f.endswith(".traj")]
        )

        for t_file in tqdm(
            train_files,
            desc=f"  [{self.name}] Collecting features",
            disable=(not _progress_enabled()),
        ):
            prob_name = t_file.replace(".traj", "")
            prob_pddl = os.path.join(train_pddl_dir, f"{prob_name}.pddl")
            traj_path = os.path.join(train_states_dir, t_file)

            if not os.path.exists(prob_pddl):
                continue

            try:
                problem = pddl.parse_problem(prob_pddl)
                objects = sorted(
                    {o.name for o in problem.objects}
                    | {o.name for o in domain.constants}
                )

                # Extract goal atoms
                goal_atoms = self._extract_goal_atoms(problem)

                # Read trajectory
                with open(traj_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    state_atoms = _PREDICATE_REGEX.findall(line.strip())
                    features = self._extract_features(state_atoms, goal_atoms, objects)
                    all_feature_keys.update(features.keys())

            except Exception:
                continue

        if not all_feature_keys:
            raise RuntimeError("No features collected during SimHash fit.")

        # 3. Build vocabulary and projection matrix
        self._feature_keys = sorted(all_feature_keys)
        self._feature_to_idx = {k: i for i, k in enumerate(self._feature_keys)}
        n_features = len(self._feature_keys)

        rng = np.random.RandomState(self.seed)
        self._projection_matrix = rng.randn(n_features, self.hash_dim).astype(
            np.float32
        )

        self._is_fitted = True
        logger.info(
            f"[{self.name}] Fitted: {n_features} features → {self.hash_dim}d hash"
        )

    def _extract_goal_atoms(self, problem) -> list[str]:
        """Extract goal atoms from a parsed pddl Problem object."""
        import pddl.logic.predicates

        goals = []

        def visit(node):
            if isinstance(node, pddl.logic.predicates.Predicate):
                args = [
                    t.name if hasattr(t, "name") else str(t) for t in node.terms
                ]
                goals.append(f"{node.name} {' '.join(args)}")
            elif hasattr(node, "operands"):
                for op in node.operands:
                    visit(op)
            elif hasattr(node, "_operands"):
                for op in node._operands:
                    visit(op)

        visit(problem.goal)
        return goals

    def _extract_features(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> dict[str, float]:
        """
        Extract a sparse feature dictionary from an ILG.

        Features are permutation-invariant (use predicate/type names,
        not object identities). Categories:

        1. Node attribute features: presence of (predicate, role) combos
        2. Edge features: (predicate, arity) counts
        3. Goal features: goal predicate patterns
        4. Structural features: object count, predicate density, etc.
        """
        features: dict[str, float] = {}

        # --- 1. State predicate features ---
        for atom_str in state_atoms:
            parts = atom_str.split()
            if not parts:
                continue
            pred = parts[0].lower()
            args = parts[1:]
            arity = len(args)

            # Count predicate occurrences
            feat_key = f"state_pred:{pred}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

            # Arity pattern
            feat_key = f"state_arity:{arity}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

            # For binary predicates, track that two objects are related
            if arity == 2:
                feat_key = f"state_binary_edge:{pred}"
                features[feat_key] = features.get(feat_key, 0) + 1.0

            # Self-referencing (arg appears twice)
            if arity == 2 and len(args) == 2 and args[0] == args[1]:
                feat_key = f"state_self_ref:{pred}"
                features[feat_key] = features.get(feat_key, 0) + 1.0

        # --- 2. Goal predicate features ---
        for atom_str in goal_atoms:
            parts = atom_str.split()
            if not parts:
                continue
            pred = parts[0].lower()
            args = parts[1:]
            arity = len(args)

            feat_key = f"goal_pred:{pred}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

            feat_key = f"goal_arity:{arity}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

        # --- 3. State-Goal interaction features ---
        state_preds = set()
        for atom_str in state_atoms:
            parts = atom_str.split()
            if parts:
                state_preds.add(parts[0].lower())

        goal_preds = set()
        for atom_str in goal_atoms:
            parts = atom_str.split()
            if parts:
                goal_preds.add(parts[0].lower())

        # Predicates appearing in both state and goal
        for pred in sorted(state_preds & goal_preds):
            feat_key = f"state_goal_overlap:{pred}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

        # Goal predicates NOT in state (unachieved goals)
        for pred in sorted(goal_preds - state_preds):
            feat_key = f"goal_missing:{pred}"
            features[feat_key] = features.get(feat_key, 0) + 1.0

        # --- 4. Structural features ---
        features["n_objects"] = float(len(objects))
        features["n_state_atoms"] = float(len(state_atoms))
        features["n_goal_atoms"] = float(len(goal_atoms))

        if len(objects) > 0:
            features["pred_density"] = float(len(state_atoms)) / float(len(objects))

        return features

    def _features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        """Convert sparse feature dict to dense vector using vocabulary."""
        vec = np.zeros(len(self._feature_keys), dtype=np.float32)
        for key, value in features.items():
            if key in self._feature_to_idx:
                vec[self._feature_to_idx[key]] = value
        return vec

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """
        Apply SimHash: extract features → project → sign().

        Returns binary vector of shape (hash_dim,) with values in {0, 1}.
        """
        self._check_fitted()

        # Parse state atoms if they contain parens
        parsed_state = []
        for a in state_atoms:
            matches = _PREDICATE_REGEX.findall(a)
            parsed_state.extend(matches)

        # Parse goal atoms similarly
        parsed_goal = []
        for a in goal_atoms:
            a_clean = a.replace("(", "").replace(")", "").strip()
            if a_clean:
                parsed_goal.append(a_clean)

        features = self._extract_features(parsed_state, parsed_goal, objects)
        dense = self._features_to_vector(features)

        # Project and binarize
        projection = dense @ self._projection_matrix  # (hash_dim,)
        binary = (projection >= 0).astype(np.float32)
        return binary

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """Embed a goal by treating it as a state with no current atoms."""
        return self.transform_state([], goal_atoms, objects)

    def get_embedding_dim(self) -> int:
        return self.hash_dim

    def save_vocabulary(self, filepath: str) -> None:
        """Save feature keys and projection matrix."""
        self._check_fitted()
        data = {
            "feature_keys": self._feature_keys,
            "hash_dim": self.hash_dim,
            "seed": self.seed,
        }
        # Save JSON metadata
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Save projection matrix as companion .npy
        matrix_path = filepath.replace(".json", "_projection.npy")
        np.save(matrix_path, self._projection_matrix)
        logger.info(f"[{self.name}] Saved vocabulary to {filepath}")

    def load_vocabulary(self, filepath: str) -> None:
        """Load feature keys and projection matrix."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self._feature_keys = data["feature_keys"]
        self._feature_to_idx = {k: i for i, k in enumerate(self._feature_keys)}
        self.hash_dim = data["hash_dim"]
        self.seed = data["seed"]
        self.embedding_dim = self.hash_dim

        matrix_path = filepath.replace(".json", "_projection.npy")
        self._projection_matrix = np.load(matrix_path)

        self._is_fitted = True
        logger.info(
            f"[{self.name}] Loaded vocabulary from {filepath}, "
            f"{len(self._feature_keys)} features → {self.hash_dim}d"
        )
