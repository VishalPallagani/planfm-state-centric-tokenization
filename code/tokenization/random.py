"""
Random baseline tokenizer for planning states.

This tokenizer intentionally avoids learned or graph-aware structure.

It is designed as a weak but still task-sensible baseline:
- the state is represented only through coarse predicate-count information
- the goal is represented as a random bag of goal atoms

Compared with a stronger grounded-atom random baseline, this removes most
object-level state structure while still giving the downstream model the
minimum conditioning needed for problem-specific planning.
"""

import hashlib
import json
import logging
import re

import numpy as np

from code.tokenization.base import TokenizationStrategy

logger = logging.getLogger(__name__)

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _normalize_atoms(raw_atoms: list[str]) -> list[str]:
    """Normalize atoms by stripping parentheses and sorting lexical tokens."""
    normalized: list[str] = []
    for atom in raw_atoms:
        if not atom:
            continue
        matches = _PREDICATE_REGEX.findall(atom)
        if matches:
            normalized.extend(m.strip().lower() for m in matches if m.strip())
            continue
        clean = atom.replace("(", "").replace(")", "").strip().lower()
        if clean:
            normalized.append(clean)
    return sorted(normalized)


def _predicate_names(normalized_atoms: list[str]) -> list[str]:
    predicates: list[str] = []
    for atom in normalized_atoms:
        parts = atom.split()
        if parts:
            predicates.append(parts[0])
    return sorted(predicates)


class RandomTokenizer(TokenizationStrategy):
    """
    Deterministic random baseline tokenizer.

    The embedding is a deterministic random baseline with asymmetric inputs:

    - state embedding: random bag of state predicate names only
    - goal embedding: random bag of exact goal atoms
    - final vectors are normalized sums

    This is intentionally weaker than a grounded-atom random baseline:
    it keeps only coarse state composition, while the explicit goal vector
    carries the problem-specific conditioning.
    """

    def __init__(self, random_dim: int = 128, seed: int = 42, normalize: bool = True):
        super().__init__(name="Random")
        self.random_dim = int(random_dim)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.embedding_dim = self.random_dim

    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        """
        No-op fit to match tokenizer interface.

        Args are accepted for API compatibility with other tokenizers.
        """
        _ = (domain_pddl_path, train_states_dir, train_pddl_dir)
        self._is_fitted = True
        logger.info(
            f"[{self.name}] Ready with dim={self.random_dim}, seed={self.seed}, "
            f"normalize={self.normalize}"
        )

    def _vector_for_token(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        local_seed = int.from_bytes(digest[:8], "big") ^ (self.seed & 0xFFFFFFFFFFFFFFFF)
        rng = np.random.default_rng(local_seed)
        vec = rng.standard_normal(self.random_dim, dtype=np.float32)
        return vec.astype(np.float32)

    def _embed_tokens(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.random_dim, dtype=np.float32)
        for token in tokens:
            vec += self._vector_for_token(token)

        if self.normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm

        return vec.astype(np.float32)

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        self._check_fitted()
        state_norm = _normalize_atoms(state_atoms)
        _ = (goal_atoms, objects)

        state_preds = _predicate_names(state_norm)
        tokens = [f"state_pred:{pred}" for pred in state_preds]
        return self._embed_tokens(tokens)

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        self._check_fitted()
        goal_norm = _normalize_atoms(goal_atoms)
        _ = objects
        tokens = [f"goal:{atom}" for atom in goal_norm]
        return self._embed_tokens(tokens)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return self.random_dim

    def save_vocabulary(self, filepath: str) -> None:
        self._check_fitted()
        payload = {
            "random_dim": self.random_dim,
            "seed": self.seed,
            "normalize": self.normalize,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[{self.name}] Saved config to {filepath}")

    def load_vocabulary(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            payload = json.load(f)
        self.random_dim = int(payload.get("random_dim", self.random_dim))
        self.seed = int(payload.get("seed", self.seed))
        self.normalize = bool(payload.get("normalize", self.normalize))
        self.embedding_dim = self.random_dim
        self._is_fitted = True
        logger.info(
            f"[{self.name}] Loaded config from {filepath} "
            f"(dim={self.random_dim}, seed={self.seed}, normalize={self.normalize})"
        )
