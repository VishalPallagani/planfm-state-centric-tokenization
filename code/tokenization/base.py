"""
Abstract base class for all tokenization strategies.

Each strategy converts an ILG (represented as state atoms + goal atoms + objects)
into a fixed-dimensional numpy embedding vector.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TokenizationStrategy(ABC):
    """
    Abstract base class for all ILG tokenization strategies.

    Each strategy must implement:
      - fit(): Build vocabulary/parameters from training data
      - transform_state(): Convert a single (state, goal, problem) to an embedding
      - transform_goal(): Convert a goal specification to an embedding
      - get_embedding_dim(): Return the fixed embedding dimensionality

    The data interface uses the same raw format as the existing pipeline:
      - state_atoms: list of strings like "(on a b)", "(clear c)"
      - goal_atoms: list of strings like "(on a b)"
      - objects: sorted list of object name strings
      - domain_pddl_path: path to domain PDDL file
      - problem_pddl_path: path to problem PDDL file
    """

    def __init__(self, name: str):
        self.name = name
        self.embedding_dim: int | None = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        """
        Build vocabulary/parameters from training data.

        Args:
            domain_pddl_path: Path to the domain PDDL file.
            train_states_dir: Directory containing training .traj files.
            train_pddl_dir: Directory containing training problem .pddl files.
        """
        pass

    @abstractmethod
    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """
        Convert a single (state, goal) pair to a fixed-dimensional embedding.

        Args:
            state_atoms: List of state atom strings, e.g. ["(on a b)", "(clear c)"]
            goal_atoms: List of goal atom strings, e.g. ["(on a b)"]
            objects: Sorted list of object names in this problem.

        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """
        Convert a goal specification to an embedding.

        Args:
            goal_atoms: List of goal atom strings.
            objects: Sorted list of object names.

        Returns:
            embedding: numpy array of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the fixed embedding dimensionality after fitting."""
        pass

    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary/parameters to disk. Override in subclasses."""
        logger.warning(
            f"{self.name}: save_vocabulary not implemented, skipping."
        )

    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary/parameters from disk. Override in subclasses."""
        logger.warning(
            f"{self.name}: load_vocabulary not implemented, skipping."
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} tokenizer has not been fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"embedding_dim={self.embedding_dim}, fitted={self._is_fitted})"
        )
