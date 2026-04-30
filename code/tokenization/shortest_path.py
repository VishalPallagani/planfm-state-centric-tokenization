"""
Shortest-path tokenizer for Instance Learning Graphs.

Builds a histogram-style embedding from:
1. Predicate occurrence features
2. Goal/state predicate interaction features
3. Object-graph shortest path length histograms (up to max_path_length)
"""

import json
import logging
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np
from tqdm import tqdm

from code.tokenization.base import TokenizationStrategy

logger = logging.getLogger(__name__)

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def _normalize_atoms(raw_atoms: list[str]) -> list[str]:
    """Normalize atom strings by removing parentheses and trimming spaces."""
    normalized: list[str] = []
    for atom in raw_atoms:
        if not atom:
            continue
        matches = _PREDICATE_REGEX.findall(atom)
        if matches:
            normalized.extend(m.strip() for m in matches if m.strip())
            continue
        clean = atom.replace("(", "").replace(")", "").strip()
        if clean:
            normalized.append(clean)
    return normalized


class ShortestPathTokenizer(TokenizationStrategy):
    """
    Shortest-path kernel style tokenizer.

    The embedding is a fixed-length nonnegative histogram over learned feature keys.
    """

    def __init__(self, max_path_length: int = 5):
        super().__init__(name="ShortestPath")
        self.max_path_length = max_path_length
        self._feature_keys: list[str] | None = None
        self._feature_to_idx: dict[str, int] | None = None

    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        import pddl

        domain = pddl.parse_domain(domain_pddl_path)
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
                goal_atoms = self._extract_goal_atoms(problem)

                with open(traj_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    state_atoms = _normalize_atoms([line.strip()])
                    features = self._extract_features(state_atoms, goal_atoms, objects)
                    all_feature_keys.update(features.keys())
            except Exception:
                continue

        # Ensure dimensionality grows with max_path_length.
        for d in range(1, self.max_path_length + 1):
            all_feature_keys.add(f"sp_len:{d}")
            all_feature_keys.add(f"goal_sp_len:{d}")

        if not all_feature_keys:
            raise RuntimeError("No features collected during ShortestPath fit.")

        self._feature_keys = sorted(all_feature_keys)
        self._feature_to_idx = {k: i for i, k in enumerate(self._feature_keys)}
        self.embedding_dim = len(self._feature_keys)
        self._is_fitted = True

        logger.info(
            f"[{self.name}] Fitted: {self.embedding_dim} features "
            f"(max_path_length={self.max_path_length})"
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

    def _build_object_graph(
        self,
        atoms: list[str],
        objects: list[str],
    ) -> dict[str, set[str]]:
        """
        Build an undirected object graph from binary predicates.
        Unary predicates are handled separately via feature counts.
        """
        graph: dict[str, set[str]] = defaultdict(set)

        for o in objects:
            graph[o]  # ensure key exists

        for atom in atoms:
            parts = atom.split()
            if not parts:
                continue
            args = parts[1:]

            if len(args) == 2:
                a, b = args
                graph[a].add(b)
                graph[b].add(a)
            elif len(args) == 1:
                arg = args[0]
                graph[arg]  # ensure singleton nodes are present

        return graph

    def _shortest_path_lengths(
        self,
        graph: dict[str, set[str]],
        start: str,
    ) -> dict[str, int]:
        """BFS shortest path lengths from one start node."""
        dist = {start: 0}
        queue = deque([start])

        while queue:
            cur = queue.popleft()
            cur_d = dist[cur]
            if cur_d >= self.max_path_length:
                continue
            for nxt in graph.get(cur, ()):
                if nxt in dist:
                    continue
                dist[nxt] = cur_d + 1
                queue.append(nxt)

        return dist

    def _path_histogram(
        self,
        graph: dict[str, set[str]],
        objects: list[str],
        prefix: str,
    ) -> dict[str, float]:
        """Histogram of pair shortest-path lengths up to max_path_length."""
        hist: dict[str, float] = {}
        objs = list(dict.fromkeys(objects))  # stable unique

        for i, src in enumerate(objs):
            dmap = self._shortest_path_lengths(graph, src)
            for dst in objs[i + 1 :]:
                d = dmap.get(dst)
                if d is None:
                    continue
                if 1 <= d <= self.max_path_length:
                    key = f"{prefix}_sp_len:{d}"
                    hist[key] = hist.get(key, 0.0) + 1.0

        return hist

    def _extract_features(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> dict[str, float]:
        """Extract nonnegative histogram features."""
        features: dict[str, float] = {}

        # Predicate count features.
        state_preds = []
        for atom in state_atoms:
            parts = atom.split()
            if not parts:
                continue
            pred = parts[0].lower()
            state_preds.append(pred)
            features[f"state_pred:{pred}"] = features.get(f"state_pred:{pred}", 0.0) + 1.0
            features[f"state_arity:{len(parts) - 1}"] = (
                features.get(f"state_arity:{len(parts) - 1}", 0.0) + 1.0
            )

        goal_preds = []
        for atom in goal_atoms:
            parts = atom.split()
            if not parts:
                continue
            pred = parts[0].lower()
            goal_preds.append(pred)
            features[f"goal_pred:{pred}"] = features.get(f"goal_pred:{pred}", 0.0) + 1.0
            features[f"goal_arity:{len(parts) - 1}"] = (
                features.get(f"goal_arity:{len(parts) - 1}", 0.0) + 1.0
            )

        state_set = set(state_preds)
        goal_set = set(goal_preds)
        for pred in sorted(state_set & goal_set):
            features[f"state_goal_overlap:{pred}"] = 1.0
        for pred in sorted(goal_set - state_set):
            features[f"goal_missing:{pred}"] = 1.0

        # Graph shortest-path histograms.
        state_graph = self._build_object_graph(state_atoms, objects)
        goal_graph = self._build_object_graph(goal_atoms, objects)
        features.update(self._path_histogram(state_graph, objects, prefix=""))
        features.update(self._path_histogram(goal_graph, objects, prefix="goal"))

        # Structural scalars.
        features["n_objects"] = float(len(objects))
        features["n_state_atoms"] = float(len(state_atoms))
        features["n_goal_atoms"] = float(len(goal_atoms))

        return features

    def _features_to_vector(self, features: dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self._feature_keys), dtype=np.float32)
        for key, value in features.items():
            idx = self._feature_to_idx.get(key)
            if idx is not None:
                vec[idx] = float(value)
        return vec

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        self._check_fitted()

        parsed_state = _normalize_atoms(state_atoms)
        parsed_goal = _normalize_atoms(goal_atoms)
        features = self._extract_features(parsed_state, parsed_goal, objects)
        return self._features_to_vector(features)

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        return self.transform_state([], goal_atoms, objects)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return int(self.embedding_dim)

    def save_vocabulary(self, filepath: str) -> None:
        self._check_fitted()
        data = {
            "feature_keys": self._feature_keys,
            "max_path_length": self.max_path_length,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[{self.name}] Saved vocabulary to {filepath}")

    def load_vocabulary(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = json.load(f)

        self._feature_keys = list(data["feature_keys"])
        self._feature_to_idx = {k: i for i, k in enumerate(self._feature_keys)}
        self.max_path_length = int(data.get("max_path_length", self.max_path_length))
        self.embedding_dim = len(self._feature_keys)
        self._is_fitted = True

        logger.info(f"[{self.name}] Loaded vocabulary from {filepath}")
