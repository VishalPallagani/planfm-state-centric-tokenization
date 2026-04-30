"""
WL (Weisfeiler-Leman) tokenizer wrapping the existing wlplan-based pipeline.

This does NOT reimplement WL - it delegates to the wlplan library that the
codebase already uses in generate_graph_embeddings.py.
"""

import logging
import os
import re
import sys

import numpy as np
from tqdm import tqdm

from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator, load_feature_generator
from wlplan.planning import Atom, State, parse_domain, parse_problem

from code.tokenization.base import TokenizationStrategy

logger = logging.getLogger(__name__)

# Regex to parse "(on a b)" -> "on a b"
_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


def _parse_traj_line_to_state(line: str, pred_map: dict) -> State:
    """Parse a line like '(on a b) (clear c)' into a wlplan State object."""
    line = line.strip()
    if not line:
        return State([])

    matches = re.findall(r"\(([\w-]+(?: [\w-]+)*)\)", line)
    atoms = []
    for m in matches:
        parts = m.split()
        pred_name = parts[0]
        objs = parts[1:]
        if pred_name in pred_map:
            atoms.append(Atom(pred_map[pred_name], objs))
    return State(atoms)


def _atoms_strings_to_wl_atoms(atom_strings: list[str], pred_map: dict) -> list[Atom]:
    """Convert list of atom strings like '(on a b)' to wlplan Atom objects."""
    atoms = []
    for a_str in atom_strings:
        content = a_str.replace("(", "").replace(")", "")
        parts = content.split()
        if not parts:
            continue
        p_name = parts[0]
        p_args = parts[1:]
        if p_name in pred_map:
            atoms.append(Atom(pred_map[p_name], p_args))
    return atoms


class WLTokenizer(TokenizationStrategy):
    """
    Weisfeiler-Leman color refinement tokenization.

    Wraps the existing wlplan library to provide a consistent interface
    with other tokenization strategies. The fit/transform cycle mirrors
    what generate_graph_embeddings.py does.
    """

    def __init__(self, iterations: int = 2):
        super().__init__(name="WL")
        self.iterations = iterations
        self._feature_gen = None
        self._wl_domain = None
        self._pred_map: dict | None = None
        self._domain_pddl_path: str | None = None

    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        """
        Build WL vocabulary from training trajectories.

        Mirrors the vocabulary collection in generate_graph_embeddings.py:
        parse domain → init feature generator → collect from training data.
        """
        self._domain_pddl_path = domain_pddl_path

        # 1. Parse domain
        self._wl_domain = parse_domain(domain_pddl_path)
        self._pred_map = {p.name: p for p in self._wl_domain.predicates}

        # 2. Initialize feature generator (ILG + WL)
        self._feature_gen = init_feature_generator(
            feature_algorithm="wl",
            domain=self._wl_domain,
            graph_representation="ilg",
            iterations=self.iterations,
            pruning="none",
            multiset_hash=True,
        )

        # 3. Load training data
        train_files = sorted(
            [f for f in os.listdir(train_states_dir) if f.endswith(".traj")]
        )

        wl_problems = []
        for t_file in tqdm(
            train_files,
            desc=f"  [{self.name}] Parsing train",
            disable=(not _progress_enabled()),
        ):
            prob_name = t_file.replace(".traj", "")
            prob_pddl = os.path.join(train_pddl_dir, f"{prob_name}.pddl")
            traj_path = os.path.join(train_states_dir, t_file)

            if not os.path.exists(prob_pddl):
                continue

            try:
                wl_prob = parse_problem(domain_pddl_path, prob_pddl)
                with open(traj_path, "r") as f:
                    lines = f.readlines()

                states = [
                    _parse_traj_line_to_state(line, self._pred_map) for line in lines
                ]
                wl_problems.append(ProblemDataset(wl_prob, states))
            except Exception:
                continue

        if not wl_problems:
            raise RuntimeError("No valid training data found for WL vocabulary.")

        # 4. Collect vocabulary
        full_train_ds = DomainDataset(self._wl_domain, wl_problems)
        self._feature_gen.collect(full_train_ds)
        self.embedding_dim = self._feature_gen.get_n_features()
        self._is_fitted = True
        logger.info(f"[{self.name}] Vocabulary size: {self.embedding_dim}")

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        """
        Embed a single state using the WL feature generator.

        Args:
            state_atoms: List of atom strings for the current state.
            goal_atoms: Not used directly (wlplan reads goal from problem PDDL).
            objects: Not used directly (wlplan reads objects from problem PDDL).
            problem_pddl_path: Path to problem PDDL (required for wlplan).
            _wl_prob: Pre-parsed wlplan problem object (optimization to avoid re-parsing).
        """
        self._check_fitted()

        if _wl_prob is None:
            if problem_pddl_path is None:
                raise ValueError("WLTokenizer requires problem_pddl_path for transform.")
            _wl_prob = parse_problem(self._domain_pddl_path, problem_pddl_path)

        wl_atoms = _atoms_strings_to_wl_atoms(state_atoms, self._pred_map)
        state = State(wl_atoms)
        ds = DomainDataset(self._wl_domain, [ProblemDataset(_wl_prob, [state])])
        embs = self._feature_gen.embed(ds)
        return np.array(embs[0], dtype=np.float32)

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        """
        Embed a goal state. Creates a wlplan State from goal atoms and embeds it.
        """
        self._check_fitted()

        if _wl_prob is None:
            if problem_pddl_path is None:
                raise ValueError("WLTokenizer requires problem_pddl_path for transform_goal.")
            _wl_prob = parse_problem(self._domain_pddl_path, problem_pddl_path)

        wl_atoms = _atoms_strings_to_wl_atoms(goal_atoms, self._pred_map)
        goal_state = State(wl_atoms)
        ds = DomainDataset(self._wl_domain, [ProblemDataset(_wl_prob, [goal_state])])
        embs = self._feature_gen.embed(ds)
        return np.array(embs[0], dtype=np.float32)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return self.embedding_dim

    def save_vocabulary(self, filepath: str) -> None:
        """Save the wlplan feature generator to JSON."""
        self._check_fitted()
        self._feature_gen.save(filepath)
        logger.info(f"[{self.name}] Saved vocabulary to {filepath}")

    def load_vocabulary(self, filepath: str) -> None:
        """Load a previously saved wlplan feature generator."""
        self._feature_gen = load_feature_generator(filepath)
        self.embedding_dim = self._feature_gen.get_n_features()
        self._is_fitted = True
        logger.info(
            f"[{self.name}] Loaded vocabulary from {filepath}, "
            f"dim={self.embedding_dim}"
        )

    def set_domain(self, domain_pddl_path: str) -> None:
        """Set domain info needed for transform calls after load_vocabulary."""
        self._domain_pddl_path = domain_pddl_path
        self._wl_domain = parse_domain(domain_pddl_path)
        self._pred_map = {p.name: p for p in self._wl_domain.predicates}
