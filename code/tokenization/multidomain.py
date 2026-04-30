"""
Utilities for all-domain tokenizer training and loading.

This module supports two cross-domain strategies:

1. pooled:
   Learn one shared tokenizer space from all training domains when the
   underlying tokenizer supports it naturally.

2. union:
   Fit one tokenizer per domain and place each domain embedding into a
   fixed block of a larger global vector. This is used as a safe fallback
   for tokenizers such as the current wlplan-backed WL implementation.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pddl
import pddl.logic.predicates

from code.tokenization.base import TokenizationStrategy
from code.tokenization.factory import create_tokenizer

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _extract_goal_atoms(problem) -> list[str]:
    """Extract goal atoms from a parsed problem as bare strings."""
    goals: list[str] = []

    def visit(node):
        if isinstance(node, pddl.logic.predicates.Predicate):
            args = [t.name if hasattr(t, "name") else str(t) for t in node.terms]
            goals.append(f"{node.name} {' '.join(args)}")
        elif hasattr(node, "operands"):
            for op in node.operands:
                visit(op)
        elif hasattr(node, "_operands"):
            for op in node._operands:
                visit(op)

    visit(problem.goal)
    return goals


def _get_objects(problem, domain) -> list[str]:
    objs = {o.name for o in problem.objects}
    objs.update(o.name for o in domain.constants)
    return sorted(objs)


def _parse_traj_line(line: str) -> list[str]:
    return [m.strip() for m in _PREDICATE_REGEX.findall(line.strip()) if m.strip()]


def build_domain_specs(data_dir: str, domains: list[str]) -> list[dict]:
    specs: list[dict] = []
    for domain_name in domains:
        specs.append(
            {
                "domain": domain_name,
                "domain_pddl_path": os.path.join(data_dir, "pddl", domain_name, "domain.pddl"),
                "train_states_dir": os.path.join(data_dir, "states", domain_name, "train"),
                "train_pddl_dir": os.path.join(data_dir, "pddl", domain_name, "train"),
            }
        )
    return specs


class MultiDomainUnionTokenizer(TokenizationStrategy):
    """
    Block-union tokenizer over one fitted tokenizer per domain.

    Each domain keeps its own tokenizer and embedding dimension. The global
    vector is the concatenation of domain-specific subspaces, and a single
    active domain selects which block is populated at transform time.
    """

    def __init__(self, base_tokenizer_name: str, tokenizer_kwargs: dict | None = None):
        super().__init__(name=f"MultiDomainUnion[{base_tokenizer_name}]")
        self.base_tokenizer_name = base_tokenizer_name
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.domain_tokenizers: dict[str, TokenizationStrategy] = {}
        self.domain_dimensions: dict[str, int] = {}
        self.domain_offsets: dict[str, int] = {}
        self.domain_order: list[str] = []
        self.active_domain: str | None = None

    def fit(self, domain_pddl_path: str, train_states_dir: str, train_pddl_dir: str) -> None:
        raise NotImplementedError(
            "Use fit_from_domain_specs() for MultiDomainUnionTokenizer."
        )

    def fit_from_domain_specs(self, domain_specs: list[dict]) -> None:
        self.domain_tokenizers = {}
        self.domain_dimensions = {}
        self.domain_offsets = {}
        self.domain_order = []

        offset = 0
        for spec in domain_specs:
            domain_name = spec["domain"]
            tokenizer = create_tokenizer(self.base_tokenizer_name, **self.tokenizer_kwargs)
            tokenizer.fit(
                spec["domain_pddl_path"],
                spec["train_states_dir"],
                spec["train_pddl_dir"],
            )
            dim = int(tokenizer.get_embedding_dim())
            self.domain_tokenizers[domain_name] = tokenizer
            self.domain_dimensions[domain_name] = dim
            self.domain_offsets[domain_name] = offset
            self.domain_order.append(domain_name)
            offset += dim

        self.embedding_dim = offset
        self._is_fitted = True

    def set_active_domain(self, domain_name: str, domain_pddl_path: str | None = None) -> None:
        self._check_fitted()
        if domain_name not in self.domain_tokenizers:
            raise KeyError(f"Unknown domain '{domain_name}' for union tokenizer.")
        self.active_domain = domain_name
        tokenizer = self.domain_tokenizers[domain_name]
        if domain_pddl_path and hasattr(tokenizer, "set_domain"):
            tokenizer.set_domain(domain_pddl_path)

    def _active_tokenizer(self) -> tuple[str, TokenizationStrategy]:
        self._check_fitted()
        if self.active_domain is None:
            raise RuntimeError(
                "Active domain not set. Call set_active_domain(domain_name, domain_pddl_path) first."
            )
        return self.active_domain, self.domain_tokenizers[self.active_domain]

    def _place_into_block(self, domain_name: str, local_vec: np.ndarray) -> np.ndarray:
        global_vec = np.zeros(self.embedding_dim, dtype=np.float32)
        offset = self.domain_offsets[domain_name]
        width = self.domain_dimensions[domain_name]
        global_vec[offset : offset + width] = local_vec.astype(np.float32)
        return global_vec

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        domain_name, tokenizer = self._active_tokenizer()
        try:
            local_vec = tokenizer.transform_state(
                state_atoms,
                goal_atoms,
                objects,
                problem_pddl_path=problem_pddl_path,
                _wl_prob=_wl_prob,
            )
        except TypeError:
            try:
                local_vec = tokenizer.transform_state(
                    state_atoms,
                    goal_atoms,
                    objects,
                    problem_pddl_path=problem_pddl_path,
                )
            except TypeError:
                local_vec = tokenizer.transform_state(state_atoms, goal_atoms, objects)
        return self._place_into_block(domain_name, local_vec)

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        domain_name, tokenizer = self._active_tokenizer()
        try:
            local_vec = tokenizer.transform_goal(
                goal_atoms,
                objects,
                problem_pddl_path=problem_pddl_path,
                _wl_prob=_wl_prob,
            )
        except TypeError:
            try:
                local_vec = tokenizer.transform_goal(
                    goal_atoms,
                    objects,
                    problem_pddl_path=problem_pddl_path,
                )
            except TypeError:
                local_vec = tokenizer.transform_goal(goal_atoms, objects)
        return self._place_into_block(domain_name, local_vec)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return int(self.embedding_dim)

    def save_vocabulary(self, filepath: str) -> None:
        self._check_fitted()
        manifest_path = Path(filepath)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        component_dir = manifest_path.with_name(f"{manifest_path.stem}_files")
        component_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        for domain_name in self.domain_order:
            tok_path = component_dir / f"{domain_name}.json"
            self.domain_tokenizers[domain_name].save_vocabulary(str(tok_path))
            files[domain_name] = os.path.relpath(tok_path, manifest_path.parent)

        payload = {
            "representation_type": "multi_domain_union",
            "base_tokenizer_name": self.base_tokenizer_name,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "embedding_dim": self.embedding_dim,
            "domain_order": self.domain_order,
            "domain_dimensions": self.domain_dimensions,
            "domain_offsets": self.domain_offsets,
            "files": files,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_vocabulary(self, filepath: str) -> None:
        manifest_path = Path(filepath)
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("representation_type") != "multi_domain_union":
            raise ValueError(f"{filepath} is not a multi-domain union tokenizer manifest.")

        self.base_tokenizer_name = payload["base_tokenizer_name"]
        self.tokenizer_kwargs = dict(payload.get("tokenizer_kwargs", {}))
        self.embedding_dim = int(payload["embedding_dim"])
        self.domain_order = list(payload["domain_order"])
        self.domain_dimensions = {
            key: int(value) for key, value in payload["domain_dimensions"].items()
        }
        self.domain_offsets = {
            key: int(value) for key, value in payload["domain_offsets"].items()
        }
        self.domain_tokenizers = {}

        for domain_name in self.domain_order:
            rel_path = str(payload["files"][domain_name]).replace("\\", "/")
            tok_path = manifest_path.parent / rel_path
            tokenizer = create_tokenizer(self.base_tokenizer_name, **self.tokenizer_kwargs)
            tokenizer.load_vocabulary(str(tok_path))
            self.domain_tokenizers[domain_name] = tokenizer

        self.active_domain = None
        self._is_fitted = True


def fit_pooled_standard_tokenizer(
    tokenizer_name: str,
    domain_specs: list[dict],
    **tokenizer_kwargs,
) -> TokenizationStrategy:
    tokenizer = create_tokenizer(tokenizer_name, **tokenizer_kwargs)

    if tokenizer_name == "random":
        tokenizer.fit("", "", "")
        return tokenizer

    if tokenizer_name == "simhash":
        all_feature_keys: set[str] = set()
        for spec in domain_specs:
            domain = pddl.parse_domain(spec["domain_pddl_path"])
            train_files = sorted(
                f for f in os.listdir(spec["train_states_dir"]) if f.endswith(".traj")
            )
            for traj_file in train_files:
                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(spec["train_pddl_dir"], f"{prob_name}.pddl")
                traj_path = os.path.join(spec["train_states_dir"], traj_file)
                if not os.path.exists(prob_pddl):
                    continue
                try:
                    problem = pddl.parse_problem(prob_pddl)
                    objects = _get_objects(problem, domain)
                    goal_atoms = tokenizer._extract_goal_atoms(problem)
                    with open(traj_path, "r", encoding="utf-8") as f:
                        for line in f:
                            state_atoms = _parse_traj_line(line)
                            features = tokenizer._extract_features(
                                state_atoms,
                                goal_atoms,
                                objects,
                            )
                            all_feature_keys.update(features.keys())
                except Exception:
                    continue

        if not all_feature_keys:
            raise RuntimeError("No features collected during pooled SimHash fit.")

        tokenizer._feature_keys = sorted(all_feature_keys)
        tokenizer._feature_to_idx = {
            key: idx for idx, key in enumerate(tokenizer._feature_keys)
        }
        rng = np.random.RandomState(tokenizer.seed)
        tokenizer._projection_matrix = rng.randn(
            len(tokenizer._feature_keys),
            tokenizer.hash_dim,
        ).astype(np.float32)
        tokenizer.embedding_dim = tokenizer.hash_dim
        tokenizer._is_fitted = True
        return tokenizer

    if tokenizer_name == "shortest_path":
        all_feature_keys: set[str] = set()
        for spec in domain_specs:
            domain = pddl.parse_domain(spec["domain_pddl_path"])
            train_files = sorted(
                f for f in os.listdir(spec["train_states_dir"]) if f.endswith(".traj")
            )
            for traj_file in train_files:
                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(spec["train_pddl_dir"], f"{prob_name}.pddl")
                traj_path = os.path.join(spec["train_states_dir"], traj_file)
                if not os.path.exists(prob_pddl):
                    continue
                try:
                    problem = pddl.parse_problem(prob_pddl)
                    objects = _get_objects(problem, domain)
                    goal_atoms = tokenizer._extract_goal_atoms(problem)
                    with open(traj_path, "r", encoding="utf-8") as f:
                        for line in f:
                            state_atoms = _parse_traj_line(line)
                            features = tokenizer._extract_features(
                                state_atoms,
                                goal_atoms,
                                objects,
                            )
                            all_feature_keys.update(features.keys())
                except Exception:
                    continue

        for length in range(1, tokenizer.max_path_length + 1):
            all_feature_keys.add(f"sp_len:{length}")
            all_feature_keys.add(f"goal_sp_len:{length}")

        if not all_feature_keys:
            raise RuntimeError("No features collected during pooled ShortestPath fit.")

        tokenizer._feature_keys = sorted(all_feature_keys)
        tokenizer._feature_to_idx = {
            key: idx for idx, key in enumerate(tokenizer._feature_keys)
        }
        tokenizer.embedding_dim = len(tokenizer._feature_keys)
        tokenizer._is_fitted = True
        return tokenizer

    if tokenizer_name == "graphbpe":
        all_graphs: list[dict] = []
        for spec in domain_specs:
            domain = pddl.parse_domain(spec["domain_pddl_path"])
            train_files = sorted(
                f for f in os.listdir(spec["train_states_dir"]) if f.endswith(".traj")
            )
            for traj_file in train_files:
                prob_name = traj_file.replace(".traj", "")
                prob_pddl = os.path.join(spec["train_pddl_dir"], f"{prob_name}.pddl")
                traj_path = os.path.join(spec["train_states_dir"], traj_file)
                if not os.path.exists(prob_pddl):
                    continue
                try:
                    problem = pddl.parse_problem(prob_pddl)
                    objects = _get_objects(problem, domain)
                    goal_atoms = tokenizer._extract_goal_atoms(problem)
                    with open(traj_path, "r", encoding="utf-8") as f:
                        for line in f:
                            state_atoms = _parse_traj_line(line)
                            graph = tokenizer._build_labeled_graph(
                                state_atoms,
                                goal_atoms,
                                objects,
                            )
                            all_graphs.append(graph)
                except Exception:
                    continue

        if not all_graphs:
            raise RuntimeError("No graphs collected during pooled GraphBPE fit.")

        tokenizer._run_bpe(all_graphs)
        all_tokens: set[str] = set()
        for graph in all_graphs:
            all_tokens.update(graph["labels"].values())

        tokenizer._vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        tokenizer.embedding_dim = len(tokenizer._vocabulary)
        tokenizer._is_fitted = True
        return tokenizer

    raise ValueError(
        f"Tokenizer '{tokenizer_name}' does not support pooled fitting in this module."
    )


def build_all_domain_tokenizer(
    tokenizer_name: str,
    domain_specs: list[dict],
    strategy: str = "auto",
    **tokenizer_kwargs,
) -> tuple[TokenizationStrategy, str]:
    resolved_strategy = strategy
    if resolved_strategy == "auto":
        resolved_strategy = "union" if tokenizer_name == "wl" else "pooled"

    if resolved_strategy == "pooled":
        if tokenizer_name == "wl":
            raise ValueError(
                "The current wl tokenizer uses a single-domain wlplan feature generator. "
                "Use strategy='union' or strategy='auto' for all-domain WL."
            )
        return (
            fit_pooled_standard_tokenizer(
                tokenizer_name,
                domain_specs,
                **tokenizer_kwargs,
            ),
            resolved_strategy,
        )

    if resolved_strategy == "union":
        tokenizer = MultiDomainUnionTokenizer(
            base_tokenizer_name=tokenizer_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        tokenizer.fit_from_domain_specs(domain_specs)
        return tokenizer, resolved_strategy

    raise ValueError(f"Unknown all-domain tokenizer strategy '{strategy}'.")


def save_tokenizer_manifest(
    tokenizer: TokenizationStrategy,
    manifest_path: str,
    tokenizer_name: str,
    domains: list[str],
    fit_strategy: str,
    tokenizer_kwargs: dict | None = None,
) -> str:
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tokenizer, MultiDomainUnionTokenizer):
        tokenizer.save_vocabulary(str(manifest))
        with open(manifest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.update(
            {
                "tokenizer_name": tokenizer_name,
                "tokenizer_kwargs": tokenizer_kwargs,
                "fit_strategy": fit_strategy,
                "domains": list(domains),
            }
        )
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return str(manifest)

    vocab_path = manifest.with_name(f"{manifest.stem}_vocab.json")
    tokenizer.save_vocabulary(str(vocab_path))
    payload = {
        "representation_type": "standard_tokenizer",
        "tokenizer_name": tokenizer_name,
        "tokenizer_kwargs": tokenizer_kwargs,
        "fit_strategy": fit_strategy,
        "domains": list(domains),
        "embedding_dim": int(tokenizer.get_embedding_dim()),
        "vocab_relpath": os.path.relpath(vocab_path, manifest.parent),
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(manifest)


def load_tokenizer_from_manifest(manifest_path: str) -> TokenizationStrategy:
    manifest = Path(manifest_path)
    with open(manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)

    representation_type = payload.get("representation_type")
    if representation_type == "multi_domain_union":
        tokenizer = MultiDomainUnionTokenizer(
            base_tokenizer_name=payload["base_tokenizer_name"],
            tokenizer_kwargs=payload.get("tokenizer_kwargs", {}),
        )
        tokenizer.load_vocabulary(str(manifest))
        return tokenizer

    if representation_type == "standard_tokenizer":
        tokenizer = create_tokenizer(
            payload["tokenizer_name"],
            **payload.get("tokenizer_kwargs", {}),
        )
        vocab_relpath = str(payload["vocab_relpath"]).replace("\\", "/")
        vocab_path = manifest.parent / vocab_relpath
        tokenizer.load_vocabulary(str(vocab_path))
        return tokenizer

    raise ValueError(f"Unsupported tokenizer manifest type in {manifest_path}.")


