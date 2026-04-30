"""
Graph BPE (Byte-Pair Encoding) tokenizer for Instance Learning Graphs.

Implements a self-contained BPE algorithm on graph structures, inspired by
the GraphBPE method (Shen & Poczos, ICML 2024 AI4Science Workshop).

The original GraphBPE repo (https://github.com/A-Chicharito-S/GraphBPE)
is specialized for molecular graphs with heavy dependencies (torch_geometric,
hydra, hgraph2graph). This implementation adapts the core BPE concept
for planning ILGs without those dependencies.

Algorithm:
1. Initialize vocabulary: each unique node label is a token.
2. Iteratively merge the most frequent adjacent token pair into a new token.
3. After training, tokenize a graph by applying learned merges.
4. Create histogram embedding over the final vocabulary.

Reference: Shen, Y., & Poczos, B. (2024). GraphBPE: Molecular Graphs Meet
Byte-Pair Encoding. ICML 2024 AI for Science Workshop.
"""

import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from code.tokenization.base import TokenizationStrategy

logger = logging.getLogger(__name__)

_PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


def _progress_enabled() -> bool:
    return bool(sys.stdout.isatty())


class GraphBPETokenizer(TokenizationStrategy):
    """
    Graph BPE tokenizer: learns frequent substructure patterns via
    iterative merging of adjacent node labels.

    Vocabulary is built by iteratively merging the most frequent pair
    of adjacent labels in the training set. At inference, learned merges
    are applied to produce a token sequence, then a histogram is created.
    """

    def __init__(self, vocab_size: int = 1000, num_iterations: int = 100):
        super().__init__(name="GraphBPE")
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations

        # Learned merge rules: [(token_a, token_b, merged_token), ...]
        self._merges: list[tuple[str, str, str]] = []
        # Final vocabulary: {token_string -> index}
        self._vocabulary: dict[str, int] | None = None
        self._domain_info: dict[str, int] | None = None

    def fit(
        self,
        domain_pddl_path: str,
        train_states_dir: str,
        train_pddl_dir: str,
    ) -> None:
        """
        Learn BPE merge rules from training data.

        1. Extract initial token sequences from all training ILGs.
        2. Iteratively merge the most frequent adjacent pair.
        3. Build final vocabulary from all remaining tokens.
        """
        import pddl

        domain = pddl.parse_domain(domain_pddl_path)
        self._domain_info = {p.name.lower(): p.arity for p in domain.predicates}

        # 1. Extract all labeled graphs from training data
        all_graphs: list[dict] = []

        train_files = sorted(
            [f for f in os.listdir(train_states_dir) if f.endswith(".traj")]
        )

        for t_file in tqdm(
            train_files,
            desc=f"  [{self.name}] Parsing graphs",
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
                    state_atoms = _PREDICATE_REGEX.findall(line.strip())
                    graph = self._build_labeled_graph(state_atoms, goal_atoms, objects)
                    all_graphs.append(graph)

            except Exception:
                continue

        if not all_graphs:
            raise RuntimeError("No graphs collected during GraphBPE fit.")

        # 2. Run BPE merge iterations
        logger.info(
            f"[{self.name}] Running BPE on {len(all_graphs)} graphs, "
            f"target vocab={self.vocab_size}, max_iter={self.num_iterations}"
        )
        self._run_bpe(all_graphs)

        # 3. Build final vocabulary
        all_tokens: set[str] = set()
        for graph in all_graphs:
            for label in graph["labels"].values():
                all_tokens.add(label)

        sorted_tokens = sorted(all_tokens)
        self._vocabulary = {t: i for i, t in enumerate(sorted_tokens)}
        self.embedding_dim = len(self._vocabulary)
        self._is_fitted = True

        logger.info(
            f"[{self.name}] Fitted: {len(self._merges)} merges, "
            f"vocabulary size: {self.embedding_dim}"
        )

    def _extract_goal_atoms(self, problem) -> list[str]:
        """Extract goal atoms from a pddl Problem object."""
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

    def _build_labeled_graph(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> dict:
        """
        Build a graph with labeled nodes and edges.

        Returns dict with:
          - 'labels': {obj_name -> initial_label}
          - 'adj': {obj_name -> set of neighbor obj_names}
        """
        # Initial labels: based on predicates involving each object
        obj_state_preds: dict[str, list[str]] = defaultdict(list)
        obj_goal_preds: dict[str, list[str]] = defaultdict(list)
        adj: dict[str, set[str]] = {obj: set() for obj in objects}

        # Process state atoms
        for atom_str in state_atoms:
            parts = atom_str.split()
            if not parts:
                continue
            pred = parts[0].lower()
            args = [a.lower() for a in parts[1:]]

            for arg in args:
                if arg in adj:
                    obj_state_preds[arg].append(f"s:{pred}")

            if len(args) >= 2:
                for i in range(len(args)):
                    for j in range(i + 1, len(args)):
                        if args[i] in adj and args[j] in adj:
                            adj[args[i]].add(args[j])
                            adj[args[j]].add(args[i])

        # Process goal atoms
        for atom_str in goal_atoms:
            parts = atom_str.split()
            if not parts:
                continue
            pred = parts[0].lower()
            args = [a.lower() for a in parts[1:]]

            for arg in args:
                if arg in adj:
                    obj_goal_preds[arg].append(f"g:{pred}")

            if len(args) >= 2:
                for i in range(len(args)):
                    for j in range(i + 1, len(args)):
                        if args[i] in adj and args[j] in adj:
                            adj[args[i]].add(args[j])
                            adj[args[j]].add(args[i])

        # Create initial labels (sorted predicate sets for determinism)
        labels = {}
        for obj in objects:
            s_preds = sorted(set(obj_state_preds.get(obj, [])))
            g_preds = sorted(set(obj_goal_preds.get(obj, [])))
            label_parts = s_preds + g_preds
            labels[obj] = "|".join(label_parts) if label_parts else "EMPTY"

        return {"labels": labels, "adj": adj}

    def _run_bpe(self, graphs: list[dict]) -> None:
        """
        Run BPE merge iterations on the graphs.

        At each step:
        1. Count all adjacent label pair frequencies across all graphs.
        2. Pick the most frequent pair.
        3. Merge: for every edge where both endpoints have these labels,
           combine them into one super-node with merged label.
        """
        self._merges = []

        for iteration in range(self.num_iterations):
            # Count adjacent pairs across all graphs
            pair_counts: Counter = Counter()
            for graph in graphs:
                labels = graph["labels"]
                adj = graph["adj"]
                for node, neighbors in adj.items():
                    if node not in labels:
                        continue
                    node_label = labels[node]
                    for neighbor in neighbors:
                        if neighbor not in labels:
                            continue
                        nbr_label = labels[neighbor]
                        # Canonical ordering
                        pair = tuple(sorted([node_label, nbr_label]))
                        pair_counts[pair] += 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # No frequent pairs left

            merged_label = f"({best_pair[0]}+{best_pair[1]})"
            self._merges.append((best_pair[0], best_pair[1], merged_label))

            # Apply merge to all graphs
            for graph in graphs:
                self._apply_merge(graph, best_pair[0], best_pair[1], merged_label)

            if len(self._vocabulary or {}) + len(self._merges) >= self.vocab_size:
                break

    def _apply_merge(
        self,
        graph: dict,
        label_a: str,
        label_b: str,
        merged_label: str,
    ) -> None:
        """
        Apply a single merge to a graph.

        Finds all edges where one endpoint has label_a and the other label_b,
        contracts the first such edge found, and repeats.
        """
        labels = graph["labels"]
        adj = graph["adj"]

        # Find all edges eligible for this merge
        merged = True
        while merged:
            merged = False
            for node in list(labels.keys()):
                if node not in labels:
                    continue
                if labels[node] not in (label_a, label_b):
                    continue

                for neighbor in list(adj.get(node, set())):
                    if neighbor not in labels:
                        continue
                    # Check if this edge matches the pair
                    pair = tuple(sorted([labels[node], labels[neighbor]]))
                    if pair == tuple(sorted([label_a, label_b])):
                        # Contract: keep 'node', remove 'neighbor'
                        labels[node] = merged_label

                        # Transfer neighbor's edges to node
                        for n_nbr in adj.get(neighbor, set()):
                            if n_nbr != node and n_nbr in adj:
                                adj[node].add(n_nbr)
                                adj[n_nbr].discard(neighbor)
                                adj[n_nbr].add(node)

                        # Remove neighbor
                        del labels[neighbor]
                        adj[node].discard(neighbor)
                        if neighbor in adj:
                            del adj[neighbor]

                        merged = True
                        break

                if merged:
                    break

    def _apply_merges_to_graph(self, graph: dict) -> dict:
        """Apply all learned merges to a new graph (inference)."""
        import copy

        g = copy.deepcopy(graph)
        for label_a, label_b, merged_label in self._merges:
            self._apply_merge(g, label_a, label_b, merged_label)
        return g

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """
        Tokenize an ILG using learned BPE merges and create histogram.
        """
        self._check_fitted()

        # Parse atoms
        parsed_state = []
        for a in state_atoms:
            matches = _PREDICATE_REGEX.findall(a)
            parsed_state.extend(matches)

        parsed_goal = []
        for a in goal_atoms:
            a_clean = a.replace("(", "").replace(")", "").strip()
            if a_clean:
                parsed_goal.append(a_clean)

        # Build graph and apply merges
        graph = self._build_labeled_graph(parsed_state, parsed_goal, objects)
        merged_graph = self._apply_merges_to_graph(graph)

        # Create histogram
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for label in merged_graph["labels"].values():
            if label in self._vocabulary:
                embedding[self._vocabulary[label]] += 1.0

        return embedding

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
    ) -> np.ndarray:
        """Embed a goal by treating it as a state with no current atoms."""
        return self.transform_state([], goal_atoms, objects)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return self.embedding_dim

    def save_vocabulary(self, filepath: str) -> None:
        """Save merges and vocabulary to JSON."""
        self._check_fitted()
        data = {
            "vocab_size": self.vocab_size,
            "num_iterations": self.num_iterations,
            "merges": self._merges,
            "vocabulary": self._vocabulary,
            "embedding_dim": self.embedding_dim,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[{self.name}] Saved vocabulary to {filepath}")

    def load_vocabulary(self, filepath: str) -> None:
        """Load merges and vocabulary from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.num_iterations = data["num_iterations"]
        self._merges = [tuple(m) for m in data["merges"]]
        self._vocabulary = data["vocabulary"]
        self.embedding_dim = data["embedding_dim"]
        self._is_fitted = True

        logger.info(
            f"[{self.name}] Loaded: {len(self._merges)} merges, "
            f"vocab size: {self.embedding_dim}"
        )
