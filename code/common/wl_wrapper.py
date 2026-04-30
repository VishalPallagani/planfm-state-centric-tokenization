import glob
import hashlib
import os
import re

import numpy as np
import pddl
import pddl.logic.base
import pddl.logic.predicates
from tqdm import tqdm

# Regex to parse "(on a b)" -> "on a b"
PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


class WLEncoder:
    def __init__(self, domain_pddl_path, iterations=2):
        self.domain_pddl_path = domain_pddl_path
        self.iterations = iterations

        # Vocabulary: Map specific WL hash strings to integer indices
        self.vocab = {}
        self.is_collected = False

        # Parse domain using the 'pddl' library (robust)
        print(f"  [GC-WL] Parsing Domain: {domain_pddl_path}")
        self.pddl_domain = pddl.parse_domain(domain_pddl_path)

        # Cache domain predicates to know arity (unary vs binary)
        # p.name is string, p.arity is int
        self.domain_info = {
            p.name.lower(): p.arity for p in self.pddl_domain.predicates
        }

    def _get_initial_graph(self, objects, state_atoms, goal_atoms):
        """
        Builds a graph where nodes = objects.
        Features include current state AND goal state info.
        """
        # 1. Initialize Nodes
        # Graph structure: {obj_name: {'attributes': [], 'neighbors': []}}
        graph = {obj: {"attributes": [], "neighbors": []} for obj in objects}

        # 2. Process State Atoms
        for atom in state_atoms:
            parts = atom.replace("(", "").replace(")", "").lower().split()
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]

            if pred not in self.domain_info:
                continue  # Skip unknown predicates (e.g. equality)

            arity = self.domain_info[pred]

            if arity == 1 and len(args) == 1:
                # Unary State Feature: clear(a) -> a has attr "state-clear"
                if args[0] in graph:
                    graph[args[0]]["attributes"].append(f"state-{pred}")
            elif arity == 2 and len(args) == 2:
                # Binary State Edge: on(a, b) -> a -[state-on]-> b
                u, v = args
                if u in graph and v in graph:
                    graph[u]["neighbors"].append((f"state-{pred}", v))
                    # We treat edges as directed. For WL, we can add inverse if needed,
                    # but standard directed WL is usually fine for planning.

        # 3. Process Goal Atoms (The "Goal-Aware" part)
        for atom in goal_atoms:
            parts = atom.replace("(", "").replace(")", "").lower().split()
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]

            if pred not in self.domain_info:
                continue

            arity = self.domain_info[pred]

            if arity == 1 and len(args) == 1:
                # Unary Goal Feature: goal-clear(a)
                if args[0] in graph:
                    graph[args[0]]["attributes"].append(f"goal-{pred}")
            elif arity == 2 and len(args) == 2:
                # Binary Goal Edge: goal-on(a, b)
                u, v = args
                if u in graph and v in graph:
                    graph[u]["neighbors"].append((f"goal-{pred}", v))

        # 4. Sort attributes for determinism
        for obj in graph:
            graph[obj]["attributes"].sort()
            graph[obj]["neighbors"].sort()

        return graph

    def _compute_wl_hashes(self, graph):
        """
        Runs k-iterations of Weisfeiler-Leman.
        Returns a list of all colors found in the final graph.
        """
        # Initial Coloring: Hash of attributes
        # current_colors: {obj_name: hash_string}
        current_colors = {}
        for obj, data in graph.items():
            # Hash the sorted list of attributes
            attr_str = "|".join(data["attributes"])
            current_colors[obj] = hashlib.md5(attr_str.encode()).hexdigest()

        # Iterations
        for _ in range(self.iterations):
            new_colors = {}
            for obj in graph:
                # Collect neighbor colors
                # neighbor_desc = list of (edge_label, neighbor_color)
                neighbors = graph[obj]["neighbors"]
                neighbor_descriptors = []
                for label, neighbor in neighbors:
                    neighbor_descriptors.append(f"{label}:{current_colors[neighbor]}")

                # Sort to ensure invariance to neighbor order
                neighbor_descriptors.sort()

                # Aggregate: (SelfColor, Neighbors)
                aggregate_str = (
                    current_colors[obj] + "||" + ",".join(neighbor_descriptors)
                )
                new_hash = hashlib.md5(aggregate_str.encode()).hexdigest()
                new_colors[obj] = new_hash

            current_colors = new_colors

        # Return all node colors (multiset)
        return list(current_colors.values())

    def parse_state_string_to_atoms(self, state_str_or_list):
        if isinstance(state_str_or_list, str):
            return PREDICATE_REGEX.findall(state_str_or_list)
        return state_str_or_list

    def parse_pddl_goal(self, problem_path):
        """Extracts goal atoms and objects from PDDL using the 'pddl' library."""
        # Parse problem
        problem = pddl.parse_problem(problem_path)

        # Collect objects (problem objects + domain constants)
        objects = set()
        for o in problem.objects:
            objects.add(o.name)
        for o in self.pddl_domain.constants:
            objects.add(o.name)

        # Extract Goal Atoms recursively
        goals = []

        def visit(node):
            if isinstance(node, pddl.logic.predicates.Predicate):
                # node.name is predicate name, node.terms are arguments
                # Handle terms that might be objects or just strings
                args = [t.name if hasattr(t, "name") else str(t) for t in node.terms]
                s = f"({node.name} {' '.join(args)})"
                goals.append(s)
            elif hasattr(node, "operands"):  # Handle And, Or, etc.
                for op in node.operands:
                    visit(op)
            elif hasattr(node, "_operands"):  # Fallback for older pddl versions
                for op in node._operands:
                    visit(op)
            # Note: We ignore 'Not' for graph edges usually, or can be handled it if needed.

        visit(problem.goal)

        return sorted(list(objects)), goals

    def collect_vocabulary(self, train_states_dir):
        print(f"  [GC-WL] Collecting vocabulary from {train_states_dir}...")
        self.vocab = {}
        unique_hashes = set()

        train_files = sorted(glob.glob(os.path.join(train_states_dir, "*.traj")))
        pddl_train_dir = train_states_dir.replace("states", "pddl")

        for traj_file in tqdm(train_files, desc="  Parsing Traces"):
            prob_name = os.path.splitext(os.path.basename(traj_file))[0]
            prob_pddl = os.path.join(pddl_train_dir, f"{prob_name}.pddl")

            if not os.path.exists(prob_pddl):
                continue

            try:
                # 1. Get Objects and Goal
                objects, goal_atoms = self.parse_pddl_goal(prob_pddl)

                # 2. Read Trajectory
                with open(traj_file, "r") as f:
                    lines = f.read().strip().split("\n")

                # 3. Process states
                for line in lines:
                    state_atoms = self.parse_state_string_to_atoms(line)
                    graph = self._get_initial_graph(objects, state_atoms, goal_atoms)
                    colors = self._compute_wl_hashes(graph)
                    unique_hashes.update(colors)

            except Exception:
                # print(f"Error reading {prob_name}: {e}")
                pass

        # Build Vocab Map
        sorted_hashes = sorted(list(unique_hashes))
        self.vocab = {h: i for i, h in enumerate(sorted_hashes)}
        self.is_collected = True
        print(f"  [GC-WL] Vocabulary collected. Size: {len(self.vocab)}")

    def embed_state(self, state_atoms_or_obj, problem_pddl_path):
        if not self.is_collected:
            raise RuntimeError("Vocab not collected")

        # Handle input types (if legacy code passes State objects)
        if hasattr(state_atoms_or_obj, "atoms"):
            # Extract string representation from wlplan State object if passed
            state_atoms = []
            for atom in state_atoms_or_obj.atoms:
                args = " ".join(atom.objects)
                state_atoms.append(f"({atom.predicate.name} {args})")
        else:
            state_atoms = self.parse_state_string_to_atoms(state_atoms_or_obj)

        # We need objects and goals again
        objects, goal_atoms = self.parse_pddl_goal(problem_pddl_path)

        # Build Graph
        graph = self._get_initial_graph(objects, state_atoms, goal_atoms)

        # Run WL
        colors = self._compute_wl_hashes(graph)

        # Vectorize (Histogram)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for c in colors:
            if c in self.vocab:
                vec[self.vocab[c]] += 1.0

        return vec

    # Adapter methods to match existing interface
    def parse_state_string_to_wl_state(self, s):
        return s

    def parse_pddl_goal_to_wl_state(self, p):
        _, goals = self.parse_pddl_goal(p)
        return goals
