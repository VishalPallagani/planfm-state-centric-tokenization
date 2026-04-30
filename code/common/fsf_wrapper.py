import re

import numpy as np
import pddl

# Special Values
VAL_PAD = -99.0  # Slot is unused (outside problem size)
VAL_DONTCARE = -10.0  # Goal value when variable is not specified


class FSFEncoder:
    def __init__(self, domain_name, domain_pddl_path, max_objects):
        """
        max_objects: The count of distinct objects in the largest problem. The actual vector size will be max_objects + 1 (for Global Slot 0).
        """
        self.domain_name = domain_name
        self.domain_pddl = pddl.parse_domain(domain_pddl_path)

        # Vector Size = Max Objects + 1 (Index 0 is reserved for Global/Robot)
        self.vector_size = max_objects + 1

        print(
            f"  [FSF] Initialized with {self.vector_size} slots (Max Objects: {max_objects} + 1 Global)"
        )

        # Regex to correctly parse "(pred arg1 arg2)" groups
        self.predicate_regex = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")

    def _get_sorted_objects(self, problem_path):
        """
        Parses problem to get all objects, including domain constants.
        Returns sorted list of object names.
        """
        problem = pddl.parse_problem(problem_path)

        objs = set()
        for o in problem.objects:
            objs.add(o.name)
        for o in self.domain_pddl.constants:
            objs.add(o.name)

        return sorted(list(objs))

    def _get_object_indices(self, objects):
        # Index 0 is Global. Objects start at 1.
        return {o: i + 1 for i, o in enumerate(objects)}

    def parse_state_atoms(self, state_lines):
        """
        Parses list of strings using Regex to separate predicates.
        Input: ['(on a b) (clear c)']
        Output: [('on', 'a', 'b'), ('clear', 'c')]
        """
        atoms = []
        for line in state_lines:
            # Find all matches of (pred arg1 arg2 ...)
            matches = self.predicate_regex.findall(line)
            for m in matches:
                parts = m.split()
                if parts:
                    atoms.append(tuple([p.lower() for p in parts]))
        return atoms

    def embed_trajectory(self, problem_path, trajectory_file, verbose=False):
        """
        Reads a .traj file and returns [T, MAX_OBJECTS] matrix.
        """
        objects = self._get_sorted_objects(problem_path)
        obj_map = self._get_object_indices(objects)

        if verbose:
            print(f"\nDEBUG: {problem_path}")
            print(f"Objects ({len(objects)}): {objects[:5]} ...")
            print(f"Map: {list(obj_map.items())[:5]} ...")

        with open(trajectory_file, "r") as f:
            lines = f.readlines()

        vectors = []
        for i, line in enumerate(lines):
            atoms = self.parse_state_atoms([line])

            # Debug first and last step
            is_debug_step = verbose and (i == 0 or i == len(lines) - 1)

            vec = self._state_to_vector(atoms, objects, obj_map, debug=is_debug_step)
            vectors.append(vec)

        return np.array(vectors, dtype=np.float32)

    def embed_goal(self, problem_path):
        """
        Parses problem goal and returns [MAX_OBJECTS] vector.
        Unspecified variables are set to VAL_DONTCARE.
        """
        objects = self._get_sorted_objects(problem_path)
        obj_map = self._get_object_indices(objects)

        problem = pddl.parse_problem(problem_path)

        # Extract goal atoms
        goal_atoms = []

        def visit(node):
            if hasattr(node, "name") and hasattr(node, "terms"):
                # Predicate
                args = [t.name if hasattr(t, "name") else str(t) for t in node.terms]
                goal_atoms.append(tuple([node.name] + args))
            elif hasattr(node, "operands"):
                for op in node.operands:
                    visit(op)

        visit(problem.goal)

        # Generate vector with DONTCARE as default
        return self._state_to_vector(goal_atoms, objects, obj_map, is_goal=True)

    def _state_to_vector(self, atoms, objects, obj_map, is_goal=False, debug=False):
        """
        Core Logic: Maps atoms to vector based on domain rules.
        """
        # 1. Initialize
        # If it's a goal, default is DONTCARE.
        # If it's a state, default is PAD (we fill valid slots below).
        default_fill = VAL_DONTCARE if is_goal else VAL_PAD

        # Create vector of size N+1
        vec = np.full(self.vector_size, default_fill, dtype=np.float32)

        # Initialize Valid Slots to 0.0 for states
        if not is_goal:
            # Global Slot (0) defaults to 0
            vec[0] = 0.0
            # Object Slots (1..N) default to 0.0 (e.g. Table/Unvisited/Free)
            for i in range(len(objects)):
                slot = i + 1
                if slot < self.vector_size:
                    vec[slot] = 0.0

        # Helper to get value (index) of an object
        def get_val(name):
            return float(obj_map.get(name, 0))

        # Helper to get slot (index) of an object
        def get_slot(name):
            s = obj_map.get(name, -1)
            if s >= self.vector_size:
                # This should theoretically not happen if we scanned correctly
                return -1
            return s

        if debug:
            print(f"Processing Atoms: {atoms}")

        # DOMAIN SPECIFIC LOGIC

        if "blocks" in self.domain_name:
            # Slot i = Block i
            # Values: 0 (Table), -1 (Held), k (On block k)

            # First pass: Set held
            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "holding":
                    # (holding a) -> V[a] = -1
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = -1.0
                        if debug:
                            print(f"  Set {args[0]} (slot {slot}) = -1.0 (Held)")

                elif name == "on":
                    # (on a b) -> V[a] = Index(b)
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = get_val(args[1])
                        if debug:
                            print(
                                f"  Set {args[0]} (slot {slot}) = {get_val(args[1])} (On {args[1]})"
                            )
                elif name == "ontable":
                    # (ontable a) -> V[a] = 0
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = 0.0

        elif "gripper" in self.domain_name:
            # Slot 0: Robot Location (Room Index)
            # Slot i (Ball): Room Index OR -1 * Gripper Index
            # Slot i (Gripper): 0 (Free) or Ball Index (Holding)

            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "at-robby":
                    vec[0] = get_val(args[0])
                    if debug:
                        print(
                            f"  Set Global (slot 0) = {get_val(args[0])} (Robby at {args[0]})"
                        )
                elif name == "at":
                    # (at ball room) -> V[ball] = Index(room)
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = get_val(args[1])
                        if debug:
                            print(
                                f"  Set {args[0]} (slot {slot}) = {get_val(args[1])} (At {args[1]})"
                            )
                elif name == "carry":
                    # (carry ball gripper)
                    ball, gripper = args
                    b_slot = get_slot(ball)
                    g_slot = get_slot(gripper)
                    if b_slot != -1:
                        vec[b_slot] = -1.0 * get_val(gripper)
                        if debug:
                            print(
                                f"  Set {ball} (slot {b_slot}) = {-1.0 * get_val(gripper)} (Carried)"
                            )
                    if g_slot != -1:
                        vec[g_slot] = get_val(ball)
                        if debug:
                            print(
                                f"  Set {gripper} (slot {g_slot}) = {get_val(ball)} (Holding)"
                            )

        elif "logistics" in self.domain_name:
            # Slot i (Pkg/Truck/Plane): Location Index
            # Slot i (Pkg in vehicle): -1 * Vehicle Index

            for pred in atoms:
                name = pred[0]
                args = pred[1:]
                if name == "at":
                    # (at obj loc)
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = get_val(args[1])
                        if debug:
                            print(
                                f"  Set {args[0]} (slot {slot}) = {get_val(args[1])} (At {args[1]})"
                            )
                elif name == "in":
                    # (in pkg vehicle)
                    pkg, veh = args
                    slot = get_slot(pkg)
                    if slot != -1:
                        vec[slot] = -1.0 * get_val(veh)
                        if debug:
                            print(
                                f"  Set {pkg} (slot {slot}) = {-1.0 * get_val(veh)} (In {veh})"
                            )

        elif "visit" in self.domain_name:  # visitall
            # Slot 0: Robot Location (Cell Index)
            # Slot i (Cell): 0 (Unvisited), 1 (Visited)

            for pred in atoms:
                name = pred[0]
                args = pred[1:]

                if name == "at-robot":
                    vec[0] = get_val(args[0])
                    if debug:
                        print(
                            f"  Set Global (slot 0) = {get_val(args[0])} (Robot at {args[0]})"
                        )
                elif name == "visited":
                    # (visited cell) -> V[cell] = 1
                    slot = get_slot(args[0])
                    if slot != -1:
                        vec[slot] = 1.0
                        if debug:
                            print(f"  Set {args[0]} (slot {slot}) = 1.0 (Visited)")

        return vec
