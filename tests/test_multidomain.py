import os
import re
import shutil
import uuid

import numpy as np
import pddl
import pddl.logic.predicates
import pytest

from code.tokenization.multidomain import (
    MultiDomainUnionTokenizer,
    build_all_domain_tokenizer,
    build_domain_specs,
    load_tokenizer_from_manifest,
    save_tokenizer_manifest,
)


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOMAINS = ["blocks", "gripper"]
PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")


@pytest.fixture(scope="module")
def has_data():
    for domain in DOMAINS:
        domain_pddl = os.path.join(DATA_DIR, "pddl", domain, "domain.pddl")
        train_dir = os.path.join(DATA_DIR, "states", domain, "train")
        if not os.path.exists(domain_pddl) or not os.path.exists(train_dir):
            pytest.skip(f"Missing data for {domain}")
    return True


def _extract_goal_atoms(problem) -> list[str]:
    goals: list[str] = []

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
    return goals


def _load_sample(domain: str):
    domain_pddl = os.path.join(DATA_DIR, "pddl", domain, "domain.pddl")
    train_pddl_dir = os.path.join(DATA_DIR, "pddl", domain, "train")
    train_states_dir = os.path.join(DATA_DIR, "states", domain, "train")

    traj_file = sorted(f for f in os.listdir(train_states_dir) if f.endswith(".traj"))[0]
    prob_name = traj_file.replace(".traj", "")
    problem_pddl = os.path.join(train_pddl_dir, f"{prob_name}.pddl")
    traj_path = os.path.join(train_states_dir, traj_file)

    parsed_domain = pddl.parse_domain(domain_pddl)
    problem = pddl.parse_problem(problem_pddl)
    objects = sorted({o.name for o in problem.objects} | {o.name for o in parsed_domain.constants})
    goal_atoms = _extract_goal_atoms(problem)

    with open(traj_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    state_atoms = [f"({match})" for match in PREDICATE_REGEX.findall(first_line)]

    return {
        "domain_pddl": domain_pddl,
        "problem_pddl": problem_pddl,
        "objects": objects,
        "goal_atoms": goal_atoms,
        "state_atoms": state_atoms,
    }


def test_union_manifest_roundtrip_random(has_data):
    specs = build_domain_specs(DATA_DIR, DOMAINS)
    tokenizer, strategy = build_all_domain_tokenizer(
        "random",
        specs,
        strategy="union",
        random_dim=32,
        seed=17,
    )
    assert strategy == "union"
    assert isinstance(tokenizer, MultiDomainUnionTokenizer)

    unique_id = uuid.uuid4().hex
    manifest_path = os.path.join(
        BASE_DIR,
        "tests",
        f"multidomain_manifest_{unique_id}.json",
    )
    component_dir = os.path.join(
        BASE_DIR,
        "tests",
        f"multidomain_manifest_{unique_id}_files",
    )
    try:
        manifest_path = save_tokenizer_manifest(
            tokenizer=tokenizer,
            manifest_path=manifest_path,
            tokenizer_name="random",
            domains=DOMAINS,
            fit_strategy=strategy,
            tokenizer_kwargs={"random_dim": 32, "seed": 17},
        )

        loaded = load_tokenizer_from_manifest(manifest_path)
        assert isinstance(loaded, MultiDomainUnionTokenizer)
        assert loaded.get_embedding_dim() == tokenizer.get_embedding_dim()

        for domain in DOMAINS:
            sample = _load_sample(domain)
            tokenizer.set_active_domain(domain, sample["domain_pddl"])
            loaded.set_active_domain(domain, sample["domain_pddl"])

            emb_expected = tokenizer.transform_state(
                sample["state_atoms"],
                sample["goal_atoms"],
                sample["objects"],
                problem_pddl_path=sample["problem_pddl"],
            )
            emb_loaded = loaded.transform_state(
                sample["state_atoms"],
                sample["goal_atoms"],
                sample["objects"],
                problem_pddl_path=sample["problem_pddl"],
            )
            np.testing.assert_allclose(emb_expected, emb_loaded)
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        shutil.rmtree(component_dir, ignore_errors=True)


