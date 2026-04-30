"""
Unit tests for all tokenization strategies.

Tests are designed to work with the actual data in data/states/ and data/pddl/.
Run with: uv run python -m pytest tests/test_tokenizers.py -v
"""

import os
import re
import tempfile

import numpy as np
import pytest

# Base paths
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOMAIN = "blocks"
DOMAIN_PDDL = os.path.join(DATA_DIR, "pddl", DOMAIN, "domain.pddl")
TRAIN_STATES = os.path.join(DATA_DIR, "states", DOMAIN, "train")
TRAIN_PDDL = os.path.join(DATA_DIR, "pddl", DOMAIN, "train")

# Sample atoms for transform tests
SAMPLE_STATE_ATOMS = [
    "(on a b)",
    "(on b c)",
    "(clear a)",
    "(on-table c)",
    "(arm-empty)",
]
SAMPLE_GOAL_ATOMS = ["(on a b)", "(on b c)"]
SAMPLE_OBJECTS = ["a", "b", "c"]


# ---- Fixtures ----


@pytest.fixture(scope="module")
def has_data():
    """Check that training data exists; skip if not."""
    if not os.path.exists(DOMAIN_PDDL):
        pytest.skip(f"Data not found: {DOMAIN_PDDL}")
    if not os.path.exists(TRAIN_STATES):
        pytest.skip(f"Data not found: {TRAIN_STATES}")
    return True


# ---- SimHash Tests ----


class TestRandomTokenizer:
    def test_fixed_dimension(self, has_data):
        from code.tokenization.random import RandomTokenizer

        for dim in [64, 128, 256]:
            tok = RandomTokenizer(random_dim=dim, seed=42)
            tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
            emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
            assert emb.shape == (dim,), f"Expected ({dim},), got {emb.shape}"

    def test_determinism(self, has_data):
        from code.tokenization.random import RandomTokenizer

        tok1 = RandomTokenizer(random_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        tok2 = RandomTokenizer(random_dim=128, seed=42)
        tok2.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_seeds_differ(self, has_data):
        from code.tokenization.random import RandomTokenizer

        tok1 = RandomTokenizer(random_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        tok2 = RandomTokenizer(random_dim=128, seed=99)
        tok2.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        assert not np.array_equal(emb1, emb2)

    def test_save_load_roundtrip(self, has_data, tmp_path):
        from code.tokenization.random import RandomTokenizer

        tok1 = RandomTokenizer(random_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        save_path = os.path.join(str(tmp_path), "random_vocab.json")
        tok1.save_vocabulary(save_path)

        tok2 = RandomTokenizer()
        tok2.load_vocabulary(save_path)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)


class TestSimHash:
    def test_fixed_dimension(self, has_data):
        """Embedding dimension matches hash_dim regardless of input."""
        from code.tokenization.simhash import SimHashTokenizer

        for dim in [64, 128, 256]:
            tok = SimHashTokenizer(hash_dim=dim, seed=42)
            tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
            emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
            assert emb.shape == (dim,), f"Expected ({dim},), got {emb.shape}"

    def test_determinism(self, has_data):
        """Same seed produces identical embeddings."""
        from code.tokenization.simhash import SimHashTokenizer

        tok1 = SimHashTokenizer(hash_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        tok2 = SimHashTokenizer(hash_dim=128, seed=42)
        tok2.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)

    def test_different_seeds_differ(self, has_data):
        """Different seeds produce different embeddings."""
        from code.tokenization.simhash import SimHashTokenizer

        tok1 = SimHashTokenizer(hash_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        tok2 = SimHashTokenizer(hash_dim=128, seed=99)
        tok2.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        assert not np.array_equal(emb1, emb2)

    def test_binary_output(self, has_data):
        """Output values are 0 or 1 (binary hash)."""
        from code.tokenization.simhash import SimHashTokenizer

        tok = SimHashTokenizer(hash_dim=128, seed=42)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert set(np.unique(emb)).issubset({0.0, 1.0})

    def test_vocabulary_nonempty(self, has_data):
        """Feature vocabulary should be non-empty after fitting."""
        from code.tokenization.simhash import SimHashTokenizer

        tok = SimHashTokenizer(hash_dim=128, seed=42)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        assert len(tok._feature_keys) > 0

    def test_save_load_roundtrip(self, has_data, tmp_path):
        """Vocabulary save/load produces identical embeddings."""
        from code.tokenization.simhash import SimHashTokenizer

        tok1 = SimHashTokenizer(hash_dim=128, seed=42)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        save_path = os.path.join(str(tmp_path), "simhash_vocab.json")
        tok1.save_vocabulary(save_path)

        tok2 = SimHashTokenizer()
        tok2.load_vocabulary(save_path)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)


# ---- ShortestPath Tests ----


class TestShortestPath:
    def test_fixed_dimension(self, has_data):
        """Embedding dimension consistent with vocabulary size."""
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok = ShortestPathTokenizer(max_path_length=5)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert emb.shape == (tok.get_embedding_dim(),)

    def test_output_nonnegative(self, has_data):
        """Histogram features should be non-negative."""
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok = ShortestPathTokenizer(max_path_length=5)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert np.all(emb >= 0)

    def test_vocabulary_nonempty(self, has_data):
        """Path pattern vocabulary should be non-empty."""
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok = ShortestPathTokenizer(max_path_length=5)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        assert tok.get_embedding_dim() > 0

    def test_max_path_length_varies(self, has_data):
        """Different max_path_length should produce different vocab sizes."""
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok_short = ShortestPathTokenizer(max_path_length=2)
        tok_short.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)

        tok_long = ShortestPathTokenizer(max_path_length=10)
        tok_long.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)

        # Longer paths should produce equal or larger vocabulary
        assert tok_long.get_embedding_dim() >= tok_short.get_embedding_dim()

    def test_save_load_roundtrip(self, has_data, tmp_path):
        """Vocabulary save/load produces identical embeddings."""
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok1 = ShortestPathTokenizer(max_path_length=5)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        save_path = os.path.join(str(tmp_path), "sp_vocab.json")
        tok1.save_vocabulary(save_path)

        tok2 = ShortestPathTokenizer()
        tok2.load_vocabulary(save_path)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)


# ---- GraphBPE Tests ----


class TestGraphBPE:
    def test_fixed_dimension(self, has_data):
        """Embedding dimension matches vocabulary size."""
        from code.tokenization.graphbpe import GraphBPETokenizer

        tok = GraphBPETokenizer(vocab_size=500, num_iterations=10)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert emb.shape == (tok.get_embedding_dim(),)

    def test_output_nonnegative(self, has_data):
        """Histogram features should be non-negative."""
        from code.tokenization.graphbpe import GraphBPETokenizer

        tok = GraphBPETokenizer(vocab_size=500, num_iterations=10)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb = tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert np.all(emb >= 0)

    def test_merges_learned(self, has_data):
        """BPE should learn at least some merge operations."""
        from code.tokenization.graphbpe import GraphBPETokenizer

        tok = GraphBPETokenizer(vocab_size=500, num_iterations=10)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        assert len(tok._merges) > 0

    def test_save_load_roundtrip(self, has_data, tmp_path):
        """Vocabulary save/load produces identical embeddings."""
        from code.tokenization.graphbpe import GraphBPETokenizer

        tok1 = GraphBPETokenizer(vocab_size=500, num_iterations=10)
        tok1.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)
        emb1 = tok1.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        save_path = os.path.join(str(tmp_path), "bpe_vocab.json")
        tok1.save_vocabulary(save_path)

        tok2 = GraphBPETokenizer()
        tok2.load_vocabulary(save_path)
        emb2 = tok2.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

        np.testing.assert_array_equal(emb1, emb2)


# ---- Goal Embedding Tests ----


class TestGoalEmbedding:
    """Verify that all tokenizers produce valid goal embeddings."""

    @pytest.mark.parametrize("tokenizer_cls,kwargs", [
        ("RandomTokenizer", {"random_dim": 64, "seed": 42}),
        ("SimHashTokenizer", {"hash_dim": 64, "seed": 42}),
        ("ShortestPathTokenizer", {"max_path_length": 3}),
        ("GraphBPETokenizer", {"vocab_size": 100, "num_iterations": 5}),
    ])
    def test_goal_produces_valid_embedding(self, has_data, tokenizer_cls, kwargs):
        import importlib

        # Import the right module
        name_map = {
            "RandomTokenizer": "code.tokenization.random",
            "SimHashTokenizer": "code.tokenization.simhash",
            "ShortestPathTokenizer": "code.tokenization.shortest_path",
            "GraphBPETokenizer": "code.tokenization.graphbpe",
        }
        mod = importlib.import_module(name_map[tokenizer_cls])
        cls = getattr(mod, tokenizer_cls)

        tok = cls(**kwargs)
        tok.fit(DOMAIN_PDDL, TRAIN_STATES, TRAIN_PDDL)

        goal_emb = tok.transform_goal(SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)
        assert goal_emb.shape == (tok.get_embedding_dim(),)
        assert goal_emb.dtype == np.float32


# ---- Unfitted Error Tests ----


class TestUnfittedErrors:
    """Verify that unfitted tokenizers raise RuntimeError."""

    def test_simhash_unfitted(self):
        from code.tokenization.simhash import SimHashTokenizer

        tok = SimHashTokenizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

    def test_random_unfitted(self):
        from code.tokenization.random import RandomTokenizer

        tok = RandomTokenizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

    def test_shortest_path_unfitted(self):
        from code.tokenization.shortest_path import ShortestPathTokenizer

        tok = ShortestPathTokenizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)

    def test_graphbpe_unfitted(self):
        from code.tokenization.graphbpe import GraphBPETokenizer

        tok = GraphBPETokenizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tok.transform_state(SAMPLE_STATE_ATOMS, SAMPLE_GOAL_ATOMS, SAMPLE_OBJECTS)


class TestTokenizerFactory:
    def test_graphs_alias_maps_to_wl(self):
        pytest.importorskip("wlplan")
        from code.tokenization.factory import create_tokenizer
        from code.tokenization.wl import WLTokenizer

        tok = create_tokenizer("graphs", iterations=2)
        assert isinstance(tok, WLTokenizer)

    def test_random_factory(self):
        from code.tokenization.factory import create_tokenizer
        from code.tokenization.random import RandomTokenizer

        tok = create_tokenizer("random", random_dim=64, seed=123)
        assert isinstance(tok, RandomTokenizer)
