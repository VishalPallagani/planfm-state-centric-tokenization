"""
Centralized experiment configuration for multi-tokenization comparison.

Defines tokenizer parameters, model hyperparameters, domains, and splits.
"""

DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS_EVAL = ["validation", "test-interpolation", "test-extrapolation"]
SPLITS_ALL = ["train"] + SPLITS_EVAL

TOKENIZATION_CONFIGS = {
    "wl": {
        "module": "code.tokenization.wl",
        "class": "WLTokenizer",
        "params": {"iterations": 2},
        "description": "Weisfeiler-Leman (k=2) color refinement",
        "encoding_dir": "graphs",  # Maps to existing data/encodings/graphs
    },
    "simhash": {
        "module": "code.tokenization.simhash",
        "class": "SimHashTokenizer",
        "params": {"hash_dim": 128, "seed": 42},
        "description": "Random projection hashing (SimHash)",
        "encoding_dir": "simhash",
    },
    "shortest_path": {
        "module": "code.tokenization.shortest_path",
        "class": "ShortestPathTokenizer",
        "params": {"max_path_length": 5},
        "description": "Shortest-path kernel features",
        "encoding_dir": "shortest_path",
    },
    "graphbpe": {
        "module": "code.tokenization.graphbpe",
        "class": "GraphBPETokenizer",
        "params": {"vocab_size": 1000, "num_iterations": 100},
        "description": "Byte-pair encoding on graph structures",
        "encoding_dir": "graphbpe",
    },
    "random": {
        "module": "code.tokenization.random",
        "class": "RandomTokenizer",
        "params": {"random_dim": 128, "seed": 42},
        "description": "Deterministic random baseline embeddings",
        "encoding_dir": "random",
    },
}

MODEL_CONFIGS = {
    "lstm": {
        "state_mode": {
            "hidden_dim": 256,
            "epochs": 500,
            "batch_size": 32,
            "lr": 1e-2,
            "no_projection": False,
        },
        "delta_mode": {
            "hidden_dim": 256,
            "epochs": 500,
            "batch_size": 32,
            "lr": 1e-2,
            "no_projection": False,
        },
    },
    "xgboost": {
        "state_mode": {
            "n_estimators": 1000,
            "max_depth": 8,
            "lr": 0.1,
            "early_stopping": 10,
        },
        "delta_mode": {
            "n_estimators": 1000,
            "max_depth": 8,
            "lr": 0.1,
            "early_stopping": 10,
        },
    },
}

SEED = 13
