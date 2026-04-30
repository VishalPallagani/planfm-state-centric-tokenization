# Code Directory

This package contains the executable code for the tokenizer study. The package name is `code`, matching the experiment commands.

| File or directory | Purpose |
| --- | --- |
| `__init__.py` | Marks `code/` as an importable package for commands such as `python -m code.experiments.run_tokenizer_study`. |
| `common/` | Shared reproducibility, validation, WL, and fixed-size compatibility helpers. |
| `encoding_generation/` | Scripts that fit tokenizers and write `.npy` trajectory/goal embeddings. |
| `experiments/` | Main orchestration and statistical analysis code. |
| `modeling/` | LSTM/XGBoost training, inference, dataset loading, and neural model definitions. |
| `tokenization/` | Tokenizer implementations and all-domain tokenizer composition utilities. |

The package covers tokenizer fitting, representation generation, predictor training, inference, analysis, and tests for this study.
