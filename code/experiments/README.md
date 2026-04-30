# Experiment Orchestration

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `__init__.py` | Makes `code.experiments` importable. |
| `config.py` | Central experiment configuration: domains, evaluation splits, tokenizer hyperparameters, model hyperparameters, and the default seed. |
| `run_tokenizer_study.py` | Main end-to-end runner for the tokenizer study. It builds embeddings, trains predictors, runs inference, validates plans, and launches analysis. |
| `analyze_tokenizer_study.py` | Aggregates completed run outputs into CSV, JSON, and Markdown summaries, including pairwise tokenizer and regime comparisons. |

The main commands in this directory are the full study runner and the analysis rerunner.
