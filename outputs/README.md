# Outputs

This directory stores compact result summaries from a completed tokenizer-study run.

| File or directory | Purpose |
| --- | --- |
| `README.md` | This guide. |
| `analysis/` | Aggregated CSV, JSON, and Markdown outputs from `code.experiments.analyze_tokenizer_study`. |
| `statistical_summaries/` | Secondary statistical summaries, search-effort diagnostics, and method-comparison tables. |
| `baselines/` | Classical-planner baseline CSVs for contextual comparisons. |

New full-study runs can write directly into `outputs/`. Full runs also create generated subdirectories such as `all_domains/`, `domain_dependent/`, `logs/`, and `repro/` for NumPy representation arrays, tokenizer model files, trained model checkpoints, per-problem inference JSON files, VAL validation records, command logs, and reproducibility metadata. Those generated subdirectories are ignored by Git.
