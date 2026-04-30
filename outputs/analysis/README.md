# Analysis Outputs

These files are compact CSV/JSON/Markdown outputs from the tokenizer-study analysis pass.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `aggregate_split_summary.csv` | Aggregated solved/executable/plan-length summaries grouped by regime, tokenizer, model, mode, domain, and split. |
| `best_config_by_regime_model_mode.csv` | Best tokenizer configuration for each regime/model/formulation family. |
| `embedding_stats.json` | Fitted tokenizer dimensions and fit-strategy metadata for domain-dependent and all-domain representations. |
| `overall_summary.csv` | Overall method summaries aggregated across splits and domains. |
| `pairwise_tokenizer_significance_by_domain_split.csv` | Pairwise tokenizer significance comparisons resolved by domain and split. |
| `pairwise_tokenizer_significance_overall.csv` | Overall pairwise tokenizer significance comparisons. |
| `study_report.md` | Markdown narrative summary produced by the analysis script. |
| `regime_significance_by_domain_split.csv` | Domain/split-level comparisons between domain-dependent and all-domain training. |
| `regime_significance_overall.csv` | Overall domain-dependent versus all-domain significance comparisons. |
| `split_summary.csv` | Seed-level and method-level split summaries used as the base for many aggregate reports. |
| `tokenizer_comparison_aggregate.csv` | Aggregate tokenizer comparisons across seeds. |
| `tokenizer_comparison_seeded.csv` | Seed-resolved tokenizer comparison rows. |
