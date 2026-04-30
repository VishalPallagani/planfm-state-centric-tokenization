# Encoding Generation

This directory contains the scripts that convert symbolic trajectories into model-ready numeric arrays.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `__init__.py` | Makes `code.encoding_generation` importable. |
| `generate_multi_embeddings.py` | Fits one tokenizer per domain and writes domain-dependent trajectory and goal embeddings for each split. This is the domain-dependent data path used by the full study. |
| `generate_all_domain_embeddings.py` | Fits or loads the all-domain tokenizer representation and writes pooled/all-domain trajectory and goal embeddings for every domain and split. |

Both scripts output `.npy` files consumed by `code.modeling.dataset.PlanningTrajectoryDataset` and by the XGBoost flattening loader. Full runs write generated embedding arrays under ignored output directories.
