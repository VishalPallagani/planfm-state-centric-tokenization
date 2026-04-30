# Tests

The tests are unit, integration, and regression checks for this repository. They are intentionally smaller than a full study run.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `test_tokenizers.py` | Unit tests for random, SimHash, shortest-path, GraphBPE, WL import/factory behavior, deterministic output, shape contracts, and save/load round trips. |
| `test_multidomain.py` | Tests all-domain union tokenizer manifest round trips and basic multidomain behavior on small domain samples. |
| `test_integration.py` | End-to-end tokenizer-to-`.npy` integration checks and dataset compatibility checks. |
| `test_experiment_suite.py` | Runs a reduced temporary-data study-runner check using XGBoost and two light tokenizers. |

Run with:

```bash
python -m pytest tests -q
```

Optional packages such as `wlplan`, `torch`, and `xgboost` are imported only where needed; tests skip rather than fail when optional runtime dependencies are unavailable.

