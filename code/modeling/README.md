# Modeling

This directory contains predictor training and inference code. Tokenizers emit fixed-dimensional state and goal vectors; these modules learn or apply transition predictors over those vectors.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `__init__.py` | Makes `code.modeling` importable. |
| `dataset.py` | Loads trajectory `.npy` files plus `_goal.npy` files, pads variable-length sequences for LSTM training, and flattens transitions for XGBoost. |
| `models.py` | Defines the LSTM predictor variants used for state and delta prediction. |
| `train_lstm.py` | Trains recurrent LSTM transition models with validation-based checkpoint selection. |
| `train_xgb.py` | Trains XGBoost multi-output regressors for state or delta prediction. |
| `inference_lstm.py` | Runs LSTM-guided symbolic beam search, scores grounded successors in tokenizer space, and records validated plan outcomes. |
| `inference_xgb.py` | Runs the corresponding XGBoost-guided symbolic beam search and records validated plan outcomes. |

The inference files use `pyperplan` to enumerate applicable symbolic successors and optionally use VAL through `code.common.utils.validate_plan`.
