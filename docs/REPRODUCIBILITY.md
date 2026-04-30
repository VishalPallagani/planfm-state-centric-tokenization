# Reproducibility Notes

## Supported Reproduction Paths

This repository supports two primary reproduction paths.

1. **Code and data checks.** Run `python -m pytest tests -q` to verify tokenizer contracts, all-domain tokenizer behavior, representation array compatibility, and the reduced runner path used by the test suite.
2. **Full study regeneration.** Run the full command in the root `README.md` with CUDA-capable PyTorch, CUDA-capable XGBoost, and VAL. This regenerates tokenizer model files, NumPy representation arrays, trained model checkpoints, per-problem inference JSON files, VAL validation records, and CSV/JSON/Markdown analysis summaries.

If a completed full run already exists, the analysis-only command in the root `README.md` reruns the numerical aggregation and statistical tests without retraining models.

## Data

The benchmark inputs are `data/pddl/` and `data/states/`. PDDL files define domains and problem instances. `.traj` files contain expert state trajectories consumed by tokenizer fitting and model training. `data/FILE_MANIFEST.tsv` gives one row per benchmark file, including split, domain, file type, and size.

Splits have the following roles:

- `train`: tokenizer fitting and model training.
- `validation`: model selection and auxiliary reporting.
- `test-interpolation`: held-out problems within the training size regime.
- `test-extrapolation`: held-out problems beyond the training size regime.

## Main Runner

`code.experiments.run_tokenizer_study` orchestrates the full study. It:

- writes runtime environment metadata,
- fits domain-dependent tokenizers,
- generates domain-dependent representation arrays,
- fits all-domain tokenizers,
- generates all-domain representation arrays,
- trains LSTM and XGBoost predictors,
- runs symbolic inference for every requested domain, split, method, and seed,
- validates plans with VAL unless `--skip_validation` is explicitly set,
- runs `code.experiments.analyze_tokenizer_study`.

The script writes into the selected run directory. The root README configures the full run to use `outputs/` directly. The `.gitignore` ignores generated tokenizer models, NumPy arrays, trained model checkpoints, inference JSON files, validation records, and command logs while allowing compact analysis summaries to be versioned.

## Analysis

`code.experiments.analyze_tokenizer_study` consumes a completed run directory and writes:

- split-level and aggregate solved-rate summaries,
- tokenizer comparisons,
- all-domain versus domain-dependent comparisons,
- bootstrap intervals and paired tests,
- compact CSV, JSON, and Markdown summaries.

The `outputs/analysis/` directory contains compact analysis outputs from the completed run. The `outputs/statistical_summaries/` directory contains secondary statistical summaries.

## VAL

VAL is external and must be installed separately. The official source repository is `https://github.com/KCL-Planning/VAL`, and the KCL Planning VAL page is `https://nms.kcl.ac.uk/planning/software/val.html`.

The runner auto-detects these local paths:

```text
VAL/build/bin/Validate.exe
VAL/build/bin/Validate
VAL/bin/Validate.exe
VAL/bin/Validate
```

If VAL is installed elsewhere, pass `--val_path /path/to/Validate`. In full study reproduction, solved rate means VAL-approved plan validity, not merely internal search termination.

## Hardware

The included summaries were generated with one CUDA GPU, CUDA-enabled PyTorch, CUDA-enabled XGBoost, and eight CPU workers for validation and XGBoost jobs. CPU-only machines can run the unit and integration tests, but full reproduction is expected to be substantially slower without GPU acceleration.
