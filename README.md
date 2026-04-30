# Tokenizer Choice Shapes Generalization In State-Centric Learning for Planning

This repository contains code for **state-centric tokenization in learned transition-model planning**. It includes the experiment runner, tokenizer implementations, benchmark PDDL instances, expert trajectory files, tests, and compact CSV/JSON/Markdown result summaries.

The uploaded files are sufficient to run the test suite, regenerate tokenizer representations, train LSTM and XGBoost transition models, run symbolic inference, validate plans with VAL, and recompute the included analysis tables.

## Study Overview

The implementation studies how symbolic-state tokenization changes the behavior of learned transition-model planning. The planner works over PDDL-style symbolic states. For each state and goal, a tokenizer produces a fixed-dimensional representation. A learned model predicts either the next state representation or the delta from the current state representation. Search then evaluates symbolic successor states by comparing their tokenizer representations against the learned prediction.

The main question is not whether a single tokenizer can solve a single domain. The study compares tokenizer families under a controlled factorial design so the effect of the representation can be separated from model family, prediction target, domain, training protocol, and random seed.

The study design is:

| Factor | Values |
| --- | --- |
| Tokenizer | `wl`, `simhash`, `shortest_path`, `graphbpe`, `random` |
| Predictor | `lstm`, `xgboost` |
| Prediction target | `state`, `delta` |
| Training protocol | `domain_dependent`, `all_domains` |
| Domain | `blocks`, `gripper`, `logistics`, `visitall-from-everywhere` |
| Seed | `13`, `23`, `37` |
| Evaluation split | `validation`, `test-interpolation`, `test-extrapolation` |

The full runner regenerates tokenizer files, builds representation arrays, trains predictors, runs symbolic inference, validates produced plans with VAL, and writes aggregate analysis tables.

## Pipeline In One Pass

The experiment pipeline is organized around five stages:

1. **Input preparation.** `data/pddl/` provides PDDL domain and problem files. `data/states/` provides expert symbolic state trajectories aligned to those problems.
2. **Tokenizer fitting.** Domain-dependent tokenizers are fit separately per planning domain. All-domain tokenizers are fit over pooled domains or represented as a union of per-domain tokenizers, depending on tokenizer support.
3. **Representation generation.** The fitted tokenizers turn trajectory states and goals into numeric arrays used by the learned predictors.
4. **Model training and inference.** LSTM and XGBoost predictors are trained for `state` and `delta` targets. Inference performs bounded symbolic search while using the learned representation prediction to rank successor states.
5. **Validation and analysis.** Candidate plans are checked with VAL when available. The analysis script aggregates solved rates, per-split comparisons, bootstrap intervals, and paired sign-flip tests.

## Repository Map

```text
code/
  common/                 Shared file, JSON, VAL, WL, and utility helpers.
  encoding_generation/    Domain-dependent and all-domain representation builders.
  experiments/            Full-study runner, configuration, and statistical analysis.
  modeling/               LSTM and XGBoost training and inference modules.
  tokenization/           Tokenizer implementations and tokenizer factory code.
data/
  pddl/                   PDDL domains, problem files, and available reference plans.
  states/                 Expert state trajectories used for fitting and training.
outputs/
  analysis/               Compact analysis outputs from the completed run.
  statistical_summaries/ Compact statistical tables for secondary analyses.
  baselines/             Classical-planner baseline summaries.
scripts/                  Full-study, analysis-only, and baseline command wrappers.
tests/                    Unit, integration, multidomain, and reduced runner checks.
docs/                     Reproducibility notes and supporting documentation.
```

Root-level files:

| File | Purpose |
| --- | --- |
| `.gitignore` | Prevents local caches, environment directories, external VAL builds, generated NumPy arrays, trained model checkpoints, and inference-result directories from entering version control. |
| `.python-version` | Python version hint for local environments. |
| `environment.yml` | Conda environment specification for the study code. |
| `pyproject.toml` | Python project metadata and dependency list for `uv` or pip workflows. |
| `LICENSE` | Repository license. |
| `README.md` | Main guide and reproduction entry point. |
| `FILE_CATALOG.md` | File-by-file inventory for the repository. |

Every file is documented in the nearest directory README and in `FILE_CATALOG.md`. The benchmark corpus also has a machine-readable manifest at `data/FILE_MANIFEST.tsv`.

## Important File Groups

`code/tokenization/` contains the representation methods under study. `base.py` defines the common interface. `factory.py` maps tokenizer names to implementations. `wl.py`, `simhash.py`, `shortest_path.py`, `graphbpe.py`, and `random.py` implement the tokenizer families. `multidomain.py` handles all-domain tokenizer manifests and union behavior.

`code/modeling/` contains the learned transition models. `dataset.py` builds supervised examples from trajectory representations. `models.py` defines neural modules. `train_lstm.py` and `train_xgb.py` fit predictors. `inference_lstm.py` and `inference_xgb.py` run symbolic search guided by those predictors.

`code/experiments/` contains the orchestrating scripts. `run_tokenizer_study.py` is the main entry point for full regeneration. `analyze_tokenizer_study.py` reads completed run directories and writes compact tables in `outputs/`. `config.py` centralizes tokenizer and domain settings.

`data/` contains the benchmark inputs used by the code. The PDDL files define domains and held-out problems. The `.traj` files contain expert state trajectories used to fit tokenizers and train predictors. The split names indicate how each problem participates in model selection or evaluation.

`outputs/` contains compact summaries from the completed run. Full reruns create additional NumPy representation arrays, tokenizer model files, trained model checkpoints, per-problem inference JSON files, VAL validation records, and command logs.

## Generated Files

Full runs create the following generated paths. They are ignored by Git because the code and benchmark data can regenerate them:

- `outputs/domain_dependent/`: domain-specific tokenizer model files, NumPy representation arrays, trained model checkpoints, per-problem inference JSON files, and VAL validation records.
- `outputs/all_domains/`: pooled tokenizer model files, NumPy representation arrays, trained model checkpoints, per-problem inference JSON files, and VAL validation records.
- `outputs/logs/`: generated command logs from full runs.
- `outputs/repro/`: reproducibility metadata written by the runner.
- `data/encodings/`: optional precomputed representation arrays. The main runner writes new arrays under `outputs/`.
- `VAL/`: external VAL validator source, build directories, and executable files.
- Trained LSTM checkpoint files, XGBoost model files, and NumPy representation matrices created by new runs.

The compact CSV, JSON, and Markdown summaries needed to inspect the numerical results are included under `outputs/analysis/`, `outputs/statistical_summaries/`, and `outputs/baselines/`.

## Installation

The code targets Python 3.10 or newer. Create the environment from one of the provided specifications.

With `uv`:

```bash
uv sync --group dev
```

With conda:

```bash
conda env create -f environment.yml
conda activate state-centric-tokenization
```

For CUDA reproduction, install a PyTorch build matching your driver and CUDA stack if the default wheel is CPU-only on your platform. The reported full run used CUDA-enabled PyTorch, CUDA-enabled XGBoost, and one CUDA GPU.

## Quick Verification

Run the test suite:

```bash
python -m pytest tests -q
```

The tests use the included benchmark data and temporary output directories. Optional packages such as `wlplan`, `torch`, and `xgboost` are imported only where needed. Tests that require unavailable optional runtime dependencies skip rather than fail.

## Install VAL

VAL is the external plan validator used to decide whether generated action sequences solve the PDDL problems. It is not vendored in this repository.

Useful VAL links:

- Official VAL source repository: `https://github.com/KCL-Planning/VAL`
- KCL Planning VAL page: `https://nms.kcl.ac.uk/planning/software/val.html`

The upstream VAL repository describes OS-specific binary downloads from its current CI build page and source builds for Windows, Linux, and macOS. The runner only needs the `Validate` executable.

Recommended layout for this repository:

```text
VAL/bin/Validate
VAL/bin/Validate.exe
VAL/build/bin/Validate
VAL/build/bin/Validate.exe
```

Any one of those paths is auto-detected. If your executable is elsewhere, pass it explicitly:

```bash
python -m code.experiments.run_tokenizer_study \
  --val_path /absolute/path/to/Validate \
  ...
```

On Windows, the official VAL README lists CMake, MinGW-w64, Strawberry Perl, LLVM, and Doxygen as build requirements, and provides scripts under `scripts/windows/`. The produced executable may live inside a platform-specific build directory under `build/win64/`. Copy `Validate.exe` to `VAL/bin/Validate.exe` or pass that full path with `--val_path`.

On Linux, the official VAL README lists Debian packages such as `cmake`, `make`, `g++`, `mingw-w64`, `flex`, and `bison`, and provides `scripts/linux/build_linux64.sh`. After building, copy or link the produced `Validate` executable to `VAL/bin/Validate`, or pass the build path with `--val_path`.

On macOS, the official VAL README points to the macOS build scripts and Xcode command-line tool requirements. As on other platforms, either place the resulting `Validate` executable under `VAL/bin/` or pass the executable path explicitly.

A minimal sanity check after installation is:

```bash
VAL/bin/Validate \
  data/pddl/blocks/domain.pddl \
  data/pddl/blocks/test-interpolation/probBLOCKS-5-0.pddl \
  data/pddl/blocks/test-interpolation/probBLOCKS-5-0.pddl.soln
```

Use the `.exe` suffix on Windows. If VAL prints that the plan is valid, the executable is ready for the full experiment.

## Reproduce The Full Study

Install VAL first for validator-backed solved-rate evaluation. Then run:

```bash
python -m code.experiments.run_tokenizer_study \
  --output_root . \
  --study_name outputs \
  --resume_existing \
  --tokenizers wl simhash shortest_path graphbpe random \
  --domains blocks gripper logistics visitall-from-everywhere \
  --models lstm xgboost \
  --modes state delta \
  --seeds 13 23 37 \
  --device cuda \
  --num_workers 8 \
  --xgb_n_jobs 8 \
  --fast \
  --validation_workers 8 \
  --analysis_seed 13 \
  --inference_steps_per_object 10
```

PowerShell helper:

```powershell
.\scripts\run_tokenizer_study_full.ps1
```

This writes into `outputs/`. Full runs create subdirectories for NumPy representation arrays, tokenizer model files, trained model checkpoints, command logs, reproducibility metadata, per-problem inference JSON files, and VAL validation records. Those generated directories are ignored by version control.

## Reproduce Analysis From A Completed Run

If the full run already exists, rerun only the numerical aggregation and statistical analysis:

```bash
python -m code.experiments.analyze_tokenizer_study \
  --run_root outputs \
  --seed 13
```

PowerShell helper:

```powershell
.\scripts\run_tokenizer_analysis_only.ps1 -RunRoot .\outputs
```

The key outputs are:

| Output | Meaning |
| --- | --- |
| `analysis/split_summary.csv` | Per-domain, per-split solved-rate summaries by tokenizer, model, mode, seed, and training protocol. |
| `analysis/aggregate_split_summary.csv` | Aggregated split-level summaries across seeds and domains. |
| `analysis/tokenizer_comparison_seeded.csv` | Seed-aware tokenizer comparisons. |
| `analysis/tokenizer_comparison_aggregate.csv` | Aggregate tokenizer comparison table. |
| `analysis/overall_summary.csv` | Top-level solved-rate summary across the study factors. |
| `analysis/pairwise_tokenizer_significance_overall.csv` | Overall paired tokenizer tests. |
| `analysis/pairwise_tokenizer_significance_by_domain_split.csv` | Paired tokenizer tests broken down by domain and split. |
| `analysis/regime_significance_overall.csv` | Overall all-domain versus domain-dependent comparisons. |
| `analysis/regime_significance_by_domain_split.csv` | Regime comparisons broken down by domain and split. |
| `analysis/embedding_stats.json` | Representation dimensionality and tokenizer metadata summaries. |
| `analysis/study_report.md` | Markdown analysis report generated by the analysis script. |

## Classical Baselines

Classical-planner baselines are generated with:

```bash
python scripts/run_classical_baselines.py \
  --pddl_root data/pddl \
  --baselines gbf:hff astar:lmcut \
  --timeout_sec 10 \
  --workers 8 \
  --output_dir outputs/baselines
```

The included baseline summaries are:

- `outputs/baselines/classical_baseline_raw.csv`
- `outputs/baselines/classical_baseline_summary.csv`

## Notes On Validation Metrics

Solved rate in the output tables means VAL-approved plan success. Internal search success alone is not treated as the solved-rate metric. The `--skip_validation` flag exists for local debugging when VAL is unavailable, but it should not be used when regenerating validator-backed solved-rate results.
