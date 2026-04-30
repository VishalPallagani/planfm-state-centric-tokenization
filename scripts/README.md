# Scripts

These scripts are convenience entry points around the package modules.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `run_tokenizer_study_full.ps1` | PowerShell wrapper for the full tokenizer-study configuration. |
| `run_tokenizer_analysis_only.ps1` | PowerShell wrapper for rerunning analysis on an existing completed study directory. |
| `run_classical_baselines.py` | Runs contextual classical-planning baselines with `pyperplan` over held-out splits and writes baseline CSV summaries. |

The scripts assume they are run from the repository root so relative paths such as `outputs` resolve correctly.
