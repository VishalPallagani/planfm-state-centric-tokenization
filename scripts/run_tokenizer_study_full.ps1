$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$StudyName = "outputs"

python -m code.experiments.run_tokenizer_study `
  --output_root . `
  --study_name $StudyName `
  --resume_existing `
  --tokenizers wl simhash shortest_path graphbpe random `
  --domains blocks gripper logistics visitall-from-everywhere `
  --models lstm xgboost `
  --modes state delta `
  --seeds 13 23 37 `
  --device cuda `
  --num_workers 8 `
  --xgb_n_jobs 8 `
  --fast `
  --validation_workers 8 `
  --analysis_seed 13 `
  --inference_steps_per_object 10
