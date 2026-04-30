param(
  [Parameter(Mandatory = $true)]
  [string]$RunRoot,

  [int]$Seed = 13
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

python -m code.experiments.analyze_tokenizer_study `
  --run_root $RunRoot `
  --seed $Seed
