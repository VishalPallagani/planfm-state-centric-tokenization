# Tokenizer Study Report

## Scope

- Domains: blocks, gripper, logistics, visitall-from-everywhere
- Tokenizers: wl, simhash, shortest_path, graphbpe, random
- Models: lstm, xgboost
- Modes: state, delta
- Seeds: 13, 23, 37
- Device request: cuda
- All-domain tokenizer strategy: auto
- Skip VAL validation: False

## Statistical Protocol

- Aggregate metrics are reported as means and population standard deviations across seeds.
- Pairwise tokenizer and regime tests use exact McNemar for single-seed comparisons.
- Multi-seed comparisons use a paired sign-flip randomization test on per-problem seed-mean solved outcomes.
- Mean-difference confidence intervals are nonparametric bootstrap 95% intervals over per-problem differences.
- Holm correction is applied within each significance family.

## Top Configurations

| Rank | Regime | Tokenizer | Model | Mode | Weighted Solved Mean | Weighted Solved Std |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | domain_dependent | wl | xgboost | delta | 89.94% | 0.00% |
| 2 | all_domains | wl | xgboost | delta | 83.80% | 0.00% |
| 3 | all_domains | shortest_path | xgboost | state | 80.73% | 0.00% |
| 4 | all_domains | wl | lstm | delta | 80.45% | 2.57% |
| 5 | domain_dependent | shortest_path | lstm | delta | 80.26% | 0.80% |
| 6 | all_domains | shortest_path | lstm | state | 79.52% | 0.13% |
| 7 | domain_dependent | shortest_path | lstm | state | 79.42% | 0.13% |
| 8 | domain_dependent | random | xgboost | delta | 79.33% | 0.00% |
| 9 | all_domains | shortest_path | xgboost | delta | 79.33% | 0.00% |
| 10 | all_domains | shortest_path | lstm | delta | 79.05% | 0.99% |
| 11 | domain_dependent | wl | lstm | delta | 78.68% | 0.13% |
| 12 | domain_dependent | shortest_path | xgboost | delta | 78.49% | 0.00% |

## Best Tokenizer Per Regime/Model/Mode

| Regime | Model | Mode | Best Tokenizer | Weighted Solved Mean | Weighted Exec Mean |
| --- | --- | --- | --- | --- | --- |
| all_domains | lstm | delta | wl | 80.45% | 100.00% |
| all_domains | lstm | state | shortest_path | 79.52% | 100.00% |
| all_domains | xgboost | delta | wl | 83.80% | 100.00% |
| all_domains | xgboost | state | shortest_path | 80.73% | 100.00% |
| domain_dependent | lstm | delta | shortest_path | 80.26% | 100.00% |
| domain_dependent | lstm | state | shortest_path | 79.42% | 100.00% |
| domain_dependent | xgboost | delta | wl | 89.94% | 100.00% |
| domain_dependent | xgboost | state | shortest_path | 76.82% | 100.00% |

## Regime Comparisons

| Tokenizer | Model | Mode | Domain-Dependent | All-Domains | Diff (all-domains minus domain-dependent) | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wl | lstm | state | 47.58% | 74.58% | 27.00% | 5e-05 | 5e-05 |
| wl | lstm | delta | 78.68% | 80.45% | 1.77% | 0.106 | 0.106 |
| wl | xgboost | state | 32.68% | 34.36% | 1.68% | 0.1805 | 0.1805 |
| wl | xgboost | delta | 89.94% | 83.80% | -6.15% | 5e-05 | 5e-05 |
| simhash | lstm | state | 39.39% | 43.85% | 4.47% | 0.01975 | 0.01975 |
| simhash | lstm | delta | 22.07% | 16.11% | -5.96% | 0.0001 | 0.0001 |
| simhash | xgboost | state | 35.47% | 31.56% | -3.91% | 0.1414 | 0.1414 |
| simhash | xgboost | delta | 22.35% | 16.48% | -5.87% | 0.0015 | 0.0015 |
| shortest_path | lstm | state | 79.42% | 79.52% | 0.09% | 1 | 1 |
| shortest_path | lstm | delta | 80.26% | 79.05% | -1.21% | 0.0079 | 0.0079 |
| shortest_path | xgboost | state | 76.82% | 80.73% | 3.91% | 0.012 | 0.012 |
| shortest_path | xgboost | delta | 78.49% | 79.33% | 0.84% | 0.3743 | 0.3743 |
| graphbpe | lstm | state | 23.74% | 24.12% | 0.37% | 0.2537 | 0.2537 |
| graphbpe | lstm | delta | 23.56% | 23.28% | -0.28% | 0.2756 | 0.2756 |
| graphbpe | xgboost | state | 23.74% | 24.39% | 0.65% | 0.2469 | 0.2469 |
| graphbpe | xgboost | delta | 23.74% | 24.58% | 0.84% | 0.2478 | 0.2478 |
| random | lstm | state | 17.60% | 14.71% | -2.89% | 0.00835 | 0.00835 |
| random | lstm | delta | 43.30% | 23.37% | -19.93% | 5e-05 | 5e-05 |
| random | xgboost | state | 16.20% | 70.39% | 54.19% | 5e-05 | 5e-05 |
| random | xgboost | delta | 79.33% | 60.61% | -18.72% | 5e-05 | 5e-05 |