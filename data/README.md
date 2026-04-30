# Benchmark Data

This directory contains the benchmark inputs needed by the tokenizer study.

| File or directory | Purpose |
| --- | --- |
| `README.md` | This data guide. |
| `FILE_MANIFEST.tsv` | Generated file-by-file catalog for every PDDL and trajectory file under `pddl/` and `states/`. |
| `pddl/` | PDDL domain files, train/validation/test-interpolation/test-extrapolation problem files, and available `.soln` reference plans. |
| `states/` | Expert state trajectories aligned to PDDL problem names. Each `.traj` file is a sequence of symbolic state lines used for tokenizer fitting and predictor training/evaluation. |

The included domains are:

| Domain | Role in the study |
| --- | --- |
| `blocks` | Blocksworld-style manipulation with stack/support structure. |
| `gripper` | Two-room transport with grippers and balls. |
| `logistics` | Truck/airplane transport across cities and locations. |
| `visitall-from-everywhere` | Large graph-traversal benchmark with broad size variation. |

The included splits are:

| Split | Use |
| --- | --- |
| `train` | Fitting tokenizers and training transition predictors. |
| `validation` | Model selection and auxiliary validation reports. |
| `test-interpolation` | Held-out in-distribution/intermediate-size evaluation. |
| `test-extrapolation` | Held-out extrapolation evaluation. |

Full runs write generated embedding arrays through `code.encoding_generation.generate_multi_embeddings` and `code.encoding_generation.generate_all_domain_embeddings` inside a run directory.
