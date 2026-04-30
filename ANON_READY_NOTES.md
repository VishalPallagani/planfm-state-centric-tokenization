# Anonymous Review Packaging Notes

This repository mirrors the code, documentation, tests, compact outputs, PDDL files,
and benchmark state trajectories needed for review.

To keep Anonymous GitHub from timing out while indexing the largest trajectory split,
`data/states/visitall-from-everywhere/test-extrapolation/` is packaged as:

`data/states/visitall-from-everywhere/test-extrapolation.zip`

Extract that zip in place to restore the original directory layout before running
experiments that need the full visitall extrapolation trajectories.
