# RQ6 Experiment Notes

## Goal
Evaluate whether a policy trained on one workload family retains useful cache decisions when tested on a different workload family.

## Workload families
- Static Zipf
- Bursty/trending workloads
- Drift workloads (train on one Zipf alpha, test on another)

## Transfer protocol
For each source-target pair, the optimizer is trained on the source workload and then evaluated on the target workload's untouched test split. The same optimizer is also trained and evaluated on the target workload to compute the in-domain baseline.

## Reporting
The script saves:
- `results/rq6_results.json`
- `results/rq6_summary.csv`
- `results/figures/rq6_<source>_to_<target>.png`

This benchmark is designed to quantify generalization and identify whether the tuned policy overfits a single workload family.
