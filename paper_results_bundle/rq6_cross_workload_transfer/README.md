# RQ6: Cross-workload transfer and generalization

## Research question
How well do the tuned policies generalize across workload families when they are trained on one distribution and evaluated on another?

## Why this matters
A policy that is only reliable on the exact training workload may not be useful in deployed cloud systems, where popularity distributions shift across time, regions, and request mixes. This RQ tests whether the algorithms transfer beyond a single request pattern.

## Experimental design
- Train each algorithm on one workload family
- Evaluate on a different workload family or a shifted distribution
- Compute the transfer score and same-family score
- Report the performance gap as the transfer loss

## Evidence
This folder includes the generated benchmark outputs and figures:
- `results/rq6_results.json`
- `results/rq6_summary.csv`
- `results/figures/`

The benchmark runner is:
- `../../rq6_experiments/run_rq6_benchmark.py`

## Key metrics
- transfer score
- same-domain score
- transfer gap = same_domain_score - transfer_score
- total CHR
- total penalty
- retrieval latency

## Paper framing
This subsection answers whether the optimizer learns a transferable policy or overfits to one workload family. A small transfer gap suggests stronger generalization and more practical value in production cloud caching.
