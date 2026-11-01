# RQ7 Experiment Notes

## Goal
Compare the tuned nature-inspired policies against standard classical cache replacement baselines under the same workload families.

## Baselines
- LRU
- LFU
- Random replacement

## Workload families
- Static Zipf
- Bursty/trending workload
- Drift workload

## Protocol
Each metaheuristic policy is trained on the train split and evaluated on the corresponding test split. Classical policies are evaluated directly on the same test split without a separate tuning phase.

## Reporting
The script saves:
- `results/rq7_results.json`
- `results/rq7_summary.csv`
- `results/figures/rq7_<workload>_comparison.png`

This benchmark answers whether the metaheuristic policies have an empirical advantage over simple but common eviction schemes.
