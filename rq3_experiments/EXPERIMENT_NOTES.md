# RQ3 workload-family consistency benchmark

This folder evaluates the nature-based cache optimizers under a consistent three-family workload protocol:

1. Static Zipf workloads
2. Bursty/trending workloads
3. Drift workloads with train/test distribution shift

All algorithms are compared under the same cache-sizing and fitness setup used elsewhere in the repository:

- same fitness objective: 0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency
- same 10% cache sizing logic from the training catalog
- same seed for fairness across algorithms
- same evaluation style: train on the workload train split, then evaluate on the workload test split

The benchmark runner is:

```bash
python -m rq3_experiments.run_rq3_benchmark
```

This benchmark is intentionally compact enough to complete in a reasonable local run window while still comparing the full set of nature-inspired algorithms: GA, PSO, SFO, DSOA, ABC, HO, and WOA.
