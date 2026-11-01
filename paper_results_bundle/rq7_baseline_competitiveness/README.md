# RQ7: Baseline competitiveness against classical cache policies

## Research question
Do the nature-inspired algorithms consistently outperform classical replacement policies under fair evaluation conditions?

## Why this matters
A metaheuristic optimizer is only valuable if it beats practical baselines that are widely used in production systems, such as LRU, LFU, and Random replacement.

## Baselines included
- LRU
- LFU
- Random replacement

## Experimental design
- Use the same workload families as the main benchmark: static Zipf, bursty/trending, and drift
- Train each optimizer using the same protocol as the tuned methods
- Evaluate the classical baselines on the same test workloads
- Compare final CHR, penalty, and latency

## Evidence
This folder includes the generated benchmark outputs and figures:
- `results/rq7_results.json`
- `results/rq7_summary.csv`
- `results/figures/`

The benchmark runner is:
- `../../rq7_experiments/run_rq7_benchmark.py`

## Key metrics
- total CHR
- HPC CHR
- total penalty
- average retrieval time
- score gap to the best baseline
- workload-wise win rate

## Paper framing
This section establishes practical significance. It answers whether the additional optimization complexity is justified in real caching workloads, or whether simple policies remain competitive enough to be preferred in operational settings.
