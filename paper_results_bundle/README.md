# Paper-ready benchmark bundle

This folder organizes the project outputs in the order used for the research paper.

## Paper order
1. Abstract
2. Introduction
3. Related Work
4. Experimental Setup and Fairness Protocol
5. RQ1: Matched real-trace benchmark
6. RQ2: Synthetic workload robustness (Zipf, bursty, trending)
7. RQ3: Drift robustness under train/test distribution shift
8. RQ4: Efficiency, stability, and scalability
9. RQ5: Sensitivity and reproducibility
10. RQ6: Cross-workload transfer and generalization
11. RQ7: Baseline competitiveness against classical cache policies
12. Discussion and limitations
13. Conclusion

## Folder structure
- `rq1_real_trace/` — matched real-trace 10k holdout benchmark and comparison tables
- `rq2_workload_robustness/` — synthetic workload family summary and figure directory
- `rq3_drift/` — drift benchmark summary and generated plot folder
- `rq4_efficiency/` — runtime-vs-quality and scalability analysis
- `rq5_sensitivity/` — seed sensitivity and reproducibility analysis
- `rq6_cross_workload_transfer/` — generalization and transfer analysis
- `rq7_baseline_competitiveness/` — classical policy comparison against LRU/LFU/Random

## Primary evidence
- `rq1_real_trace/PSO_SFO_10K_COMPARISON.md`
- `rq3_drift/rq3_summary.json`
- `rq3_drift/figures/`

This is the ordering to follow when drafting the paper narrative.
