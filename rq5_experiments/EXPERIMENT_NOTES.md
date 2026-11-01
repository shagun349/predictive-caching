# RQ5: Parameter sensitivity and reproducibility

## Research question
How stable are the nature-inspired optimizers across repeated runs and different random seeds?

## Workload used
The sensitivity benchmark uses a static Zipf workload with a fixed train/test split to isolate instability due to optimizer stochasticity rather than workload variation.

## Protocol
- Run the same optimizer on the same workload across multiple seeds
- Keep the workload, cache model, and optimization objective fixed
- Measure the score variance and runtime variance across seeds
- Report both mean and standard deviation to quantify reproducibility

## Metrics reported
- mean score
- score standard deviation
- mean runtime
- runtime variation
- seed-wise rank stability

## Interpretation
An optimizer that is highly sensitive to initialization or random perturbations may appear strong in a single lucky run but be unreliable in repeated deployment settings. This benchmark captures that risk directly.

## Output files
- `results/rq5_results.json`
- `results/rq5_summary.csv`
- `results/figures/rq5_seed_stability.png`

## Paper framing
RQ5 addresses robustness under stochasticity. It strengthens the paper by showing whether the observed performance is a stable result or a seed-dependent artifact.
