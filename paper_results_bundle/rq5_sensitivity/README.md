# RQ5: Parameter sensitivity and reproducibility

## Research question
How stable are the nature-inspired optimizers across repeated runs and different random seeds?

## Experimental design
For each optimizer, run the same workload over multiple seeds and measure:
- mean fitness score
- standard deviation of score
- runtime variation

This quantifies reproducibility and robustness to stochastic initialization.

## Key metrics
- mean score
- std score
- mean runtime
- rank stability across seeds

## Paper framing
A method that wins only on one lucky seed is less convincing than one that is consistently competitive across repeated runs.

## Experiment notes
See the benchmark notes in `../../rq5_experiments/EXPERIMENT_NOTES.md` for the workload protocol, metrics, and interpretation.
