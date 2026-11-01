# RQ4: Efficiency, stability, and scalability

## Research question
How much quality gain do the nature-inspired algorithms achieve per unit of optimization cost, and how do they scale across workload families?

## Workloads used
To keep the comparison fair, RQ4 evaluates each optimizer on the same workload families used elsewhere in the benchmark suite:
- static Zipf workload
- bursty/trending workload
- drift workload

## Protocol
- Train each optimizer under a common optimization loop and fixed objective function
- Measure optimization time separately from test evaluation time
- Report both score and runtime to separate quality from computational cost
- Use the same generation and population sizes as the benchmark protocol so the comparison remains meaningful

## Metrics reported
- total score on the test workload
- optimization time
- evaluation time
- total runtime
- quality-per-second efficiency

## Interpretation
This benchmark is meant to answer the practical deployment question: does a small gain in cache hit ratio justify the increased optimization cost? A method that is marginally better but substantially slower may not be preferable in a production cloud cache.

## Output files
- `results/rq4_results.json`
- `results/rq4_summary.csv`
- `results/figures/static_zipf_rq4_runtime_quality.png`
- `results/figures/bursty_5x_rq4_runtime_quality.png`
- `results/figures/drift_08_to_16_rq4_runtime_quality.png`

## Paper framing
RQ4 supports the practical significance argument: the best method is not simply the one with the highest score, but the one that achieves strong performance without excessive optimization cost.
