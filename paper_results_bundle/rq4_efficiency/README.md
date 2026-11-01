# RQ4: Efficiency, stability, and scalability

## Research question
How much quality gain do the nature-inspired algorithms achieve per unit of optimization cost, and how do they scale across workload families?

## Experimental design
This benchmark measures runtime against score for each optimizer on:
- static Zipf workload
- bursty/trending workload
- drift workload

The output is a runtime-vs-quality summary that answers if the extra optimization time is worth the performance gain.

## Key metrics
- score on the test workload
- optimization training time
- total runtime
- quality-per-second efficiency score

## Paper framing
This section enables a practical argument: an optimizer that is slightly better but much slower may not be preferable in deployment.

## Experiment notes
See the benchmark notes in `../../rq4_experiments/EXPERIMENT_NOTES.md` for the workload protocol, metrics, and interpretation.
