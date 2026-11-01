# RQ2: Synthetic workload robustness

This section organizes the workload robustness study across:

1. static Zipf workloads
2. bursty/trending workloads
3. workload-shift or skew-change scenarios

## Workload families
- static Zipf: stationary popularity with varying heavy-tail skew
- bursty/trending: temporary hot-object bursts and attention spikes
- drift: train on one distribution and test on another

## Evidence folder
The synthetic benchmark scaffolding lives in the repo under `rq2_experiments/` and `rq3_experiments/`.

This folder is reserved for the polished paper-ready figures and tables to be exported once the full RQ2 sweep is run at publication scale.

## Planned paper narrative
- Compare all algorithms under static Zipf workloads
- Compare the same algorithms under bursty/trending attention shifts
- Emphasize robustness, adaptation speed, and stability under non-stationary request patterns
