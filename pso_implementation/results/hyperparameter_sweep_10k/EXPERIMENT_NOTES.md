# PSO 10k Hyperparameter Sweep Notes

## What was run

The sweep script, `pso_implementation/run_pso_hyperparameter_sweep.py`, compares
ten PSO configurations on `Dataset/cluster07(00).csv`. Each configuration uses
the same 10,000-request training block, 40,000-request validation block, and
40,000-request holdout block:

| Split | Request indices (1-based, inclusive) | Requests |
| --- | ---: | ---: |
| Training | 1–10,000 | 10,000 |
| Validation | 10,001–50,000 | 40,000 |
| Holdout | 50,001–90,000 | 40,000 |

Every configuration uses 30 particles, 25 PSO iterations, and random seed 42.
The three seeded starting particles (QM-ARC-style, LRU, and LFU) and the
remaining randomized particles are unchanged between configurations. The
fitness weights are also fixed across the sweep: total CHR 0.40, HPC CHR 0.35,
normalized penalty 0.15, and normalized latency 0.10. The fitness formula is
`0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`.

## Hyperparameter changes and validation results

`w_start` and `w_end` are the beginning and ending inertia weights; inertia
decays linearly during each run. `c1` is the cognitive (personal-best)
coefficient, and `c2` is the social (global-best) coefficient. Each row changes
one setting from the baseline except `balanced_18`, which changes both
acceleration coefficients. The validation score is the selection metric; a
higher score is better.

| Configuration | `w_start` | `w_end` | `c1` | `c2` | Best training fitness | Validation score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.9 | 0.4 | 1.5 | 1.5 | 0.445288 | 0.453127 |
| inertia_08_04 | 0.8 | 0.4 | 1.5 | 1.5 | 0.445721 | 0.453112 |
| inertia_10_04 | 1.0 | 0.4 | 1.5 | 1.5 | 0.445572 | 0.451903 |
| inertia_09_02 | 0.9 | 0.2 | 1.5 | 1.5 | 0.444677 | 0.451443 |
| inertia_09_06 | 0.9 | 0.6 | 1.5 | 1.5 | 0.446367 | **0.453811** |
| cognitive_10 | 0.9 | 0.4 | 1.0 | 1.5 | 0.445672 | 0.452593 |
| cognitive_20 | 0.9 | 0.4 | 2.0 | 1.5 | 0.443750 | 0.450221 |
| social_10 | 0.9 | 0.4 | 1.5 | 1.0 | 0.445019 | 0.451015 |
| social_20 | 0.9 | 0.4 | 1.5 | 2.0 | 0.445143 | 0.452279 |
| balanced_18 | 0.9 | 0.4 | 1.8 | 1.8 | 0.445445 | 0.452262 |

`inertia_09_06` was selected using validation only. Its holdout results were
67.62% total CHR, 69.96% HPC CHR, 66.59% LPC CHR, 309,275 total penalty, and
0.03437 seconds average retrieval time.

The hyperparameters above are held constant for the duration of each
configuration's run. The 25 PSO iterations are optimization generations in
which particle positions/velocities and best-fitness values evolve; they are
not 25 additional manual hyperparameter changes. The per-generation best and
average fitness and elapsed time are recorded in
[`training_histories.json`](./training_histories.json). Full per-configuration
metrics and the selected holdout result are in
[`final_results.json`](./final_results.json); the progress checkpoint is
[`sweep_progress.json`](./sweep_progress.json).

## Simulated cache and storage model

These are simulator parameters, not measurements of a physical machine:

| Modeled resource | Value | Basis |
| --- | ---: | --- |
| Unique-object catalog in training block | 6,135,728 bytes | Sum of unique training-object sizes |
| Total cache capacity | 613,572 bytes | 10% of that training catalog |
| DRAM cache tier | 122,714 bytes | 20% of total cache, integer-truncated |
| SSD cache tier | 490,857 bytes | 80% of total cache, integer-truncated |
| DRAM latency | 100 ns | Simulator constant |
| DRAM read bandwidth | 40 GB/s | Simulator constant |
| DRAM write bandwidth | 20 GB/s | Simulator constant |
| SSD latency | 10 µs | Simulator constant |
| SSD read bandwidth | 2 GB/s | `ssd_throughput_gb=2.0` |
| SSD write bandwidth | 1 GB/s | Half the configured SSD read throughput |
| Simulator slot size | Mean object size in the preprocessed trace | Passed to the simulator; exact value was not saved in the result JSON |
| Upstream response time | Per trace record | `response_time` from the dataset; no single fixed RTT |

The independently truncated tier limits add to 613,571 bytes, one byte less
than total cache capacity. This follows the simulator's separate integer
calculations for the 20% and 80% tiers.

The SSD figure above is the capacity of the **simulated SSD cache tier**, not
the size of an SSD installed in the computer.

## Physical run-host specifications

The saved sweep outputs do not record the physical run host's CPU model/core
count, installed RAM, operating system, Python version, or SSD/HDD model and
capacity. Therefore these details cannot be reliably stated from this run's
artifacts. The simulation defines DRAM/SSD capacities, latencies, and
bandwidths in software; these modeled values should not be mistaken for the
computer's actual hardware specifications.

## Reproducibility

From the repository root, run:

```bash
python -m pso_implementation.run_pso_hyperparameter_sweep
```

The script writes the result JSON files and plots under
`pso_implementation/results/hyperparameter_sweep_10k/`. Runtime durations in
those files are observations from the original run and can vary with the
machine and its load.
