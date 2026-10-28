# GA 10k Hyperparameter Sweep

## Protocol

This sweep uses the same trace, preprocessing, data partitions, cache sizing,
and objective weights as the PSO and SFO 10k sweeps.

| Split | Request indices (1-based, inclusive) | Requests |
| --- | ---: | ---: |
| Training | 1–10,000 | 10,000 |
| Validation | 10,001–50,000 | 40,000 |
| Holdout | 50,001–90,000 | 40,000 |

The trace is `Dataset/cluster07(00).csv`, preprocessed with a 30% high-priority
ratio and seed 42. Each configuration uses a population of 30, 25 generations,
and seed 42. The selected configuration is chosen using validation score only;
the holdout block is evaluated once after selection.

This is one seeded run, not an average across repeated seeds.

The fixed fitness is
`0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`.
Penalty is normalized by 20 times the number of requests in the corresponding
split, and latency is normalized by 150 seconds. The same formula is used for
training and validation selection.

## GA settings compared

The baseline is the existing GA setup: crossover rate 0.85, mutation rate 0.20,
three elites, tournament size three, SBX crossover, and Gaussian mutation.
Each other configuration changes one setting while keeping population,
generations, seed, fitness, and trace splits fixed.

| Configuration | Changed setting |
| --- | --- |
| baseline | Existing GA defaults |
| crossover_70 | Crossover rate 0.70 |
| crossover_95 | Crossover rate 0.95 |
| mutation_10 | Mutation rate 0.10 |
| mutation_30 | Mutation rate 0.30 |
| elite_1 | Elites: 1 |
| elite_5 | Elites: 5 |
| tournament_2 | Tournament size: 2 |
| tournament_5 | Tournament size: 5 |
| blend_gaussian | BLX-style blend crossover |

## Validation results

The winner was selected by validation score; higher is better.

| Configuration | Best training fitness | Validation score | Training time (s) |
| --- | ---: | ---: | ---: |
| baseline | 0.4456 | 0.452122 | 263.6 |
| crossover_70 | 0.4433 | 0.449071 | 274.0 |
| crossover_95 | 0.4453 | 0.451118 | 304.4 |
| mutation_10 | 0.4446 | 0.452249 | 294.1 |
| mutation_30 | 0.4453 | 0.452210 | 299.8 |
| elite_1 | 0.4448 | 0.452099 | 286.4 |
| elite_5 | 0.4457 | **0.452464** | 306.3 |
| tournament_2 | 0.4453 | 0.450967 | 295.1 |
| tournament_5 | 0.4449 | 0.451303 | 329.2 |
| blend_gaussian | 0.4441 | 0.448571 | 316.7 |

`elite_5` was selected with five elites, crossover rate 0.85, mutation rate
0.20, tournament size three, SBX crossover, and Gaussian mutation. Its untouched
holdout results were 67.38% total CHR, 69.96% HPC CHR, 66.24% LPC CHR, 310,920
total penalty, and 34.61 ms average retrieval time.

Results and generation histories are stored in `final_results.json`,
`sweep_progress.json`, and `training_histories.json`. The `figures` directory
contains validation metric comparisons and the selected policy's holdout total
CHR. Runtime observations vary with system load.

## Reproducibility

From the repository root, run:

```bash
python -m genetic_algorithm_implementation.run_ga_hyperparameter_sweep
```

To continue an interrupted sweep from its last completed configuration:

```bash
python -m genetic_algorithm_implementation.run_ga_hyperparameter_sweep --resume
```
