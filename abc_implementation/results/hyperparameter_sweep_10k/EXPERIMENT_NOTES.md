# ABC 10k Hyperparameter Sweep Notes

## What was run

The sweep script, `abc_implementation/run_abc_hyperparameter_sweep.py`, compares
seven ABC configurations on `Dataset/cluster07(00).csv`. Each configuration uses
the same 10,000-request training block, 40,000-request validation block, and
40,000-request holdout block:

| Split | Request indices (1-based, inclusive) | Requests |
| --- | ---: | ---: |
| Training | 1–10,000 | 10,000 |
| Validation | 10,001–50,000 | 40,000 |
| Holdout | 50,001–90,000 | 40,000 |

Every configuration uses 30 bees by default unless a configuration changes the
colony size; all runs use 25 iterations and random seed 42. The fitness weights
are fixed across the sweep: total CHR 0.40, HPC CHR 0.35, normalized penalty
0.15, and normalized latency 0.10. The selection objective is
`0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`.

## Hyperparameter changes and validation results

The ABC search space is driven by the colony size and scout abandonment limit.
The validation score is the selection metric; a higher score is better.

| Configuration | Colony size | Scout limit | Best training fitness | Validation score |
| --- | ---: | ---: | ---: | ---: |
| baseline | 30 | 8 | 0.443263 | 0.450722 |
| limit_04 | 30 | 4 | 0.442232 | 0.449833 |
| limit_12 | 30 | 12 | 0.442126 | 0.447278 |
| colony_20 | 20 | 8 | 0.443802 | 0.450816 |
| colony_40 | 40 | 8 | 0.444687 | 0.451052 |
| combined_20_12 | 20 | 12 | 0.442436 | 0.449094 |
| combined_40_12 | 40 | 12 | 0.444385 | **0.451080** |

`combined_40_12` was selected using validation only. Its untouched holdout
results were 67.26% total CHR, 69.69% HPC CHR, 66.19% LPC CHR, 310,965 total
penalty, and 34.74 ms average retrieval time.

The sweep stores per-configuration validation metrics in
`final_results.json`, per-iteration histories in `training_histories.json`, and
progress state in `sweep_progress.json`. The figures directory contains the
validation-score plot and the selected-policy output summary.

## Simulated cache and storage model

These are simulator parameters, not measurements of a physical machine:

| Modeled resource | Value | Basis |
| --- | ---: | --- |
| Unique-object catalog in training block | 6,135,728 bytes | Sum of unique training-object sizes |
| Total cache capacity | 613,572 bytes | 10% of training catalog |
| DRAM cache tier | 122,714 bytes | 20% of total cache |
| SSD cache tier | 490,857 bytes | 80% of total cache |
| DRAM latency | 100 ns | Simulator constant |
| DRAM read bandwidth | 40 GB/s | Simulator constant |
| DRAM write bandwidth | 20 GB/s | Simulator constant |
| SSD latency | 10 µs | Simulator constant |
| SSD read bandwidth | 2 GB/s | Simulator constant |
| SSD write bandwidth | 1 GB/s | Half the configured read throughput |
| Simulator slot size | Mean object size in the trace | Passed to the simulator |

## Reproducibility

From the repository root, run:

```bash
python -m abc_implementation.run_abc_hyperparameter_sweep
```

The script writes all result files and plots under
`abc_implementation/results/hyperparameter_sweep_10k/`.
