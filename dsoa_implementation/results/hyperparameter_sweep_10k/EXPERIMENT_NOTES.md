# DSOA 10k Hyperparameter Sweep Notes

## What was run

The DSOA sweep mirrors the matched GA/PSO/SFO 10k protocol: `Dataset/cluster07(00).csv`, seed 42, 30 agents, 25 iterations, a 10,000-request training block, a 40,000-request validation block, and a 40,000-request holdout block.

| Split | Request indices (1-based, inclusive) | Requests |
| --- | ---: | ---: |
| Training | 1–10,000 | 10,000 |
| Validation | 10,001–50,000 | 40,000 |
| Holdout | 50,001–90,000 | 40,000 |

Each configuration reuses the same cache model, fixed objective weights, and validation-selection workflow as the GA/PSO/SFO sweeps.

## DSOA hyperparameter changes

The sweep varies DSOA-specific controls while keeping the search budget fixed across configurations:

- `eta`: local foraging step-size scaling
- `satiety_decay`: satiety decay rate per iteration
- `satiety_threshold`: hunger threshold used to trigger relocation
- `levy_prob`: probability of Lévy-flight perturbation

The main objective is the same multi-objective validation score used elsewhere in this project:

`0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`

## Reproducibility

From the repository root, run:

```bash
python -m dsoa_implementation.run_dsoa_hyperparameter_sweep
```

The script writes the sweep checkpoint and plots under `dsoa_implementation/results/hyperparameter_sweep_10k/`.

## Notes

This run is intentionally aligned with the GA/PSO/SFO 10k hyperparameter sweep methodology so the selected DSOA configuration can be directly compared against the other finalist policies on the same holdout block.
