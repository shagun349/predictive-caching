# Genetic Algorithm Hyperparameter Sweep

This implementation uses the standardized 10k/40k/40k sweep protocol and the dataset `Dataset/cluster07(00).csv`.

The active GA workflow is:
- Training: first 10,000 requests
- Validation: next 40,000 requests
- Holdout: final 40,000 requests
- Selection: best validation score only
- Reporting: final holdout result only

Use the sweep entry point from the repository root:

```bash
python -m genetic_algorithm_implementation.run_ga_hyperparameter_sweep
```

The full results and notes are stored under:
`genetic_algorithm_implementation/results/hyperparameter_sweep_10k/`

The legacy sample-dataset benchmark and its older result graphs have been removed so the comparison stays on the matched hyperparameter-sweep protocol only.
cd /home/shagun-agarwal/REPOS/predictive-caching
python3 genetic_algorithm_implementation/run_ga_experiment.py
```

## 8. Matched 10k Hyperparameter Sweep

To compare GA settings on the same trace splits, cache model, and objective as
the PSO and SFO sweeps, run this from the repository root:

```bash
python -m genetic_algorithm_implementation.run_ga_hyperparameter_sweep
```

The sweep uses 10,000 training requests, 40,000 validation requests, and a
40,000-request holdout from `Dataset/cluster07(00).csv`. It selects the GA
configuration on validation and evaluates the winner once on holdout. Results,
histories, charts, and the full protocol are saved under
`genetic_algorithm_implementation/results/hyperparameter_sweep_10k/`.
Interrupted sweeps can be continued with `--resume`.
