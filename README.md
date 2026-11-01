# Predictive Multi-Tier Caching

A high-performance benchmark suite evaluating bio-inspired metaheuristic optimization and predictive caching algorithms against standard multi-tier replacement policies on the matched 10k hyperparameter-sweep trace (`Dataset/cluster07(00).csv`).

---

## Implemented Algorithms

The benchmark suite features a comprehensive suite of metaheuristic and baseline caching algorithms categorized into three tiers:

### 1. Newer Algorithms
- **Sharpbelly Fish Optimization Algorithm (SFO)**: Bio-inspired metaheuristic simulating the predatory hunting, foraging, and school-formation behaviors of sharpbelly fish to optimize multi-tier cache utility weights.
- **Dove Swarm Optimization Algorithm (DSOA-EC)**: Novel swarm intelligence framework incorporating exploration-competition dynamics, dove flock foraging, satiety decay mechanisms, and Lévy flight perturbations.
- **Hippopotamus Optimization (HO)**: Nature-inspired metaheuristic modeling the defense, territorial dominance, and evasion behaviors of hippopotamuses in aquatic/terrestrial habitats.

### 2. Established Algorithms
- **Genetic Algorithm (GA)**: Evolutionary search framework with tournament selection, uniform crossover, dynamic mutation, and multi-objective Pareto-like fitness optimization.
- **Particle Swarm Optimization (PSO)**: Continuous velocity-displacement swarm algorithm tracking personal bests ($p_{best}$) and global bests ($g_{best}$) for multi-tier parameter search.
- **Artificial Bee Colony (ABC)**: Swarm metaheuristic modeling the foraging behaviors of employed bees, onlooker bees, and scout bees around nectar source locations.
- **Whale Optimization Algorithm (WOA)**: Swarm optimizer mimicking the bubble-net hunting strategy, encircling prey mechanisms, and search phase of humpback whales.

### 3. Baseline Algorithms
- **Least Recently Used (LRU)**: Standard recency-driven multi-tier eviction with promotion from SSD to DRAM upon cache hit.
- **Least Frequently Used (LFU)**: Frequency-counting multi-tier eviction targeting items with minimal cumulative access counts.
- **Random Replacement (RR)**: Uniform random eviction baseline serving as an unbiased control reference across storage tiers.

---

## Evaluation Benchmark Protocol

The comparison is performed under one matched experimental protocol so that GA, PSO, SFO, and DSOA are evaluated fairly on the same request split and cache model:

- **Trace Dataset**: `Dataset/cluster07(00).csv`
- **Priority preprocessing**: same 30% high-priority ratio and `seed = 42`
- **Request split**:
  - **Training**: first 10,000 requests
  - **Validation**: next 40,000 requests
  - **Holdout**: final 40,000 requests
- **Multi-Tier Hardware Specifications**:
  - Cache capacity: 10% of the training catalog
  - DRAM tier: 20% of cache; SSD tier: 80% of cache
  - Same latency and bandwidth assumptions across all tuned algorithms
- **Objective**: `0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`
- **Selection rule**: choose the configuration with the best validation score, then evaluate that single policy once on the untouched holdout split

This is the shared protocol used by the GA, PSO, SFO, and DSOA sweeps, and it is the one we compare against each other.

---

## Running the Matched Hyperparameter Sweeps

Use the hyperparameter sweep scripts only; the legacy sample-dataset benchmark path is no longer the active comparison flow.

```bash
python -m genetic_algorithm_implementation.run_ga_hyperparameter_sweep
python -m pso_implementation.run_pso_hyperparameter_sweep
python -m sfo_implementation.run_sfo_hyperparameter_sweep
python -m dsoa_implementation.run_dsoa_hyperparameter_sweep
python -m abc_implementation.run_abc_hyperparameter_sweep
python -m ho_implementation.run_ho_hyperparameter_sweep
python -m woa_implementation.run_woa_hyperparameter_sweep
python -m rq6_experiments.run_rq6_benchmark
python -m rq7_experiments.run_rq7_benchmark
```

## Shared Hyperparameter Sweep Protocol

All tuned algorithms now use the same evaluation protocol so their selected
configurations are comparable and not contaminated by test leakage:

- Dataset: `Dataset/cluster07(00).csv`
- Preprocessing: same priority tagging (`high_priority_ratio=0.30`) and same seed 42
- Train: first 10,000 requests
- Validation: next 40,000 requests
- Holdout: final 40,000 requests
- Cache model: fixed total cache size equal to 10% of the training catalog, split 20% DRAM / 80% SSD
- Fitness objective: `0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`
- Search budget: 30 agents, 25 iterations/generations, same seed for each algorithm
- Selection rule: choose the configuration with the best validation score only; evaluate the chosen config once on the holdout block

This is the protocol used by the GA, PSO, SFO, DSOA, ABC, HO, and WOA sweeps, and it is the
one we compare against each other.

The 10k PSO hyperparameter sweep configurations, validation results, modeled
cache/storage parameters, and physical-host metadata availability are documented in
[the PSO sweep notes](pso_implementation/results/hyperparameter_sweep_10k/EXPERIMENT_NOTES.md).
The matching SFO sweep uses the same dataset, request splits, cache model, and
fitness weights; see [the SFO sweep notes](sfo_implementation/results/hyperparameter_sweep_10k/EXPERIMENT_NOTES.md)
and run it with `python -m sfo_implementation.run_sfo_hyperparameter_sweep`.
The DSOA sweep follows the same 10k train/validation/holdout protocol and can be
reproduced with `python -m dsoa_implementation.run_dsoa_hyperparameter_sweep`.
The ABC sweep follows the same 10k train/validation/holdout protocol and can be
reproduced with `python -m abc_implementation.run_abc_hyperparameter_sweep`.
The HO sweep follows the same 10k train/validation/holdout protocol and can be
reproduced with `python -m ho_implementation.run_ho_hyperparameter_sweep`.
The WOA sweep follows the same 10k train/validation/holdout protocol and can be
reproduced with `python -m woa_implementation.run_woa_hyperparameter_sweep`.
The selected-policy holdout comparison for GA, PSO, SFO, DSOA, ABC, HO, and WOA is summarized in
[PSO_SFO_10K_COMPARISON.md](results/PSO_SFO_10K_COMPARISON.md).

## Why we use validation

Validation is there to prevent test-set leakage and to prevent a configuration
from being chosen because it happened to fit the final holdout data by chance.
Without a validation stage, the optimizer could tune hyperparameters on the same
set we later report as "final performance", which would overstate real-world
performance.

The workflow is therefore:

1. Train candidate configurations on the training split.
2. Measure each configuration on the validation split.
3. Select the best validation configuration.
4. Freeze it.
5. Evaluate it once on the untouched holdout split.

The holdout result is the only final-number we should trust for comparison,
while validation is the selection mechanism used during tuning.

---

## Matched Holdout Comparison (fixed protocol)

All values below are from the same 40,000-request holdout block (requests 50,001–90,000), after each policy was selected using validation only.

| Policy | Total CHR | HPC CHR | LPC CHR | Total Penalty | Avg Retrieval Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSO** | **67.62%** | 69.96% | **66.59%** | **309,275** | **34.37** |
| **SFO** | 66.99% | **70.11%** | 65.61% | 311,155 | 34.90 |
| **GA** | 67.38% | 69.96% | 66.24% | 310,920 | 34.61 |
| **DSOA** | 67.13% | 70.05% | 65.84% | 310,710 | 34.93 |
| **ABC** | 67.26% | 69.69% | 66.19% | 310,965 | 34.74 |
| **HO** | 67.62% | 69.61% | 66.75% | 312,470 | 34.41 |
| **WOA** | 67.41% | 69.76% | 66.38% | 311,530 | 34.52 |

This is the comparison that should be used when discussing algorithm performance under the shared protocol. Older benchmark numbers from different request splits are not directly comparable to this matched 10k/validation/holdout sweep.
