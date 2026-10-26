# Predictive Multi-Tier Caching

Benchmarking metaheuristic predictive caching algorithms against standard multi-tier replacement policies on real-world trace data (`Dataset/memcached-sample`).

---

## Supported Algorithms

Only the following 7 caching algorithms are supported and benchmarked:

1. **DSOA-Predictive**: Dove Swarm Optimization Algorithm with Exploration-Competition (DSOA-EC).
2. **SFO-Predictive**: Sailfish Optimizer predictive multi-tier caching policy.
3. **PSO-Predictive**: Particle Swarm Optimization predictive policy.
4. **GA-Predictive**: Genetic Algorithm evolved predictive policy.
5. **LRU**: Multi-tier Least Recently Used caching policy.
6. **LFU**: Multi-tier Least Frequently Used caching policy.
7. **Random**: Multi-tier Random Eviction policy.

---

## Directory Structure

```
├── Dataset/
│   └── memcached-sample                   # Real-world Memcached trace (20,000 requests)
├── dsoa_implementation/                   # Dove Swarm Optimization (DSOA-EC)
│   ├── dsoa_dove.py                       # Dove agent model & satiety dynamics
│   ├── dsoa_engine.py                     # DSOA foraging & competition engine
│   ├── dsoa_predictive_policy.py          # Predictive utility policy
│   ├── run_dsoa_experiment.py             # DSOA benchmark runner
│   └── results/                           # Results & publication figures
├── sfo_implementation/                    # Sailfish Optimizer (SFO)
│   ├── sfo_fish.py                        # Sharpbelly fish & sailfish models
│   ├── sfo_engine.py                      # SFO search engine
│   ├── sfo_predictive_policy.py           # SFO predictive caching policy
│   ├── run_sfo_experiment.py              # SFO benchmark runner
│   └── results/                           # Results & publication figures
├── pso_implementation/                    # Particle Swarm Optimization (PSO)
│   ├── pso_particle.py                    # PSO particle model
│   ├── pso_engine.py                      # PSO search engine
│   ├── pso_predictive_policy.py           # PSO predictive policy
│   ├── run_pso_experiment.py              # PSO benchmark runner
│   └── results/                           # Results & publication figures
├── genetic_algorithm_implementation/      # Genetic Algorithm (GA)
│   ├── cache_simulator.py                 # Pure Python Multi-Tier Cache Simulator
│   ├── trace_preprocessor.py              # QoS tagging & trace preprocessing
│   ├── baseline_policies.py               # Baseline policy definitions (LRU, LFU, Rand)
│   ├── ga_chromosome.py                   # Chromosome representation
│   ├── ga_engine.py                       # GA evolution engine
│   ├── ga_predictive_policy.py            # GA predictive caching policy
│   ├── run_ga_experiment.py               # GA benchmark runner
│   └── results/                           # Results & publication figures
├── lru_implementation/                    # Dedicated LRU Implementation
│   ├── lru_policy.py                      # Multi-tier LRU policy
│   ├── run_lru_experiment.py              # LRU runner
│   └── results/                           # Results & publication figures
├── lfu_implementation/                    # Dedicated LFU Implementation
│   ├── lfu_policy.py                      # Multi-tier LFU policy
│   ├── run_lfu_experiment.py              # LFU runner
│   └── results/                           # Results & publication figures
├── random_implementation/                 # Dedicated Random Implementation
│   ├── random_policy.py                   # Multi-tier Random policy
│   ├── run_random_experiment.py           # Random runner
│   └── results/                           # Results & publication figures
├── main.py                                # Master comparative benchmark runner (all 7 policies)
├── results/                               # Consolidated results and figures
│   ├── consolidated_results.json          # Metrics across all 7 policies
│   ├── summary_table.txt                  # Tabular summary
│   └── figures/                           # Comparative publication-quality plots
└── README.md
```

---

## Evaluation Benchmark Protocol

- **Dataset**: `Dataset/memcached-sample` (20,000 trace requests)
- **Train/Test Partition**:
  - **Train**: First 5,000 requests (strictly used for optimization/hyperparameter search)
  - **Test**: Remaining 15,000 requests (held-out unseen requests for zero-leakage evaluation)
- **Hierarchy Configuration**:
  - Total Cache: 10% of total catalog size
  - DRAM Tier: 20% of total cache (100 ns latency, 40 GB/s bandwidth)
  - SSD Tier: 80% of total cache (10 µs latency, 2 GB/s bandwidth)

---

## Running the Benchmark

To evaluate all 7 policies and generate comparison figures:

```bash
python3 main.py
```

To run individual implementations separately:

```bash
python3 -m dsoa_implementation.run_dsoa_experiment
python3 -m sfo_implementation.run_sfo_experiment
python3 -m pso_implementation.run_pso_experiment
python3 -m genetic_algorithm_implementation.run_ga_experiment
python3 -m lru_implementation.run_lru_experiment
python3 -m lfu_implementation.run_lfu_experiment
python3 -m random_implementation.run_random_experiment
```

---

## Summary of Results (15,000 Unseen Test Requests)

| Policy | Total CHR | HPC CHR | LPC CHR | Total SLA Penalty | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DSOA-Predictive** | **41.41%** | **45.64%** | **39.49%** | **205,790** | **61.511** |
| **SFO-Predictive** | 41.41% | 45.47% | 39.58% | 206,350 | 61.430 |
| **GA-Predictive** | 40.63% | 46.04% | 38.17% | 206,320 | 62.216 |
| **PSO-Predictive** | 38.04% | 47.22% | 33.88% | 207,565 | 64.601 |
| **LRU** | 37.95% | 38.92% | 37.50% | 226,145 | 65.119 |
| **Random** | 35.18% | 35.86% | 34.87% | 238,720 | 68.007 |
| **LFU** | 33.08% | 34.43% | 32.47% | 238,735 | 67.637 |
