# Predictive Multi-Tier Caching

A high-performance benchmark suite evaluating bio-inspired metaheuristic optimization and predictive caching algorithms against standard multi-tier replacement policies on real-world trace data (`Dataset/memcached-sample`).

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

- **Trace Dataset**: `Dataset/memcached-sample` (20,000 requests with 31.2% High-Priority HPC and 68.8% Low-Priority LPC requests).
- **Strict Zero-Leakage Train/Test Split**:
  - **Training Trace**: First 5,000 requests (dedicated exclusively to metaheuristic parameter evolution).
  - **Test Trace**: Remaining 15,000 requests (unseen evaluation data; zero data leakage).
- **Multi-Tier Hardware Specifications**:
  - Total Cache Capacity: 10% of total catalog size (220,946 bytes).
  - DRAM Tier (20% cache): 44,189 bytes, 100 ns latency, 40 GB/s bandwidth.
  - SSD Tier (80% cache): 176,756 bytes, 10 µs latency, 2 GB/s bandwidth.
  - External Repository: 100 ms round-trip latency.

---

## Running the Benchmark Suite

To execute the master comparative benchmark:

```bash
python3 main.py
```

To run individual algorithms independently:

```bash
# Newer Algorithms
python3 -m dsoa_implementation.run_dsoa_experiment
python3 -m sfo_implementation.run_sfo_experiment

# Established Algorithms
python3 -m genetic_algorithm_implementation.run_ga_experiment
python3 -m pso_implementation.run_pso_experiment

# Baseline Algorithms
python3 -m lru_implementation.run_lru_experiment
python3 -m lfu_implementation.run_lfu_experiment
python3 -m random_implementation.run_random_experiment
```

---

## Comparative Performance Summary (15,000 Unseen Test Requests)

| Category | Policy | Total CHR | HPC CHR | LPC CHR | Total SLA Penalty | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Newer** | **DSOA-Predictive (DSOA-EC)** | **41.41%** | **45.64%** | **39.49%** | **205,790** | **61.511** |
| **Newer** | **SFO-Predictive (SFO)** | 41.41% | 45.47% | 39.58% | 206,350 | 61.430 |
| **Established** | **GA-Predictive (GA)** | 40.63% | 46.04% | 38.17% | 206,320 | 62.216 |
| **Established** | **PSO-Predictive (PSO)** | 38.04% | 47.22% | 33.88% | 207,565 | 64.601 |
| **Baseline** | **LRU (Least Recently Used)** | 37.95% | 38.92% | 37.50% | 226,145 | 65.119 |
| **Baseline** | **Random Replacement (RR)** | 35.18% | 35.86% | 34.87% | 238,720 | 68.007 |
| **Baseline** | **LFU (Least Frequently Used)** | 33.08% | 34.43% | 32.47% | 238,735 | 67.637 |
