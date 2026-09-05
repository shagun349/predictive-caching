# Predictive Caching — ARC vs LRU Performance Analysis

This project implements and compares the **Adaptive Replacement Cache (ARC)** algorithm against standard **LRU** caching using real-world datasets from the MultiTierCacheSimulator.

## Features

- **ARC (Adaptive Replacement Cache)**: Self-tuning cache that dynamically balances recency and frequency via ghost lists and an adaptive parameter `p`.
- **LRU (Least Recently Used)**: Classic baseline eviction policy.
- **Dataset Loading**: Reads CSV trace files from the MultiTierCacheSimulator project.
- **Workload Augmentation**: Extends small datasets using Zipf-like frequency-biased sampling for realistic evaluation.
- **Visualization**: Generates side-by-side plots of hit rate vs. cache capacity and ARC's dynamic `p` parameter adaptation.

## Project Structure

```
predictive-caching/
├── main.py                          # Main script (ARC, LRU, experiments, plots)
├── results.png                      # Generated performance comparison plots
├── README.md
├── .gitignore
└── MultiTierCacheSimulator-main/    # Dataset source (CSV trace files)
```

## How to Run

```bash
pip install numpy matplotlib
python main.py
```

## Results

The script generates `results.png` with four plots:
1. **Hit Rate vs Cache Capacity** for each dataset (ARC vs LRU)
2. **ARC Parameter `p` Adaptation** over time for each dataset

## Datasets

Uses CSV traces from the [MultiTierCacheSimulator](https://github.com/...) project:
- `lru_better.csv` — workload favoring recency-based policies
- `lfu_better.csv` — workload favoring frequency-based policies
