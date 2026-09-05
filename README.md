# Predictive Caching — Base Paper Implementation

## Multi-Tier NDN Cache Simulator: Q-Learning Enhanced QoS-Aware ARC

This project implements and compares **7 caching policies** from the research paper on multi-tier content store management for Named Data Networking (NDN):

### Policies Implemented

| Policy | Description | Key Feature |
|--------|-------------|-------------|
| **M_ARC** | Multi-tier ARC | Extends ARC across DRAM+SSD with proportional β-based p adaptation |
| **QM_ARC** | Quality-aware M_ARC | Differentiates high/low priority content using α-based insertion |
| **QL_QM_ARC** | Q-Learning QM_ARC | Learns optimal p adaptation via reinforcement learning |
| **LRU** | Least Recently Used | Baseline recency-based eviction |
| **LFU** | Least Frequently Used | Baseline frequency-based eviction |
| **Random** | Random Eviction | Baseline random eviction |
| **PriorityLRU** | Priority-aware LRU | High-priority content prefetching |

### Architecture

- **Discrete-event simulation** using SimPy
- **Multi-tier storage**: DRAM (fast, small) + SSD (slower, large)
- **NDN packet forwarding** with Pending Interest Table (PIT) and Content Store (CS)
- **Realistic hardware parameters**: DRAM @ 40GB/s read, SSD @ 2GB/s read

## Project Structure

```
predictive-caching/
├── main.py                          # Full simulation runner (all 7 policies)
├── generate_trace.py                # Synthetic trace dataset generator
├── results/                         # Simulation outputs
│   ├── consolidated_results.json    # All metrics in one JSON
│   ├── figures/                     # Publication-quality comparison plots
│   │   ├── fig1_chr_total.png       # Total Cache Hit Ratio
│   │   ├── fig2_chr_hpc.png         # High-Priority CHR
│   │   ├── fig3_chr_lpc.png         # Low-Priority CHR
│   │   ├── fig4_penalty.png         # Penalty Costs
│   │   ├── fig5_retrieval_time.png  # Data Retrieval Time
│   │   ├── fig6_ql_rewards.png      # Q-Learning Convergence
│   │   ├── fig7_reads.png           # Read Operations
│   │   ├── fig8_writes.png          # Write Operations
│   │   └── summary_table.png        # Summary Table
│   └── *.txt                        # Per-policy raw JSON results
├── MultiTierCacheSimulator-main/    # Core simulator engine
│   └── MultiTierCacheSimulator-main/
│       ├── policies/                # All caching policy implementations
│       │   ├── MARC/                # M_ARC (multi-tier ARC)
│       │   ├── QM_ARC/              # QM_ARC (quality-aware)
│       │   ├── QL_QM_ARC/           # QL_QM_ARC (Q-Learning)
│       │   ├── lru_policy.py        # LRU
│       │   ├── lfu_policy.py        # LFU
│       │   └── random_policy.py     # Random
│       ├── forwarder_structures/    # NDN forwarder, tiers, index, PIT
│       ├── traces/                  # Trace generation and parsing
│       ├── simulation.py            # SimPy simulation engine
│       └── resources/               # Dataset storage
└── README.md
```

## How to Run

### 1. Install Dependencies

```bash
pip install numpy matplotlib pandas simpy
```

### 2. Generate Synthetic Trace (run once)

```bash
python generate_trace.py
```

This creates a synthetic NDN traffic trace with:
- 100 unique content items, Zipf(α=0.8) popularity
- Poisson arrivals (λ=5 req/s), 15-minute traffic period (~4500 requests)
- 30% high-priority, 70% low-priority content

### 3. Run All Simulations

```bash
python main.py
```

Runs all 7 policies and generates comparison plots in `results/figures/`.

## Key Results

| Policy | Total CHR | HPC CHR | LPC CHR | Penalty HPC | Penalty LPC |
|--------|-----------|---------|---------|-------------|-------------|
| M_ARC | 0.2595 | 0.4222 | 0.1559 | 63110 | 29155 |
| **QM_ARC** | **0.2666** | **0.5175** | 0.1067 | **53695** | 30875 |
| QL_QM_ARC | 0.1812 | 0.3512 | 0.0729 | 72315 | 32255 |
| LRU | 0.2133 | 0.3320 | 0.1376 | 72650 | 29845 |
| LFU | 0.2133 | 0.3320 | 0.1376 | 72650 | 29845 |
| Random | 0.2029 | 0.3134 | 0.1325 | 74760 | 30105 |
| PriorityLRU | 0.2222 | 0.3749 | 0.1250 | 67295 | 30295 |

### Key Observations

- **QM_ARC achieves the best high-priority CHR** (0.5175) — 55% better than LRU baseline
- **M_ARC outperforms all baselines** in total CHR (0.2595 vs 0.2133 for LRU)
- **QM_ARC has the lowest penalty cost** for high-priority content (53,695)
- **PriorityLRU** shows moderate improvement via priority-aware prefetching

## Datasets

Synthetic trace generated using Zipf-distributed popularity with Poisson arrivals, stored in:
`MultiTierCacheSimulator-main/MultiTierCacheSimulator-main/resources/dataset_synthetic/`

## Paper Reference

Multi-tier content store management using Q-Learning enhanced QoS-aware ARC for NDN.
