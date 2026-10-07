# Genetic Algorithm Predictive Caching for Multi-Tier Storage

Implementation of a **Genetic Algorithm (GA) Driven Predictive Caching Strategy** evaluated on the real-world **Memcached Workload Trace** (`Dataset/memcached-sample`).

This implementation is modeled after and benchmarked against the research paper:
> **"QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy"**  
> *Future Generation Computer Systems (2025)*

---

## 1. Overview & Problem Formulation

In distributed data-centric networks and multi-tier Content Stores (DRAM + SSD), traditional caching heuristics face fundamental trade-offs:
- **LRU / LFU** focus solely on recency or frequency, treating all content identically and ignoring Quality-of-Service (QoS) Service Level Agreements (SLAs).
- **PriorityLRU** aggressively protects high-priority content but causes severe cache starvation for low-priority content, crashing the total hit ratio.
- **M-ARC / QM-ARC** introduce dynamic recency/frequency adaptation ($p$) and static QoS insertion ($\alpha = 0.2$), but rely on fixed rules that cannot adapt across heterogeneous workloads.

### The Genetic Algorithm Solution
This project implements a metaheuristic **Genetic Algorithm Optimization Engine** that evolves a multi-objective predictive caching chromosome $\mathbf{C}$:
$$\mathbf{C} = \left[ w_{\text{rec}}, w_{\text{freq}}, \lambda_{\text{decay}}, w_{\text{size}}, w_{\text{prio}}, w_{\text{pen}}, w_{\text{reuse}}, \theta_{\text{dram}}, \alpha_{\text{qos}}, p_{\text{target}} \right]$$

The real-time predictive utility score of each cached content item $x$ is:
$$U(x) = \left( \frac{1}{1 + \lambda_{\text{decay}} \cdot \Delta t} \right) \cdot \left[ w_{\text{rec}} + w_{\text{freq}} \cdot \ln(1 + \text{count}) + w_{\text{prio}} \cdot \mathbb{I}_{\text{hpc}} + w_{\text{pen}} \cdot \frac{\text{Penalty}(x)}{75} + w_{\text{reuse}} \cdot P_{\text{reuse}} \right] - w_{\text{size}} \cdot \frac{\text{size}}{\text{slot}}$$

**Key Insight — Recency-Modulated Priority Protection**:
By modulating the priority and SLA penalty boost with the temporal recency decay factor, **active High-Priority Content (HPC)** is strongly preserved against eviction, while **stale HPC smoothly ages out**, preventing cache pollution and maintaining high hit ratios for Low-Priority Content (LPC).

---

## 2. Multi-Objective Fitness Function

The Genetic Algorithm optimizes a multi-objective fitness function balancing throughput, SLA QoS, and latency:
$$\text{Fitness} = w_1 \cdot \text{CHR}_{\text{total}} + w_2 \cdot \text{CHR}_{\text{HPC}} - w_3 \cdot \frac{\text{Penalty}}{\text{MaxPenalty}} - w_4 \cdot \frac{\text{Latency}}{\text{MaxLatency}}$$

- $w_1 = 0.40$: Maximizes Total Cache Hit Ratio
- $w_2 = 0.35$: Rewards High-Priority QoS Hit Ratio (HPC CHR)
- $w_3 = 0.15$: Minimizes SLA Miss Penalty violations
- $w_4 = 0.10$: Minimizes End-to-End Data Retrieval Latency

---

## 3. Architecture & Multi-Tier Storage Parameters

Matching the research paper hardware model:
- **DRAM Tier (Fast)**:
  - Capacity: **20%** of total cache
  - Latency: $100\text{ ns}$ ($10^{-7}\text{ s}$)
  - Read Throughput: $40\text{ GB/s}$
  - Write Throughput: $20\text{ GB/s}$
- **SSD Tier (Secondary)**:
  - Capacity: **80%** of total cache
  - Latency: $10\text{ }\mu\text{s}$ ($10^{-5}\text{ s}$)
  - Read Throughput: $2\text{ GB/s}$
  - Write Throughput: $1\text{ GB/s}$
- **Pending Interest Table (PIT)**:
  - In-flight request aggregation with $4.0\text{ s}$ interest lifetime
- **Cache Sizing**:
  - Total catalog size: **2,209,468 bytes** (2.11 MB) across 9,641 unique objects
  - Total cache size: **220,946 bytes** (exact **10%** catalog capacity)
  - DRAM: 44,189 bytes, SSD: 176,756 bytes

---

## 4. Key Experimental Results

Evaluated on the full **20,000 requests** of `Dataset/memcached-sample` with 31.2% HPC and 68.8% LPC:

| Policy | Total CHR | HPC CHR | LPC CHR | Penalty HPC | Penalty LPC | Total Penalty | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GA-Predictive (Ours)** | **41.87%** | **47.68%** | **39.22%** | **177,125** | **90,565** | **267,690** | **60.86 ms** |
| **M_ARC** | 39.29% | 40.40% | 38.79% | 200,930 | 91,185 | 292,115 | 63.43 ms |
| **QM_ARC** | 39.29% | 40.40% | 38.79% | 202,055 | 91,185 | 293,240 | 63.43 ms |
| **LRU** | 39.01% | 40.14% | 38.49% | 203,750 | 91,640 | 295,390 | 63.79 ms |
| **Random** | 35.66% | 36.78% | 35.14% | 217,415 | 96,580 | 313,995 | 67.00 ms |
| **QL_QM_ARC** | 32.75% | 34.36% | 32.01% | 216,805 | 99,630 | 316,435 | 67.60 ms |
| **LFU** | 31.90% | 34.11% | 30.90% | 218,975 | 100,615 | 319,590 | 67.55 ms |
| **PriorityLRU** | 23.08% | 48.50% | 11.53% | 175,110 | 128,005 | 303,115 | 75.58 ms |

### Key Achievements:
1. **Best Total Cache Hit Ratio (41.87%)**: Outperforms M_ARC/QM_ARC (+2.58% absolute gain) and LRU (+2.86% absolute gain).
2. **Massive High-Priority CHR Boost (47.68%)**: An 18.0% relative improvement over M_ARC/QM_ARC (40.40%), providing enterprise-grade SLA assurance.
3. **Lowest SLA Penalty (267,690)**: Saves over **24,425 penalty points** compared to M_ARC, and **25,550 points** compared to QM_ARC.
4. **Lowest Average Retrieval Latency (60.86 ms)**: Fastest response time among all 8 policies.
5. **Balanced LPC Protection**: Unlike PriorityLRU which collapsed LPC CHR to 11.53%, GA-Predictive preserves 39.22% LPC CHR.

---

## 5. Evolved Best Chromosome Genes

The Genetic Algorithm converged to the following optimal gene configuration:
- `w_recency = +0.5478` (Balanced recency weighting)
- `w_frequency = +0.4451` (Frequency accumulation reward)
- `w_freq_decay = +0.7075` (Active temporal aging factor)
- `w_size = +0.0614` (Slight penalty for large content)
- `w_priority = +2.1160` (Strong QoS boost for high-priority items)
- `w_penalty = +1.2894` (High sensitivity to SLA miss penalty costs)
- `w_predicted_reuse = -0.8840` (Filtering out single-access burst items)
- `theta_dram = +0.8938` (Selective DRAM admission threshold)
- `alpha_qos = +0.7815` (QoS differentiation factor)
- `target_p = +0.7199` (Recency-to-frequency balance target)

---

## 6. Project Structure

```
genetic_algorithm_implementation/
├── trace_preprocessor.py        # Preprocesses 'Dataset/memcached-sample' into NDN format
├── ga_chromosome.py             # Chromosome encoding, gene specs, SBX crossover, mutation
├── ga_engine.py                 # GA optimization engine (tournament, elitism, fitness)
├── cache_simulator.py           # Two-tier simulator (DRAM+SSD, PIT, latency, penalty)
├── ga_predictive_policy.py      # Evolved predictive caching policy
├── baseline_policies.py         # 7 paper baselines (LRU, LFU, Random, PriorityLRU, M_ARC, QM_ARC, QL_QM_ARC)
├── run_ga_experiment.py         # End-to-end experiment runner
├── README.md                    # Comprehensive documentation and results
└── results/                     # All outputs strictly stored in this folder
    ├── preprocessed_memcached_trace.csv
    ├── ga_training_history.json
    ├── consolidated_results.json
    ├── summary_table.txt
    └── figures/
        ├── fig1_ga_convergence.png
        ├── fig2_chr_total_comparison.png
        ├── fig3_chr_hpc_comparison.png
        ├── fig4_chr_lpc_comparison.png
        ├── fig5_penalty_comparison.png
        ├── fig6_retrieval_time_comparison.png
        ├── fig7_io_operations.png
        ├── fig8_chromosome_weights.png
        └── fig9_summary_table.png
```

---

## 7. How to Reproduce

To re-run the entire Genetic Algorithm evolution and generate all results and figures:

```bash
cd /home/shagun-agarwal/REPOS/predictive-caching
python3 genetic_algorithm_implementation/run_ga_experiment.py
```
