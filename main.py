#!/usr/bin/env python3
"""
Predictive Multi-Tier Caching: Unified Benchmark Runner
======================================================
Implements and evaluates the 7 supported caching policies:
  1. DSOA-Predictive  — Dove Swarm Optimization Algorithm with Exploration-Competition
  2. SFO-Predictive   — Sailfish Optimizer Predictive Policy
  3. PSO-Predictive   — Particle Swarm Optimization Predictive Policy
  4. GA-Predictive    — Genetic Algorithm Predictive Policy
  5. LRU              — Least Recently Used Baseline
  6. LFU              — Least Frequently Used Baseline
  7. Random           — Random Eviction Baseline

Protocol:
  - Dataset: 'Dataset/memcached-sample' (20,000 requests)
  - Train Partition: First 5,000 requests (for metaheuristic hyperparameter search)
  - Test Partition: Remaining 15,000 requests (held-out UNSEEN evaluation)
  - Multi-tier Cache: 10% catalog size (20% DRAM, 80% SSD)
"""

import json
import os
import sys
import time
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator

# Policy Imports
from lru_implementation.lru_policy import MultiTierLRUPolicy
from lfu_implementation.lfu_policy import MultiTierLFUPolicy
from random_implementation.random_policy import MultiTierRandomPolicy
from genetic_algorithm_implementation.ga_chromosome import Chromosome
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from pso_implementation.pso_particle import Particle
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy
from sfo_implementation.sfo_fish import Fish
from sfo_implementation.sfo_predictive_policy import SFOPredictivePolicy
from dsoa_implementation.dsoa_dove import Dove
from dsoa_implementation.dsoa_predictive_policy import DSOAPredictivePolicy
from ho_implementation.ho_hippo import Hippo
from ho_implementation.ho_predictive_policy import HOPredictivePolicy

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

POLICY_ORDER = [
    "HO-Predictive",
    "DSOA-Predictive",
    "SFO-Predictive",
    "PSO-Predictive",
    "GA-Predictive",
    "LRU",
    "LFU",
    "Random"
]

POLICY_COLORS = {
    "HO-Predictive": "#2E7D32",
    "DSOA-Predictive": "#D81B60",
    "SFO-Predictive": "#00897B",
    "PSO-Predictive": "#0288D1",
    "GA-Predictive": "#5E35B1",
    "LRU": "#E53935",
    "LFU": "#6D4C41",
    "Random": "#757575"
}



def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })


def load_best_weights():
    """Load pre-trained metaheuristic weights from respective folders."""
    # 1. GA
    ga_path = os.path.join(PROJECT_ROOT, "genetic_algorithm_implementation", "results", "consolidated_results.json")
    with open(ga_path, "r") as f:
        ga_data = json.load(f)
    ga_weights = ga_data["ga_optimization"]["best_genes"]

    # 2. PSO
    pso_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "pso_best_weights.json")
    if os.path.exists(pso_path):
        with open(pso_path, "r") as f:
            pso_weights = json.load(f)
    else:
        pso_weights = ga_weights

    # 3. SFO
    sfo_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "sfo_best_weights.json")
    if os.path.exists(sfo_path):
        with open(sfo_path, "r") as f:
            sfo_weights = json.load(f)
    else:
        sfo_weights = ga_weights

    # 4. DSOA
    dsoa_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "dsoa_best_weights.json")
    if os.path.exists(dsoa_path):
        with open(dsoa_path, "r") as f:
            dsoa_weights = json.load(f)
    else:
        dsoa_weights = ga_weights

    # 5. HO
    ho_path = os.path.join(PROJECT_ROOT, "ho_implementation", "results", "ho_best_weights.json")
    if os.path.exists(ho_path):
        with open(ho_path, "r") as f:
            ho_weights = json.load(f)
    else:
        ho_weights = ga_weights

    return ga_weights, pso_weights, sfo_weights, dsoa_weights, ho_weights



def generate_bar_chart(policies, values, title, ylabel, filename):
    setup_plot_style()
    plt.figure(figsize=(10, 5.5))
    colors = [POLICY_COLORS.get(p, "#424242") for p in policies]
    bars = plt.bar(policies, values, color=colors, edgecolor="#212121", linewidth=0.8, width=0.55)
    plt.title(title, pad=12)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right", fontweight="medium")

    is_pct = "%" in ylabel or "Ratio" in title
    for bar in bars:
        h = bar.get_height()
        fmt = f"{h:.2f}%" if is_pct else f"{int(h):,}"
        plt.annotate(fmt, (bar.get_x() + bar.get_width() / 2, h + (max(values)*0.01)),
                     ha="center", fontsize=9.5, fontweight="bold")

    plt.ylim(0, max(values) * 1.15)
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()


def generate_all_plots(results: Dict[str, Any]):
    setup_plot_style()
    policies = [p for p in POLICY_ORDER if p in results]

    # 1. Total CHR
    generate_bar_chart(
        policies, [results[p]["chr_total"] * 100 for p in policies],
        "Total Cache Hit Ratio (CHR) Across All 7 Policies", "Hit Ratio (%)", "fig1_chr_total.png"
    )

    # 2. HPC CHR
    generate_bar_chart(
        policies, [results[p]["chr_hpc"] * 100 for p in policies],
        "High-Priority Content (HPC) Hit Ratio Comparison", "HPC Hit Ratio (%)", "fig2_chr_hpc.png"
    )

    # 3. LPC CHR
    generate_bar_chart(
        policies, [results[p]["chr_lpc"] * 100 for p in policies],
        "Low-Priority Content (LPC) Hit Ratio Comparison", "LPC Hit Ratio (%)", "fig3_chr_lpc.png"
    )

    # 4. Total Penalty
    generate_bar_chart(
        policies, [results[p]["penalty_total"] for p in policies],
        "Total SLA Penalty Cost by Policy (Lower is Better)", "Penalty Cost ($)", "fig4_penalty.png"
    )

    # 5. Average Latency
    generate_bar_chart(
        policies, [results[p]["avg_retrieval_time"] * 1000 for p in policies],
        "Average End-to-End Latency by Policy (Lower is Better)", "Latency (ms)", "fig5_retrieval_time.png"
    )

    # 6. DRAM vs SSD Reads
    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(policies))
    w = 0.55
    dram_reads = [results[p]["reads_dram"] for p in policies]
    ssd_reads = [results[p]["reads_ssd"] for p in policies]
    plt.bar(x, dram_reads, w, label="DRAM Reads", color="#1E88E5", edgecolor="black", linewidth=0.5)
    plt.bar(x, ssd_reads, w, bottom=dram_reads, label="SSD Reads", color="#43A047", edgecolor="black", linewidth=0.5)
    plt.xticks(x, policies, rotation=25, ha="right", fontweight="medium")
    plt.ylabel("Read Operations")
    plt.title("Cache Read Operations Distribution by Tier", pad=12)
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, "fig7_reads.png"))
    plt.close()

    # 7. Writes
    plt.figure(figsize=(10, 5.5))
    dram_writes = [results[p]["writes_dram"] for p in policies]
    ssd_writes = [results[p]["writes_ssd"] for p in policies]
    plt.bar(x, dram_writes, w, label="DRAM Writes", color="#1E88E5", edgecolor="black", linewidth=0.5)
    plt.bar(x, ssd_writes, w, bottom=dram_writes, label="SSD Writes", color="#FB8C00", edgecolor="black", linewidth=0.5)
    plt.xticks(x, policies, rotation=25, ha="right", fontweight="medium")
    plt.ylabel("Write Operations")
    plt.title("Cache Write Operations Distribution by Tier", pad=12)
    plt.legend()
    plt.savefig(os.path.join(FIGURE_DIR, "fig8_writes.png"))
    plt.close()


def main():
    print("=" * 80)
    print("  PREDICTIVE CACHING MASTER BENCHMARK: 7 SUPPORTED POLICIES")
    print("  Algorithms: DSOA, SFO, PSO, GA, LRU, LFU, Random")
    print("  Dataset: 'Dataset/memcached-sample' (Strict 5k Train / 15k Unseen Test Split)")
    print("=" * 80)

    # 1. Preprocess Trace
    raw_dataset = os.path.join(PROJECT_ROOT, "Dataset", "memcached-sample")
    preprocessor = TracePreprocessor(raw_dataset, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process()

    unique_key_sizes = {r["key"]: r["size"] for r in records}
    total_catalog_bytes = sum(unique_key_sizes.values())
    total_cache_size = int(total_catalog_bytes * 0.10)
    dram_fraction = 0.20
    mean_obj_size = int(stats["mean_object_size"])

    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=dram_fraction,
        ssd_throughput_gb=2.0,
        slot_size=mean_obj_size
    )

    train_size = 5000
    test_trace = records[train_size:]
    print(f"\nEvaluating all 8 policies on {len(test_trace)} UNSEEN test requests...")

    # Load weights
    ga_w, pso_w, sfo_w, dsoa_w, ho_w = load_best_weights()

    ga_chrom = Chromosome()
    ga_chrom.genes = list(ga_w.values())

    pso_part = Particle(position_seed=pso_w)
    sfo_fish = Fish(position_seed=sfo_w)
    dsoa_dove = Dove(position_seed=dsoa_w)
    ho_hippo = Hippo(position_seed=ho_w)

    policies = {
        "HO-Predictive": HOPredictivePolicy(ho_hippo),
        "DSOA-Predictive": DSOAPredictivePolicy(dsoa_dove),
        "SFO-Predictive": SFOPredictivePolicy(sfo_fish),
        "PSO-Predictive": PSOPredictivePolicy(pso_part),
        "GA-Predictive": GAPredictivePolicy(ga_chrom),
        "LRU": MultiTierLRUPolicy(),
        "LFU": MultiTierLFUPolicy(),
        "Random": MultiTierRandomPolicy(seed=42)
    }


    results = {}
    for pol_name in POLICY_ORDER:
        pol_inst = policies[pol_name]
        t0 = time.time()
        res = simulator.run(pol_inst, test_trace)
        elapsed = time.time() - t0
        results[pol_name] = res
        print(f"  [{pol_name:17s}] Total CHR: {res['chr_total']*100:5.2f}% | HPC CHR: {res['chr_hpc']*100:5.2f}% | LPC CHR: {res['chr_lpc']*100:5.2f}% | Penalty: {res['penalty_total']:7d} | Time: {elapsed:.2f}s")

    # Save consolidated results
    with open(os.path.join(OUTPUT_DIR, "consolidated_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save summary table
    summary_path = os.path.join(OUTPUT_DIR, "summary_table.txt")
    header = f"{'Policy':18s} | {'Total CHR':9s} | {'HPC CHR':9s} | {'LPC CHR':9s} | {'Penalty HPC':11s} | {'Penalty LPC':11s} | {'Total Penalty':13s} | {'Latency (ms)':12s}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for p in POLICY_ORDER:
        m = results[p]
        lines.append(
            f"{p:18s} | {m['chr_total']*100:8.2f}% | {m['chr_hpc']*100:8.2f}% | {m['chr_lpc']*100:8.2f}% | "
            f"{m['penalty_hpc']:11d} | {m['penalty_lpc']:11d} | {m['penalty_total']:13d} | {m['avg_retrieval_time']*1000:12.3f}"
        )
    lines.append(sep)
    summary_text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text + "\n")

    # Generate comparative figures
    generate_all_plots(results)
    print(f"All figures generated successfully in: {FIGURE_DIR}")
    print("=" * 80)
    print("MASTER BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()