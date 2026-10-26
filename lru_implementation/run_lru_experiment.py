"""
Experiment Runner: LRU (Least Recently Used) Caching Evaluation
==============================================================
Runs evaluation on 'Dataset/memcached-sample':
- 5,000 requests training trace partition
- 15,000 requests UNSEEN test trace partition
- Evaluates Multi-Tier LRU policy
- Generates summary tables and figures in lru_implementation/results/
"""

import json
import os
import sys
import time
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from lru_implementation.lru_policy import MultiTierLRUPolicy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


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
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def main():
    print("=" * 70)
    print("  LRU MULTI-TIER CACHING EVALUATION BENCHMARK")
    print("  Dataset: 'Dataset/memcached-sample'")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 70)

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
    print(f"Evaluating LRU on {len(test_trace)} UNSEEN test requests...")

    lru_policy = MultiTierLRUPolicy()
    t0 = time.time()
    res = simulator.run(lru_policy, test_trace)
    elapsed = time.time() - t0

    print(f"LRU Completed in {elapsed:.2f}s")
    print(f"Total CHR: {res['chr_total']*100:.2f}% | HPC CHR: {res['chr_hpc']*100:.2f}% | LPC CHR: {res['chr_lpc']*100:.2f}%")
    print(f"Total SLA Penalty: {res['penalty_total']:,} | Avg Latency: {res['avg_retrieval_time']*1000:.3f} ms")

    # Save results
    eval_output = {"LRU": res}
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_output, f, indent=2)

    # Summary table
    summary_path = os.path.join(RESULTS_DIR, "summary_table.txt")
    header = f"{'Policy':14s} | {'Total CHR':9s} | {'HPC CHR':9s} | {'LPC CHR':9s} | {'Penalty HPC':11s} | {'Penalty LPC':11s} | {'Total Penalty':13s} | {'Latency (ms)':12s}"
    sep = "-" * len(header)
    row = f"{'LRU':14s} | {res['chr_total']*100:8.2f}% | {res['chr_hpc']*100:8.2f}% | {res['chr_lpc']*100:8.2f}% | {res['penalty_hpc']:11d} | {res['penalty_lpc']:11d} | {res['penalty_total']:13d} | {res['avg_retrieval_time']*1000:12.3f}"
    with open(summary_path, "w") as f:
        f.write("\n".join([sep, header, sep, row, sep]) + "\n")

    # Generate Figures
    setup_plot_style()

    # Fig 1: Hit Ratio by Tier
    fig, ax = plt.subplots(figsize=(6, 5))
    tiers = ["DRAM", "SSD", "Total"]
    vals = [res["chr_dram"] * 100, res["chr_ssd"] * 100, res["chr_total"] * 100]
    bars = ax.bar(tiers, vals, color=["#1E88E5", "#43A047", "#00897B"], width=0.5, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h + 0.5), ha="center", fontweight="bold")
    ax.set_ylabel("Cache Hit Ratio (%)")
    ax.set_title("LRU Cache Hit Ratio by Tier", pad=12)
    ax.set_ylim(0, max(vals) * 1.2)
    plt.savefig(os.path.join(FIGURES_DIR, "1_lru_tier_chr.png"))
    plt.close()

    # Fig 2: QoS Breakdown (HPC vs LPC)
    fig, ax = plt.subplots(figsize=(6, 5))
    cats = ["HPC CHR", "LPC CHR", "Overall CHR"]
    vals = [res["chr_hpc"] * 100, res["chr_lpc"] * 100, res["chr_total"] * 100]
    bars = ax.bar(cats, vals, color=["#FB8C00", "#7E57C2", "#26A69A"], width=0.5, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", (bar.get_x() + bar.get_width() / 2, h + 0.5), ha="center", fontweight="bold")
    ax.set_ylabel("Hit Ratio (%)")
    ax.set_title("LRU QoS Tier Performance (HPC vs LPC)", pad=12)
    ax.set_ylim(0, max(vals) * 1.2)
    plt.savefig(os.path.join(FIGURES_DIR, "2_lru_qos_breakdown.png"))
    plt.close()

    # Fig 3: Penalty & Latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(["HPC Penalty", "LPC Penalty"], [res["penalty_hpc"], res["penalty_lpc"]], color=["#E53935", "#FF7043"], width=0.5, edgecolor="black")
    ax1.set_ylabel("Penalty Cost ($)")
    ax1.set_title("LRU SLA Penalty Breakdown", pad=12)
    ax1.annotate(f"{res['penalty_hpc']:,}", (0, res["penalty_hpc"] + 1500), ha="center", fontweight="bold")
    ax1.annotate(f"{res['penalty_lpc']:,}", (1, res["penalty_lpc"] + 1500), ha="center", fontweight="bold")

    ax2.bar(["HPC Latency", "LPC Latency"], [res["avg_hpc_retrieval_time"] * 1000, res["avg_lpc_retrieval_time"] * 1000], color=["#039BE5", "#26C6DA"], width=0.5, edgecolor="black")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("LRU Latency by Priority", pad=12)
    ax2.annotate(f"{res['avg_hpc_retrieval_time']*1000:.2f} ms", (0, res["avg_hpc_retrieval_time"] * 1000 + 1), ha="center", fontweight="bold")
    ax2.annotate(f"{res['avg_lpc_retrieval_time']*1000:.2f} ms", (1, res["avg_lpc_retrieval_time"] * 1000 + 1), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_lru_penalty_latency.png"))
    plt.close()

    # Fig 4: Storage utilization
    fig, ax = plt.subplots(figsize=(6, 5))
    storage = ["DRAM Used", "SSD Used"]
    sizes_kb = [res["used_size_dram"] / 1024, res["used_size_ssd"] / 1024]
    bars = ax.bar(storage, sizes_kb, color=["#5C6BC0", "#8D6E63"], width=0.5, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} KB", (bar.get_x() + bar.get_width() / 2, h + 2), ha="center", fontweight="bold")
    ax.set_ylabel("Storage Occupation (KB)")
    ax.set_title("LRU Tier Storage Occupation", pad=12)
    ax.set_ylim(0, max(sizes_kb) * 1.2)
    plt.savefig(os.path.join(FIGURES_DIR, "4_lru_storage_utilization.png"))
    plt.close()

    print(f"All LRU results and figures generated in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
