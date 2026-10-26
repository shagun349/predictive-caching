"""
Main Experiment Runner: Whale Optimization Algorithm (WOA) Predictive Caching
==============================================================================
Runs the end-to-end benchmark on 'Dataset/memcached-sample':
  1. Preprocesses 20,000 requests with QoS tagging (30% HPC, 70% LPC).
  2. Partitions dataset strictly into:
       - Training trace: first 5,000 requests (for WOA parameter evolution)
       - Test trace: remaining 15,000 requests (held-out UNSEEN test set)
  3. Trains the Whale Optimization Algorithm (WOA) Engine.
  4. Evaluates WOA-Predictive against ABC, HO, DSOA, SFO, PSO, GA, LRU, LFU, and Random.
  5. Generates publication-grade figures, consolidated JSON results, and summary tables.
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Any

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
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

from abc_implementation.abc_bee import FoodSource
from abc_implementation.abc_predictive_policy import ABCPredictivePolicy

from lru_implementation.lru_policy import MultiTierLRUPolicy
from lfu_implementation.lfu_policy import MultiTierLFUPolicy
from random_implementation.random_policy import MultiTierRandomPolicy

from woa_implementation.woa_whale import Whale
from woa_implementation.woa_engine import WOAEngine
from woa_implementation.woa_predictive_policy import WOAPredictivePolicy

# Output directories inside woa_implementation
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def generate_bar_chart(
    policies: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filename: str,
    bar_colors: List[str]
):
    setup_plot_style()
    plt.figure(figsize=(11, 6))
    bars = plt.bar(policies, values, color=bar_colors, edgecolor="#212121", linewidth=0.8, width=0.6)
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=35, ha="right", fontweight="medium")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    is_pct = "%" in ylabel or "Ratio" in title
    for bar in bars:
        yval = bar.get_height()
        text_format = f"{yval:.2f}%" if is_pct else f"{int(yval):,}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(values) * 0.01),
            text_format,
            va='bottom',
            ha='center',
            fontsize=9.5,
            fontweight="bold"
        )

    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()


def generate_woa_convergence_plot(history: List[Dict]):
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    iterations = [h["iteration"] for h in history]
    best_fits = [h["best_fitness"] for h in history]
    mean_fits = [h["mean_fitness"] for h in history]
    best_chrs = [h["best_chr_total"] * 100 for h in history]
    best_hpcs = [h["best_chr_hpc"] * 100 for h in history]

    line1 = ax1.plot(iterations, best_fits, "o-", color="#00838F", linewidth=2.4, label="Best Whale Fitness")
    line2 = ax1.plot(iterations, mean_fits, "s--", color="#4DD0E1", linewidth=1.6, alpha=0.85, label="Pod Mean Fitness")
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Multi-Objective Fitness", fontweight="bold", color="#006064")
    ax1.tick_params(axis="y", labelcolor="#006064")

    ax2 = ax1.twinx()
    line3 = ax2.plot(iterations, best_chrs, "^-", color="#1E88E5", linewidth=1.8, label="Best Total CHR (%)")
    line4 = ax2.plot(iterations, best_hpcs, "d-.", color="#00897B", linewidth=1.8, label="Best HPC CHR (%)")
    ax2.set_ylabel("Cache Hit Ratio (%)", fontweight="bold", color="#1E88E5")
    ax2.tick_params(axis="y", labelcolor="#1E88E5")
    ax2.grid(False)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.92)
    plt.title("Whale Optimization Algorithm (WOA) Multi-Objective Convergence Curve", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "6_woa_convergence.png"))
    plt.close()


def generate_comparative_summary_plot(results: Dict[str, Any]):
    setup_plot_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    policies = list(results.keys())

    colors_map = {
        "WOA-Predictive (Ours)": "#00ACC1",  # Vibrant Cyan / Whale Ocean
        "ABC-Predictive": "#FF8F00",         # Warm Amber / Bee Gold
        "HO-Predictive": "#2E7D32",          # Forest Green
        "DSOA-Predictive": "#D81B60",        # Rose
        "SFO-Predictive": "#00897B",         # Teal
        "PSO-Predictive": "#0288D1",         # Light Blue
        "GA-Predictive": "#5E35B1",          # Deep Purple
        "LRU": "#E53935",                    # Red
        "LFU": "#6D4C41",                    # Brown
        "Random": "#757575"                  # Grey
    }
    palette = [colors_map.get(p, "#424242") for p in policies]

    # Panel 1: Total CHR
    chrs = [results[p]["chr_total"] * 100 for p in policies]
    bars1 = ax1.bar(policies, chrs, color=palette, edgecolor="black", linewidth=0.6)
    ax1.set_title("(a) Total Cache Hit Ratio (%)", pad=10)
    ax1.set_ylabel("Hit Ratio (%)")
    ax1.set_xticks(range(len(policies)))
    ax1.set_xticklabels(policies, rotation=35, ha="right")
    for b, v in zip(bars1, chrs):
        ax1.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v + 0.5), ha="center", fontsize=8, fontweight="bold")

    # Panel 2: HPC CHR
    hpcs = [results[p]["chr_hpc"] * 100 for p in policies]
    bars2 = ax2.bar(policies, hpcs, color=palette, edgecolor="black", linewidth=0.6)
    ax2.set_title("(b) High-Priority Content (HPC) Hit Ratio", pad=10)
    ax2.set_ylabel("HPC Hit Ratio (%)")
    ax2.set_xticks(range(len(policies)))
    ax2.set_xticklabels(policies, rotation=35, ha="right")
    for b, v in zip(bars2, hpcs):
        ax2.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v + 0.5), ha="center", fontsize=8, fontweight="bold")

    # Panel 3: Penalty
    pens = [results[p]["penalty_total"] for p in policies]
    bars3 = ax3.bar(policies, pens, color=palette, edgecolor="black", linewidth=0.6)
    ax3.set_title("(c) Total SLA Penalty Cost (Lower is Better)", pad=10)
    ax3.set_ylabel("Penalty ($)")
    ax3.set_xticks(range(len(policies)))
    ax3.set_xticklabels(policies, rotation=35, ha="right")
    for b, v in zip(bars3, pens):
        ax3.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v + 1500), ha="center", fontsize=7.5, fontweight="bold")

    # Panel 4: Latency
    lats = [results[p]["avg_retrieval_time"] * 1000 for p in policies]
    bars4 = ax4.bar(policies, lats, color=palette, edgecolor="black", linewidth=0.6)
    ax4.set_title("(d) Average End-to-End Latency (ms)", pad=10)
    ax4.set_ylabel("Latency (ms)")
    ax4.set_xticks(range(len(policies)))
    ax4.set_xticklabels(policies, rotation=35, ha="right")
    for b, v in zip(bars4, lats):
        ax4.annotate(f"{v:.1f}ms", (b.get_x() + b.get_width() / 2, v + 0.5), ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Multi-Tier Predictive Caching: Whale Optimization Algorithm (WOA) Benchmark", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "7_all_algos_comparison.png"))
    plt.close()


def generate_woa_bubble_net_dynamics_plot(history: List[Dict]):
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    iterations = [h["iteration"] for h in history]
    spiral = np.array([h["spiral_maneuvers"] for h in history])
    encircling = np.array([h["encircling_maneuvers"] for h in history])
    search = np.array([h["search_maneuvers"] for h in history])
    a_param = [h["a_parameter"] for h in history]

    bar_width = 0.65
    b1 = ax1.bar(iterations, spiral, width=bar_width, label="Bubble-Net Spiral Attack", color="#00ACC1", edgecolor="#006064", linewidth=0.7)
    b2 = ax1.bar(iterations, encircling, width=bar_width, bottom=spiral, label="Encircling Prey", color="#26A69A", edgecolor="#004D40", linewidth=0.7)
    b3 = ax1.bar(iterations, search, width=bar_width, bottom=spiral + encircling, label="Search for Prey (Exploration)", color="#FFA726", edgecolor="#E65100", linewidth=0.7)

    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Hunting Maneuver Count per Pod Iteration", fontweight="bold")
    ax1.set_ylim(0, 35)

    ax2 = ax1.twinx()
    line_a = ax2.plot(iterations, a_param, "r--", linewidth=2.0, label="Exploration Decay Parameter (a)")
    ax2.set_ylabel("Parameter a (2 → 0)", fontweight="bold", color="#D32F2F")
    ax2.tick_params(axis="y", labelcolor="#D32F2F")
    ax2.set_ylim(0, 2.2)
    ax2.grid(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", framealpha=0.92)

    plt.title("WOA Hunting Dynamics: Transition from Exploration (Search) to Exploitation (Bubble-Net)", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "8_woa_bubble_net_dynamics.png"))
    plt.close()


def main():
    print("=" * 80)
    print("  WHALE OPTIMIZATION ALGORITHM (WOA) PREDICTIVE CACHING BENCHMARK")
    print("  Dataset: 'Dataset/memcached-sample' (20,000 requests)")
    print("  Protocol: Strict Fair Train/Test Split (5k Train / 15k Unseen Test)")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 80)

    # 1. Preprocess Dataset
    raw_dataset = os.path.join(PROJECT_ROOT, "Dataset", "memcached-sample")
    print(f"\n[1/5] Preprocessing dataset from: {raw_dataset} ...")
    preprocessor = TracePreprocessor(raw_dataset, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process()

    total_requests = len(records)
    hpc_count = sum(1 for r in records if r["priority"] == "h")
    lpc_count = total_requests - hpc_count
    print(f"  Total requests processed: {total_requests}")
    print(f"  Unique catalog items: {stats['unique_objects']}")
    print(f"  HPC requests: {hpc_count} ({hpc_count/total_requests*100:.2f}%)")
    print(f"  LPC requests: {lpc_count} ({lpc_count/total_requests*100:.2f}%)")
    print(f"  Mean object size: {stats['mean_object_size']:.2f} bytes")

    # 2. Hardware and Cache Configuration
    unique_key_sizes = {r["key"]: r["size"] for r in records}
    total_catalog_bytes = sum(unique_key_sizes.values())
    total_cache_size = int(total_catalog_bytes * 0.10)
    dram_fraction = 0.20
    dram_bytes = int(total_cache_size * dram_fraction)
    ssd_bytes = int(total_cache_size * (1.0 - dram_fraction))
    mean_obj_size = int(stats["mean_object_size"])

    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=dram_fraction,
        ssd_throughput_gb=2.0,
        slot_size=mean_obj_size
    )

    # 3. Training Trace Partition
    train_size = min(5000, len(records))
    training_trace = records[:train_size]
    test_trace = records[train_size:]

    print(f"\n[3/5] Training WOA Engine on {train_size} requests (excluding test data)...")
    woa_engine = WOAEngine(population_size=30, iterations=25, b=1.0, seed=42)

    t0_woa = time.time()
    best_whale = woa_engine.optimize(simulator, training_trace, verbose=True)
    woa_duration = time.time() - t0_woa
    print(f"  WOA Training successfully completed in {woa_duration:.2f} seconds.")
    print(f"  Best Evolved Fitness: {best_whale.fitness:.4f}")

    # Save training history and best weights
    with open(os.path.join(RESULTS_DIR, "woa_training_history.json"), "w") as f:
        json.dump(woa_engine.history, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "woa_best_weights.json"), "w") as f:
        json.dump(best_whale.position, f, indent=2)

    # 4. Evaluation on UNSEEN test set
    print(f"\n[4/5] Fair Evaluation of ALL Policies on {len(test_trace)} UNSEEN Test Requests...")
    print("  Ensuring zero data leakage: NO policy is evaluated on its training trace.")

    # Load peer weights
    ga_path = os.path.join(PROJECT_ROOT, "genetic_algorithm_implementation", "results", "consolidated_results.json")
    with open(ga_path, "r") as f:
        ga_data = json.load(f)
    ga_weights = ga_data["ga_optimization"]["best_genes"]
    ga_chrom = Chromosome()
    ga_chrom.genes = list(ga_weights.values())

    dsoa_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "dsoa_best_weights.json")
    with open(dsoa_path, "r") as f:
        dsoa_w = json.load(f)
    dsoa_dove = Dove(position_seed=dsoa_w)

    sfo_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "sfo_best_weights.json")
    with open(sfo_path, "r") as f:
        sfo_w = json.load(f)
    sfo_fish = Fish(position_seed=sfo_w)

    pso_path = os.path.join(PROJECT_ROOT, "dsoa_implementation", "results", "pso_best_weights.json")
    with open(pso_path, "r") as f:
        pso_w = json.load(f)
    pso_part = Particle(position_seed=pso_w)

    ho_path = os.path.join(PROJECT_ROOT, "ho_implementation", "results", "ho_best_weights.json")
    with open(ho_path, "r") as f:
        ho_w = json.load(f)
    ho_hippo = Hippo(position_seed=ho_w)

    abc_path = os.path.join(PROJECT_ROOT, "abc_implementation", "results", "abc_best_weights.json")
    with open(abc_path, "r") as f:
        abc_w = json.load(f)
    abc_fs = FoodSource(position_seed=abc_w)

    policies = {
        "WOA-Predictive (Ours)": WOAPredictivePolicy(best_whale),
        "ABC-Predictive": ABCPredictivePolicy(abc_fs),
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
    print("\n  Evaluating policies sequentially on unseen test workload:")
    for pol_name, pol_inst in policies.items():
        t0 = time.time()
        res = simulator.run(pol_inst, test_trace)
        elapsed = time.time() - t0
        results[pol_name] = res
        print(f"    [{pol_name:21s}] Total CHR: {res['chr_total']*100:5.2f}% | "
              f"HPC CHR: {res['chr_hpc']*100:5.2f}% | LPC CHR: {res['chr_lpc']*100:5.2f}% | "
              f"Penalty: {res['penalty_total']:7d} | Latency: {res['avg_retrieval_time']*1000:6.2f} ms | Time: {elapsed:.2f}s")

    # 5. Save results and generate figures
    print(f"\n[5/5] Generating publication figures and consolidated summaries...")

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    consolidated = {
        "experiment_metadata": {
            "dataset": "Dataset/memcached-sample",
            "total_trace_requests": total_requests,
            "training_samples": train_size,
            "unseen_test_samples": len(test_trace),
            "catalog_bytes": total_catalog_bytes,
            "cache_size_bytes": total_cache_size,
            "dram_bytes": dram_bytes,
            "ssd_bytes": ssd_bytes,
            "woa_training_time_sec": woa_duration
        },
        "woa_evolved_weights": best_whale.position,
        "results": results
    }
    with open(os.path.join(RESULTS_DIR, "consolidated_comparison.json"), "w") as f:
        json.dump(consolidated, f, indent=2)

    # ASCII Summary Table
    summary_txt_path = os.path.join(RESULTS_DIR, "summary_table.txt")
    header = f"{'Policy':23s} | {'Total CHR':9s} | {'HPC CHR':9s} | {'LPC CHR':9s} | {'Penalty HPC':11s} | {'Penalty LPC':11s} | {'Total Penalty':13s} | {'Latency (ms)':12s}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for p, m in results.items():
        lines.append(
            f"{p:23s} | {m['chr_total']*100:8.2f}% | {m['chr_hpc']*100:8.2f}% | {m['chr_lpc']*100:8.2f}% | "
            f"{m['penalty_hpc']:11d} | {m['penalty_lpc']:11d} | {m['penalty_total']:13d} | {m['avg_retrieval_time']*1000:12.3f}"
        )
    lines.append(sep)
    summary_text = "\n".join(lines)
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print("\n" + summary_text + "\n")

    # Generate Figures
    policy_names = list(results.keys())
    colors_map = {
        "WOA-Predictive (Ours)": "#00ACC1",
        "ABC-Predictive": "#FF8F00",
        "HO-Predictive": "#2E7D32",
        "DSOA-Predictive": "#D81B60",
        "SFO-Predictive": "#00897B",
        "PSO-Predictive": "#0288D1",
        "GA-Predictive": "#5E35B1",
        "LRU": "#E53935",
        "LFU": "#6D4C41",
        "Random": "#757575"
    }
    palette = [colors_map.get(p, "#424242") for p in policy_names]

    # 1. Total CHR
    generate_bar_chart(
        policy_names, [results[p]["chr_total"] * 100 for p in policy_names],
        "Total Cache Hit Ratio (CHR) on Unseen Test Workload",
        "Hit Ratio (%)", "1_total_chr.png", palette
    )

    # 2. HPC CHR
    generate_bar_chart(
        policy_names, [results[p]["chr_hpc"] * 100 for p in policy_names],
        "High Priority Content (HPC) Cache Hit Ratio (SLA Assurance)",
        "HPC Hit Ratio (%)", "2_hpc_chr.png", palette
    )

    # 3. LPC CHR
    generate_bar_chart(
        policy_names, [results[p]["chr_lpc"] * 100 for p in policy_names],
        "Low Priority Content (LPC) Cache Hit Ratio",
        "LPC Hit Ratio (%)", "3_lpc_chr.png", palette
    )

    # 4. Total Penalty
    generate_bar_chart(
        policy_names, [results[p]["penalty_total"] for p in policy_names],
        "Total SLA Penalty Cost (Lower is Better)",
        "Penalty Cost", "4_total_penalty.png", palette
    )

    # 5. Average Latency
    generate_bar_chart(
        policy_names, [results[p]["avg_retrieval_time"] * 1000 for p in policy_names],
        "Average End-to-End Data Retrieval Latency (Lower is Better)",
        "Latency (ms)", "5_avg_latency.png", palette
    )

    # 6. WOA Convergence Curve
    generate_woa_convergence_plot(woa_engine.history)

    # 7. Comparative 4-panel summary
    generate_comparative_summary_plot(results)

    # 8. WOA Bubble-Net Dynamics
    generate_woa_bubble_net_dynamics_plot(woa_engine.history)

    print(f"All figures and results generated in: {FIGURES_DIR}")
    print("=" * 80)
    print("WOA EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
