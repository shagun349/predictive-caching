"""
Main Experiment Runner: Dove Swarm Optimization Algorithm (DSOA-EC) Predictive Caching
======================================================================================
Runs the complete end-to-end benchmark on 'Dataset/memcached-sample':
  1. Preprocesses the 20,000 trace requests with QoS tagging (30% HPC, 70% LPC).
  2. Partitions dataset strictly into:
       - Training trace: first 5,000 requests (used ONLY for DSOA parameter search)
       - Test trace: remaining 15,000 requests (held-out UNSEEN data for evaluation)
  3. Trains the Dove Swarm Optimization (DSOA-EC) Engine.
  4. Evaluates DSOA-Predictive against SFO-Predictive, PSO-Predictive, GA-Predictive,
     LRU, LFU, and Random strictly on the UNSEEN test set.
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
from genetic_algorithm_implementation.baseline_policies import (
    LRUPolicy, LFUPolicy, RandPolicy
)
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from genetic_algorithm_implementation.ga_chromosome import Chromosome

from pso_implementation.pso_particle import Particle
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy
from pso_implementation.pso_engine import PSOEngine

from sfo_implementation.sfo_fish import Fish
from sfo_implementation.sfo_engine import SFOEngine
from sfo_implementation.sfo_predictive_policy import SFOPredictivePolicy

from dsoa_implementation.dsoa_dove import Dove
from dsoa_implementation.dsoa_engine import DSOAEngine
from dsoa_implementation.dsoa_predictive_policy import DSOAPredictivePolicy

# Output directories inside dsoa_implementation
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def setup_plot_style():
    """Configure clean matplotlib aesthetic parameters."""
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
    """Generate a clean annotated bar chart."""
    setup_plot_style()
    plt.figure(figsize=(11, 6))
    bars = plt.bar(policies, values, color=bar_colors, edgecolor="#212121", linewidth=0.8, width=0.6)
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=40, ha="right", fontweight="medium")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    is_pct = "%" in ylabel or "Ratio" in title
    for bar in bars:
        yval = bar.get_height()
        text_format = f"{yval:.2f}%" if is_pct else f"{int(yval):,}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(values) * 0.015),
            text_format,
            va='bottom',
            ha='center',
            fontsize=9.5,
            fontweight="bold"
        )

    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()


def generate_dsoa_convergence_plot(history: List[Dict[str, Any]]):
    """Generate convergence curve showing fitness, CHR, and satiety progression."""
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(9, 5))

    iters = [h["iteration"] for h in history]
    best_fits = [h["best_fitness"] for h in history]
    avg_fits = [h["avg_fitness"] for h in history]
    best_chrs = [h["best_chr_total"] * 100 for h in history]
    best_hpcs = [h["best_chr_hpc"] * 100 for h in history]

    line1 = ax1.plot(iters, best_fits, "o-", color="#6D4C41", linewidth=2.2, label="Best Fitness")
    line2 = ax1.plot(iters, avg_fits, "s--", color="#A1887F", linewidth=1.5, alpha=0.8, label="Average Fitness")
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Multi-Objective Fitness", fontweight="bold", color="#3E2723")
    ax1.tick_params(axis="y", labelcolor="#3E2723")

    ax2 = ax1.twinx()
    line3 = ax2.plot(iters, best_chrs, "^-", color="#E65100", linewidth=1.8, label="Best Total CHR (%)")
    line4 = ax2.plot(iters, best_hpcs, "d-.", color="#1E88E5", linewidth=1.8, label="Best HPC CHR (%)")
    ax2.set_ylabel("Hit Ratio (%)", fontweight="bold", color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax2.grid(False)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)
    plt.title("Dove Swarm Optimization (DSOA-EC) Convergence History", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "6_dsoa_convergence.png"), dpi=200)
    plt.close()


def generate_dsoa_satiety_plot(history: List[Dict[str, Any]]):
    """Generate satiety & exploration dynamics plot unique to DSOA."""
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(9, 5))

    iters = [h["iteration"] for h in history]
    avg_satiety = [h["avg_satiety"] for h in history]
    exploring = [h["exploring_doves"] for h in history]

    line1 = ax1.plot(iters, avg_satiety, "o-", color="#6D4C41", linewidth=2.2, label="Avg Satiety")
    ax1.set_xlabel("Iteration", fontweight="bold")
    ax1.set_ylabel("Average Satiety Level", fontweight="bold", color="#3E2723")
    ax1.tick_params(axis="y", labelcolor="#3E2723")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    line2 = ax2.bar(iters, exploring, alpha=0.4, color="#FF7043", label="Exploring Doves")
    ax2.set_ylabel("# Exploring Doves (Hungry)", fontweight="bold", color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax2.grid(False)

    lines = line1 + [line2]
    labels = ["Avg Satiety", "Exploring Doves"]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    plt.title("DSOA-EC Satiety & Exploration Dynamics", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "8_dsoa_satiety_dynamics.png"), dpi=200)
    plt.close()


def generate_comparative_summary_plot(results: Dict[str, Any]):
    """Generate side-by-side comparison of Total CHR, HPC CHR, and LPC CHR."""
    setup_plot_style()
    policies = list(results.keys())
    chr_total = [results[p]["chr_total"] * 100 for p in policies]
    chr_hpc = [results[p]["chr_hpc"] * 100 for p in policies]
    chr_lpc = [results[p]["chr_lpc"] * 100 for p in policies]

    x = np.arange(len(policies))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6.5))
    rects1 = ax.bar(x - width, chr_total, width, label="Total CHR (%)", color="#1E88E5", edgecolor="#1565C0")
    rects2 = ax.bar(x, chr_hpc, width, label="HPC CHR (%)", color="#FB8C00", edgecolor="#E65100")
    rects3 = ax.bar(x + width, chr_lpc, width, label="LPC CHR (%)", color="#43A047", edgecolor="#2E7D32")

    ax.set_ylabel("Cache Hit Ratio (%)", fontweight="bold")
    ax.set_title("Comprehensive Cache Hit Ratio Comparison Across All Evaluated Policies (Unseen Test Trace)", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=35, ha="right", fontweight="medium")
    ax.legend(framealpha=0.9, loc="upper right")
    ax.set_ylim(0, max(chr_hpc) * 1.20)

    # Highlight DSOA
    for i, p in enumerate(policies):
        if "DSOA" in p:
            ax.get_xticklabels()[i].set_fontweight("bold")
            ax.get_xticklabels()[i].set_color("#6D4C41")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "7_all_algos_comparison.png"), dpi=200)
    plt.close()


def main():
    print("=" * 80)
    print("  DOVE SWARM OPTIMIZATION ALGORITHM (DSOA-EC) PREDICTIVE CACHING BENCHMARK")
    print("  Dataset: 'Dataset/memcached-sample' (20,000 requests)")
    print("  Protocol: Strict Fair Train/Test Split (5k Train / 15k Unseen Test)")
    print(f"  Results Directory: {RESULTS_DIR}")
    print("=" * 80)

    # 1. Dataset Loading and Preprocessing
    raw_dataset = os.path.join(PROJECT_ROOT, "Dataset", "memcached-sample")
    print(f"\n[1/5] Preprocessing dataset from: {raw_dataset} ...")
    preprocessor = TracePreprocessor(raw_dataset, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process()

    total_requests = len(records)
    print(f"  Total requests processed: {total_requests}")
    print(f"  Unique catalog items: {stats['unique_objects']}")
    print(f"  HPC requests: {stats['hpc_requests']} ({stats['hpc_ratio']*100:.2f}%)")
    print(f"  LPC requests: {stats['lpc_requests']} ({stats['lpc_ratio']*100:.2f}%)")
    print(f"  Mean object size: {stats['mean_object_size']:.2f} bytes")

    # 2. Configure Cache Sizing (exact 10% catalog capacity matching paper)
    unique_key_sizes = {r["key"]: r["size"] for r in records}
    total_catalog_bytes = sum(unique_key_sizes.values())
    total_cache_size = int(total_catalog_bytes * 0.10)  # 10% of catalog
    dram_fraction = 0.20
    mean_obj_size = int(stats["mean_object_size"])

    print(f"\n[2/5] Hardware & Multi-Tier Storage Configuration:")
    print(f"  Total Catalog: {total_catalog_bytes:,} bytes ({total_catalog_bytes/1024/1024:.2f} MB)")
    print(f"  Total Cache (10%): {total_cache_size:,} bytes ({total_cache_size/1024:.2f} KB)")
    print(f"  DRAM (20%): {int(total_cache_size * dram_fraction):,} bytes (100 ns latency, 40 GB/s)")
    print(f"  SSD (80%): {int(total_cache_size * (1 - dram_fraction)):,} bytes (10 us latency, 2 GB/s)")

    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=dram_fraction,
        ssd_throughput_gb=2.0,
        slot_size=mean_obj_size
    )

    # 3. Train DSOA-EC Engine on Training Split ONLY
    train_size = 5000
    training_trace = records[:train_size]
    test_trace = records[train_size:]

    dsoa_weights_file = os.path.join(RESULTS_DIR, "dsoa_best_weights.json")
    dsoa_history_file = os.path.join(RESULTS_DIR, "dsoa_training_history.json")
    pso_weights_file = os.path.join(RESULTS_DIR, "pso_best_weights.json")
    sfo_weights_file = os.path.join(RESULTS_DIR, "sfo_best_weights.json")

    force_retrain = "--retrain" in sys.argv
    need_dsoa_train = force_retrain or not os.path.exists(dsoa_weights_file) or not os.path.exists(dsoa_history_file)

    if need_dsoa_train:
        print(f"\n[3/5] Training DSOA-EC Engine on {train_size} requests (excluding test data)...")
        print("  DSOA-EC Configuration: 30 doves, 25 iterations, eta=0.6, satiety_decay=0.85, levy_prob=0.15, seed=42")

        dsoa_engine = DSOAEngine(
            num_doves=30,
            iterations=25,
            eta=0.6,
            satiety_decay=0.85,
            satiety_threshold=0.3,
            levy_prob=0.15,
            w_chr=0.40,
            w_hpc=0.35,
            w_pen=0.15,
            w_lat=0.10,
            seed=42
        )

        t0_dsoa = time.time()
        best_dove = dsoa_engine.optimize(simulator, training_trace, verbose=True)
        dsoa_train_time = time.time() - t0_dsoa
        print(f"  DSOA-EC Training successfully completed in {dsoa_train_time:.2f} seconds.")
        print(f"  Best Evolved Fitness: {best_dove.fitness:.4f}")

        # Save DSOA training history & weights
        with open(dsoa_history_file, "w") as f:
            json.dump(dsoa_engine.history, f, indent=2)

        with open(dsoa_weights_file, "w") as f:
            json.dump(best_dove.position, f, indent=2)
        dsoa_history = dsoa_engine.history
    else:
        print(f"\n[3/5] Loading pre-trained DSOA-EC weights from: {dsoa_weights_file} (pass --retrain to re-run search)...")
        with open(dsoa_weights_file, "r") as f:
            dsoa_pos = json.load(f)
        best_dove = Dove(position_seed=dsoa_pos)
        with open(dsoa_history_file, "r") as f:
            dsoa_history = json.load(f)
        dsoa_train_time = 0.0
        print("  DSOA-EC Best Weights loaded successfully.")

    # 4. Fair Evaluation on UNSEEN Test Set Only
    print(f"\n[4/5] Fair Evaluation of ALL Policies on {len(test_trace)} UNSEEN Test Requests...")
    print("  Ensuring zero data leakage: NO policy is evaluated on its training trace.")

    # Configure GA chromosome with evolved best weights
    ga_best_genes = {
        "w_recency": 0.5478194149316628,
        "w_frequency": 0.44510894440121584,
        "w_freq_decay": 0.707548695492647,
        "w_size": 0.06138790081789007,
        "w_priority": 2.1159543598956696,
        "w_penalty": 1.2893921108912005,
        "w_predicted_reuse": -0.883952162238629,
        "theta_dram": 0.8938328267508344,
        "alpha_qos": 0.7815057707433745,
        "target_p": 0.7199189990523559
    }
    ga_chromosome = Chromosome(genes=list(ga_best_genes.values()))

    # Configure PSO particle: load cached or train
    if os.path.exists(pso_weights_file) and not force_retrain:
        print(f"  Loading pre-trained PSO best weights from: {pso_weights_file} ...")
        with open(pso_weights_file, "r") as f:
            pso_pos = json.load(f)
        best_particle = Particle(position_seed=pso_pos)
    else:
        print("  Training PSO-Predictive baseline (30 particles, 25 iterations)...")
        pso_engine = PSOEngine(num_particles=30, iterations=25, seed=42)
        best_particle = pso_engine.optimize(simulator, training_trace, verbose=False)
        with open(pso_weights_file, "w") as f:
            json.dump(best_particle.position, f, indent=2)

    # Configure SFO fish: load cached or train
    if os.path.exists(sfo_weights_file) and not force_retrain:
        print(f"  Loading pre-trained SFO best weights from: {sfo_weights_file} ...")
        with open(sfo_weights_file, "r") as f:
            sfo_pos = json.load(f)
        best_fish = Fish(position_seed=sfo_pos)
    else:
        print("  Training SFO-Predictive baseline (30 fish, 25 iterations)...")
        sfo_engine = SFOEngine(num_fish=30, iterations=25, seed=42)
        best_fish = sfo_engine.optimize(simulator, training_trace, verbose=False)
        with open(sfo_weights_file, "w") as f:
            json.dump(best_fish.position, f, indent=2)

    policies: Dict[str, Any] = {
        "DSOA-Predictive (Ours)": DSOAPredictivePolicy(best_dove),
        "SFO-Predictive": SFOPredictivePolicy(best_fish),
        "PSO-Predictive": PSOPredictivePolicy(best_particle),
        "GA-Predictive": GAPredictivePolicy(ga_chromosome),
        "LRU": LRUPolicy(),
        "LFU": LFUPolicy(),
        "Random": RandPolicy()
    }

    results: Dict[str, Any] = {}
    print("\n  Evaluating policies sequentially on unseen test workload:")
    for pol_name, pol_inst in policies.items():
        t_eval = time.time()
        res = simulator.run(pol_inst, test_trace)
        eval_duration = time.time() - t_eval
        results[pol_name] = res
        print(
            f"    [{pol_name:23s}] Total CHR: {res['chr_total']*100:5.2f}% | "
            f"HPC CHR: {res['chr_hpc']*100:5.2f}% | "
            f"LPC CHR: {res['chr_lpc']*100:5.2f}% | "
            f"Penalty: {res['penalty_total']:7d} | "
            f"Latency: {res['avg_retrieval_time']*1000:6.2f} ms | "
            f"Time: {eval_duration:.2f}s"
        )

    # 5. Save Results, Summary Table, and Publication Figures
    print(f"\n[5/5] Generating publication figures and consolidated summaries...")

    # Save evaluation_results.json
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Consolidated comparison json
    consolidated_output = {
        "experiment_metadata": {
            "dataset": "Dataset/memcached-sample",
            "total_trace_requests": total_requests,
            "training_samples": train_size,
            "unseen_test_samples": len(test_trace),
            "catalog_bytes": total_catalog_bytes,
            "cache_size_bytes": total_cache_size,
            "dram_bytes": int(total_cache_size * dram_fraction),
            "ssd_bytes": int(total_cache_size * (1 - dram_fraction)),
            "dsoa_training_time_sec": dsoa_train_time
        },
        "dsoa_evolved_weights": best_dove.position,
        "results": results
    }
    with open(os.path.join(RESULTS_DIR, "consolidated_comparison.json"), "w") as f:
        json.dump(consolidated_output, f, indent=2)

    # Summary table
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

    # Custom color palette with DSOA highlighted
    colors_map = {
        "DSOA-Predictive (Ours)": "#6D4C41",   # Rich Brown (Dove)
        "SFO-Predictive": "#00897B",            # Teal
        "PSO-Predictive": "#0288D1",            # Light Blue
        "GA-Predictive": "#5E35B1",             # Deep Purple
        "LRU": "#E53935",                       # Red
        "LFU": "#FB8C00",                       # Amber
        "Random": "#757575"                     # Grey
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

    # 6. DSOA Convergence Curve
    generate_dsoa_convergence_plot(dsoa_history)

    # 7. Side-by-side comparative summary
    generate_comparative_summary_plot(results)

    # 8. DSOA Satiety Dynamics (unique to DSOA)
    generate_dsoa_satiety_plot(dsoa_history)

    print(f"All figures and results generated in: {RESULTS_DIR}")
    print("=" * 80)
    print("DSOA-EC EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
