"""
Main Experiment Runner: Genetic Algorithm Predictive Caching
============================================================
Runs the end-to-end experiment using 'Dataset/memcached-sample':
  1. Preprocesses the 20,000 memcached trace requests.
  2. Runs Genetic Algorithm to evolve the optimal predictive caching chromosome.
  3. Benchmarks GA-Predictive Cache against all 7 paper baseline policies.
  4. Saves all results and publication-quality plots strictly inside this folder!
"""

import json
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
from genetic_algorithm_implementation.ga_chromosome import Chromosome, GENE_NAMES
from genetic_algorithm_implementation.ga_engine import GeneticAlgorithmEngine
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.baseline_policies import (
    LRUPolicy, LFUPolicy, RandPolicy
)

# Output directories inside genetic_algorithm_implementation only
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def setup_plot_style():
    """Configure clean matplotlib style."""
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
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def generate_plots(history: List[Dict[str, Any]], results: Dict[str, Any], best_chromosome: Chromosome):
    """Generate all publication-quality comparison figures."""
    setup_plot_style()
    policies = list(results.keys())

    # Color palette
    colors = {
        "GA-Predictive": "#00897B",  # Deep Teal
        "LRU": "#E53935",            # Red
        "LFU": "#6D4C41",            # Brown
        "Random": "#757575",         # Grey
    }
    policy_palette = [colors.get(p, "#424242") for p in policies]

    # --------------------------------------------------------------------------
    # Fig 1: GA Convergence Curve
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    gens = [h["generation"] for h in history]
    best_fits = [h["best_fitness"] for h in history]
    mean_fits = [h["mean_fitness"] for h in history]
    best_chrs = [h["best_chr_total"] * 100 for h in history]
    best_hpcs = [h["best_chr_hpc"] * 100 for h in history]

    line1 = ax1.plot(gens, best_fits, "o-", color="#00897B", linewidth=2.2, label="Best Fitness")
    line2 = ax1.plot(gens, mean_fits, "s--", color="#26A69A", linewidth=1.5, alpha=0.8, label="Mean Fitness")
    ax1.set_xlabel("Generation", fontweight="bold")
    ax1.set_ylabel("Multi-Objective Fitness", fontweight="bold", color="#004D40")
    ax1.tick_params(axis="y", labelcolor="#004D40")

    ax2 = ax1.twinx()
    line3 = ax2.plot(gens, best_chrs, "^-", color="#E65100", linewidth=1.8, label="Best CHR (%)")
    line4 = ax2.plot(gens, best_hpcs, "d-.", color="#1E88E5", linewidth=1.8, label="Best HPC CHR (%)")
    ax2.set_ylabel("Hit Ratio (%)", fontweight="bold", color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax2.grid(False)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", framealpha=0.9)
    plt.title("Genetic Algorithm Optimization Convergence Across Generations", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_ga_convergence.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 2: Total Cache Hit Ratio (Stacked DRAM + SSD)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(policies))
    width = 0.55

    dram_vals = [results[p]["chr_dram"] * 100 for p in policies]
    ssd_vals = [results[p]["chr_ssd"] * 100 for p in policies]

    ax.bar(x, dram_vals, width, label="DRAM Hit Ratio", color="#1E88E5", edgecolor="black", linewidth=0.5)
    ax.bar(x, ssd_vals, width, bottom=dram_vals, label="SSD Hit Ratio", color="#FFB300", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Cache Hit Ratio (%)", fontweight="bold")
    ax.set_title("Total Cache Hit Ratio (DRAM + SSD) across Policies", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    for i in range(len(policies)):
        tot = dram_vals[i] + ssd_vals[i]
        ax.annotate(f"{tot:.2f}%", (x[i], tot + 0.4), ha="center", fontsize=9.5, fontweight="bold")

    max_tot = max([d + s for d, s in zip(dram_vals, ssd_vals)])
    ax.set_ylim(0, max_tot * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_chr_total_comparison.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 3: High Priority Content CHR (HPC)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    hpc_vals = [results[p]["chr_hpc"] * 100 for p in policies]
    bars = ax.bar(x, hpc_vals, width, color=policy_palette, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("High-Priority CHR (%)", fontweight="bold")
    ax.set_title("High-Priority Content (HPC) Cache Hit Ratio Comparison", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontweight="bold")

    for bar, val in zip(bars, hpc_vals):
        ax.annotate(f"{val:.2f}%", (bar.get_x() + bar.get_width() / 2, val + 0.4),
                    ha="center", fontsize=9.5, fontweight="bold")

    ax.set_ylim(0, max(hpc_vals) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_chr_hpc_comparison.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 4: Low Priority Content CHR (LPC)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    lpc_vals = [results[p]["chr_lpc"] * 100 for p in policies]
    bars = ax.bar(x, lpc_vals, width, color=policy_palette, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Low-Priority CHR (%)", fontweight="bold")
    ax.set_title("Low-Priority Content (LPC) Cache Hit Ratio Comparison", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontweight="bold")

    for bar, val in zip(bars, lpc_vals):
        ax.annotate(f"{val:.2f}%", (bar.get_x() + bar.get_width() / 2, val + 0.4),
                    ha="center", fontsize=9.5, fontweight="bold")

    ax.set_ylim(0, max(lpc_vals) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_chr_lpc_comparison.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 5: Penalty Comparison (HPC + LPC SLA Miss Penalty)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    pen_hpc = [results[p]["penalty_hpc"] for p in policies]
    pen_lpc = [results[p]["penalty_lpc"] for p in policies]

    ax.bar(x, pen_hpc, width, label="HPC SLA Penalty", color="#D32F2F", edgecolor="black", linewidth=0.5)
    ax.bar(x, pen_lpc, width, bottom=pen_hpc, label="LPC SLA Penalty", color="#FF7043", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Miss Penalty Cost (SLA Violation Units)", fontweight="bold")
    ax.set_title("SLA Miss Penalty Costs across Policies (Lower is Better)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    for i in range(len(policies)):
        tot_pen = pen_hpc[i] + pen_lpc[i]
        ax.annotate(f"{tot_pen:,}", (x[i], tot_pen * 1.01), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, max([h + l for h, l in zip(pen_hpc, pen_lpc)]) * 1.12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_penalty_comparison.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 6: Data Retrieval Latency
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.26
    hpc_lat = [results[p]["avg_hpc_retrieval_time"] * 1000 for p in policies]
    lpc_lat = [results[p]["avg_lpc_retrieval_time"] * 1000 for p in policies]
    tot_lat = [results[p]["avg_retrieval_time"] * 1000 for p in policies]

    ax.bar(x - w, hpc_lat, w, label="Avg HPC Latency (ms)", color="#0288D1", edgecolor="black", linewidth=0.5)
    ax.bar(x, lpc_lat, w, label="Avg LPC Latency (ms)", color="#7E57C2", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, tot_lat, w, label="Avg Overall Latency (ms)", color="#26A69A", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Retrieval Latency (milliseconds)", fontweight="bold")
    ax.set_title("Average Data Retrieval Time (Lower is Better)", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_retrieval_time_comparison.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 7: I/O Operations (Reads and Writes)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    reads_d = [results[p]["reads_dram"] for p in policies]
    reads_s = [results[p]["reads_ssd"] for p in policies]
    writes_d = [results[p]["writes_dram"] for p in policies]
    writes_s = [results[p]["writes_ssd"] for p in policies]

    ax1.bar(x, reads_d, width, label="DRAM Reads", color="#1E88E5")
    ax1.bar(x, reads_s, width, bottom=reads_d, label="SSD Reads", color="#FFB300")
    ax1.set_title("Read Operations Breakdown", fontweight="bold")
    ax1.set_ylabel("Read Operations", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, rotation=35, ha="right", fontsize=8.5)
    ax1.legend(loc="upper left")

    ax2.bar(x, writes_d, width, label="DRAM Writes", color="#43A047")
    ax2.bar(x, writes_s, width, bottom=writes_d, label="SSD Writes", color="#FB8C00")
    ax2.set_title("Write Operations Breakdown", fontweight="bold")
    ax2.set_ylabel("Write Operations", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=35, ha="right", fontsize=8.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_io_operations.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 8: Evolved Chromosome Gene Weights
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    gene_dict = best_chromosome.get_gene_dict()
    g_names = list(gene_dict.keys())
    g_vals = list(gene_dict.values())
    bar_colors = ["#00897B" if v >= 0 else "#E53935" for v in g_vals]

    y = np.arange(len(g_names))
    ax.barh(y, g_vals, height=0.6, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(g_names, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Evolved Gene Weight / Parameter Value", fontweight="bold")
    ax.set_title("Best Evolved Chromosome: Feature Weights & Thresholds", pad=12)

    for i, val in enumerate(g_vals):
        offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        ax.annotate(f"{val:+.3f}", (val + offset, i), va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig8_chromosome_weights.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Fig 9: Rendered Visual Summary Table
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")

    table_data = [
        ["Policy", "Total CHR", "HPC CHR", "LPC CHR", "Penalty HPC", "Penalty LPC", "Total Penalty", "Latency (ms)"]
    ]
    for p in policies:
        m = results[p]
        table_data.append([
            p,
            f"{m['chr_total']*100:.2f}%",
            f"{m['chr_hpc']*100:.2f}%",
            f"{m['chr_lpc']*100:.2f}%",
            f"{m['penalty_hpc']:,}",
            f"{m['penalty_lpc']:,}",
            f"{m['penalty_total']:,}",
            f"{m['avg_retrieval_time']*1000:.2f}"
        ])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.6)

    # Style header row
    for j in range(len(table_data[0])):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", weight="bold")

    # Highlight GA row
    for j in range(len(table_data[0])):
        cell = table[1, j]
        cell.set_facecolor("#E0F2F1")
        cell.set_text_props(weight="bold", color="#004D40")

    plt.title("Performance Summary Table: GA-Predictive Cache vs Paper Baselines", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig9_summary_table.png"))
    plt.close()

    print(f"All 9 publication figures saved successfully to: {FIGURES_DIR}")


def main():
    print("=" * 70)
    print("  GENETIC ALGORITHM PREDICTIVE CACHING SIMULATION")
    print("  Dataset: 'Dataset/memcached-sample'")
    print("  Output Directory: genetic_algorithm_implementation/results/")
    print("=" * 70)

    # 1. Preprocess Dataset
    raw_dataset = os.path.join(PROJECT_ROOT, "Dataset", "memcached-sample")
    print(f"\n[1/5] Preprocessing dataset from: {raw_dataset} ...")
    preprocessor = TracePreprocessor(raw_dataset, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process()

    preprocessed_csv = os.path.join(RESULTS_DIR, "preprocessed_memcached_trace.csv")
    preprocessor.save_ndn_csv(records, preprocessed_csv)
    print(f"  Processed {stats['total_requests']} requests ({stats['unique_objects']} unique objects)")
    print(f"  HPC requests: {stats['hpc_requests']} ({stats['hpc_ratio']*100:.1f}%), LPC: {stats['lpc_requests']} ({stats['lpc_ratio']*100:.1f}%)")
    print(f"  Preprocessed trace saved to: {preprocessed_csv}")

    # 2. Configure Cache Sizes (10% catalog capacity as defined in paper)
    unique_key_sizes = {}
    for r in records:
        unique_key_sizes[r["key"]] = r["size"]
    total_catalog_bytes = sum(unique_key_sizes.values())
    total_cache_size = int(total_catalog_bytes * 0.10)  # 10% catalog
    dram_fraction = 0.20  # DRAM 20%, SSD 80%
    mean_obj_size = int(stats["mean_object_size"])

    print(f"\n[2/5] Configuring Multi-Tier Cache:")
    print(f"  Total Catalog Size: {total_catalog_bytes:,} bytes ({total_catalog_bytes/1024/1024:.2f} MB)")
    print(f"  Total Cache Size (10%): {total_cache_size:,} bytes ({total_cache_size/1024:.1f} KB)")
    print(f"  DRAM (20%): {int(total_cache_size * dram_fraction):,} bytes")
    print(f"  SSD  (80%): {int(total_cache_size * (1 - dram_fraction)):,} bytes")
    print(f"  Mean Object Size: {mean_obj_size} bytes")

    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=dram_fraction,
        ssd_throughput_gb=2.0,
        slot_size=mean_obj_size
    )

    # 3. Train Genetic Algorithm
    print(f"\n[3/5] Training Genetic Algorithm...")
    # Training subset for evolution
    train_size = min(5000, len(records))
    training_trace = records[:train_size]

    ga_engine = GeneticAlgorithmEngine(
        population_size=30,
        generations=25,
        crossover_rate=0.85,
        mutation_rate=0.20,
        elite_count=3,
        tournament_size=3,
        crossover_method="sbx",
        seed=42,
        w_chr=0.40,
        w_hpc=0.35,
        w_pen=0.15,
        w_lat=0.10
    )

    t_ga_start = time.time()
    best_chromosome = ga_engine.evolve(simulator, training_trace, verbose=True)
    ga_duration = time.time() - t_ga_start

    # Save GA training history
    history_file = os.path.join(RESULTS_DIR, "ga_training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(ga_engine.history, f, indent=2)
    print(f"  GA training history saved to: {history_file}")

    # 4. Evaluate ALL Policies on UNSEEN test set only (no data leakage)
    test_trace = records[train_size:]
    print(f"\n[4/5] Evaluating all policies on {len(test_trace)} UNSEEN test requests (excluding {train_size} training requests)...")
    evaluation_trace = test_trace

    policies_to_evaluate = {
        "GA-Predictive": GAPredictivePolicy(best_chromosome),
        "LRU": LRUPolicy(),
        "LFU": LFUPolicy(),
        "Random": RandPolicy()
    }

    all_results = {}
    for pol_name, pol_inst in policies_to_evaluate.items():
        t0 = time.time()
        res = simulator.run(pol_inst, evaluation_trace)
        elapsed = time.time() - t0
        all_results[pol_name] = res
        print(f"  [{pol_name:13s}] Total CHR: {res['chr_total']*100:5.2f}% | HPC CHR: {res['chr_hpc']*100:5.2f}% | Penalty: {res['penalty_total']:7d} | Time: {elapsed:.2f}s")

    # 5. Save Results & Generate Plots
    print(f"\n[5/5] Generating figures and consolidated summary...")
    consolidated_output = {
        "workload_statistics": stats,
        "cache_configuration": {
            "total_catalog_bytes": total_catalog_bytes,
            "total_cache_size": total_cache_size,
            "dram_size": int(total_cache_size * dram_fraction),
            "ssd_size": int(total_cache_size * (1 - dram_fraction)),
            "slot_size": mean_obj_size,
            "dram_fraction": dram_fraction
        },
        "ga_optimization": {
            "training_samples": train_size,
            "generations": 25,
            "population_size": 30,
            "duration_seconds": ga_duration,
            "best_fitness": best_chromosome.fitness,
            "best_genes": best_chromosome.get_gene_dict()
        },
        "policy_results": all_results
    }

    results_json_file = os.path.join(RESULTS_DIR, "consolidated_results.json")
    with open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(consolidated_output, f, indent=2)
    print(f"  Consolidated results saved to: {results_json_file}")

    # Generate ASCII summary table
    summary_txt_file = os.path.join(RESULTS_DIR, "summary_table.txt")
    header = f"{'Policy':14s} | {'Total CHR':9s} | {'HPC CHR':9s} | {'LPC CHR':9s} | {'Penalty HPC':11s} | {'Penalty LPC':11s} | {'Total Penalty':13s} | {'Latency (ms)':12s}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for p, m in all_results.items():
        lines.append(
            f"{p:14s} | {m['chr_total']*100:8.2f}% | {m['chr_hpc']*100:8.2f}% | {m['chr_lpc']*100:8.2f}% | {m['penalty_hpc']:11d} | {m['penalty_lpc']:11d} | {m['penalty_total']:13d} | {m['avg_retrieval_time']*1000:12.3f}"
        )
    lines.append(sep)
    summary_text = "\n".join(lines)
    with open(summary_txt_file, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"  Summary table saved to: {summary_txt_file}")
    print("\n" + summary_text + "\n")

    # Generate all plots
    generate_plots(ga_engine.history, all_results, best_chromosome)

    print("\n" + "=" * 70)
    print("EXPERIMENT SUCCESSFULLY COMPLETED!")
    print(f"All code and results are located strictly in: {SCRIPT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
