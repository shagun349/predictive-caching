#!/usr/bin/env python3
"""
Base Paper Implementation: Multi-Tier NDN Cache Simulator
=========================================================

Implements and compares 7 caching policies from the research paper:
  1. M_ARC      — Multi-tier Adaptive Replacement Cache
  2. QM_ARC     — Quality-aware Multi-tier ARC (priority-based insertion)
  3. QL_QM_ARC  — Q-Learning enhanced QM_ARC (learned p adaptation)
  4. LRU        — Least Recently Used (baseline)
  5. LFU        — Least Frequently Used (baseline)
  6. Random     — Random eviction (baseline)
  7. PriorityLRU— Priority-aware LRU (high-priority prefetching)

Uses the MultiTierCacheSimulator infrastructure with SimPy discrete-event simulation,
multi-tier storage (DRAM + SSD), and realistic hardware parameters.

Metrics collected:
  - Cache Hit Ratio (total, high-priority, low-priority) per tier
  - Data Retrieval Time (high/low priority) per tier
  - Penalty costs for cache misses
  - Read/Write operations and storage utilization
  - Q-Learning reward convergence (for QL_QM_ARC)
"""

import json
import math
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─── Path Setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(BASE_DIR, "MultiTierCacheSimulator-main", "MultiTierCacheSimulator-main")
sys.path.insert(0, SIM_DIR)

# Suppress SimPy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── Imports from MultiTierCacheSimulator ────────────────────────────────────
import simpy
import importlib
from forwarder_structures.content_store.index import Index
from forwarder_structures.content_store.tier import Tier
from forwarder_structures.forwarder import Forwarder
from forwarder_structures.pit import PIT
from simulation import Simulation

from policies.lfu_policy import LFUPolicy
from policies.lru_policy import LRUPolicy
from policies.random_policy import RandPolicy

from traces.trace_reading.arc_trace import ARCTrace
from traces.trace_reading.common_trace import CommonTrace
from traces.trace_reading.priority_trace import PriorityTrace


# ─── Configuration ───────────────────────────────────────────────────────────
# Cache size proportions (paper: DRAM=20%, SSD=80%)
SIZE_PROPORTION = [2 / 10]   # DRAM gets 20% of total cache
THROUGHPUT = [2]              # SSD read throughput multiplier (2 GB/s)

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_traces():
    """Load trace data from the synthetic dataset."""
    print("Loading trace data...")

    arc_trace = ARCTrace()
    arc_trace.gen_data()

    priority_trace = PriorityTrace()
    priority_trace.gen_data()

    common_trace = CommonTrace()
    common_trace.gen_data()

    print(f"  ARC trace: {len(arc_trace.data)} requests")
    print(f"  Priority trace: {len(priority_trace.data)} requests")
    print(f"  Common trace: {len(common_trace.data)} requests")

    return arc_trace, priority_trace, common_trace


def compute_cache_sizes(trace):
    """Compute slot_size, nb_objects, and total_size from trace data."""
    slot_size = max([int(line[3]) for line in trace.data])
    nb_objects = len(list(set([line[2] for line in trace.data])))
    # Total cache = 10% of catalog size
    total_size = [(slot_size * nb_objects * 10) / 100]
    return slot_size, nb_objects, total_size


def run_arc_policy(name, policies, fromlist, slot_size, size_proportion,
                   total_size, throughput, trace, output_folder, logs_enabled=False):
    """Run an ARC-based policy (M_ARC, QM_ARC, QL_QM_ARC)."""
    results = []
    for i in size_proportion:
        for j in throughput:
            for k in total_size:
                print(f"\n{'='*50}")
                print(f"Running: {name}")
                print(f"  DRAM size: {int(k * i)}, SSD size: {int(k - k * i)}")

                env = simpy.Environment()
                index = Index(env)

                # BOSS tier (abstract coordinator for ARC-based policies)
                boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0,
                            read_throughput=0, write_throughput=0, target_occupation=0)

                # DRAM: latency=100ns, read=40GB/s, write=20GB/s
                dram = Tier(name="DRAM", max_size=int(k * i), granularity=1, latency=1e-7,
                            read_throughput=40000000000, write_throughput=20000000000,
                            target_occupation=1)

                # SSD: latency=10μs, read=j GB/s, write=j/2 GB/s
                ssd = Tier(name="SSD", max_size=int(k - k * i), granularity=512, latency=1e-5,
                           read_throughput=j * 1000000000, write_throughput=(j / 2) * 1000000000,
                           target_occupation=1.0)

                tiers = [boss, dram, ssd]
                pit = PIT()
                forwarder = Forwarder(env, index, tiers, pit, slot_size, default_tier_index=1)

                # Assign policies via dynamic import
                for module_name, policy_class_name, tier in zip(fromlist, policies, tiers):
                    module = importlib.import_module(module_name, package=None)
                    getattr(module, policy_class_name)(env, forwarder, tier)

                log_file = os.path.join(output_folder, f"latest{name}.log")
                sim = Simulation([trace], forwarder, env, log_file=log_file,
                                 logs_enabled=logs_enabled)

                print("  Starting simulation...")
                t0 = time.time()
                result_json = sim.run()
                elapsed = time.time() - t0
                print(f"  Completed in {elapsed:.2f}s")

                # Save results
                results_file = os.path.join(output_folder, f"{name}.txt")
                with open(results_file, "w") as f:
                    f.write(result_json)

                results.append(json.loads(result_json))

    return results


def run_standard_policy(name, policy_class, slot_size, size_proportion,
                        total_size, throughput, trace, output_folder, logs_enabled=False):
    """Run a standard policy (LRU, LFU, Random, PriorityLRU)."""
    results = []
    for i in size_proportion:
        for j in throughput:
            for k in total_size:
                print(f"\n{'='*50}")
                print(f"Running: {name}")
                print(f"  DRAM size: {int(k * i)}, SSD size: {int(k - k * i)}")

                env = simpy.Environment()
                index = Index(env)

                # DRAM tier
                dram = Tier(name="DRAM", max_size=int(k * i), granularity=1, latency=1e-7,
                            read_throughput=40000000000, write_throughput=20000000000,
                            target_occupation=1)

                # SSD tier
                ssd = Tier(name="SSD", max_size=int(k - k * i), granularity=512, latency=1e-5,
                           read_throughput=j * 1000000000, write_throughput=(j / 2) * 1000000000,
                           target_occupation=1.0)

                tiers = [dram, ssd]
                pit = PIT()
                forwarder = Forwarder(env, index, tiers, pit, slot_size, default_tier_index=0)

                # Assign same policy to both tiers
                policy_class(env, forwarder, dram)
                policy_class(env, forwarder, ssd)

                log_file = os.path.join(output_folder, f"latest{name}.log")
                sim = Simulation([trace], forwarder, env, log_file=log_file,
                                 logs_enabled=logs_enabled)

                print("  Starting simulation...")
                t0 = time.time()
                result_json = sim.run()
                elapsed = time.time() - t0
                print(f"  Completed in {elapsed:.2f}s")

                # Save results
                results_file = os.path.join(output_folder, f"{name}.txt")
                with open(results_file, "w") as f:
                    f.write(result_json)

                results.append(json.loads(result_json))

    return results


def extract_ql_rewards(slot_size, size_proportion, total_size, throughput, trace):
    """
    Run QL_QM_ARC separately to extract Q-Learning reward history.
    Returns the list of rewards over time.
    """
    for i in size_proportion:
        for j in throughput:
            for k in total_size:
                env = simpy.Environment()
                index = Index(env)

                boss = Tier(name="BOSS", max_size=0, granularity=0, latency=0,
                            read_throughput=0, write_throughput=0, target_occupation=0)
                dram = Tier(name="DRAM", max_size=int(k * i), granularity=1, latency=1e-7,
                            read_throughput=40000000000, write_throughput=20000000000,
                            target_occupation=1)
                ssd = Tier(name="SSD", max_size=int(k - k * i), granularity=512, latency=1e-5,
                           read_throughput=j * 1000000000, write_throughput=(j / 2) * 1000000000,
                           target_occupation=1.0)

                tiers = [boss, dram, ssd]
                pit = PIT()
                forwarder = Forwarder(env, index, tiers, pit, slot_size, default_tier_index=1)

                # Import and assign QL_QM_ARC
                ql_qm_arc_classes = ["AbstractQLQMARCPolicy", "QLQMARCPolicy", "QLQMARCPolicy"]
                ql_qm_arc_modules = [
                    "policies.QL_QM_ARC.abstract_ql_qm_arc_policy",
                    "policies.QL_QM_ARC.tier_ql_qm_arc_policy",
                    "policies.QL_QM_ARC.tier_ql_qm_arc_policy"
                ]

                ql_agent = None
                for module_name, policy_class_name, tier in zip(ql_qm_arc_modules, ql_qm_arc_classes, tiers):
                    module = importlib.import_module(module_name, package=None)
                    policy_inst = getattr(module, policy_class_name)(env, forwarder, tier)
                    # Extract the Q-Learning agent from the abstract policy
                    if hasattr(policy_inst, 'q_learning_agent'):
                        ql_agent = policy_inst.q_learning_agent

                log_file = os.path.join(OUTPUT_DIR, "latestQL_rewards.log")
                sim = Simulation([trace], forwarder, env, log_file=log_file,
                                 logs_enabled=False)
                sim.run()

                if ql_agent:
                    return ql_agent.rewards
    return []


def parse_results(all_results):
    """
    Parse the collected results into a structured dict for plotting.

    all_results: dict of { policy_name: [ [tier_data_list], ... ] }
    """
    parsed = {}
    for policy_name, result_list in all_results.items():
        dram_data = None
        ssd_data = None
        for run in result_list:
            for tier_data in run:
                if tier_data['tier_name'] == 'DRAM':
                    dram_data = tier_data
                elif tier_data['tier_name'] == 'SSD':
                    ssd_data = tier_data

        if dram_data:
            parsed[policy_name] = {
                'chr_dram': dram_data.get('cache hit ratio', 0),
                'chr_ssd': ssd_data.get('cache hit ratio', 0) if ssd_data else 0,
                'chr_hpc_dram': dram_data.get('cache hit ratio hpc', 0),
                'chr_hpc_ssd': ssd_data.get('cache hit ratio hpc', 0) if ssd_data else 0,
                'chr_lpc_dram': dram_data.get('cache hit ratio lpc', 0),
                'chr_lpc_ssd': ssd_data.get('cache hit ratio lpc', 0) if ssd_data else 0,
                'penalty_hpc': dram_data.get('penalty_hpc', 0),
                'penalty_lpc': dram_data.get('penalty_lpc', 0),
                'avg_hpc_retrieval': dram_data.get('avg_hpc_retrieval_time', 0),
                'avg_lpc_retrieval': dram_data.get('avg_lpc_retrieval_time', 0),
                'avg_retrieval_dram': dram_data.get('avg_v_retrieval_time', 0),
                'avg_retrieval_ssd': ssd_data.get('avg_v_retrieval_time', 0) if ssd_data else 0,
                'reads_dram': dram_data.get('number_of_reads', 0),
                'reads_ssd': ssd_data.get('number_of_reads', 0) if ssd_data else 0,
                'writes_dram': dram_data.get('number_of_write', 0),
                'writes_ssd': ssd_data.get('number_of_write', 0) if ssd_data else 0,
                'used_size_dram': dram_data.get('used_size', 0),
                'used_size_ssd': ssd_data.get('used_size', 0) if ssd_data else 0,
            }

    return parsed


# ─── Plotting Functions ─────────────────────────────────────────────────────

# Publication-quality plot style
COLORS = {
    'DRAM': '#2196F3',      # Blue
    'SSD': '#FF9800',       # Orange
    'primary': '#1565C0',   # Dark Blue
    'secondary': '#E65100', # Dark Orange
    'accent1': '#4CAF50',   # Green
    'accent2': '#9C27B0',   # Purple
    'accent3': '#F44336',   # Red
    'accent4': '#00BCD4',   # Cyan
}

POLICY_COLORS = ['#1565C0', '#E65100', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548']
POLICY_ORDER = ['M_ARC', 'QM_ARC', 'QL_QM_ARC', 'LRU', 'LFU', 'Rand', 'PriorityLRU']


def setup_plot_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_stacked_bar(parsed, metric_dram, metric_ssd, ylabel, title, filename):
    """Create a stacked bar chart (DRAM + SSD) for all policies."""
    policies = [p for p in POLICY_ORDER if p in parsed]
    dram_vals = [parsed[p][metric_dram] for p in policies]
    ssd_vals = [parsed[p][metric_ssd] for p in policies]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(policies))
    width = 0.5

    bars1 = ax.bar(x, dram_vals, width, label='DRAM', color=COLORS['DRAM'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, ssd_vals, width, bottom=dram_vals, label='SSD', color=COLORS['SSD'],
                   edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + h / 2.,
                    f'{h:.3f}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + h / 2.,
                    f'{h:.3f}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.set_xlabel('Caching Policy', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


def plot_grouped_bar(parsed, metrics, labels, ylabel, title, filename):
    """Create a grouped bar chart for comparing metrics across policies."""
    policies = [p for p in POLICY_ORDER if p in parsed]
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(policies))
    width = 0.8 / n_metrics

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [parsed[p][metric] for p in policies]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=POLICY_COLORS[i % len(POLICY_COLORS)],
                      edgecolor='white', linewidth=0.5)

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Caching Policy', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


def plot_ql_rewards(rewards, filename):
    """Plot Q-Learning reward convergence."""
    if not rewards:
        print("  No Q-Learning rewards to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.3, linewidth=0.5, color='#1565C0')
    # Moving average
    window = min(100, len(rewards) // 5)
    if window > 0:
        moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax1.plot(moving_avg, linewidth=2, color='#E65100', label=f'{window}-step Moving Avg')
    ax1.set_xlabel('Request Number', fontweight='bold')
    ax1.set_ylabel('Reward', fontweight='bold')
    ax1.set_title('Q-Learning Reward Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Cumulative reward
    cum_rewards = np.cumsum(rewards)
    ax2.plot(cum_rewards, linewidth=2, color='#4CAF50')
    ax2.set_xlabel('Request Number', fontweight='bold')
    ax2.set_ylabel('Cumulative Reward', fontweight='bold')
    ax2.set_title('Q-Learning Cumulative Reward', fontsize=14, fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


def plot_summary_table(parsed, filename):
    """Create a summary comparison table."""
    policies = [p for p in POLICY_ORDER if p in parsed]
    data = []
    for p in policies:
        d = parsed[p]
        total_chr = d['chr_dram'] + d['chr_ssd']
        data.append({
            'Policy': p,
            'Total CHR': f"{total_chr:.4f}",
            'CHR DRAM': f"{d['chr_dram']:.4f}",
            'CHR SSD': f"{d['chr_ssd']:.4f}",
            'HPC CHR': f"{d['chr_hpc_dram'] + d['chr_hpc_ssd']:.4f}",
            'LPC CHR': f"{d['chr_lpc_dram'] + d['chr_lpc_ssd']:.4f}",
            'Penalty HPC': f"{d['penalty_hpc']:.0f}",
            'Penalty LPC': f"{d['penalty_lpc']:.0f}",
        })

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(data)
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center',
                     loc='center', colColours=['#E3F2FD'] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')

    ax.set_title('Policy Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


def create_all_plots(parsed, ql_rewards):
    """Generate all publication-quality comparison plots."""
    setup_plot_style()
    print("\nGenerating comparison plots...")

    # Fig 1: Total Cache Hit Ratio (stacked DRAM + SSD)
    plot_stacked_bar(parsed, 'chr_dram', 'chr_ssd',
                     'Cache Hit Ratio', 'Total Cache Hit Ratio by Policy',
                     'fig1_chr_total.png')

    # Fig 2: High-Priority CHR
    plot_stacked_bar(parsed, 'chr_hpc_dram', 'chr_hpc_ssd',
                     'High Priority CHR', 'High-Priority Content Cache Hit Ratio',
                     'fig2_chr_hpc.png')

    # Fig 3: Low-Priority CHR
    plot_stacked_bar(parsed, 'chr_lpc_dram', 'chr_lpc_ssd',
                     'Low Priority CHR', 'Low-Priority Content Cache Hit Ratio',
                     'fig3_chr_lpc.png')

    # Fig 4: Penalty comparison
    plot_grouped_bar(parsed,
                     ['penalty_hpc', 'penalty_lpc'],
                     ['High Priority Penalty', 'Low Priority Penalty'],
                     'Penalty ($)', 'Cache Miss Penalty by Policy',
                     'fig4_penalty.png')

    # Fig 5: Data Retrieval Time (DRAM + SSD)
    plot_stacked_bar(parsed, 'avg_retrieval_dram', 'avg_retrieval_ssd',
                     'Avg Data Retrieval Time', 'Average Data Retrieval Time by Policy',
                     'fig5_retrieval_time.png')

    # Fig 6: Q-Learning reward convergence
    plot_ql_rewards(ql_rewards, 'fig6_ql_rewards.png')

    # Fig 7: Read/Write operations
    plot_stacked_bar(parsed, 'reads_dram', 'reads_ssd',
                     'Number of Reads', 'Read Operations by Policy',
                     'fig7_reads.png')
    plot_stacked_bar(parsed, 'writes_dram', 'writes_ssd',
                     'Number of Writes', 'Write Operations by Policy',
                     'fig8_writes.png')

    # Summary table
    plot_summary_table(parsed, 'summary_table.png')


def print_results_summary(parsed):
    """Print a tabular summary of results to console."""
    policies = [p for p in POLICY_ORDER if p in parsed]

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Policy':<15} | {'Total CHR':>10} | {'HPC CHR':>10} | {'LPC CHR':>10} | {'Penalty HPC':>12} | {'Penalty LPC':>12}"
    print(header)
    print("-" * 90)

    for p in policies:
        d = parsed[p]
        total_chr = d['chr_dram'] + d['chr_ssd']
        hpc_chr = d['chr_hpc_dram'] + d['chr_hpc_ssd']
        lpc_chr = d['chr_lpc_dram'] + d['chr_lpc_ssd']
        print(f"{p:<15} | {total_chr:>10.4f} | {hpc_chr:>10.4f} | {lpc_chr:>10.4f} | "
              f"{d['penalty_hpc']:>12.0f} | {d['penalty_lpc']:>12.0f}")

    print("=" * 90)


# ─── Main Execution ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  MULTI-TIER NDN CACHE SIMULATOR — BASE PAPER IMPLEMENTATION")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Figure directory: {FIGURE_DIR}")

    # Load traces
    arc_trace, priority_trace, common_trace = load_traces()

    # Compute cache sizes from trace
    slot_size, nb_objects, total_size = compute_cache_sizes(common_trace)
    print(f"\n  Slot size: {slot_size} bytes")
    print(f"  Unique objects: {nb_objects}")
    print(f"  Total cache size: {total_size[0]:.0f} bytes")

    all_results = {}

    # ─── 1. M_ARC ────────────────────────────────────────────────────────
    m_arc_classes = ["AbstractMARCPolicy", "MARCPolicy", "MARCPolicy"]
    m_arc_modules = [
        "policies.MARC.abstract_m_arc_policy",
        "policies.MARC.tier_m_arc_policy",
        "policies.MARC.tier_m_arc_policy"
    ]
    all_results["M_ARC"] = run_arc_policy(
        "M_ARC", m_arc_classes, m_arc_modules,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        arc_trace, OUTPUT_DIR)

    # ─── 2. QM_ARC ───────────────────────────────────────────────────────
    qm_arc_classes = ["AbstractQMARCPolicy", "QMARCPolicy", "QMARCPolicy"]
    qm_arc_modules = [
        "policies.QM_ARC.abstract_qm_arc_policy",
        "policies.QM_ARC.tier_qm_arc_policy",
        "policies.QM_ARC.tier_qm_arc_policy"
    ]
    all_results["QM_ARC"] = run_arc_policy(
        "QM_ARC", qm_arc_classes, qm_arc_modules,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        arc_trace, OUTPUT_DIR)

    # ─── 3. QL_QM_ARC ────────────────────────────────────────────────────
    ql_qm_arc_classes = ["AbstractQLQMARCPolicy", "QLQMARCPolicy", "QLQMARCPolicy"]
    ql_qm_arc_modules = [
        "policies.QL_QM_ARC.abstract_ql_qm_arc_policy",
        "policies.QL_QM_ARC.tier_ql_qm_arc_policy",
        "policies.QL_QM_ARC.tier_ql_qm_arc_policy"
    ]
    all_results["QL_QM_ARC"] = run_arc_policy(
        "QL_QM_ARC", ql_qm_arc_classes, ql_qm_arc_modules,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        arc_trace, OUTPUT_DIR)

    # ─── 4. PriorityLRU ──────────────────────────────────────────────────
    all_results["PriorityLRU"] = run_standard_policy(
        "PriorityLRU", LRUPolicy,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        priority_trace, OUTPUT_DIR)

    # ─── 5. LRU ──────────────────────────────────────────────────────────
    all_results["LRU"] = run_standard_policy(
        "LRU", LRUPolicy,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        common_trace, OUTPUT_DIR)

    # ─── 6. LFU ──────────────────────────────────────────────────────────
    all_results["LFU"] = run_standard_policy(
        "LFU", LFUPolicy,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        common_trace, OUTPUT_DIR)

    # ─── 7. Random ────────────────────────────────────────────────────────
    all_results["Rand"] = run_standard_policy(
        "Rand", RandPolicy,
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT,
        common_trace, OUTPUT_DIR)

    # ─── Parse Results ────────────────────────────────────────────────────
    parsed = parse_results(all_results)

    # ─── Print Summary ───────────────────────────────────────────────────
    print_results_summary(parsed)

    # ─── Extract Q-Learning Rewards ──────────────────────────────────────
    print("\nExtracting Q-Learning reward data...")
    ql_rewards = extract_ql_rewards(
        slot_size, SIZE_PROPORTION, total_size, THROUGHPUT, arc_trace)
    print(f"  Collected {len(ql_rewards)} reward samples")

    # ─── Generate Plots ──────────────────────────────────────────────────
    create_all_plots(parsed, ql_rewards)

    # ─── Save consolidated results ───────────────────────────────────────
    consolidated = {
        'config': {
            'slot_size': slot_size,
            'nb_objects': nb_objects,
            'total_size': total_size[0],
            'size_proportion': SIZE_PROPORTION,
            'throughput': THROUGHPUT,
        },
        'results': parsed
    }
    with open(os.path.join(OUTPUT_DIR, "consolidated_results.json"), "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nConsolidated results saved to: {os.path.join(OUTPUT_DIR, 'consolidated_results.json')}")

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE — All 7 policies evaluated!")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()