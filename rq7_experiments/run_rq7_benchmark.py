"""RQ7 baseline competitiveness benchmark against LRU, LFU, and Random."""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from lru_implementation.lru_policy import MultiTierLRUPolicy
from lfu_implementation.lfu_policy import MultiTierLFUPolicy
from random_implementation.random_policy import MultiTierRandomPolicy
from rq2_experiments.run_rq2_benchmarks import evaluate_algorithm
from rq2_experiments.synthetic_workload_generator import (
    generate_bursty_trace,
    generate_zipf_trace,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ALGORITHMS = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]
BASELINES = ["LRU", "LFU", "Random"]


def validation_score(metrics: Dict[str, Any], request_count: int) -> float:
    penalty_norm = min(1.0, metrics["penalty_total"] / (request_count * 20.0))
    latency_norm = min(1.0, metrics["avg_retrieval_time"] / 150.0)
    return (
        0.40 * metrics["chr_total"]
        + 0.35 * metrics["chr_hpc"]
        - 0.15 * penalty_norm
        - 0.10 * latency_norm
    )


def get_cache_size(trace: List[Dict[str, Any]]) -> Tuple[int, int]:
    unique_sizes = {}
    for record in trace:
        unique_sizes[record["key"]] = int(record["size"])
    catalog_bytes = sum(unique_sizes.values())
    total_cache = max(1, int(catalog_bytes * 0.10))
    slot_size = max(1, int(catalog_bytes / max(1, len(unique_sizes))))
    return total_cache, slot_size


def build_simulator(train_trace: List[Dict[str, Any]]) -> MultiTierCacheSimulator:
    total_cache, slot_size = get_cache_size(train_trace)
    return MultiTierCacheSimulator(
        total_cache_size=total_cache,
        dram_fraction=0.20,
        ssd_throughput_gb=2.0,
        slot_size=slot_size,
    )


def evaluate_baseline(name: str, train_trace: List[Dict[str, Any]], test_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    simulator = build_simulator(train_trace)
    if name == "LRU":
        policy = MultiTierLRUPolicy()
    elif name == "LFU":
        policy = MultiTierLFUPolicy()
    elif name == "Random":
        policy = MultiTierRandomPolicy(seed=42)
    else:
        raise ValueError(f"Unsupported baseline: {name}")
    metrics = simulator.run(policy, test_trace)
    return {
        "name": name,
        "score": validation_score(metrics, len(test_trace)),
        "chr_total": metrics["chr_total"],
        "chr_hpc": metrics["chr_hpc"],
        "penalty_total": metrics["penalty_total"],
        "avg_retrieval_time": metrics["avg_retrieval_time"],
    }


def build_workloads() -> List[Dict[str, Any]]:
    zipf_trace = generate_zipf_trace(alpha=1.2, num_objects=800, num_requests=2000, seed=42)
    burst_trace = generate_bursty_trace(
        alpha=1.2,
        num_objects=800,
        num_requests=2000,
        burst_windows=6,
        hot_fraction=0.12,
        burst_multiplier=5.0,
        seed=99,
    )
    drift_train = generate_zipf_trace(alpha=0.8, num_objects=800, num_requests=1600, seed=300)
    drift_test = generate_zipf_trace(alpha=1.6, num_objects=800, num_requests=1600, seed=400)

    return [
        {
            "name": "static_zipf_alpha_12",
            "workload_family": "static_zipf",
            "train": zipf_trace[:1400],
            "test": zipf_trace[1400:],
        },
        {
            "name": "bursty_5x",
            "workload_family": "bursty_trending",
            "train": burst_trace[:1400],
            "test": burst_trace[1400:],
        },
        {
            "name": "drift_08_to_16",
            "workload_family": "drift",
            "train": drift_train,
            "test": drift_test,
        },
    ]


def run_suite() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workloads = build_workloads()
    all_rows: List[Dict[str, Any]] = []

    for workload in workloads:
        workload_rows = []
        for algorithm in ALGORITHMS:
            result = evaluate_algorithm(
                algorithm,
                workload["test"],
                workload["train"],
                seed=42,
            )
            workload_rows.append({
                "workload": workload["name"],
                "workload_family": workload["workload_family"],
                "algorithm": algorithm,
                "score": result["score"],
                "chr_total": result["chr_total"],
                "penalty_total": result["penalty_total"],
                "avg_retrieval_time": result["avg_retrieval_time"],
                "type": "metaheuristic",
            })
        for baseline in BASELINES:
            result = evaluate_baseline(baseline, workload["train"], workload["test"])
            workload_rows.append({
                "workload": workload["name"],
                "workload_family": workload["workload_family"],
                "algorithm": baseline,
                "score": result["score"],
                "chr_total": result["chr_total"],
                "penalty_total": result["penalty_total"],
                "avg_retrieval_time": result["avg_retrieval_time"],
                "type": "baseline",
            })
        all_rows.extend(workload_rows)

    with open(os.path.join(RESULTS_DIR, "rq7_results.json"), "w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq7_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["workload", "workload_family", "algorithm", "score", "chr_total", "penalty_total", "avg_retrieval_time", "type"])
        for row in all_rows:
            writer.writerow([
                row["workload"],
                row["workload_family"],
                row["algorithm"],
                row["score"],
                row["chr_total"],
                row["penalty_total"],
                row["avg_retrieval_time"],
                row["type"],
            ])

    for workload in workloads:
        subset = [row for row in all_rows if row["workload"] == workload["name"]]
        names = [row["algorithm"] for row in subset]
        values = [row["score"] for row in subset]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4E79A7" if row["type"] == "metaheuristic" else "#E15759" for row in subset]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(f"RQ7 classical baseline competitiveness — {workload['name']}")
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"rq7_{workload['name']}_comparison.png"), dpi=200)
        plt.close(fig)

    print(f"RQ7 benchmark complete: {len(all_rows)} algorithm/baseline evaluations across {len(workloads)} workloads.")


if __name__ == "__main__":
    run_suite()
