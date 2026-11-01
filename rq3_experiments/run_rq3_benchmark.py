"""Run a consistency benchmark across static Zipf, bursty/trending, and drift workloads."""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rq2_experiments.synthetic_workload_generator import (
    generate_bursty_trace,
    generate_trending_trace,
    generate_zipf_trace,
)
from rq2_experiments.run_rq2_benchmarks import evaluate_algorithm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ALGORITHMS = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]


def build_static_zipf_workloads() -> List[Dict[str, Any]]:
    trace = generate_zipf_trace(
        alpha=1.2,
        num_objects=800,
        num_requests=2000,
        seed=42,
    )
    train_size = int(len(trace) * 0.7)
    return [{
        "name": "static_zipf_alpha_12",
        "workload_family": "static_zipf",
        "train": trace[:train_size],
        "test": trace[train_size:],
    }]


def build_bursty_trending_workloads() -> List[Dict[str, Any]]:
    burst_trace = generate_bursty_trace(
        alpha=1.2,
        num_objects=800,
        num_requests=2000,
        burst_windows=6,
        hot_fraction=0.12,
        burst_multiplier=5.0,
        seed=99,
    )
    trend_trace = generate_trending_trace(
        alpha=1.1,
        num_objects=800,
        num_requests=2000,
        trend_windows=4,
        hot_fraction=0.08,
        trend_multiplier=8.0,
        seed=150,
    )
    burst_train_size = int(len(burst_trace) * 0.7)
    trend_train_size = int(len(trend_trace) * 0.7)
    return [
        {
            "name": "bursty_5x",
            "workload_family": "bursty_trending",
            "train": burst_trace[:burst_train_size],
            "test": burst_trace[burst_train_size:],
        },
        {
            "name": "trending_8x",
            "workload_family": "bursty_trending",
            "train": trend_trace[:trend_train_size],
            "test": trend_trace[trend_train_size:],
        },
    ]


def build_drift_workloads() -> List[Dict[str, Any]]:
    train_trace = generate_zipf_trace(
        alpha=0.8,
        num_objects=800,
        num_requests=1200,
        seed=300,
    )
    test_trace = generate_zipf_trace(
        alpha=1.6,
        num_objects=800,
        num_requests=1000,
        seed=400,
    )
    return [{
        "name": "drift_08_to_16",
        "workload_family": "drift",
        "train": train_trace,
        "test": test_trace,
    }]


def run_benchmark() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workloads = []
    workloads.extend(build_static_zipf_workloads())
    workloads.extend(build_bursty_trending_workloads())
    workloads.extend(build_drift_workloads())

    all_results = []
    for workload in workloads:
        result_rows = []
        for algorithm in ALGORITHMS:
            row = evaluate_algorithm(
                algorithm,
                workload["test"],
                workload["train"],
                seed=42,
            )
            result_rows.append(row)
        all_results.append({
            "workload": workload["name"],
            "workload_family": workload["workload_family"],
            "results": result_rows,
        })

    with open(os.path.join(RESULTS_DIR, "rq3_results.json"), "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    summary_rows = []
    for workload in all_results:
        scores = {row["name"]: row["score"] for row in workload["results"]}
        best_name = max(scores, key=scores.get)
        summary_rows.append({
            "workload": workload["workload"],
            "workload_family": workload["workload_family"],
            "best_algorithm": best_name,
            "best_score": scores[best_name],
            "scores": scores,
        })

    with open(os.path.join(RESULTS_DIR, "rq3_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq3_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["workload", "workload_family", "best_algorithm", "best_score"])
        for row in summary_rows:
            writer.writerow([row["workload"], row["workload_family"], row["best_algorithm"], row["best_score"]])

    for workload in all_results:
        names = [result["name"] for result in workload["results"]]
        values = [result["score"] for result in workload["results"]]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, values, color=["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"])
        plt.title(f"RQ3 Score Comparison — {workload['workload']}")
        plt.ylabel("Fitness score")
        plt.xticks(rotation=20)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{workload['workload']}_scores.png"), dpi=200)
        plt.close()

    print(f"RQ3 benchmark complete: {len(all_results)} workloads evaluated.")


if __name__ == "__main__":
    run_benchmark()
