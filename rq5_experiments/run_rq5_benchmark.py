"""RQ5 sensitivity benchmark: score variance across seeds for each optimizer."""

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

from rq2_experiments.synthetic_workload_generator import generate_zipf_trace
from rq4_experiments.run_rq4_benchmark import evaluate_algorithm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ALGORITHMS = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]
SEEDS = [42, 43, 44]


def build_workload() -> Dict[str, Any]:
    trace = generate_zipf_trace(alpha=1.2, num_objects=600, num_requests=1500, seed=42)
    train = trace[:1050]
    test = trace[1050:]
    return {"name": "static_zipf_seed_sensitivity", "train": train, "test": test}


def run_suite() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workload = build_workload()
    rows: List[Dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        scores = []
        for seed in SEEDS:
            result = evaluate_algorithm(algorithm, workload["train"], workload["test"], seed=seed)
            scores.append({"seed": seed, "score": result["score"], "total_seconds": result["total_seconds"]})
        rows.append({
            "algorithm": algorithm,
            "scores": scores,
            "mean_score": sum(item["score"] for item in scores) / len(scores),
            "std_score": (sum((item["score"] - (sum(s["score"] for s in scores) / len(scores))) ** 2 for item in scores) / len(scores)) ** 0.5,
            "mean_runtime": sum(item["total_seconds"] for item in scores) / len(scores),
        })

    with open(os.path.join(RESULTS_DIR, "rq5_results.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq5_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "mean_score", "std_score", "mean_runtime"])
        for row in rows:
            writer.writerow([row["algorithm"], row["mean_score"], row["std_score"], row["mean_runtime"]])

    names = [row["algorithm"] for row in rows]
    mean_scores = [row["mean_score"] for row in rows]
    std_scores = [row["std_score"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, mean_scores, yerr=std_scores, capsize=5, color="#4E79A7")
    ax.set_title("RQ5 reproducibility across seeds")
    ax.set_ylabel("Mean score")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "rq5_seed_stability.png"), dpi=200)
    plt.close(fig)

    print(f"RQ5 benchmark complete for {len(rows)} algorithms across {len(SEEDS)} seeds.")


if __name__ == "__main__":
    run_suite()
