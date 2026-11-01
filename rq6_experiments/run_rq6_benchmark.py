"""RQ6 cross-workload transfer and generalization benchmark."""

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

from rq2_experiments.run_rq2_benchmarks import evaluate_algorithm
from rq2_experiments.synthetic_workload_generator import (
    generate_bursty_trace,
    generate_zipf_trace,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ALGORITHMS = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]


def build_workloads() -> Dict[str, Dict[str, List[Any]]]:
    static_train = generate_zipf_trace(
        alpha=1.2,
        num_objects=500,
        num_requests=1200,
        seed=42,
    )
    static_test = generate_zipf_trace(
        alpha=1.2,
        num_objects=500,
        num_requests=1200,
        seed=43,
    )

    burst_train = generate_bursty_trace(
        alpha=1.2,
        num_objects=500,
        num_requests=1200,
        burst_windows=5,
        hot_fraction=0.12,
        burst_multiplier=4.0,
        seed=99,
    )
    burst_test = generate_bursty_trace(
        alpha=1.2,
        num_objects=500,
        num_requests=1200,
        burst_windows=5,
        hot_fraction=0.12,
        burst_multiplier=4.0,
        seed=100,
    )

    drift_train = generate_zipf_trace(
        alpha=0.8,
        num_objects=500,
        num_requests=1000,
        seed=300,
    )
    drift_test = generate_zipf_trace(
        alpha=1.6,
        num_objects=500,
        num_requests=1000,
        seed=400,
    )

    return {
        "static_zipf": {
            "name": "static_zipf",
            "train": static_train[:800],
            "test": static_test[800:],
        },
        "bursty_trending": {
            "name": "bursty_trending",
            "train": burst_train[:800],
            "test": burst_test[800:],
        },
        "drift": {
            "name": "drift",
            "train": drift_train,
            "test": drift_test,
        },
    }


def transfer_pairs() -> List[Tuple[str, str]]:
    return [
        ("static_zipf", "bursty_trending"),
        ("static_zipf", "drift"),
        ("bursty_trending", "static_zipf"),
        ("bursty_trending", "drift"),
        ("drift", "static_zipf"),
        ("drift", "bursty_trending"),
    ]


def run_suite() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workloads = build_workloads()
    rows: List[Dict[str, Any]] = []

    for source_name, target_name in transfer_pairs():
        source = workloads[source_name]
        target = workloads[target_name]

        transfer_results = []
        same_domain_results = []
        for algorithm in ALGORITHMS:
            transfer_result = evaluate_algorithm(
                algorithm,
                target["test"],
                source["train"],
                seed=42,
            )
            in_domain_result = evaluate_algorithm(
                algorithm,
                target["test"],
                target["train"],
                seed=42,
            )
            transfer_results.append({
                "algorithm": algorithm,
                "transfer_score": transfer_result["score"],
                "transfer_chr_total": transfer_result["chr_total"],
                "transfer_penalty": transfer_result["penalty_total"],
            })
            same_domain_results.append({
                "algorithm": algorithm,
                "same_domain_score": in_domain_result["score"],
                "gap": in_domain_result["score"] - transfer_result["score"],
            })

        for transfer_entry in transfer_results:
            match = next(item for item in same_domain_results if item["algorithm"] == transfer_entry["algorithm"])
            entry = {
                "source_workload": source_name,
                "target_workload": target_name,
                "algorithm": transfer_entry["algorithm"],
                "transfer_score": transfer_entry["transfer_score"],
                "same_domain_score": match["same_domain_score"],
                "gap": match["gap"],
                "transfer_chr_total": transfer_entry["transfer_chr_total"],
                "transfer_penalty": transfer_entry["transfer_penalty"],
            }
            rows.append(entry)

        transfer_names = [entry["algorithm"] for entry in transfer_results]
        transfer_scores = [entry["transfer_score"] for entry in transfer_results]
        same_scores = [entry["same_domain_score"] for entry in same_domain_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(transfer_names)))
        ax.bar([i - 0.18 for i in x], transfer_scores, width=0.35, color="#4E79A7", label="Transfer")
        ax.bar([i + 0.18 for i in x], same_scores, width=0.35, color="#F28E2B", label="In-domain")
        ax.set_xticks(x)
        ax.set_xticklabels(transfer_names, rotation=20)
        ax.set_ylabel("Score")
        ax.set_title(f"RQ6 transfer: {source_name} -> {target_name}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for i, value in enumerate(transfer_scores):
            ax.text(i - 0.18, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        for i, value in enumerate(same_scores):
            ax.text(i + 0.18, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"rq6_{source_name}_to_{target_name}.png"), dpi=200)
        plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "rq6_results.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq6_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_workload",
            "target_workload",
            "algorithm",
            "transfer_score",
            "same_domain_score",
            "gap",
        ])
        for row in rows:
            writer.writerow([
                row["source_workload"],
                row["target_workload"],
                row["algorithm"],
                row["transfer_score"],
                row["same_domain_score"],
                row["gap"],
            ])

    print(f"RQ6 benchmark complete: {len(rows)} transfer evaluations across {len(transfer_pairs())} pairs.")


if __name__ == "__main__":
    run_suite()
