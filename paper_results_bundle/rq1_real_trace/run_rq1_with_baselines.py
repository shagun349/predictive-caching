"""Evaluate LRU, LFU, and Random on the same RQ1 real-trace protocol and
compare them against the nature-inspired algorithms reported in the paper.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from lfu_implementation.lfu_policy import MultiTierLFUPolicy
from lru_implementation.lru_policy import MultiTierLRUPolicy
from random_implementation.random_policy import MultiTierRandomPolicy

DATASET_PATH = ROOT / "Dataset" / "cluster07(00).csv"
RESULTS_DIR = Path(__file__).resolve().parent
FIGURE_PATH = RESULTS_DIR / "rq1_nature_vs_baselines_total_chr.png"
CSV_PATH = RESULTS_DIR / "rq1_nature_vs_baselines_summary.csv"

ALGORITHM_PATHS = {
    "PSO": ROOT / "pso_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "SFO": ROOT / "sfo_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "GA": ROOT / "genetic_algorithm_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "DSOA": ROOT / "dsoa_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "ABC": ROOT / "abc_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "HO": ROOT / "ho_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
    "WOA": ROOT / "woa_implementation" / "results" / "hyperparameter_sweep_10k" / "final_results.json",
}


def load_nature_results() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, path in ALGORITHM_PATHS.items():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload["selected_holdout_metrics"]
        rows.append(
            {
                "algorithm": name,
                "type": "nature-inspired",
                "total_chr_pct": metrics["chr_total"] * 100.0,
                "hpc_chr_pct": metrics["chr_hpc"] * 100.0,
                "lpc_chr_pct": metrics["chr_lpc"] * 100.0,
                "penalty_total": metrics["penalty_total"],
                "avg_retrieval_ms": metrics["avg_retrieval_time"] * 1000.0,
            }
        )
    return rows


def evaluate_baselines(records: List[Dict[str, Any]], cache_size: int, slot_size: int) -> List[Dict[str, Any]]:
    holdout = records[50_000:90_000]
    simulator = MultiTierCacheSimulator(
        total_cache_size=cache_size,
        dram_fraction=0.20,
        ssd_throughput_gb=2.0,
        slot_size=slot_size,
    )
    rows: List[Dict[str, Any]] = []
    for name, policy in (
        ("LRU", MultiTierLRUPolicy()),
        ("LFU", MultiTierLFUPolicy()),
        ("Random", MultiTierRandomPolicy(seed=42)),
    ):
        metrics = simulator.run(policy, holdout)
        rows.append(
            {
                "algorithm": name,
                "type": "baseline",
                "total_chr_pct": metrics["chr_total"] * 100.0,
                "hpc_chr_pct": metrics["chr_hpc"] * 100.0,
                "lpc_chr_pct": metrics["chr_lpc"] * 100.0,
                "penalty_total": metrics["penalty_total"],
                "avg_retrieval_ms": metrics["avg_retrieval_time"] * 1000.0,
            }
        )
    return rows


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    preprocessor = TracePreprocessor(str(DATASET_PATH), high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process(max_records=90_000)
    if len(records) < 90_000:
        raise ValueError(f"Dataset only has {len(records):,} records; 90,000 required.")

    reference = json.loads((ALGORITHM_PATHS["PSO"]).read_text(encoding="utf-8"))
    cache_size = reference["cache"]["capacity_bytes"]
    slot_size = max(1, int(stats["mean_object_size"]))

    all_rows = load_nature_results() + evaluate_baselines(records, cache_size, slot_size)
    all_rows.sort(key=lambda row: row["total_chr_pct"], reverse=True)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "algorithm",
            "type",
            "total_chr_pct",
            "hpc_chr_pct",
            "lpc_chr_pct",
            "penalty_total",
            "avg_retrieval_ms",
        ])
        for row in all_rows:
            writer.writerow([
                row["algorithm"],
                row["type"],
                f"{row['total_chr_pct']:.2f}",
                f"{row['hpc_chr_pct']:.2f}",
                f"{row['lpc_chr_pct']:.2f}",
                row["penalty_total"],
                f"{row['avg_retrieval_ms']:.2f}",
            ])

    labels = [row["algorithm"] for row in all_rows]
    values = [row["total_chr_pct"] for row in all_rows]
    colors = ["#4c78a8" if row["type"] == "nature-inspired" else "#e45756" for row in all_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("RQ1 real-trace comparison: nature-inspired vs LRU/LFU/Random")
    ax.set_ylabel("Total CHR (%)")
    ax.set_ylim(0, max(values) * 1.10 + 5)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.4,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)

    print(f"Saved summary CSV to {CSV_PATH}")
    print(f"Saved figure to {FIGURE_PATH}")
    for row in all_rows:
        print(
            f"{row['algorithm']:>5} | total CHR={row['total_chr_pct']:.2f}% | "
            f"HPC CHR={row['hpc_chr_pct']:.2f}% | LPC CHR={row['lpc_chr_pct']:.2f}% | "
            f"penalty={row['penalty_total']:,} | avg latency={row['avg_retrieval_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
