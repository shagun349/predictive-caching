"""RQ4 efficiency benchmark: runtime vs quality across workload families."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from rq2_experiments.synthetic_workload_generator import (
    generate_bursty_trace,
    generate_zipf_trace,
)
from woa_implementation.woa_engine import WOAEngine
from woa_implementation.woa_predictive_policy import WOAPredictivePolicy
from ho_implementation.ho_engine import HOEngine
from ho_implementation.ho_predictive_policy import HOPredictivePolicy
from pso_implementation.pso_engine import PSOEngine
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy
from sfo_implementation.sfo_engine import SFOEngine
from sfo_implementation.sfo_predictive_policy import SFOPredictivePolicy
from dsoa_implementation.dsoa_engine import DSOAEngine
from dsoa_implementation.dsoa_predictive_policy import DSOAPredictivePolicy
from abc_implementation.abc_engine import ABCEngine
from abc_implementation.abc_predictive_policy import ABCPredictivePolicy
from genetic_algorithm_implementation.ga_engine import GeneticAlgorithmEngine
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ALGORITHMS = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]


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


def build_workloads() -> List[Dict[str, Any]]:
    zipf = generate_zipf_trace(alpha=1.2, num_objects=800, num_requests=2000, seed=42)
    zipf_train = zipf[:1400]
    zipf_test = zipf[1400:]

    burst = generate_bursty_trace(
        alpha=1.2,
        num_objects=800,
        num_requests=2000,
        burst_windows=6,
        hot_fraction=0.12,
        burst_multiplier=5.0,
        seed=99,
    )
    burst_train = burst[:1400]
    burst_test = burst[1400:]

    train = generate_zipf_trace(alpha=0.8, num_objects=800, num_requests=1200, seed=300)
    test = generate_zipf_trace(alpha=1.6, num_objects=800, num_requests=1000, seed=400)

    return [
        {"name": "static_zipf", "family": "static", "train": zipf_train, "test": zipf_test},
        {"name": "bursty_5x", "family": "bursty", "train": burst_train, "test": burst_test},
        {"name": "drift_08_to_16", "family": "drift", "train": train, "test": test},
    ]


def evaluate_algorithm(name: str, train_trace: List[Dict[str, Any]], test_trace: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    simulator = build_simulator(train_trace)
    start = time.time()
    if name == "GA":
        engine = GeneticAlgorithmEngine(population_size=30, generations=25, seed=seed)
        best = engine.evolve(simulator, train_trace, verbose=False)
        policy = GAPredictivePolicy(best)
    elif name == "PSO":
        engine = PSOEngine(num_particles=30, iterations=25, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = PSOPredictivePolicy(best)
    elif name == "SFO":
        engine = SFOEngine(num_fish=30, iterations=25, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = SFOPredictivePolicy(best)
    elif name == "DSOA":
        engine = DSOAEngine(num_doves=30, iterations=25, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = DSOAPredictivePolicy(best)
    elif name == "ABC":
        engine = ABCEngine(colony_size=30, iterations=25, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = ABCPredictivePolicy(best)
    elif name == "HO":
        engine = HOEngine(population_size=30, iterations=25, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = HOPredictivePolicy(best)
    elif name == "WOA":
        engine = WOAEngine(population_size=30, iterations=25, b=1.5, seed=seed)
        best = engine.optimize(simulator, train_trace, verbose=False)
        policy = WOAPredictivePolicy(best)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    train_seconds = time.time() - start
    eval_start = time.time()
    metrics = simulator.run(policy, test_trace)
    eval_seconds = time.time() - eval_start
    return {
        "algorithm": name,
        "score": validation_score(metrics, len(test_trace)),
        "training_seconds": train_seconds,
        "evaluation_seconds": eval_seconds,
        "total_seconds": train_seconds + eval_seconds,
        "chr_total": metrics["chr_total"],
        "chr_hpc": metrics["chr_hpc"],
        "penalty_total": metrics["penalty_total"],
        "avg_retrieval_time": metrics["avg_retrieval_time"],
    }


def run_suite() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workloads = build_workloads()
    rows: List[Dict[str, Any]] = []

    for workload in workloads:
        for algorithm in ALGORITHMS:
            result = evaluate_algorithm(algorithm, workload["train"], workload["test"], seed=42)
            rows.append({
                "workload": workload["name"],
                "workload_family": workload["family"],
                **result,
            })

    with open(os.path.join(RESULTS_DIR, "rq4_results.json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq4_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(["workload", "family", "algorithm", "score", "training_seconds", "evaluation_seconds", "total_seconds"])
        for row in rows:
            writer.writerow([
                row["workload"],
                row["workload_family"],
                row["algorithm"],
                row["score"],
                row["training_seconds"],
                row["evaluation_seconds"],
                row["total_seconds"],
            ])

    for family in sorted({row["workload_family"] for row in rows}):
        subset = [row for row in rows if row["workload_family"] == family]
        workloads_in_family = sorted({row["workload"] for row in subset})
        for workload_name in workloads_in_family:
            workload_rows = [row for row in subset if row["workload"] == workload_name]
            names = [row["algorithm"] for row in workload_rows]
            scores = [row["score"] for row in workload_rows]
            runtimes = [row["total_seconds"] for row in workload_rows]
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.bar(names, scores, color="#4E79A7")
            ax1.set_ylabel("Score")
            ax1.set_title(f"RQ4 score vs runtime — {workload_name}")
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=20)

            ax2 = ax1.twinx()
            ax2.plot(names, runtimes, color="#E15759", marker="o", linewidth=2)
            ax2.set_ylabel("Total runtime (s)")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f"{workload_name}_rq4_runtime_quality.png"), dpi=200)
            plt.close(fig)

    print(f"RQ4 benchmark complete: {len(rows)} algorithm-workload evaluations.")


if __name__ == "__main__":
    run_suite()
