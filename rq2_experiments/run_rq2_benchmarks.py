"""Run RQ2 robustness benchmark across synthetic Zipf and bursty workloads."""

from __future__ import annotations

import csv
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
    generate_trending_trace,
    generate_zipf_trace,
    partition_requests,
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


def validation_score(metrics: Dict[str, Any], request_count: int) -> float:
    penalty_norm = min(1.0, metrics["penalty_total"] / (request_count * 20.0))
    latency_norm = min(1.0, metrics["avg_retrieval_time"] / 150.0)
    return (
        0.40 * metrics["chr_total"]
        + 0.35 * metrics["chr_hpc"]
        - 0.15 * penalty_norm
        - 0.10 * latency_norm
    )


def get_cache_size(trace: List[Dict[str, object]]) -> Tuple[int, int, int]:
    unique_sizes = {}
    for record in trace:
        unique_sizes[record["key"]] = int(record["size"])
    catalog_bytes = sum(unique_sizes.values())
    total_cache = max(1, int(catalog_bytes * 0.10))
    return total_cache, max(1, int(catalog_bytes / max(1, len(unique_sizes)))), len(unique_sizes)


def build_simulator(trace: List[Dict[str, object]]) -> MultiTierCacheSimulator:
    total_cache, slot_size, _ = get_cache_size(trace)
    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache,
        dram_fraction=0.20,
        ssd_throughput_gb=2.0,
        slot_size=slot_size,
    )
    return simulator


def evaluate_algorithm(name: str, trace: List[Dict[str, object]], train_trace: List[Dict[str, object]], seed: int = 42) -> Dict[str, Any]:
    simulator = build_simulator(train_trace)
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

    metrics = simulator.run(policy, trace)
    return {
        "name": name,
        "score": validation_score(metrics, len(trace)),
        "chr_total": metrics["chr_total"],
        "chr_hpc": metrics["chr_hpc"],
        "chr_lpc": metrics["chr_lpc"],
        "penalty_total": metrics["penalty_total"],
        "avg_retrieval_time": metrics["avg_retrieval_time"],
        "metrics": metrics,
    }


def run_suite() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    workloads = []

    for alpha in [0.8, 1.0, 1.2, 1.5]:
        trace = generate_zipf_trace(alpha=alpha, num_objects=2000, num_requests=20000, seed=42 + int(alpha * 100))
        split = partition_requests(trace, train_ratio=0.5, validation_ratio=0.2)
        workloads.append({
            "name": f"zipf_{alpha}",
            "type": "static_zipf",
            "trace": trace,
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        })

    for burst_multiplier in [3, 5, 10]:
        trace = generate_bursty_trace(
            alpha=1.2,
            num_objects=2000,
            num_requests=20000,
            burst_windows=10,
            hot_fraction=0.10,
            burst_multiplier=float(burst_multiplier),
            seed=42 + burst_multiplier,
        )
        split = partition_requests(trace, train_ratio=0.5, validation_ratio=0.2)
        workloads.append({
            "name": f"bursty_{burst_multiplier}x",
            "type": "bursty",
            "trace": trace,
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        })

    for trend_mult in [4, 8, 12]:
        trace = generate_trending_trace(
            alpha=1.1,
            num_objects=2000,
            num_requests=20000,
            trend_windows=8,
            hot_fraction=0.08,
            trend_multiplier=float(trend_mult),
            seed=100 + trend_mult,
        )
        split = partition_requests(trace, train_ratio=0.5, validation_ratio=0.2)
        workloads.append({
            "name": f"trending_{trend_mult}x",
            "type": "trending",
            "trace": trace,
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
        })

    all_results: List[Dict[str, Any]] = []
    algorithms = ["GA", "PSO", "SFO", "DSOA", "ABC", "HO", "WOA"]

    for workload in workloads:
        workload_results = []
        for algorithm in algorithms:
            result = evaluate_algorithm(
                algorithm,
                workload["test"],
                workload["train"],
                seed=42,
            )
            workload_results.append(result)
        all_results.append({
            "workload": workload["name"],
            "workload_type": workload["type"],
            "results": workload_results,
        })

    result_path = os.path.join(RESULTS_DIR, "rq2_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    summary_rows = []
    for workload in all_results:
        names = [result["name"] for result in workload["results"]]
        values = [result["score"] for result in workload["results"]]
        summary_rows.append({
            "workload": workload["workload"],
            "workload_type": workload["workload_type"],
            "best_algorithm": max(workload["results"], key=lambda item: item["score"])["name"],
            "best_score": max(values),
            "scores": dict(zip(names, values)),
        })

    with open(os.path.join(RESULTS_DIR, "rq2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "rq2_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["workload", "workload_type", "best_algorithm", "best_score"])
        for row in summary_rows:
            writer.writerow([row["workload"], row["workload_type"], row["best_algorithm"], row["best_score"]])

    for workload in all_results:
        names = [result["name"] for result in workload["results"]]
        values = [result["score"] for result in workload["results"]]
        plt.figure(figsize=(10, 5))
        plt.bar(names, values, color=["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"])
        plt.title(f"RQ2 Score Comparison — {workload['workload']}")
        plt.ylabel("Validation score")
        plt.xticks(rotation=20)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"{workload['workload']}_scores.png"), dpi=200)
        plt.close()

    print(f"RQ2 benchmark complete: {len(all_results)} workloads evaluated.")


if __name__ == "__main__":
    run_suite()
