"""Compare PSO against LRU, LFU, and Random on the large memcached trace."""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from genetic_algorithm_implementation.baseline_policies import (
    LFUPolicy,
    LRUPolicy,
    RandPolicy,
)
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from pso_implementation.pso_particle import Particle
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy


DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "cluster07(00).csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "cluster07_comparison"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
EVALUATION_START_REQUEST = 1_050_001
EVALUATION_REQUESTS = 100_000
SWEEP_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "hyperparameter_sweep_10k"
)


def generate_bar_chart(
    names: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filename: str,
    color: str,
) -> None:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=color)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.2f}%" if ylabel == "Hit Ratio (%)" else f"{value:,.0f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=220)
    plt.close()


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    previous_results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    sweep_results_path = os.path.join(SWEEP_RESULTS_DIR, "final_results.json")
    sweep_history_path = os.path.join(SWEEP_RESULTS_DIR, "training_histories.json")
    if not os.path.isfile(previous_results_path):
        raise FileNotFoundError(
            f"Existing comparison results required to preserve cache settings: "
            f"{previous_results_path}"
        )
    if not os.path.isfile(sweep_results_path):
        raise FileNotFoundError(
            f"Completed PSO hyperparameter sweep not found: {sweep_results_path}"
        )

    with open(previous_results_path, "r", encoding="utf-8") as source:
        previous_results = json.load(source)
    with open(sweep_results_path, "r", encoding="utf-8") as source:
        sweep_results = json.load(source)
    if sweep_results["dataset"].replace("\\", "/") != "Dataset/cluster07(00).csv":
        raise ValueError("The saved PSO sweep belongs to a different dataset.")
    if sweep_results["request_ranges"]["training"] != [1, 10_000]:
        raise ValueError("Expected the PSO model trained on requests 1–10,000.")

    selected_name = sweep_results["selected_configuration"]
    selected_result = next(
        result
        for result in sweep_results["validation_sweep"]
        if result["name"] == selected_name
    )
    selected_particle = Particle(
        position_seed=selected_result["best_particle_position"]
    )
    selected_particle.fitness = selected_result["best_training_fitness"]
    best_policy = PSOPredictivePolicy(selected_particle)
    cache_size = previous_results["total_cache_size_bytes"]

    requested = EVALUATION_START_REQUEST + EVALUATION_REQUESTS - 1
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(
        f"Reading through request {requested:,}; evaluating the previously "
        f"unused block {EVALUATION_START_REQUEST:,}–{requested:,} "
        f"({EVALUATION_REQUESTS:,} requests)...",
        flush=True,
    )
    started = time.time()
    preprocessor = TracePreprocessor(DATASET_PATH, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process(max_records=requested)
    if len(records) < requested:
        raise ValueError(
            f"Trace contains {len(records):,} valid requests; {requested:,} "
            "are required for this fresh-block comparison."
        )

    evaluation_start = EVALUATION_START_REQUEST - 1
    evaluation_trace = records[evaluation_start:requested]
    if len(evaluation_trace) != EVALUATION_REQUESTS:
        raise ValueError(
            f"Trace has only {len(evaluation_trace):,} requests in the requested "
            f"fresh block; {EVALUATION_REQUESTS:,} are required."
        )

    simulator = MultiTierCacheSimulator(
        total_cache_size=cache_size,
        dram_fraction=0.20,
        ssd_throughput_gb=2.0,
        slot_size=max(1, int(stats["mean_object_size"])),
    )
    print(
        f"Loaded {len(records):,} records in {time.time() - started:.1f}s. "
        f"Preserving the previous shared cache capacity: {cache_size:,} bytes.",
        flush=True,
    )
    print(
        f"Using selected PSO configuration {selected_name} "
        f"(validation score {sweep_results['selected_validation_score']:.6f}); "
        "no retraining.",
        flush=True,
    )

    policies = {
        "PSO-Predictive": best_policy,
        "LRU": LRUPolicy(),
        "LFU": LFUPolicy(),
        "Random": RandPolicy(),
    }
    results: Dict[str, Dict[str, Any]] = {}
    evaluation_seconds: Dict[str, float] = {}
    for name, policy in policies.items():
        print(
            f"\nEvaluating {name} on the same {len(evaluation_trace):,} "
            "fresh requests...",
            flush=True,
        )
        random.seed(42)
        started = time.time()
        results[name] = simulator.run(policy, evaluation_trace)
        evaluation_seconds[name] = time.time() - started
        result = results[name]
        print(
            f"  Total CHR: {result['chr_total'] * 100:.2f}% | "
            f"HPC CHR: {result['chr_hpc'] * 100:.2f}% | "
            f"LPC CHR: {result['chr_lpc'] * 100:.2f}% | "
            f"Penalty: {result['penalty_total']:,} | "
            f"Elapsed: {evaluation_seconds[name]:.1f}s",
            flush=True,
        )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    names = list(results)
    values = list(results.values())
    generate_bar_chart(
        names,
        [result["chr_total"] * 100 for result in values],
        "Total Cache Hit Ratio",
        "Hit Ratio (%)",
        "1_total_chr.png",
        "steelblue",
    )
    generate_bar_chart(
        names,
        [result["chr_hpc"] * 100 for result in values],
        "High Priority Cache Hit Ratio",
        "Hit Ratio (%)",
        "2_hpc_chr.png",
        "darkorange",
    )
    generate_bar_chart(
        names,
        [result["chr_lpc"] * 100 for result in values],
        "Low Priority Cache Hit Ratio",
        "Hit Ratio (%)",
        "3_lpc_chr.png",
        "mediumseagreen",
    )
    generate_bar_chart(
        names,
        [result["penalty_total"] for result in values],
        "Total SLA Penalty Cost",
        "Penalty Cost",
        "4_total_penalty.png",
        "crimson",
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(
        os.path.join(RESULTS_DIR, "evaluation_results.json"), "w", encoding="utf-8"
    ) as output:
        json.dump(
            {
                "dataset": os.path.relpath(DATASET_PATH, PROJECT_ROOT),
                "pso_training_request_range": sweep_results["request_ranges"]["training"],
                "pso_validation_request_range": sweep_results["request_ranges"]["validation"],
                "selected_pso_configuration": selected_name,
                "selected_pso_parameters": sweep_results["selected_parameters"],
                "selected_pso_validation_score": sweep_results[
                    "selected_validation_score"
                ],
                "evaluation_requests": len(evaluation_trace),
                "evaluation_starts_at_request": EVALUATION_START_REQUEST,
                "evaluation_ends_at_request": requested,
                "evaluation_block_note": (
                    "Fresh block after the previous comparison ended at request 1,050,000."
                ),
                "cache_catalog_basis": previous_results["cache_catalog_basis"],
                "training_catalog_bytes": previous_results["training_catalog_bytes"],
                "total_cache_size_bytes": cache_size,
                "evaluation_seconds": evaluation_seconds,
                "results": results,
            },
            output,
            indent=2,
        )
    with open(sweep_history_path, "r", encoding="utf-8") as source:
        sweep_histories = json.load(source)
    with open(
        os.path.join(RESULTS_DIR, "pso_training_history.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(sweep_histories[selected_name], output, indent=2)

    print(
        f"\nResults and four comparison figures saved to: {RESULTS_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
