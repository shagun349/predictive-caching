"""Tune ABC settings on validation, then evaluate the winner on a holdout trace."""

import json
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abc_implementation.abc_engine import ABCEngine
from abc_implementation.abc_predictive_policy import ABCPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor

DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "cluster07(00).csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "hyperparameter_sweep_10k"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TRAIN_REQUESTS = 10_000
VALIDATION_REQUESTS = 40_000
HOLDOUT_REQUESTS = 40_000
COLONY_SIZE = 30
ITERATIONS = 25
SEED = 42

CONFIGURATIONS = [
    {"name": "baseline", "colony_size": 30, "limit": 8},
    {"name": "limit_04", "colony_size": 30, "limit": 4},
    {"name": "limit_12", "colony_size": 30, "limit": 12},
    {"name": "colony_20", "colony_size": 20, "limit": 8},
    {"name": "colony_40", "colony_size": 40, "limit": 8},
    {"name": "combined_20_12", "colony_size": 20, "limit": 12},
    {"name": "combined_40_12", "colony_size": 40, "limit": 12},
]


def validation_score(metrics: Dict[str, Any], request_count: int) -> float:
    penalty_norm = min(1.0, metrics["penalty_total"] / (request_count * 20.0))
    latency_norm = min(1.0, metrics["avg_retrieval_time"] / 150.0)
    return (
        0.40 * metrics["chr_total"]
        + 0.35 * metrics["chr_hpc"]
        - 0.15 * penalty_norm
        - 0.10 * latency_norm
    )


def save_json(path: str, payload: Any) -> None:
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)
    os.replace(temp, path)


def generate_bar_chart(
    names: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filename: str,
    color: str,
) -> None:
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, color=color)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.2f}%" if ylabel == "Hit Ratio (%)" else f"{value:,.0f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()


def main() -> None:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    requested = TRAIN_REQUESTS + VALIDATION_REQUESTS + HOLDOUT_REQUESTS
    print(f"Dataset: {DATASET_PATH}", flush=True)
    print(
        f"Reading {TRAIN_REQUESTS:,} training + {VALIDATION_REQUESTS:,} validation "
        f"+ {HOLDOUT_REQUESTS:,} holdout requests...",
        flush=True,
    )
    started = time.time()
    preprocessor = TracePreprocessor(DATASET_PATH, high_priority_ratio=0.30, seed=SEED)
    records, stats = preprocessor.process(max_records=requested)
    if len(records) < requested:
        raise ValueError(
            f"Dataset has {len(records):,} valid requests; {requested:,} are required."
        )

    train_end = TRAIN_REQUESTS
    validation_end = train_end + VALIDATION_REQUESTS
    training_trace = records[:train_end]
    validation_trace = records[train_end:validation_end]
    holdout_trace = records[validation_end:requested]

    unique_training_objects: Dict[str, int] = {}
    for record in training_trace:
        unique_training_objects[record["key"]] = record["size"]
    catalog_bytes = sum(unique_training_objects.values())
    total_cache_size = max(1, int(catalog_bytes * 0.10))
    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=0.20,
        ssd_throughput_gb=2.0,
        slot_size=max(1, int(stats["mean_object_size"])),
    )
    print(
        f"Loaded {len(records):,} requests in {time.time() - started:.1f}s. "
        f"Fixed cache: {total_cache_size:,} bytes "
        f"(10% of the training-block catalog).",
        flush=True,
    )

    checkpoint_path = os.path.join(RESULTS_DIR, "sweep_progress.json")
    history_path = os.path.join(RESULTS_DIR, "training_histories.json")
    sweep_results: List[Dict[str, Any]] = []
    histories: Dict[str, List[Dict[str, Any]]] = {}
    selected_sources: Dict[str, Any] = {}

    for index, configuration in enumerate(CONFIGURATIONS, start=1):
        name = configuration["name"]
        print(
            f"\n[{index}/{len(CONFIGURATIONS)}] Training {name}: "
            f"colony_size={configuration['colony_size']}, limit={configuration['limit']}",
            flush=True,
        )
        engine = ABCEngine(
            colony_size=configuration["colony_size"],
            iterations=ITERATIONS,
            limit=configuration["limit"],
            seed=SEED,
        )
        training_started = time.time()
        best_food_source = engine.optimize(simulator, training_trace, verbose=True)
        training_seconds = time.time() - training_started
        histories[name] = engine.history
        selected_sources[name] = best_food_source

        print(
            f"Validating {name} on requests {TRAIN_REQUESTS + 1:,}–{validation_end:,}...",
            flush=True,
        )
        validation_started = time.time()
        validation_metrics = simulator.run(
            ABCPredictivePolicy(best_food_source), validation_trace
        )
        validation_seconds = time.time() - validation_started
        score = validation_score(validation_metrics, len(validation_trace))
        sweep_results.append(
            {
                "name": name,
                "parameters": configuration,
                "best_training_fitness": best_food_source.fitness,
                "validation_score": score,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "validation_metrics": validation_metrics,
                "best_food_source_position": best_food_source.position,
            }
        )
        print(
            f"{name} validation score={score:.5f} | "
            f"CHR={validation_metrics['chr_total'] * 100:.2f}% | "
            f"HPC={validation_metrics['chr_hpc'] * 100:.2f}% | "
            f"Penalty={validation_metrics['penalty_total']:,} | "
            f"train={training_seconds:.1f}s validate={validation_seconds:.1f}s",
            flush=True,
        )
        save_json(
            checkpoint_path,
            {
                "completed_configurations": len(sweep_results),
                "total_configurations": len(CONFIGURATIONS),
                "progress": [item["name"] for item in sweep_results],
            },
        )

    save_json(history_path, histories)

    selected = max(sweep_results, key=lambda item: item["validation_score"])
    selected_name = selected["name"]
    selected_source = selected_sources[selected_name]
    selected_holdout_started = time.time()
    selected_holdout_metrics = simulator.run(
        ABCPredictivePolicy(selected_source), holdout_trace
    )
    selected_holdout_seconds = time.time() - selected_holdout_started

    final_payload = {
        "dataset": "Dataset\\cluster07(00).csv",
        "request_ranges": {
            "training": [1, TRAIN_REQUESTS],
            "validation": [TRAIN_REQUESTS + 1, TRAIN_REQUESTS + VALIDATION_REQUESTS],
            "holdout": [TRAIN_REQUESTS + VALIDATION_REQUESTS + 1, requested],
        },
        "cache": {
            "capacity_bytes": total_cache_size,
            "catalog_bytes": catalog_bytes,
            "capacity_basis": "10% of unique-object catalog in training requests",
            "dram_fraction": 0.2,
            "slot_size_bytes": max(1, int(stats["mean_object_size"])),
            "dram_capacity_bytes": int(total_cache_size * 0.2),
            "ssd_capacity_bytes": int(total_cache_size * 0.8),
            "dram_latency_seconds": 1e-07,
            "dram_read_bandwidth_bytes_per_second": 40000000000.0,
            "dram_write_bandwidth_bytes_per_second": 20000000000.0,
            "ssd_latency_seconds": 1e-05,
            "ssd_read_bandwidth_bytes_per_second": 2000000000.0,
            "ssd_write_bandwidth_bytes_per_second": 1000000000.0,
        },
        "population": COLONY_SIZE,
        "iterations": ITERATIONS,
        "seed": SEED,
        "fitness_weights": {
            "total_chr": 0.4,
            "hpc_chr": 0.35,
            "normalized_penalty": 0.15,
            "normalized_latency": 0.1,
        },
        "selection_metric": "ABC multi-objective fitness formula computed on validation",
        "selected_configuration": selected_name,
        "selected_parameters": {
            "name": selected_name,
            **selected["parameters"],
        },
        "selected_validation_score": selected["validation_score"],
        "selected_holdout_metrics": selected_holdout_metrics,
        "selected_holdout_seconds": selected_holdout_seconds,
        "validation_sweep": sweep_results,
    }
    final_path = os.path.join(RESULTS_DIR, "final_results.json")
    save_json(final_path, final_payload)

    validation_scores = [item["validation_score"] for item in sweep_results]
    config_names = [item["name"] for item in sweep_results]
    generate_bar_chart(
        config_names,
        validation_scores,
        "ABC validation scores across hyperparameter sweep",
        "Validation score",
        "abc_validation_scores.png",
        "#f5b700",
    )

    print("\nSelected configuration:", selected_name)
    print("Validation score:", selected["validation_score"])
    print("Holdout metrics:", selected_holdout_metrics)
    print(f"Files written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
