"""Tune WOA settings on validation, then evaluate the winner on a holdout trace."""

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

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from woa_implementation.woa_engine import WOAEngine
from woa_implementation.woa_whale import Whale
from woa_implementation.woa_predictive_policy import WOAPredictivePolicy

DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "cluster07(00).csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "hyperparameter_sweep_10k"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TRAIN_REQUESTS = 10_000
VALIDATION_REQUESTS = 40_000
HOLDOUT_REQUESTS = 40_000
POPULATION = 30
ITERATIONS = 25
SEED = 42

CONFIGURATIONS = [
    {"name": "baseline", "b": 1.0},
    {"name": "spiral_05", "b": 0.5},
    {"name": "spiral_15", "b": 1.5},
    {"name": "spiral_20", "b": 2.0},
    {"name": "spiral_25", "b": 2.5},
    {"name": "gentle_03", "b": 0.3},
    {"name": "gentle_08", "b": 0.8},
    {"name": "aggressive_12", "b": 1.2},
    {"name": "aggressive_18", "b": 1.8},
    {"name": "aggressive_30", "b": 3.0},
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
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
    os.replace(temporary_path, path)


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

    validation_end = TRAIN_REQUESTS + VALIDATION_REQUESTS
    training_trace = records[:TRAIN_REQUESTS]
    validation_trace = records[TRAIN_REQUESTS:validation_end]
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
    if "--resume" in sys.argv:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Cannot resume; sweep checkpoint not found: {checkpoint_path}"
            )
        with open(checkpoint_path, encoding="utf-8") as checkpoint:
            previous = json.load(checkpoint)
        expected_names = [configuration["name"] for configuration in CONFIGURATIONS]
        if (
            previous.get("configuration_names") != expected_names
            or previous.get("training_requests") != TRAIN_REQUESTS
            or previous.get("validation_requests") != VALIDATION_REQUESTS
            or previous.get("holdout_requests") != HOLDOUT_REQUESTS
        ):
            raise ValueError("Existing WOA sweep checkpoint does not match this experiment.")
        sweep_results = previous.get("results", [])
        if len(sweep_results) != previous.get("completed_configurations"):
            raise ValueError("WOA sweep checkpoint and completed count are inconsistent.")
        if len(sweep_results) > len(CONFIGURATIONS):
            raise ValueError("WOA sweep checkpoint contains unexpected configurations.")
        if [result.get("name") for result in sweep_results] != expected_names[: len(sweep_results)]:
            raise ValueError("WOA sweep checkpoint configurations are out of order.")
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as history_file:
                histories = json.load(history_file)
        completed_names = {result["name"] for result in sweep_results}
        if not completed_names.issubset(histories):
            raise ValueError("WOA sweep checkpoint and training histories are inconsistent.")
        histories = {name: histories[name] for name in completed_names}
        print(f"Resuming after {len(sweep_results)} completed configurations.", flush=True)

    for index, configuration in enumerate(
        CONFIGURATIONS[len(sweep_results):], start=len(sweep_results) + 1
    ):
        name = configuration["name"]
        print(
            f"\n[{index}/{len(CONFIGURATIONS)}] Training {name}: "
            f"{POPULATION} whales, {ITERATIONS} iterations, b={configuration['b']}",
            flush=True,
        )
        engine = WOAEngine(
            population_size=POPULATION,
            iterations=ITERATIONS,
            b=configuration["b"],
            seed=SEED,
        )
        training_started = time.time()
        best_whale = engine.optimize(simulator, training_trace, verbose=True)
        training_seconds = time.time() - training_started
        histories[name] = engine.history

        validation_started = time.time()
        validation_metrics = simulator.run(WOAPredictivePolicy(best_whale), validation_trace)
        validation_seconds = time.time() - validation_started
        score = validation_score(validation_metrics, len(validation_trace))
        sweep_results.append(
            {
                "name": name,
                "parameters": configuration,
                "best_training_fitness": best_whale.fitness,
                "validation_score": score,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "validation_metrics": validation_metrics,
                "best_whale_genes": best_whale.position.copy(),
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

        save_json(history_path, histories)
        save_json(
            checkpoint_path,
            {
                "completed_configurations": len(sweep_results),
                "total_configurations": len(CONFIGURATIONS),
                "configuration_names": [item["name"] for item in CONFIGURATIONS],
                "training_requests": len(training_trace),
                "validation_requests": len(validation_trace),
                "holdout_requests": len(holdout_trace),
                "training_catalog_bytes": catalog_bytes,
                "total_cache_size_bytes": total_cache_size,
                "results": sweep_results,
            },
        )

    winner = max(sweep_results, key=lambda result: result["validation_score"])
    winner_whale = Whale(position_seed=winner["best_whale_genes"])
    winner_whale.fitness = winner["best_training_fitness"]
    winner_whale.chr_total = winner["validation_metrics"]["chr_total"]
    winner_whale.chr_hpc = winner["validation_metrics"]["chr_hpc"]
    winner_whale.penalty_total = winner["validation_metrics"]["penalty_total"]
    winner_policy = WOAPredictivePolicy(winner_whale)
    holdout_started = time.time()
    holdout_metrics = simulator.run(winner_policy, holdout_trace)
    holdout_seconds = time.time() - holdout_started

    final_results = {
        "dataset": "Dataset\\cluster07(00).csv",
        "request_ranges": {
            "training": [1, 10_000],
            "validation": [10_001, 50_000],
            "holdout": [50_001, 90_000],
        },
        "cache": {
            "capacity_bytes": total_cache_size,
            "catalog_bytes": catalog_bytes,
            "capacity_basis": "10% of unique-object catalog in training requests",
            "dram_fraction": 0.2,
            "slot_size_bytes": max(1, int(stats["mean_object_size"])),
            "dram_capacity_bytes": int(total_cache_size * 0.20),
            "ssd_capacity_bytes": int(total_cache_size * 0.80),
            "dram_latency_seconds": 1e-07,
            "dram_read_bandwidth_bytes_per_second": 40_000_000_000.0,
            "dram_write_bandwidth_bytes_per_second": 20_000_000_000.0,
            "ssd_latency_seconds": 1e-05,
            "ssd_read_bandwidth_bytes_per_second": 2_000_000_000.0,
            "ssd_write_bandwidth_bytes_per_second": 1_000_000_000.0,
        },
        "population": POPULATION,
        "generations": ITERATIONS,
        "seed": SEED,
        "fitness_weights": {
            "total_chr": 0.4,
            "hpc_chr": 0.35,
            "normalized_penalty": 0.15,
            "normalized_latency": 0.1,
        },
        "selection_metric": "WOA fitness formula computed on validation",
        "selected_configuration": winner["name"],
        "selected_parameters": winner["parameters"],
        "selected_validation_score": winner["validation_score"],
        "selected_holdout_metrics": holdout_metrics,
        "selected_holdout_seconds": holdout_seconds,
        "validation_sweep": sweep_results,
    }
    save_json(os.path.join(RESULTS_DIR, "final_results.json"), final_results)

    generate_bar_chart(
        [result["name"] for result in sweep_results],
        [result["validation_metrics"]["chr_total"] * 100.0 for result in sweep_results],
        "WOA validation total CHR by configuration",
        "Hit Ratio (%)",
        "1_validation_total_chr.png",
        "#5DA5DA",
    )
    generate_bar_chart(
        [result["name"] for result in sweep_results],
        [result["validation_metrics"]["chr_hpc"] * 100.0 for result in sweep_results],
        "WOA validation HPC CHR by configuration",
        "Hit Ratio (%)",
        "2_validation_hpc_chr.png",
        "#FAA43A",
    )
    generate_bar_chart(
        [result["name"] for result in sweep_results],
        [result["validation_metrics"]["chr_lpc"] * 100.0 for result in sweep_results],
        "WOA validation LPC CHR by configuration",
        "Hit Ratio (%)",
        "3_validation_lpc_chr.png",
        "#60BD68",
    )
    generate_bar_chart(
        [result["name"] for result in sweep_results],
        [result["validation_metrics"]["penalty_total"] for result in sweep_results],
        "WOA validation total penalty by configuration",
        "Penalty",
        "4_validation_penalty.png",
        "#B276B2",
    )

    plt.figure(figsize=(8, 5))
    bars = plt.bar([winner["name"]], [holdout_metrics["chr_total"] * 100.0], color="#4E79A7")
    plt.title("WOA selected policy holdout total CHR", fontsize=14)
    plt.ylabel("Hit Ratio (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_holdout_total_chr.png"), dpi=200)
    plt.close()

    print(f"\nCompleted WOA sweep. Selected configuration: {winner['name']}")
    print(
        f"Validation score: {winner['validation_score']:.6f} | "
        f"Holdout total CHR: {holdout_metrics['chr_total'] * 100:.2f}% | "
        f"HPC CHR: {holdout_metrics['chr_hpc'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
