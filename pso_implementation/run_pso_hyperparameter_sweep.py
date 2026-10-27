"""Tune PSO settings with validation, then evaluate the winner on a holdout trace."""

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
from pso_implementation.pso_engine import PSOEngine
from pso_implementation.pso_particle import Particle
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy


DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "cluster07(00).csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "hyperparameter_sweep_10k"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TRAIN_REQUESTS = 10_000
VALIDATION_REQUESTS = 40_000
HOLDOUT_REQUESTS = 40_000
PARTICLES = 30
ITERATIONS = 25
SEED = 42

# The default is the existing PSO configuration. Each other row changes one
# inertia or acceleration setting at a time, keeping the search budget equal.
CONFIGURATIONS = [
    {"name": "baseline", "w_start": 0.9, "w_end": 0.4, "c1": 1.5, "c2": 1.5},
    {"name": "inertia_08_04", "w_start": 0.8, "w_end": 0.4, "c1": 1.5, "c2": 1.5},
    {"name": "inertia_10_04", "w_start": 1.0, "w_end": 0.4, "c1": 1.5, "c2": 1.5},
    {"name": "inertia_09_02", "w_start": 0.9, "w_end": 0.2, "c1": 1.5, "c2": 1.5},
    {"name": "inertia_09_06", "w_start": 0.9, "w_end": 0.6, "c1": 1.5, "c2": 1.5},
    {"name": "cognitive_10", "w_start": 0.9, "w_end": 0.4, "c1": 1.0, "c2": 1.5},
    {"name": "cognitive_20", "w_start": 0.9, "w_end": 0.4, "c1": 2.0, "c2": 1.5},
    {"name": "social_10", "w_start": 0.9, "w_end": 0.4, "c1": 1.5, "c2": 1.0},
    {"name": "social_20", "w_start": 0.9, "w_end": 0.4, "c1": 1.5, "c2": 2.0},
    {"name": "balanced_18", "w_start": 0.9, "w_end": 0.4, "c1": 1.8, "c2": 1.8},
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


def save_json(path: str, payload: Dict[str, Any]) -> None:
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
    preprocessor = TracePreprocessor(DATASET_PATH, high_priority_ratio=0.30, seed=42)
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

    sweep_results: List[Dict[str, Any]] = []
    histories: Dict[str, List[Dict[str, Any]]] = {}
    checkpoint_path = os.path.join(RESULTS_DIR, "sweep_progress.json")
    history_path = os.path.join(RESULTS_DIR, "training_histories.json")

    for index, configuration in enumerate(CONFIGURATIONS, start=1):
        name = configuration["name"]
        print(
            f"\n[{index}/{len(CONFIGURATIONS)}] Training {name}: "
            f"{PARTICLES} particles, {ITERATIONS} iterations, "
            f"w_start={configuration['w_start']}, "
            f"w_end={configuration['w_end']}, c1={configuration['c1']}, "
            f"c2={configuration['c2']}",
            flush=True,
        )
        engine = PSOEngine(
            num_particles=PARTICLES,
            iterations=ITERATIONS,
            w_start=configuration["w_start"],
            w_end=configuration["w_end"],
            c1=configuration["c1"],
            c2=configuration["c2"],
            seed=SEED,
        )
        training_started = time.time()
        best_particle = engine.optimize(simulator, training_trace, verbose=True)
        training_seconds = time.time() - training_started
        histories[name] = engine.history

        print(
            f"Validating {name} on requests {TRAIN_REQUESTS + 1:,}–"
            f"{validation_end:,}...",
            flush=True,
        )
        validation_started = time.time()
        validation_metrics = simulator.run(
            PSOPredictivePolicy(best_particle), validation_trace
        )
        validation_seconds = time.time() - validation_started
        score = validation_score(validation_metrics, len(validation_trace))
        sweep_results.append(
            {
                "name": name,
                "parameters": configuration,
                "best_training_fitness": best_particle.fitness,
                "validation_score": score,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "validation_metrics": validation_metrics,
                "best_particle_position": best_particle.position,
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
                "training_requests": len(training_trace),
                "validation_requests": len(validation_trace),
                "holdout_requests": len(holdout_trace),
                "training_catalog_bytes": catalog_bytes,
                "total_cache_size_bytes": total_cache_size,
                "results": sweep_results,
            },
        )
        save_json(history_path, histories)

    winner = max(sweep_results, key=lambda result: result["validation_score"])
    winner_particle = Particle(position_seed=winner["best_particle_position"])
    winner_particle.fitness = winner["best_training_fitness"]
    winner_policy = PSOPredictivePolicy(winner_particle)

    print(
        f"\nSelected {winner['name']} by validation score "
        f"{winner['validation_score']:.5f}. Evaluating it once on requests "
        f"{validation_end + 1:,}–{requested:,}...",
        flush=True,
    )
    holdout_started = time.time()
    holdout_metrics = simulator.run(winner_policy, holdout_trace)
    holdout_seconds = time.time() - holdout_started
    print(
        f"Holdout: CHR={holdout_metrics['chr_total'] * 100:.2f}% | "
        f"HPC={holdout_metrics['chr_hpc'] * 100:.2f}% | "
        f"LPC={holdout_metrics['chr_lpc'] * 100:.2f}% | "
        f"Penalty={holdout_metrics['penalty_total']:,} | "
        f"Elapsed={holdout_seconds:.1f}s",
        flush=True,
    )

    names = [result["name"] for result in sweep_results]
    generate_bar_chart(
        names,
        [result["validation_metrics"]["chr_total"] * 100 for result in sweep_results],
        "PSO Hyperparameter Validation: Total CHR",
        "Hit Ratio (%)",
        "1_validation_total_chr.png",
        "steelblue",
    )
    generate_bar_chart(
        names,
        [result["validation_metrics"]["chr_hpc"] * 100 for result in sweep_results],
        "PSO Hyperparameter Validation: HPC CHR",
        "Hit Ratio (%)",
        "2_validation_hpc_chr.png",
        "darkorange",
    )
    generate_bar_chart(
        names,
        [result["validation_metrics"]["chr_lpc"] * 100 for result in sweep_results],
        "PSO Hyperparameter Validation: LPC CHR",
        "Hit Ratio (%)",
        "3_validation_lpc_chr.png",
        "mediumseagreen",
    )
    generate_bar_chart(
        names,
        [result["validation_metrics"]["penalty_total"] for result in sweep_results],
        "PSO Hyperparameter Validation: SLA Penalty",
        "Penalty Cost",
        "4_validation_penalty.png",
        "crimson",
    )
    generate_bar_chart(
        ["Selected PSO"],
        [holdout_metrics["chr_total"] * 100],
        "Selected PSO: Final Holdout Total CHR",
        "Hit Ratio (%)",
        "5_holdout_total_chr.png",
        "slateblue",
    )

    final_results = {
        "dataset": os.path.relpath(DATASET_PATH, PROJECT_ROOT),
        "request_ranges": {
            "training": [1, TRAIN_REQUESTS],
            "validation": [TRAIN_REQUESTS + 1, validation_end],
            "holdout": [validation_end + 1, requested],
        },
        "cache": {
            "capacity_bytes": total_cache_size,
            "catalog_bytes": catalog_bytes,
            "capacity_basis": "10% of unique-object catalog in training requests",
            "dram_fraction": simulator.dram_fraction,
        },
        "population": PARTICLES,
        "iterations": ITERATIONS,
        "seed": SEED,
        "selection_metric": "PSO multi-objective fitness formula computed on validation",
        "selected_configuration": winner["name"],
        "selected_parameters": winner["parameters"],
        "selected_validation_score": winner["validation_score"],
        "selected_holdout_metrics": holdout_metrics,
        "selected_holdout_seconds": holdout_seconds,
        "validation_sweep": sweep_results,
    }
    save_json(os.path.join(RESULTS_DIR, "final_results.json"), final_results)
    print(f"\nSweep, final evaluation, and figures saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
