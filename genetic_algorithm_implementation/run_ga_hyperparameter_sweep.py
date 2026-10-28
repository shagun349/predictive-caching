"""Tune GA settings on validation, then evaluate the winner on a holdout trace."""

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

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.ga_chromosome import Chromosome, GENE_NAMES
from genetic_algorithm_implementation.ga_engine import GeneticAlgorithmEngine
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor


DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "cluster07(00).csv")
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "hyperparameter_sweep_10k"
)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TRAIN_REQUESTS = 10_000
VALIDATION_REQUESTS = 40_000
HOLDOUT_REQUESTS = 40_000
POPULATION = 30
GENERATIONS = 25
SEED = 42

CONFIGURATIONS = [
    {
        "name": "baseline",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "crossover_70",
        "crossover_rate": 0.70,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "crossover_95",
        "crossover_rate": 0.95,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "mutation_10",
        "crossover_rate": 0.85,
        "mutation_rate": 0.10,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "mutation_30",
        "crossover_rate": 0.85,
        "mutation_rate": 0.30,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "elite_1",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 1,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "elite_5",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 5,
        "tournament_size": 3,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "tournament_2",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 2,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "tournament_5",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 5,
        "crossover_method": "sbx",
        "mutation_method": "gaussian",
    },
    {
        "name": "blend_gaussian",
        "crossover_rate": 0.85,
        "mutation_rate": 0.20,
        "elite_count": 3,
        "tournament_size": 3,
        "crossover_method": "blend",
        "mutation_method": "gaussian",
    },
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


def load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def generate_comparison_bar_chart(
    names: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filename: str,
    colors: List[str],
) -> None:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors, edgecolor="black", linewidth=0.7)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=0, ha="center")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.2f}%" if ylabel == "Hit Ratio (%)" else f"{value:,.0f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            max(value * 0.05, 1.0),
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
    preprocessor = TracePreprocessor(
        DATASET_PATH, high_priority_ratio=0.30, seed=SEED
    )
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
            raise ValueError("Existing GA sweep checkpoint does not match this experiment.")
        sweep_results = previous.get("results", [])
        if len(sweep_results) != previous.get("completed_configurations"):
            raise ValueError("GA sweep checkpoint and completed count are inconsistent.")
        if len(sweep_results) > len(CONFIGURATIONS):
            raise ValueError("GA sweep checkpoint contains unexpected configurations.")
        if [result.get("name") for result in sweep_results] != expected_names[
            : len(sweep_results)
        ]:
            raise ValueError("GA sweep checkpoint configurations are out of order.")
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as history_file:
                histories = json.load(history_file)
        completed_names = {result["name"] for result in sweep_results}
        if not completed_names.issubset(histories):
            raise ValueError("GA sweep checkpoint and training histories are inconsistent.")
        histories = {name: histories[name] for name in completed_names}
        print(f"Resuming after {len(sweep_results)} completed configurations.", flush=True)

    for index, configuration in enumerate(
        CONFIGURATIONS[len(sweep_results):], start=len(sweep_results) + 1
    ):
        name = configuration["name"]
        print(
            f"\n[{index}/{len(CONFIGURATIONS)}] Training {name}: "
            f"{POPULATION} individuals, {GENERATIONS} generations, "
            f"crossover={configuration['crossover_rate']}, "
            f"mutation={configuration['mutation_rate']}, "
            f"elite={configuration['elite_count']}, "
            f"tournament={configuration['tournament_size']}, "
            f"crossover_method={configuration['crossover_method']}",
            flush=True,
        )
        engine = GeneticAlgorithmEngine(
            population_size=POPULATION,
            generations=GENERATIONS,
            crossover_rate=configuration["crossover_rate"],
            mutation_rate=configuration["mutation_rate"],
            elite_count=configuration["elite_count"],
            tournament_size=configuration["tournament_size"],
            crossover_method=configuration["crossover_method"],
            mutation_method=configuration["mutation_method"],
            seed=SEED,
            w_chr=0.40,
            w_hpc=0.35,
            w_pen=0.15,
            w_lat=0.10,
            max_penalty_ref=TRAIN_REQUESTS * 20.0,
            max_latency_ref=150.0,
        )
        training_started = time.time()
        best_chromosome = engine.evolve(simulator, training_trace, verbose=True)
        training_seconds = time.time() - training_started
        histories[name] = engine.history

        validation_started = time.time()
        validation_metrics = simulator.run(
            GAPredictivePolicy(best_chromosome), validation_trace
        )
        validation_seconds = time.time() - validation_started
        score = validation_score(validation_metrics, len(validation_trace))
        sweep_results.append(
            {
                "name": name,
                "parameters": configuration,
                "best_training_fitness": best_chromosome.fitness,
                "validation_score": score,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "validation_metrics": validation_metrics,
                "best_chromosome_genes": best_chromosome.get_gene_dict(),
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
                "configuration_names": [
                    item["name"] for item in CONFIGURATIONS
                ],
                "training_requests": len(training_trace),
                "validation_requests": len(validation_trace),
                "holdout_requests": len(holdout_trace),
                "training_catalog_bytes": catalog_bytes,
                "total_cache_size_bytes": total_cache_size,
                "results": sweep_results,
            },
        )

    winner = max(sweep_results, key=lambda result: result["validation_score"])
    winner_chromosome = chromosome_from_genes(
        winner["best_chromosome_genes"], winner["best_training_fitness"]
    )
    print(
        f"\nSelected {winner['name']} by validation score "
        f"{winner['validation_score']:.5f}. Evaluating once on requests "
        f"{validation_end + 1:,}–{requested:,}...",
        flush=True,
    )
    holdout_started = time.time()
    random.seed(SEED)
    holdout_metrics = simulator.run(
        GAPredictivePolicy(winner_chromosome), holdout_trace
    )
    holdout_seconds = time.time() - holdout_started
    print(
        f"Holdout: CHR={holdout_metrics['chr_total'] * 100:.2f}% | "
        f"HPC={holdout_metrics['chr_hpc'] * 100:.2f}% | "
        f"LPC={holdout_metrics['chr_lpc'] * 100:.2f}% | "
        f"Penalty={holdout_metrics['penalty_total']:,} | "
        f"Elapsed={holdout_seconds:.1f}s",
        flush=True,
    )

    comparison_path = {
        "PSO": os.path.join(
            PROJECT_ROOT,
            "pso_implementation",
            "results",
            "hyperparameter_sweep_10k",
            "final_results.json",
        ),
        "SFO": os.path.join(
            PROJECT_ROOT,
            "sfo_implementation",
            "results",
            "hyperparameter_sweep_10k",
            "final_results.json",
        ),
    }
    comparison_results = {"GA": holdout_metrics}
    for algorithm, path in comparison_path.items():
        with open(path, encoding="utf-8") as source:
            comparison_results[algorithm] = json.load(source)["selected_holdout_metrics"]

    generate_comparison_bar_chart(
        ["GA", "PSO", "SFO"],
        [comparison_results["GA"]["chr_total"] * 100, comparison_results["PSO"]["chr_total"] * 100, comparison_results["SFO"]["chr_total"] * 100],
        "Selected-policy Holdout: Total CHR",
        "Hit Ratio (%)",
        "1_validation_total_chr.png",
        ["#4c78a8", "#f58518", "#54a24b"],
    )
    generate_comparison_bar_chart(
        ["GA", "PSO", "SFO"],
        [comparison_results["GA"]["chr_hpc"] * 100, comparison_results["PSO"]["chr_hpc"] * 100, comparison_results["SFO"]["chr_hpc"] * 100],
        "Selected-policy Holdout: HPC CHR",
        "Hit Ratio (%)",
        "2_validation_hpc_chr.png",
        ["#4c78a8", "#f58518", "#54a24b"],
    )
    generate_comparison_bar_chart(
        ["GA", "PSO", "SFO"],
        [comparison_results["GA"]["chr_lpc"] * 100, comparison_results["PSO"]["chr_lpc"] * 100, comparison_results["SFO"]["chr_lpc"] * 100],
        "Selected-policy Holdout: LPC CHR",
        "Hit Ratio (%)",
        "3_validation_lpc_chr.png",
        ["#4c78a8", "#f58518", "#54a24b"],
    )
    generate_comparison_bar_chart(
        ["GA", "PSO", "SFO"],
        [comparison_results["GA"]["penalty_total"], comparison_results["PSO"]["penalty_total"], comparison_results["SFO"]["penalty_total"]],
        "Selected-policy Holdout: SLA Penalty",
        "Penalty Cost",
        "4_validation_penalty.png",
        ["#4c78a8", "#f58518", "#54a24b"],
    )
    generate_comparison_bar_chart(
        ["GA", "PSO", "SFO"],
        [comparison_results["GA"]["avg_retrieval_time"] * 1000.0, comparison_results["PSO"]["avg_retrieval_time"] * 1000.0, comparison_results["SFO"]["avg_retrieval_time"] * 1000.0],
        "Selected-policy Holdout: Average Retrieval Time",
        "Time (ms)",
        "5_holdout_total_chr.png",
        ["#4c78a8", "#f58518", "#54a24b"],
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
            "slot_size_bytes": simulator.slot_size,
            "dram_capacity_bytes": simulator.dram_max_size,
            "ssd_capacity_bytes": simulator.ssd_max_size,
            "dram_latency_seconds": simulator.dram_latency,
            "dram_read_bandwidth_bytes_per_second": simulator.dram_read_bw,
            "dram_write_bandwidth_bytes_per_second": simulator.dram_write_bw,
            "ssd_latency_seconds": simulator.ssd_latency,
            "ssd_read_bandwidth_bytes_per_second": simulator.ssd_read_bw,
            "ssd_write_bandwidth_bytes_per_second": simulator.ssd_write_bw,
        },
        "population": POPULATION,
        "generations": GENERATIONS,
        "seed": SEED,
        "fitness_weights": {
            "total_chr": 0.40,
            "hpc_chr": 0.35,
            "normalized_penalty": 0.15,
            "normalized_latency": 0.10,
        },
        "selection_metric": "GA multi-objective fitness formula computed on validation",
        "selected_configuration": winner["name"],
        "selected_parameters": winner["parameters"],
        "selected_validation_score": winner["validation_score"],
        "selected_holdout_metrics": holdout_metrics,
        "selected_holdout_seconds": holdout_seconds,
        "selected_policy_comparison": {
            "GA": holdout_metrics,
            "PSO": comparison_results["PSO"],
            "SFO": comparison_results["SFO"],
        },
        "validation_sweep": sweep_results,
    }
    save_json(os.path.join(RESULTS_DIR, "final_results.json"), final_results)
    print(f"\nSweep, final evaluation, and figures saved to {RESULTS_DIR}", flush=True)


def chromosome_from_genes(genes: Dict[str, float], fitness: float) -> Chromosome:
    chromosome = Chromosome([genes[name] for name in GENE_NAMES])
    chromosome.fitness = fitness
    return chromosome


if __name__ == "__main__":
    main()
