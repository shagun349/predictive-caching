"""
Genetic Algorithm Optimization Engine for Predictive Caching
============================================================
Coordinates population evolution, selection, crossover, mutation, elitism,
and multi-objective fitness evaluation. Tracks convergence history and diversity.
"""

import copy
import math
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Callable

from genetic_algorithm_implementation.ga_chromosome import Chromosome, GENE_NAMES, GENE_BOUNDS
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator


class GeneticAlgorithmEngine:
    """
    Genetic Algorithm Engine for evolving predictive multi-tier caching policies.
    """

    def __init__(
        self,
        population_size: int = 30,
        generations: int = 25,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.20,
        elite_count: int = 3,
        tournament_size: int = 3,
        crossover_method: str = "sbx",
        mutation_method: str = "gaussian",
        seed: int = 42,
        # Multi-objective fitness weights
        w_chr: float = 0.40,
        w_hpc: float = 0.35,
        w_pen: float = 0.15,
        w_lat: float = 0.10
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.seed = seed

        # Fitness weights
        self.w_chr = w_chr
        self.w_hpc = w_hpc
        self.w_pen = w_pen
        self.w_lat = w_lat

        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        self.history: List[Dict[str, Any]] = []

        random.seed(seed)

    def initialize_population(self):
        """
        Initialize population with heuristic baseline seeds + diverse random individuals.
        """
        self.population = [
            Chromosome.create_lru_seed(),
            Chromosome.create_lfu_seed(),
            Chromosome.create_qm_arc_seed()
        ]
        while len(self.population) < self.population_size:
            self.population.append(Chromosome.create_random())

        self.best_chromosome = None
        self.history.clear()

    def evaluate_chromosome(
        self,
        chromosome: Chromosome,
        simulator: MultiTierCacheSimulator,
        trace_subset: List[Dict[str, Any]],
        max_penalty_ref: float = 500000.0,
        max_latency_ref: float = 0.05
    ) -> float:
        """
        Simulate chromosome policy on trace and compute multi-objective fitness.
        """
        policy = GAPredictivePolicy(chromosome)
        metrics = simulator.run(policy, trace_subset)

        chr_tot = metrics["chr_total"]
        chr_hpc = metrics["chr_hpc"]
        pen_norm = min(1.0, metrics["penalty_total"] / max(max_penalty_ref, 1.0))
        lat_norm = min(1.0, metrics["avg_retrieval_time"] / max(max_latency_ref, 1e-4))

        # Multi-objective fitness function
        fitness = (
            self.w_chr * chr_tot +
            self.w_hpc * chr_hpc -
            self.w_pen * pen_norm -
            self.w_lat * lat_norm
        )

        chromosome.fitness = fitness
        chromosome.metrics = metrics
        chromosome.evaluated = True
        return fitness

    def tournament_selection(self) -> Chromosome:
        """Tournament selection operator."""
        contenders = random.sample(self.population, self.tournament_size)
        return max(contenders, key=lambda c: c.fitness)

    def compute_population_diversity(self) -> float:
        """Compute average pairwise Euclidean distance across normalized genes."""
        if len(self.population) < 2:
            return 0.0
        n_genes = len(GENE_BOUNDS)
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist_sq = 0.0
                for g in range(n_genes):
                    min_v, max_v = GENE_BOUNDS[g]
                    g1_norm = (self.population[i].genes[g] - min_v) / (max_v - min_v)
                    g2_norm = (self.population[j].genes[g] - min_v) / (max_v - min_v)
                    dist_sq += (g1_norm - g2_norm) ** 2
                distances.append(math.sqrt(dist_sq / n_genes))
        return sum(distances) / len(distances)

    def evolve(
        self,
        simulator: MultiTierCacheSimulator,
        training_trace: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Chromosome:
        """
        Execute full Genetic Algorithm evolution cycle across generations.
        """
        self.initialize_population()
        t0 = time.time()

        # Initial evaluation
        for ind in self.population:
            if not ind.evaluated:
                self.evaluate_chromosome(ind, simulator, training_trace)

        self.population.sort(key=lambda c: c.fitness, reverse=True)
        self.best_chromosome = self.population[0].clone()

        if verbose:
            print("=" * 65)
            print(f"Starting Genetic Algorithm Optimization ({self.generations} generations)")
            print(f"Population size: {self.population_size}, Training samples: {len(training_trace)}")
            print("=" * 65)

        for gen in range(1, self.generations + 1):
            gen_start = time.time()

            # Elitism: preserve top elite individuals
            new_population = [c.clone() for c in self.population[:self.elite_count]]

            # Adaptive mutation sigma decaying slightly over generations
            decay_factor = 1.0 - (0.5 * gen / self.generations)
            current_sigma = 0.15 * decay_factor

            # Reproduction loop
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = Chromosome.crossover(p1, p2, method=self.crossover_method)
                else:
                    offspring1, offspring2 = p1.clone(), p2.clone()

                offspring1.mutate(self.mutation_rate, current_sigma, method=self.mutation_method)
                offspring2.mutate(self.mutation_rate, current_sigma, method=self.mutation_method)

                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)

            self.population = new_population

            # Evaluate newly generated individuals
            for ind in self.population:
                if not ind.evaluated:
                    self.evaluate_chromosome(ind, simulator, training_trace)

            # Sort by fitness descending
            self.population.sort(key=lambda c: c.fitness, reverse=True)

            if self.population[0].fitness > self.best_chromosome.fitness:
                self.best_chromosome = self.population[0].clone()

            # Statistics for this generation
            fitnesses = [c.fitness for c in self.population]
            best_fit = self.population[0].fitness
            mean_fit = sum(fitnesses) / len(fitnesses)
            diversity = self.compute_population_diversity()
            best_m = self.population[0].metrics
            gen_elapsed = time.time() - gen_start

            gen_stat = {
                "generation": gen,
                "best_fitness": best_fit,
                "mean_fitness": mean_fit,
                "min_fitness": min(fitnesses),
                "diversity": diversity,
                "best_chr_total": best_m["chr_total"],
                "best_chr_hpc": best_m["chr_hpc"],
                "best_chr_lpc": best_m["chr_lpc"],
                "best_penalty": best_m["penalty_total"],
                "best_avg_retrieval": best_m["avg_retrieval_time"],
                "time_sec": gen_elapsed,
                "best_genes": self.population[0].get_gene_dict()
            }
            self.history.append(gen_stat)

            if verbose:
                print(
                    f"Gen {gen:2d}/{self.generations} | "
                    f"Best Fit: {best_fit:.4f} | Mean: {mean_fit:.4f} | "
                    f"CHR: {best_m['chr_total']*100:.2f}% | "
                    f"HPC CHR: {best_m['chr_hpc']*100:.2f}% | "
                    f"Penalty: {best_m['penalty_total']:6d} | "
                    f"Time: {gen_elapsed:.2f}s"
                )

        total_elapsed = time.time() - t0
        if verbose:
            print("=" * 65)
            print(f"GA Evolution completed in {total_elapsed:.2f}s!")
            print(f"Overall Best Fitness: {self.best_chromosome.fitness:.4f}")
            print(f"Best Chromosome Genes:")
            for k, v in self.best_chromosome.get_gene_dict().items():
                print(f"  {k:20s}: {v:+.4f}")
            print("=" * 65)

        return self.best_chromosome
