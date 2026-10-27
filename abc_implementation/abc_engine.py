"""
Artificial Bee Colony (ABC) Optimization Engine
===============================================
Implements the tripartite Artificial Bee Colony algorithm:
  1. Employed Bee Phase: Local neighborhood foraging around assigned food sources.
  2. Onlooker Bee Phase: Probabilistic waggle-dance selection & refinement based on nectar yield.
  3. Scout Bee Phase: Abandonment of exhausted food sources & random exploration.

Ensures exact multi-objective fitness evaluation parity with GA, PSO, SFO, DSOA, and HO.
"""

import math
import random
import time
from typing import Dict, List, Optional
import numpy as np

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from abc_implementation.abc_bee import FoodSource
from abc_implementation.abc_predictive_policy import ABCPredictivePolicy


class ABCEngine:
    """
    Artificial Bee Colony (ABC) Metaheuristic Optimizer.
    """

    def __init__(
        self,
        colony_size: int = 30,
        iterations: int = 25,
        limit: int = 8,
        seed: Optional[int] = 42
    ):
        self.SN = colony_size       # Number of food sources (equal to number of employed bees)
        self.T_max = iterations     # Maximum cycles
        self.limit = limit          # Abandonment limit for scout bees
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.food_sources: List[FoodSource] = []
        self.best_food_source: Optional[FoodSource] = None
        self.history: List[Dict] = []

    def _evaluate_source(
        self,
        source: FoodSource,
        simulator: MultiTierCacheSimulator,
        trace: List[Dict]
    ) -> float:
        """
        Evaluate candidate food source under multi-objective caching SLA:
          CHR_total (40%) + CHR_HPC (35%) - Penalty_norm (15%) - Latency_norm (10%)
        """
        policy = ABCPredictivePolicy(source)
        metrics = simulator.run(policy, trace)

        source.chr_total = float(metrics["chr_total"])
        source.chr_hpc = float(metrics["chr_hpc"])
        source.penalty_total = int(metrics["penalty_total"])

        MAX_PENALTY = len(trace) * 20.0
        MAX_LATENCY = 150.0

        norm_penalty = min(1.0, source.penalty_total / MAX_PENALTY)
        norm_latency = min(1.0, float(metrics.get("avg_retrieval_time", 0.0)) / MAX_LATENCY)

        fitness = (
            (0.40 * source.chr_total)
            + (0.35 * source.chr_hpc)
            - (0.15 * norm_penalty)
            - (0.10 * norm_latency)
        )
        source.fitness = float(fitness)
        return source.fitness

    def _produce_neighbor(self, current_fs: FoodSource, all_sources: List[FoodSource]) -> FoodSource:
        """Generate a neighbor candidate: v_{i,j} = x_{i,j} + phi * (x_{i,j} - x_{k,j})."""
        neighbor = current_fs.clone()

        # Pick partner k != current
        candidates = [s for s in all_sources if s is not current_fs]
        partner = random.choice(candidates) if candidates else current_fs

        # Pick 1 or 2 random dimensions to mutate
        num_mutations = random.choice([1, 2])
        chosen_keys = random.sample(FoodSource.GENE_KEYS, num_mutations)

        for key in chosen_keys:
            phi = random.uniform(-1.0, 1.0)
            diff = current_fs.position[key] - partner.position[key]
            neighbor.position[key] = current_fs.position[key] + phi * diff

        neighbor.clamp_bounds()
        return neighbor

    def optimize(
        self,
        simulator: MultiTierCacheSimulator,
        training_trace: List[Dict],
        verbose: bool = True
    ) -> FoodSource:
        """
        Execute full Artificial Bee Colony optimization on the training trace partition.
        """
        if verbose:
            print(f"Evaluating initial ABC food sources ({self.SN} sources)...")

        # 1. Initialize food sources randomly
        self.food_sources = [FoodSource() for _ in range(self.SN)]
        for fs in self.food_sources:
            self._evaluate_source(fs, simulator, training_trace)

        # Track global best food source
        self.food_sources.sort(key=lambda s: s.fitness, reverse=True)
        self.best_food_source = self.food_sources[0].clone()

        if verbose:
            print(f"Initial Best Food Source Fitness: {self.best_food_source.fitness:.4f} | "
                  f"Total CHR: {self.best_food_source.chr_total*100:.2f}% | "
                  f"HPC CHR: {self.best_food_source.chr_hpc*100:.2f}%")

        self.history = []

        # 2. Main ABC Cycle Loop
        for cycle in range(1, self.T_max + 1):
            t0 = time.time()
            scout_count = 0

            # -------------------------------------------------------------
            # Stage 1: Employed Bee Phase
            # -------------------------------------------------------------
            for i in range(self.SN):
                fs = self.food_sources[i]
                candidate = self._produce_neighbor(fs, self.food_sources)
                cand_fit = self._evaluate_source(candidate, simulator, training_trace)

                if cand_fit > fs.fitness:
                    self.food_sources[i] = candidate
                    self.food_sources[i].trial = 0
                    if cand_fit > self.best_food_source.fitness:
                        self.best_food_source = candidate.clone()
                else:
                    fs.trial += 1

            # -------------------------------------------------------------
            # Stage 2: Calculate Selection Probabilities (Waggle Dance)
            # -------------------------------------------------------------
            fits = [fs.fitness for fs in self.food_sources]
            min_f = min(fits)
            nectar_amounts = [f - min_f + 1e-6 for f in fits]
            sum_nectar = sum(nectar_amounts)
            probs = [n / sum_nectar for n in nectar_amounts]

            # -------------------------------------------------------------
            # Stage 3: Onlooker Bee Phase
            # -------------------------------------------------------------
            for _ in range(self.SN):
                # Roulette wheel selection based on nectar amount
                chosen_idx = np.random.choice(self.SN, p=probs)
                chosen_fs = self.food_sources[chosen_idx]

                candidate = self._produce_neighbor(chosen_fs, self.food_sources)
                cand_fit = self._evaluate_source(candidate, simulator, training_trace)

                if cand_fit > chosen_fs.fitness:
                    self.food_sources[chosen_idx] = candidate
                    self.food_sources[chosen_idx].trial = 0
                    if cand_fit > self.best_food_source.fitness:
                        self.best_food_source = candidate.clone()
                else:
                    chosen_fs.trial += 1

            # -------------------------------------------------------------
            # Stage 4: Scout Bee Phase (Abandonment)
            # -------------------------------------------------------------
            for i in range(self.SN):
                if self.food_sources[i].trial >= self.limit:
                    scout_count += 1
                    new_fs = FoodSource()
                    new_fs.reset_random()
                    self._evaluate_source(new_fs, simulator, training_trace)
                    self.food_sources[i] = new_fs

                    if new_fs.fitness > self.best_food_source.fitness:
                        self.best_food_source = new_fs.clone()

            # Record iteration metrics
            current_fits = [fs.fitness for fs in self.food_sources]
            mean_fit = float(np.mean(current_fits))
            elapsed = time.time() - t0

            snapshot = {
                "iteration": cycle,
                "best_fitness": float(self.best_food_source.fitness),
                "mean_fitness": mean_fit,
                "best_chr_total": float(self.best_food_source.chr_total),
                "best_chr_hpc": float(self.best_food_source.chr_hpc),
                "best_penalty": int(self.best_food_source.penalty_total),
                "scout_explorations": scout_count,
                "elapsed_sec": elapsed
            }
            self.history.append(snapshot)

            if verbose:
                flag = "↑" if snapshot["best_fitness"] > (self.history[-2]["best_fitness"] if cycle > 1 else 0) else "="
                print(f"Cycle [{cycle:2d}/{self.T_max}] {flag} | "
                      f"Best Nectar Fit: {self.best_food_source.fitness:.4f} | "
                      f"Mean Fit: {mean_fit:.4f} | "
                      f"CHR: {self.best_food_source.chr_total*100:.2f}% | "
                      f"HPC CHR: {self.best_food_source.chr_hpc*100:.2f}% | "
                      f"Scouts: {scout_count:2d} | Time: {elapsed:.2f}s")

        return self.best_food_source
