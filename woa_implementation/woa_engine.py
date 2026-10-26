"""
Whale Optimization Algorithm (WOA) Engine
==========================================
Implements the Whale Optimization Algorithm metaheuristic for predictive multi-tier caching:
  1. Encircling Prey: Whales converge toward the current best candidate solution (exploitation).
  2. Bubble-Net Attacking (Spiral Maneuver): Logarithmic spiral path simulating bubble-net feeding.
  3. Search for Prey: Global exploration guided by a randomly chosen whale when |A| >= 1.

Ensures exact multi-objective fitness evaluation parity with GA, PSO, SFO, DSOA, HO, and ABC:
  Fitness = 0.40 * CHR_total + 0.35 * CHR_HPC - 0.15 * norm_penalty - 0.10 * norm_latency
"""

import math
import random
import time
from typing import Dict, List, Optional
import numpy as np

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from woa_implementation.woa_whale import Whale
from woa_implementation.woa_predictive_policy import WOAPredictivePolicy


class WOAEngine:
    """
    Whale Optimization Algorithm (WOA) Metaheuristic Optimizer.
    """

    def __init__(
        self,
        population_size: int = 30,
        iterations: int = 25,
        b: float = 1.0,
        seed: Optional[int] = 42
    ):
        self.pop_size = population_size
        self.T_max = iterations
        self.b = b  # Spiral shape parameter
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.whales: List[Whale] = []
        self.best_whale: Optional[Whale] = None
        self.history: List[Dict] = []

    def _evaluate_whale(
        self,
        whale: Whale,
        simulator: MultiTierCacheSimulator,
        trace: List[Dict]
    ) -> float:
        """
        Evaluate candidate whale under multi-objective caching SLA:
          CHR_total (40%) + CHR_HPC (35%) - Penalty_norm (15%) - Latency_norm (10%)
        """
        policy = WOAPredictivePolicy(whale)
        metrics = simulator.run(policy, trace)

        whale.chr_total = float(metrics["chr_total"])
        whale.chr_hpc = float(metrics["chr_hpc"])
        whale.penalty_total = int(metrics["penalty_total"])

        MAX_PENALTY = len(trace) * 20.0
        MAX_LATENCY = 150.0

        norm_penalty = min(1.0, whale.penalty_total / MAX_PENALTY)
        norm_latency = min(1.0, float(metrics.get("avg_retrieval_time", 0.0)) / MAX_LATENCY)

        fitness = (
            (0.40 * whale.chr_total)
            + (0.35 * whale.chr_hpc)
            - (0.15 * norm_penalty)
            - (0.10 * norm_latency)
        )
        whale.fitness = float(fitness)
        whale.update_pbest()
        return whale.fitness

    def optimize(
        self,
        simulator: MultiTierCacheSimulator,
        training_trace: List[Dict],
        verbose: bool = True
    ) -> Whale:
        """
        Execute full Whale Optimization Algorithm on the training trace partition.
        """
        if verbose:
            print(f"Evaluating initial WOA whale population ({self.pop_size} whales)...")

        # 1. Initialize random population
        self.whales = [Whale() for _ in range(self.pop_size)]
        for w in self.whales:
            self._evaluate_whale(w, simulator, training_trace)

        # Track global best whale (the prey / leader)
        self.whales.sort(key=lambda w: w.fitness, reverse=True)
        self.best_whale = self.whales[0].clone()

        if verbose:
            print(f"Initial Best Whale Fitness: {self.best_whale.fitness:.4f} | "
                  f"Total CHR: {self.best_whale.chr_total*100:.2f}% | "
                  f"HPC CHR: {self.best_whale.chr_hpc*100:.2f}%")

        self.history = []

        # 2. Main WOA Iteration Loop
        for t in range(1, self.T_max + 1):
            t0 = time.time()

            # a linearly decreases from 2 to 0
            a = 2.0 - 2.0 * (t / self.T_max)

            spiral_count = 0
            encircling_count = 0
            search_count = 0

            for i in range(self.pop_size):
                current_whale = self.whales[i]
                new_pos = {}

                # Random vectors for WOA coefficients
                r1 = random.random()
                r2 = random.random()
                A = 2.0 * a * r1 - a
                C = 2.0 * r2

                # Probability to select between spiral bubble-net and encircling/search
                p = random.random()
                l = random.uniform(-1.0, 1.0)

                if p < 0.5:
                    if abs(A) < 1.0:
                        # -------------------------------------------------------------
                        # Mode 1: Encircling Prey (Exploitation towards current best)
                        # -------------------------------------------------------------
                        encircling_count += 1
                        current_whale.last_maneuver = "encircling"
                        for key in Whale.GENE_KEYS:
                            X_star = self.best_whale.position[key]
                            X_curr = current_whale.position[key]
                            D = abs(C * X_star - X_curr)
                            new_pos[key] = X_star - A * D
                    else:
                        # -------------------------------------------------------------
                        # Mode 2: Search for Prey (Exploration towards random whale)
                        # -------------------------------------------------------------
                        search_count += 1
                        current_whale.last_maneuver = "search"
                        # Pick random whale partner
                        candidates = [w for idx, w in enumerate(self.whales) if idx != i]
                        rand_whale = random.choice(candidates) if candidates else current_whale

                        for key in Whale.GENE_KEYS:
                            X_rand = rand_whale.position[key]
                            X_curr = current_whale.position[key]
                            D = abs(C * X_rand - X_curr)
                            new_pos[key] = X_rand - A * D
                else:
                    # -------------------------------------------------------------
                    # Mode 3: Spiral Bubble-Net Attack (Exploitation along helix path)
                    # -------------------------------------------------------------
                    spiral_count += 1
                    current_whale.last_maneuver = "spiral"
                    for key in Whale.GENE_KEYS:
                        X_star = self.best_whale.position[key]
                        X_curr = current_whale.position[key]
                        D_prime = abs(X_star - X_curr)
                        # D' * e^(b*l) * cos(2*pi*l) + X*
                        new_pos[key] = D_prime * math.exp(self.b * l) * math.cos(2.0 * math.pi * l) + X_star

                # Form candidate whale & clamp bounds
                candidate_whale = Whale(position_seed=new_pos)
                cand_fit = self._evaluate_whale(candidate_whale, simulator, training_trace)

                # Greedy acceptance (if improved, replace; else retain personal best if candidate is better)
                if cand_fit > current_whale.fitness:
                    self.whales[i] = candidate_whale
                    if cand_fit > self.best_whale.fitness:
                        self.best_whale = candidate_whale.clone()

            # Record iteration history
            current_fits = [w.fitness for w in self.whales]
            mean_fit = float(np.mean(current_fits))
            elapsed = time.time() - t0

            snapshot = {
                "iteration": t,
                "a_parameter": float(a),
                "best_fitness": float(self.best_whale.fitness),
                "mean_fitness": mean_fit,
                "best_chr_total": float(self.best_whale.chr_total),
                "best_chr_hpc": float(self.best_whale.chr_hpc),
                "best_penalty": int(self.best_whale.penalty_total),
                "spiral_maneuvers": spiral_count,
                "encircling_maneuvers": encircling_count,
                "search_maneuvers": search_count,
                "elapsed_sec": elapsed
            }
            self.history.append(snapshot)

            if verbose:
                flag = "↑" if snapshot["best_fitness"] > (self.history[-2]["best_fitness"] if t > 1 else 0) else "="
                print(f"Iter [{t:2d}/{self.T_max}] {flag} | "
                      f"Best Fit: {self.best_whale.fitness:.4f} | "
                      f"Mean Fit: {mean_fit:.4f} | "
                      f"CHR: {self.best_whale.chr_total*100:.2f}% | "
                      f"HPC CHR: {self.best_whale.chr_hpc*100:.2f}% | "
                      f"Maneuvers (Spiral:{spiral_count:2d}, Encircl:{encircling_count:2d}, Search:{search_count:2d}) | "
                      f"Time: {elapsed:.2f}s")

        return self.best_whale
