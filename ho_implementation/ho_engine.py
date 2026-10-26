"""
Hippopotamus Optimization (HO) Algorithm Engine
===============================================
Implements the trinary-phase Hippopotamus Optimization metaheuristic:
  Phase 1: Position Updating in Rivers/Ponds (Herd Exploration & Dominant Bull Guidance)
  Phase 2: Defensive Response against Predators (Lévy-Flight Diversity Jumps)
  Phase 3: Evasion & Escape Tactics (Niche Exploitation & Local Convergence)

Ensures exact fitness evaluation parity with GA, PSO, SFO, and DSOA engines.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from ho_implementation.ho_hippo import Hippo
from ho_implementation.ho_predictive_policy import HOPredictivePolicy


class HOEngine:
    """
    Hippopotamus Optimization (HO) Metaheuristic Engine.
    """

    def __init__(
        self,
        population_size: int = 30,
        iterations: int = 25,
        p_defense: float = 0.5,
        beta_levy: float = 1.5,
        seed: Optional[int] = 42
    ):
        self.N = population_size
        self.T_max = iterations
        self.p_defense = p_defense
        self.beta = beta_levy
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Precompute Lévy sigma factor
        num = math.gamma(1.0 + self.beta) * math.sin(math.pi * self.beta / 2.0)
        den = math.gamma((1.0 + self.beta) / 2.0) * self.beta * (2.0 ** ((self.beta - 1.0) / 2.0))
        self.levy_sigma = (num / den) ** (1.0 / self.beta)

        self.population: List[Hippo] = []
        self.dominant_bull: Optional[Hippo] = None
        self.history: List[Dict] = []

    def _sample_levy(self) -> float:
        """Sample from stable Lévy distribution using Mantegna algorithm."""
        u = np.random.normal(0, self.levy_sigma)
        v = np.random.normal(0, 1.0)
        return float(u / (abs(v) ** (1.0 / self.beta)))

    def _evaluate_hippo(
        self,
        hippo: Hippo,
        simulator: MultiTierCacheSimulator,
        trace: List[Dict]
    ) -> float:
        """
        Evaluate objective fitness under multi-objective caching SLA:
          CHR_total (40%) + CHR_HPC (35%) + Normalized Penalty Savings (15%) + DRAM Utilization (10%)
        """
        policy = HOPredictivePolicy(hippo)
        metrics = simulator.run(policy, trace)

        hippo.chr_total = float(metrics["chr_total"])
        hippo.chr_hpc = float(metrics["chr_hpc"])
        hippo.penalty_total = int(metrics["penalty_total"])

        MAX_PENALTY = len(trace) * 20.0
        MAX_LATENCY = 150.0

        norm_penalty = min(1.0, hippo.penalty_total / MAX_PENALTY)
        norm_latency = min(1.0, float(metrics.get("avg_retrieval_time", 0.0)) / MAX_LATENCY)

        fitness = (
            (0.40 * hippo.chr_total)
            + (0.35 * hippo.chr_hpc)
            - (0.15 * norm_penalty)
            - (0.10 * norm_latency)
        )
        hippo.fitness = float(fitness)
        hippo.update_pbest()
        return hippo.fitness


    def optimize(
        self,
        simulator: MultiTierCacheSimulator,
        training_trace: List[Dict],
        verbose: bool = True
    ) -> Hippo:
        """
        Execute full Hippopotamus Optimization on the training trace partition.
        """
        if verbose:
            print(f"Evaluating initial HO population ({self.N} hippos)...")

        # Initialize herd
        self.population = [Hippo() for _ in range(self.N)]
        for hippo in self.population:
            self._evaluate_hippo(hippo, simulator, training_trace)

        # Identify dominant bull (global best)
        self.population.sort(key=lambda h: h.fitness, reverse=True)
        self.dominant_bull = self.population[0].clone()

        if verbose:
            print(f"Initial Dominant Bull Fitness: {self.dominant_bull.fitness:.4f} | "
                  f"Total CHR: {self.dominant_bull.chr_total*100:.2f}% | "
                  f"HPC CHR: {self.dominant_bull.chr_hpc*100:.2f}%")

        self.history = []

        # Optimization Loop
        for it in range(1, self.T_max + 1):
            t0 = time.time()
            T = 1.0 - (it / float(self.T_max))  # Dynamic temperature / search radius factor
            cf = 2.0 * T                        # Convergence factor for Phase 3

            phase_counts = {1: 0, 2: 0, 3: 0}

            for i in range(self.N):
                hippo = self.population[i]
                old_pos = hippo.position.copy()
                old_fit = hippo.fitness

                r_strategy = random.random()

                # -------------------------------------------------------------
                # Phase 1: Aquatic River/Pond Movement (Dominant Bull Guidance)
                # -------------------------------------------------------------
                if r_strategy < 0.40:
                    hippo.last_phase = 1
                    phase_counts[1] += 1
                    rand_partner = random.choice(self.population)

                    for k in Hippo.GENE_KEYS:
                        r1 = random.random()
                        step = r1 * (self.dominant_bull.position[k] - rand_partner.position[k]) * T
                        hippo.position[k] = self.dominant_bull.position[k] + step

                # -------------------------------------------------------------
                # Phase 2: Defensive Response Against Predators (Lévy Jumps)
                # -------------------------------------------------------------
                elif r_strategy < 0.75:
                    hippo.last_phase = 2
                    phase_counts[2] += 1

                    for k in Hippo.GENE_KEYS:
                        levy_step = self._sample_levy() * 0.05
                        diff = self.dominant_bull.position[k] - hippo.position[k]
                        hippo.position[k] = hippo.position[k] + levy_step * diff

                # -------------------------------------------------------------
                # Phase 3: Evasion & Escape Tactics (Niche Exploitation)
                # -------------------------------------------------------------
                else:
                    hippo.last_phase = 3
                    phase_counts[3] += 1

                    for k in Hippo.GENE_KEYS:
                        min_v, max_v = Hippo.GENE_BOUNDS[k]
                        r2 = random.random()
                        local_escape = cf * (r2 * (max_v - min_v) * 0.1) * (1.0 if random.random() > 0.5 else -1.0)
                        hippo.position[k] = hippo.position[k] + local_escape

                hippo.clamp_bounds()

                # Greedy selection
                new_fit = self._evaluate_hippo(hippo, simulator, training_trace)
                if new_fit < old_fit:
                    # Revert if movement did not improve fitness
                    hippo.position = old_pos
                    hippo.fitness = old_fit

                if hippo.fitness > self.dominant_bull.fitness:
                    self.dominant_bull = hippo.clone()

            # Record iteration history
            fits = [h.fitness for h in self.population]
            mean_fit = float(np.mean(fits))
            elapsed = time.time() - t0

            snapshot = {
                "iteration": it,
                "best_fitness": float(self.dominant_bull.fitness),
                "mean_fitness": mean_fit,
                "best_chr_total": float(self.dominant_bull.chr_total),
                "best_chr_hpc": float(self.dominant_bull.chr_hpc),
                "best_penalty": int(self.dominant_bull.penalty_total),
                "phase_counts": phase_counts,
                "elapsed_sec": elapsed
            }
            self.history.append(snapshot)

            if verbose:
                flag = "↑" if snapshot["best_fitness"] > (self.history[-2]["best_fitness"] if it > 1 else 0) else "="
                print(f"Iteration [{it:2d}/{self.T_max}] {flag} | "
                      f"Dominant Fit: {self.dominant_bull.fitness:.4f} | "
                      f"Avg Fit: {mean_fit:.4f} | "
                      f"CHR: {self.dominant_bull.chr_total*100:.2f}% | "
                      f"HPC CHR: {self.dominant_bull.chr_hpc*100:.2f}% | "
                      f"Phases (Aquatic:{phase_counts[1]}, Defend:{phase_counts[2]}, Evade:{phase_counts[3]}) | "
                      f"Time: {elapsed:.2f}s")

        return self.dominant_bull
