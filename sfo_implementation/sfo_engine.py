"""
Sharpbelly Fish Optimization (SFO) Engine
=========================================
Implements the Sharpbelly Fish Optimization metaheuristic algorithm modeled after
the ecological schooling and foraging behaviors of Hemiculter leucisculus:
  1. Fitness-Driven Fast Swimming (accelerating toward optimal zones)
  2. Convergence-Guided Gathering (collective aggregation toward global best)
  3. Stagnation-Triggered Dispersal (scattering when search stalls)
  4. Disturbance-Induced Escape (evasive random jumps under environmental disruption)
"""

import time
import random
from typing import List, Dict, Any, Optional

from sfo_implementation.sfo_fish import Fish
from sfo_implementation.sfo_predictive_policy import SFOPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator


class SFOEngine:
    """
    Sharpbelly Fish Optimization Engine for Multi-Tier Predictive Caching.
    """

    def __init__(
        self,
        num_fish: int = 30,
        iterations: int = 25,
        alpha: float = 0.3,
        ts: int = 5,
        pf: float = 0.4,
        w_chr: float = 0.40,
        w_hpc: float = 0.35,
        w_pen: float = 0.15,
        w_lat: float = 0.10,
        seed: int = 42
    ):
        self.num_fish = num_fish
        self.iterations = iterations
        self.alpha = alpha  # Gathering weight towards best solution
        self.ts = ts        # Stagnation count threshold before dispersal
        self.pf = pf        # Global disturbance probability
        self.w_chr = w_chr
        self.w_hpc = w_hpc
        self.w_pen = w_pen
        self.w_lat = w_lat

        random.seed(seed)
        self.history: List[Dict[str, Any]] = []

        self.gbest_position: Dict[str, float] = {}
        self.gbest_fitness: float = -float('inf')
        self.gbest_metrics: Dict[str, Any] = {}

    def _evaluate_fish(
        self,
        fish: Fish,
        simulator_env: MultiTierCacheSimulator,
        training_data: List[Dict]
    ) -> float:
        """Run the multi-tier simulator and calculate multi-objective fitness for a candidate fish."""
        policy = SFOPredictivePolicy(fish=fish)
        results = simulator_env.run(policy, training_data)

        chr_total = results["chr_total"]
        chr_hpc = results["chr_hpc"]
        penalty = results["penalty_total"]
        latency = results["avg_retrieval_time"]

        # Bound penalty and latency normalization (matching GA and PSO)
        MAX_PENALTY = len(training_data) * 20.0
        MAX_LATENCY = 150.0

        norm_penalty = min(1.0, penalty / MAX_PENALTY)
        norm_latency = min(1.0, latency / MAX_LATENCY)

        # Multi-objective fitness function (higher is better)
        fitness = (
            (self.w_chr * chr_total) +
            (self.w_hpc * chr_hpc) -
            (self.w_pen * norm_penalty) -
            (self.w_lat * norm_latency)
        )

        fish.fitness = fitness
        fish.last_results = results
        return fitness

    def optimize(
        self,
        simulator_env: MultiTierCacheSimulator,
        training_data: List[Dict],
        verbose: bool = True
    ) -> Fish:
        """
        Executes the Sharpbelly Fish Optimization algorithm across iterations.
        """
        # 1. Initialize fish population with heuristic seeds for equal comparison baseline
        swarm: List[Fish] = []
        swarm.append(Fish.create_qm_arc_seed())
        swarm.append(Fish.create_lru_seed())
        swarm.append(Fish.create_lfu_seed())

        while len(swarm) < self.num_fish:
            swarm.append(Fish())

        # Precalculate delta vector (escape jump magnitude = 0.05 * (ub - lb))
        delta: Dict[str, float] = {}
        span: Dict[str, float] = {}
        for key, (lb, ub) in Fish.GENE_BOUNDS.items():
            delta[key] = 0.05 * (ub - lb)
            span[key] = ub - lb

        # 2. Initial evaluation
        if verbose:
            print(f"Evaluating initial SFO population ({self.num_fish} fish)...")

        for f in swarm:
            fit = self._evaluate_fish(f, simulator_env, training_data)
            f.pbest_fitness = fit
            f.pbest_position = f.position.copy()

            if fit > self.gbest_fitness:
                self.gbest_fitness = fit
                self.gbest_position = f.position.copy()
                self.gbest_metrics = f.last_results

        if verbose:
            print(f"Initial Best Fitness: {self.gbest_fitness:.4f} | Total CHR: {self.gbest_metrics['chr_total']*100:.2f}% | HPC CHR: {self.gbest_metrics['chr_hpc']*100:.2f}%")

        no_improvement_count = 0

        # 3. Main SFO Iteration Loop
        for iter_num in range(1, self.iterations + 1):
            t0 = time.time()

            # Find fitness bounds for relative swimming speed scaling
            current_fits = [f.fitness for f in swarm]
            min_fit = min(current_fits)
            max_fit = max(current_fits)
            fit_range = max_fit - min_fit + 1e-12

            # Update each fish based on the 4 ecological behaviors
            for f in swarm:
                # Behavior 1: Fitness-Driven Fast Swimming
                # Higher-fitness fish swim faster toward food / best position
                swim_speed = (f.fitness - min_fit) / fit_range

                for key in Fish.GENE_KEYS:
                    lb, ub = Fish.GENE_BOUNDS[key]
                    diff = self.gbest_position[key] - f.position[key]

                    # 1. Fast swimming velocity component
                    v1 = swim_speed * random.random() * diff

                    # 2. Convergence-guided gathering component (alpha attractor)
                    v2 = self.alpha * random.random() * diff

                    # Base velocity
                    v = v1 + v2

                    # 4. Disturbance-induced escape (random perturbation)
                    if random.random() < self.pf:
                        v += delta[key] * random.gauss(0, 1.0)

                    # 3. Stagnation-triggered dispersal (scattering jump)
                    if no_improvement_count >= self.ts:
                        v += 0.2 * span[key] * (2.0 * random.random() - 1.0)

                    f.velocity[key] = v
                    f.position[key] += v

                # Clamp within boundaries
                f.clamp_bounds()

            # Evaluate all fish at new positions
            iter_best_fit = -float('inf')
            iter_sum_fit = 0.0

            for f in swarm:
                fit = self._evaluate_fish(f, simulator_env, training_data)
                iter_sum_fit += fit

                # Update personal best
                if fit > f.pbest_fitness:
                    f.pbest_fitness = fit
                    f.pbest_position = f.position.copy()

                # Update global best
                if fit > self.gbest_fitness:
                    self.gbest_fitness = fit
                    self.gbest_position = f.position.copy()
                    self.gbest_metrics = f.last_results

                if fit > iter_best_fit:
                    iter_best_fit = fit

            # Check for stagnation / improvement
            if self.gbest_fitness > (self.history[-1]["best_fitness"] if self.history else -float('inf')):
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            elapsed = time.time() - t0
            avg_fit = iter_sum_fit / len(swarm)

            iter_record = {
                "iteration": iter_num,
                "best_fitness": self.gbest_fitness,
                "avg_fitness": avg_fit,
                "best_chr_total": self.gbest_metrics.get("chr_total", 0.0),
                "best_chr_hpc": self.gbest_metrics.get("chr_hpc", 0.0),
                "penalty_total": self.gbest_metrics.get("penalty_total", 0),
                "no_improvement_count": no_improvement_count,
                "time": elapsed
            }
            self.history.append(iter_record)

            if verbose:
                print(
                    f"Iteration [{iter_num:2d}/{self.iterations}] | "
                    f"Best Fit: {self.gbest_fitness:.4f} | "
                    f"Avg Fit: {avg_fit:.4f} | "
                    f"CHR: {self.gbest_metrics['chr_total']*100:5.2f}% | "
                    f"HPC CHR: {self.gbest_metrics['chr_hpc']*100:5.2f}% | "
                    f"Stagnation: {no_improvement_count:2d} | "
                    f"Time: {elapsed:.2f}s"
                )

        best_fish = Fish(position_seed=self.gbest_position)
        best_fish.fitness = self.gbest_fitness
        return best_fish
