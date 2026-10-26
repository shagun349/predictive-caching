"""
Dove Swarm Optimization Algorithm (DSOA-EC) Engine
===================================================
Implements the Dove Swarm Optimization metaheuristic modeled after the
foraging, competition, and satisfaction-based relocation behaviors of doves:

  1. Satiety-Driven Foraging (well-fed doves exploit locally near best)
  2. Competition-Guided Movement (step-size scales with relative fitness)
  3. Hunger-Triggered Exploration (unsatisfied doves relocate randomly)
  4. Lévy-Flight Perturbation (occasional long jumps for diversity)

Reference:
  Su, M.-C. et al. (2022). "Dove Swarm Optimization Algorithm"
"""

import math
import time
import random
from typing import List, Dict, Any

from dsoa_implementation.dsoa_dove import Dove
from dsoa_implementation.dsoa_predictive_policy import DSOAPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator


class DSOAEngine:
    """
    Dove Swarm Optimization Algorithm Engine for Multi-Tier Predictive Caching.

    The DSOA-EC variant applies three core dove behaviors per iteration:
      Phase 1 — Local Foraging (Exploitation):
        Well-fed doves (high satiety) move toward the global best position
        with a step size proportional to their relative fitness.
      Phase 2 — Competition & Satiety Update:
        Each dove's satiety decays over time but increases when fitness improves.
        This creates natural exploration pressure on stagnant doves.
      Phase 3 — Exploration (Relocation):
        Hungry doves (satiety below threshold) abandon their current position
        and fly to a random location in the search space, with optional
        Lévy-flight long jumps for enhanced diversity.
    """

    def __init__(
        self,
        num_doves: int = 30,
        iterations: int = 25,
        eta: float = 0.6,           # Step-size scaling factor for local foraging
        satiety_decay: float = 0.85, # Satiety decay rate per iteration (lambda)
        satiety_threshold: float = 0.3, # Below this satiety, dove relocates
        levy_prob: float = 0.15,     # Probability of Lévy-flight perturbation
        w_chr: float = 0.40,
        w_hpc: float = 0.35,
        w_pen: float = 0.15,
        w_lat: float = 0.10,
        seed: int = 42
    ):
        self.num_doves = num_doves
        self.iterations = iterations
        self.eta = eta
        self.satiety_decay = satiety_decay
        self.satiety_threshold = satiety_threshold
        self.levy_prob = levy_prob
        self.w_chr = w_chr
        self.w_hpc = w_hpc
        self.w_pen = w_pen
        self.w_lat = w_lat

        random.seed(seed)
        self.history: List[Dict[str, Any]] = []

        self.gbest_position: Dict[str, float] = {}
        self.gbest_fitness: float = -float('inf')
        self.gbest_metrics: Dict[str, Any] = {}

    @staticmethod
    def _levy_step(beta: float = 1.5) -> float:
        """
        Generate a Lévy-flight step using Mantegna's algorithm.
        Produces occasional long jumps to escape local optima.
        """
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = random.gauss(0, sigma_u)
        v = random.gauss(0, 1)
        step = u / (abs(v) ** (1 / beta))
        return step

    def _evaluate_dove(
        self,
        dove: Dove,
        simulator_env: MultiTierCacheSimulator,
        training_data: List[Dict]
    ) -> float:
        """Run the multi-tier simulator and calculate multi-objective fitness for a candidate dove."""
        policy = DSOAPredictivePolicy(dove=dove)
        results = simulator_env.run(policy, training_data)

        chr_total = results["chr_total"]
        chr_hpc = results["chr_hpc"]
        penalty = results["penalty_total"]
        latency = results["avg_retrieval_time"]

        # Bound penalty and latency normalization (matching GA, PSO, SFO)
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

        dove.fitness = fitness
        dove.last_results = results
        return fitness

    def optimize(
        self,
        simulator_env: MultiTierCacheSimulator,
        training_data: List[Dict],
        verbose: bool = True
    ) -> Dove:
        """
        Executes the Dove Swarm Optimization Algorithm (DSOA-EC) across iterations.
        """
        # 1. Initialize dove population with heuristic seeds
        flock: List[Dove] = []
        flock.append(Dove.create_balanced_seed())
        flock.append(Dove.create_lru_seed())
        flock.append(Dove.create_lfu_seed())

        while len(flock) < self.num_doves:
            flock.append(Dove())

        # 2. Initial evaluation
        if verbose:
            print(f"Evaluating initial DSOA population ({self.num_doves} doves)...")

        for dove in flock:
            fit = self._evaluate_dove(dove, simulator_env, training_data)
            dove.pbest_fitness = fit
            dove.pbest_position = dove.position.copy()
            dove.satiety = 0.5  # Start at mid-satiety

            if fit > self.gbest_fitness:
                self.gbest_fitness = fit
                self.gbest_position = dove.position.copy()
                self.gbest_metrics = dove.last_results

        if verbose:
            print(
                f"Initial Best Fitness: {self.gbest_fitness:.4f} | "
                f"Total CHR: {self.gbest_metrics['chr_total']*100:.2f}% | "
                f"HPC CHR: {self.gbest_metrics['chr_hpc']*100:.2f}%"
            )

        # 3. Main DSOA-EC Iteration Loop
        for iter_num in range(1, self.iterations + 1):
            t0 = time.time()

            # Find fitness bounds for relative scaling
            current_fits = [d.fitness for d in flock]
            min_fit = min(current_fits)
            max_fit = max(current_fits)
            fit_range = max_fit - min_fit + 1e-12

            prev_gbest = self.gbest_fitness

            for dove in flock:
                # Compute relative fitness (0 = worst, 1 = best in flock)
                relative_fit = (dove.fitness - min_fit) / fit_range

                # Phase 1: Satiety decay
                dove.satiety *= self.satiety_decay

                if dove.satiety >= self.satiety_threshold:
                    # ─── LOCAL FORAGING (Exploitation) ─────────────────────
                    # Well-fed doves move toward the global best position
                    # Step size scales with fitness (better doves take bigger steps)
                    for key in Dove.GENE_KEYS:
                        diff = self.gbest_position[key] - dove.position[key]
                        r1 = random.random()

                        # Fitness-driven step: η * relative_fitness * r1 * (gbest - xi)
                        step = self.eta * (0.5 + 0.5 * relative_fit) * r1 * diff

                        # Optional Lévy-flight perturbation for diversity
                        if random.random() < self.levy_prob:
                            lb, ub = Dove.GENE_BOUNDS[key]
                            levy = self._levy_step() * 0.01 * (ub - lb)
                            step += levy

                        dove.velocity[key] = step
                        dove.position[key] += step
                else:
                    # ─── EXPLORATION (Hunger-Triggered Relocation) ─────────
                    # Hungry doves fly to a completely new random location
                    for key in Dove.GENE_KEYS:
                        lb, ub = Dove.GENE_BOUNDS[key]

                        # Partial relocation: blend random position with personal best
                        r2 = random.random()
                        random_pos = lb + random.random() * (ub - lb)
                        dove.position[key] = r2 * random_pos + (1 - r2) * dove.pbest_position[key]
                        dove.velocity[key] = dove.position[key] - dove.pbest_position[key]

                # Clamp within boundaries
                dove.clamp_bounds()

            # Evaluate all doves at new positions
            iter_sum_fit = 0.0

            for dove in flock:
                old_fit = dove.fitness
                fit = self._evaluate_dove(dove, simulator_env, training_data)
                iter_sum_fit += fit

                # Satiety boost: if fitness improved, dove found food → satiety increases
                if fit > old_fit:
                    improvement_ratio = (fit - old_fit) / (abs(old_fit) + 1e-12)
                    dove.satiety = min(1.0, dove.satiety + 0.3 + 0.2 * improvement_ratio)
                else:
                    # Slight additional decay on stagnation
                    dove.satiety *= 0.95

                # Update personal best
                if fit > dove.pbest_fitness:
                    dove.pbest_fitness = fit
                    dove.pbest_position = dove.position.copy()

                # Update global best
                if fit > self.gbest_fitness:
                    self.gbest_fitness = fit
                    self.gbest_position = dove.position.copy()
                    self.gbest_metrics = dove.last_results

            elapsed = time.time() - t0
            avg_fit = iter_sum_fit / len(flock)
            avg_satiety = sum(d.satiety for d in flock) / len(flock)
            exploring_count = sum(1 for d in flock if d.satiety < self.satiety_threshold)

            iter_record = {
                "iteration": iter_num,
                "best_fitness": self.gbest_fitness,
                "avg_fitness": avg_fit,
                "best_chr_total": self.gbest_metrics.get("chr_total", 0.0),
                "best_chr_hpc": self.gbest_metrics.get("chr_hpc", 0.0),
                "penalty_total": self.gbest_metrics.get("penalty_total", 0),
                "avg_satiety": avg_satiety,
                "exploring_doves": exploring_count,
                "time": elapsed
            }
            self.history.append(iter_record)

            if verbose:
                improved = "↑" if self.gbest_fitness > prev_gbest else "="
                print(
                    f"Iteration [{iter_num:2d}/{self.iterations}] {improved} | "
                    f"Best Fit: {self.gbest_fitness:.4f} | "
                    f"Avg Fit: {avg_fit:.4f} | "
                    f"CHR: {self.gbest_metrics['chr_total']*100:5.2f}% | "
                    f"HPC CHR: {self.gbest_metrics['chr_hpc']*100:5.2f}% | "
                    f"Satiety: {avg_satiety:.2f} | "
                    f"Exploring: {exploring_count:2d} | "
                    f"Time: {elapsed:.2f}s"
                )

        best_dove = Dove(position_seed=self.gbest_position)
        best_dove.fitness = self.gbest_fitness
        return best_dove
