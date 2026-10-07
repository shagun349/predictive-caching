"""
Particle Swarm Optimization Engine
==================================
Trains the PSO-Predictive policy using a subset of the trace data.
"""

import copy
import time
import random
from typing import List, Dict, Any, Tuple
from pso_implementation.pso_particle import Particle
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator

class PSOEngine:
    def __init__(
        self,
        num_particles: int = 30,
        iterations: int = 25,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        w_chr: float = 0.40,
        w_hpc: float = 0.35,
        w_pen: float = 0.15,
        w_lat: float = 0.10,
        seed: int = 42
    ):
        self.num_particles = num_particles
        self.iterations = iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.w_chr = w_chr
        self.w_hpc = w_hpc
        self.w_pen = w_pen
        self.w_lat = w_lat
        self.history = []
        
        random.seed(seed)
        
        # Global best
        self.gbest_position = {}
        self.gbest_fitness = -float('inf')
        
    def _evaluate_particle(self, particle: Particle, simulator_env: MultiTierCacheSimulator, training_data: List[Dict]) -> float:
        """Runs the simulator and calculates multi-objective fitness for a particle."""
        policy = PSOPredictivePolicy(particle=particle)
        
        # Run simulation
        results = simulator_env.run(policy, training_data)
        
        chr_total = results["chr_total"]
        chr_hpc = results["chr_hpc"]
        penalty = results["penalty_total"]
        latency = results["avg_retrieval_time"]
        
        # Normalization bounds for fitness (approximate worst cases to bound penalty/latency)
        MAX_PENALTY = len(training_data) * 20.0
        MAX_LATENCY = 150.0
        
        norm_penalty = min(1.0, penalty / MAX_PENALTY)
        norm_latency = min(1.0, latency / MAX_LATENCY)
        
        # Multi-objective fitness function (higher is better)
        fitness = (self.w_chr * chr_total) + \
                  (self.w_hpc * chr_hpc) - \
                  (self.w_pen * norm_penalty) - \
                  (self.w_lat * norm_latency)
                  
        return fitness

    def optimize(self, simulator_env: MultiTierCacheSimulator, training_data: List[Dict], verbose: bool = True) -> Particle:
        """Runs the PSO optimization loop."""
        
        # 1. Initialize Swarm
        swarm = []
        swarm.append(Particle.create_qm_arc_seed())
        swarm.append(Particle.create_lru_seed())
        swarm.append(Particle.create_lfu_seed())
        
        while len(swarm) < self.num_particles:
            swarm.append(Particle())
            
        # Initial evaluation
        for p in swarm:
            fit = self._evaluate_particle(p, simulator_env, training_data)
            p.fitness = fit
            p.pbest_fitness = fit
            p.pbest_position = p.position.copy()
            
            if fit > self.gbest_fitness:
                self.gbest_fitness = fit
                self.gbest_position = p.position.copy()
                
        if verbose:
            print(f"Initial Best Fitness: {self.gbest_fitness:.4f}")
            
        # 2. Main Optimization Loop
        for iter_num in range(1, self.iterations + 1):
            t0 = time.time()
            
            # Linear decay of inertia weight
            w = self.w_start - ((self.w_start - self.w_end) * (iter_num / self.iterations))
            
            iter_best_fit = -float('inf')
            
            for p in swarm:
                # Update velocity
                for key in Particle.GENE_BOUNDS.keys():
                    r1 = random.random()
                    r2 = random.random()
                    
                    cognitive = self.c1 * r1 * (p.pbest_position[key] - p.position[key])
                    social = self.c2 * r2 * (self.gbest_position[key] - p.position[key])
                    
                    p.velocity[key] = (w * p.velocity[key]) + cognitive + social
                    
                # Update position
                p.update_position()
                
                # Evaluate new position
                fit = self._evaluate_particle(p, simulator_env, training_data)
                p.fitness = fit
                
                # Update pbest
                if fit > p.pbest_fitness:
                    p.pbest_fitness = fit
                    p.pbest_position = p.position.copy()
                    
                # Update gbest
                if fit > self.gbest_fitness:
                    self.gbest_fitness = fit
                    self.gbest_position = p.position.copy()
                    
                if fit > iter_best_fit:
                    iter_best_fit = fit
                    
            elapsed = time.time() - t0
            
            # Record history
            avg_fit = sum(p.fitness for p in swarm) / self.num_particles
            self.history.append({
                "iteration": iter_num,
                "best_fitness": self.gbest_fitness,
                "avg_fitness": avg_fit,
                "time": elapsed
            })
            
            if verbose:
                print(f"Iter {iter_num:2d}/{self.iterations} | Best Fit: {self.gbest_fitness:.4f} | Mean: {avg_fit:.4f} | Time: {elapsed:.2f}s")
                
        best_particle = Particle(position_seed=self.gbest_position)
        best_particle.fitness = self.gbest_fitness
        return best_particle
