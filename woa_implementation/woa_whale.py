"""
Whale Agent (Whale) for Whale Optimization Algorithm (WOA)
==========================================================
Represents a humpback whale in the Whale Optimization Algorithm population.
Holds coordinates in 10-dimensional hyperparameter space for predictive multi-tier caching,
tracks personal fitness, and records hunting maneuver mode (spiral bubble-net, encircling, or search).
"""

import random
from typing import Dict, List, Optional
import numpy as np


class Whale:
    """
    Candidate solution agent in the Whale Optimization Algorithm (WOA).
    """

    GENE_BOUNDS: Dict[str, tuple] = {
        "w_recency": (-1.0, 1.0),
        "w_frequency": (-1.0, 1.0),
        "w_freq_decay": (0.01, 1.0),
        "w_size": (-1.0, 1.0),
        "w_priority": (0.0, 3.0),
        "w_penalty": (0.0, 2.0),
        "w_predicted_reuse": (-1.0, 1.0),
        "theta_dram": (0.0, 1.0),
        "alpha_qos": (0.0, 1.0),
        "target_p": (0.0, 1.0)
    }

    GENE_KEYS: List[str] = list(GENE_BOUNDS.keys())

    def __init__(self, position_seed: Optional[Dict[str, float]] = None):
        self.position: Dict[str, float] = {}

        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            if position_seed and key in position_seed:
                self.position[key] = float(np.clip(position_seed[key], min_val, max_val))
            else:
                self.position[key] = random.uniform(min_val, max_val)

        self.fitness: float = -float("inf")
        self.pbest_position: Dict[str, float] = self.position.copy()
        self.pbest_fitness: float = -float("inf")

        self.chr_total: float = 0.0
        self.chr_hpc: float = 0.0
        self.penalty_total: int = 0
        self.last_maneuver: str = "search"  # "spiral", "encircling", or "search"

    def clamp_bounds(self):
        """Ensure all parameter dimensions respect designated boundary ranges."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            self.position[key] = float(np.clip(self.position[key], min_val, max_val))

    def update_pbest(self):
        """Update personal best position if current fitness improves."""
        if self.fitness > self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()

    def clone(self) -> "Whale":
        copy_w = Whale(position_seed=self.position.copy())
        copy_w.fitness = self.fitness
        copy_w.pbest_position = self.pbest_position.copy()
        copy_w.pbest_fitness = self.pbest_fitness
        copy_w.chr_total = self.chr_total
        copy_w.chr_hpc = self.chr_hpc
        copy_w.penalty_total = self.penalty_total
        copy_w.last_maneuver = self.last_maneuver
        return copy_w
