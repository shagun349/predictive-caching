"""
Food Source / Bee Agent for Artificial Bee Colony (ABC)
======================================================
Represents a candidate food source (solution) in the Artificial Bee Colony algorithm.
Holds position coordinates in 10-dimensional space for caching policy weights,
tracks nectar amount (fitness), and monitors trial count for abandonment/scouting.
"""

import random
from typing import Dict, List, Optional
import numpy as np


class FoodSource:
    """
    Candidate solution food source in the Artificial Bee Colony (ABC) metaheuristic.
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
        self.trial: int = 0  # Number of unpromoted iterations
        self.selection_prob: float = 0.0

        self.chr_total: float = 0.0
        self.chr_hpc: float = 0.0
        self.penalty_total: int = 0

    def clamp_bounds(self):
        """Ensure all gene values stay within designated boundary constraints."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            self.position[key] = float(np.clip(self.position[key], min_val, max_val))

    def reset_random(self):
        """Scout bee regeneration: produce a new random position in search space."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            self.position[key] = random.uniform(min_val, max_val)
        self.trial = 0

    def clone(self) -> "FoodSource":
        copy_fs = FoodSource(position_seed=self.position.copy())
        copy_fs.fitness = self.fitness
        copy_fs.trial = self.trial
        copy_fs.selection_prob = self.selection_prob
        copy_fs.chr_total = self.chr_total
        copy_fs.chr_hpc = self.chr_hpc
        copy_fs.penalty_total = self.penalty_total
        return copy_fs
