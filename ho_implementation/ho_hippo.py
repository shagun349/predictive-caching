"""
Hippopotamus Agent (Hippo)
==========================
Represents an individual hippopotamus in the Hippopotamus Optimization (HO) swarm.
Each hippo holds a position in 10-dimensional space corresponding to predictive
multi-tier caching policy hyperparameters.

HO Features:
  - Dominant Bull Guidance (Global exploration in river/pond herds)
  - Defensive Counter-Maneuvers (Lévy-flight perturbation against predators)
  - Evasion Escapes (Localized niche exploitation)
"""

import random
from typing import Dict, List, Optional
import numpy as np


class Hippo:
    """
    Candidate solution agent in the Hippopotamus Optimization (HO) metaheuristic.
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
        self.chr_total: float = 0.0
        self.chr_hpc: float = 0.0
        self.penalty_total: int = 0
        self.pbest_position: Dict[str, float] = self.position.copy()
        self.pbest_fitness: float = -float("inf")
        self.last_phase: int = 1  # 1: Aquatic River, 2: Predator Defense, 3: Evasion

    def clamp_bounds(self):
        """Ensure all position dimensions strictly respect boundary constraints."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            self.position[key] = float(np.clip(self.position[key], min_val, max_val))

    def update_pbest(self):
        """Update personal best position if current fitness improves."""
        if self.fitness > self.pbest_fitness:
            self.pbest_fitness = self.fitness
            self.pbest_position = self.position.copy()

    def get_position_vector(self) -> np.ndarray:
        return np.array([self.position[k] for k in self.GENE_KEYS], dtype=np.float64)

    def set_position_vector(self, vec: np.ndarray):
        for i, k in enumerate(self.GENE_KEYS):
            min_val, max_val = self.GENE_BOUNDS[k]
            self.position[k] = float(np.clip(vec[i], min_val, max_val))

    def clone(self) -> "Hippo":
        copy_hippo = Hippo(position_seed=self.position.copy())
        copy_hippo.fitness = self.fitness
        copy_hippo.chr_total = self.chr_total
        copy_hippo.chr_hpc = self.chr_hpc
        copy_hippo.penalty_total = self.penalty_total
        copy_hippo.pbest_position = self.pbest_position.copy()
        copy_hippo.pbest_fitness = self.pbest_fitness
        copy_hippo.last_phase = self.last_phase
        return copy_hippo
