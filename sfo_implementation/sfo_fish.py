"""
Sharpbelly Fish Agent (Fish)
============================
Represents an individual sharpbelly fish in the SFO population.
Tracks candidate caching policy weights, velocities, and fitness scores.
"""

import random
from typing import Dict, List, Optional


class Fish:
    """
    Represents an individual sharpbelly fish in the SFO swarm.
    Each fish holds a position in 10-dimensional space corresponding
    to predictive multi-tier caching policy hyperparameters.
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
        self.velocity: Dict[str, float] = {}

        # Initialize position within bounds
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            if position_seed and key in position_seed:
                self.position[key] = float(position_seed[key])
            else:
                self.position[key] = random.uniform(min_val, max_val)

            # Initialize small exploratory velocity
            v_range = abs(max_val - min_val) * 0.1
            self.velocity[key] = random.uniform(-v_range, v_range)

        self.fitness: float = -float('inf')
        self.pbest_position: Dict[str, float] = self.position.copy()
        self.pbest_fitness: float = -float('inf')

    def clamp_bounds(self):
        """Clamp position to predefined parameter boundaries."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            if self.position[key] < min_val:
                self.position[key] = min_val
                self.velocity[key] *= -0.5  # Soft bounce boundary effect
            elif self.position[key] > max_val:
                self.position[key] = max_val
                self.velocity[key] *= -0.5

    def to_vector(self) -> List[float]:
        """Convert position dictionary to ordered vector."""
        return [self.position[k] for k in self.GENE_KEYS]

    def from_vector(self, vec: List[float]):
        """Load position from ordered vector and clamp to bounds."""
        for i, k in enumerate(self.GENE_KEYS):
            self.position[k] = vec[i]
        self.clamp_bounds()

    @classmethod
    def create_qm_arc_seed(cls) -> 'Fish':
        """Seed based on QM-ARC principles for fair initial population starting ground."""
        seed = {
            "w_recency": 0.5,
            "w_frequency": 0.5,
            "w_freq_decay": 0.05,
            "w_size": -0.1,
            "w_priority": 1.5,
            "w_penalty": 1.0,
            "w_predicted_reuse": 0.3,
            "theta_dram": 0.4,
            "alpha_qos": 0.2,
            "target_p": 0.5
        }
        return cls(position_seed=seed)

    @classmethod
    def create_lru_seed(cls) -> 'Fish':
        """Seed emphasizing pure recency."""
        seed = {
            "w_recency": 1.0,
            "w_frequency": 0.0,
            "w_freq_decay": 0.05,
            "w_size": 0.0,
            "w_priority": 0.0,
            "w_penalty": 0.0,
            "w_predicted_reuse": 0.0,
            "theta_dram": 0.2,
            "alpha_qos": 0.0,
            "target_p": 0.8
        }
        return cls(position_seed=seed)

    @classmethod
    def create_lfu_seed(cls) -> 'Fish':
        """Seed emphasizing pure frequency."""
        seed = {
            "w_recency": 0.0,
            "w_frequency": 1.0,
            "w_freq_decay": 0.01,
            "w_size": 0.0,
            "w_priority": 0.0,
            "w_penalty": 0.0,
            "w_predicted_reuse": 0.0,
            "theta_dram": 0.5,
            "alpha_qos": 0.0,
            "target_p": 0.2
        }
        return cls(position_seed=seed)

    def copy(self) -> 'Fish':
        """Return a deep copy of this fish."""
        f = Fish(position_seed=self.position)
        f.velocity = self.velocity.copy()
        f.fitness = self.fitness
        f.pbest_position = self.pbest_position.copy()
        f.pbest_fitness = self.pbest_fitness
        return f
