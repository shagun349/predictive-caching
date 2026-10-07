import random
from typing import Dict

class Particle:
    """
    Represents a single particle in the PSO Swarm.
    Each particle has a position (the 10 caching policy weights), a velocity,
    a personal best position, and fitness scores.
    """

    GENE_BOUNDS = {
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

    def __init__(self, position_seed: Dict[str, float] = None):
        self.position = {}
        self.velocity = {}
        
        # Initialize position
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            if position_seed and key in position_seed:
                self.position[key] = position_seed[key]
            else:
                self.position[key] = random.uniform(min_val, max_val)
            
            # Initialize velocity as 0 or a small random value
            v_range = abs(max_val - min_val) * 0.1
            self.velocity[key] = random.uniform(-v_range, v_range)
            
        self.fitness = -float('inf')
        self.pbest_position = self.position.copy()
        self.pbest_fitness = -float('inf')

    def update_position(self):
        """Updates position based on current velocity and clamps to bounds."""
        for key, (min_val, max_val) in self.GENE_BOUNDS.items():
            self.position[key] += self.velocity[key]
            
            # Clamp to bounds
            if self.position[key] < min_val:
                self.position[key] = min_val
                self.velocity[key] *= -0.5 # bounce effect
            elif self.position[key] > max_val:
                self.position[key] = max_val
                self.velocity[key] *= -0.5 # bounce effect

    @classmethod
    def create_qm_arc_seed(cls) -> 'Particle':
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
    def create_lru_seed(cls) -> 'Particle':
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
    def create_lfu_seed(cls) -> 'Particle':
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
