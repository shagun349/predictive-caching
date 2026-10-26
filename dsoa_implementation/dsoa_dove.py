"""
Dove Agent (Dove)
=================
Represents an individual dove in the Dove Swarm Optimization (DSOA-EC) population.
Each dove carries a position in 10-dimensional space corresponding to the predictive
multi-tier caching policy hyperparameters, a satiety level, and fitness history.

Reference:
  Su, M.-C. et al. (2022). "Dove Swarm Optimization Algorithm"
  Biological Metaphor: foraging, competition, and satisfaction-based relocation.
"""

import random
from typing import Dict, List, Optional


class Dove:
    """
    Represents an individual dove in the DSOA-EC swarm.
    Each dove holds a position in 10-dimensional space corresponding
    to predictive multi-tier caching policy hyperparameters.

    Unique DSOA features vs PSO/SFO:
      - satiety: tracks how 'well-fed' (fit) the dove is across iterations
      - Satiety-driven branching between exploitation (local foraging) and
        exploration (random relocation)
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

        # DSOA-specific: satiety level (0.0 = starving, 1.0 = fully fed)
        self.satiety: float = 0.5
        self.last_results: Dict = {}

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
    def create_balanced_seed(cls) -> 'Dove':
        """Seed with balanced weights for fair initial population starting ground."""
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
    def create_lru_seed(cls) -> 'Dove':
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
    def create_lfu_seed(cls) -> 'Dove':
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

    def copy(self) -> 'Dove':
        """Return a deep copy of this dove."""
        d = Dove(position_seed=self.position)
        d.velocity = self.velocity.copy()
        d.fitness = self.fitness
        d.pbest_position = self.pbest_position.copy()
        d.pbest_fitness = self.pbest_fitness
        d.satiety = self.satiety
        return d
