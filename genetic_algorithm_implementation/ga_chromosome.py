"""
Genetic Algorithm Chromosome Definition for Predictive Caching
==============================================================
Defines the gene structure, encoding, crossover, mutation, and baseline seed initialization
for evolving predictive multi-tier cache replacement policies.
"""

import copy
import math
import random
from typing import Dict, List, Tuple, Any, Optional


# Specification of each gene in the chromosome
GENE_SPECS = [
    # (name, min_val, max_val, default_val, description)
    ("w_recency", -1.0, 1.0, 0.6, "Recency weight in predictive utility score"),
    ("w_frequency", -1.0, 1.0, 0.4, "Frequency weight in predictive utility score"),
    ("w_freq_decay", 0.01, 1.0, 0.1, "Temporal frequency decay rate (aging factor)"),
    ("w_size", -1.0, 1.0, -0.2, "Object size penalty weight (negative penalizes large objects)"),
    ("w_priority", 0.0, 3.0, 1.5, "QoS priority boost for High Priority Content (HPC)"),
    ("w_penalty", 0.0, 2.0, 1.0, "Miss penalty cost sensitivity weight"),
    ("w_predicted_reuse", -1.0, 1.0, 0.5, "Predictive reuse probability weight"),
    ("theta_dram", 0.0, 1.0, 0.5, "DRAM admission threshold (DRAM vs SSD tier placement)"),
    ("alpha_qos", 0.0, 1.0, 0.2, "QoS differentiation factor for Low Priority Content (LPC)"),
    ("target_p", 0.0, 1.0, 0.5, "Target balance parameter between recency and frequency")
]

GENE_NAMES = [spec[0] for spec in GENE_SPECS]
GENE_BOUNDS = [(spec[1], spec[2]) for spec in GENE_SPECS]


class Chromosome:
    """Represents a single candidate solution (predictive caching policy configuration)."""

    def __init__(self, genes: Optional[List[float]] = None):
        if genes is None:
            self.genes = [
                random.uniform(min_v, max_v)
                for _, min_v, max_v, _, _ in GENE_SPECS
            ]
        else:
            self.genes = list(genes)
            self._clamp_genes()

        # Fitness and performance evaluation metrics
        self.fitness: float = -float("inf")
        self.metrics: Dict[str, Any] = {}
        self.evaluated: bool = False

    def _clamp_genes(self):
        """Clamp genes to their valid bounds."""
        for i, (min_v, max_v) in enumerate(GENE_BOUNDS):
            if self.genes[i] < min_v:
                self.genes[i] = min_v
            elif self.genes[i] > max_v:
                self.genes[i] = max_v

    def get_gene_dict(self) -> Dict[str, float]:
        """Return genes as a dictionary keyed by gene name."""
        return {name: self.genes[i] for i, name in enumerate(GENE_NAMES)}

    def clone(self) -> "Chromosome":
        """Create a deep copy of this chromosome."""
        c = Chromosome(self.genes)
        c.fitness = self.fitness
        c.metrics = copy.deepcopy(self.metrics)
        c.evaluated = self.evaluated
        return c

    @classmethod
    def create_random(cls) -> "Chromosome":
        """Create a purely random chromosome."""
        return cls()

    @classmethod
    def create_lru_seed(cls) -> "Chromosome":
        """Seed representing recency-dominated behavior (LRU-like)."""
        genes = [
            1.0,   # w_recency
            0.0,   # w_frequency
            0.05,  # w_freq_decay
            0.0,   # w_size
            0.0,   # w_priority
            0.0,   # w_penalty
            0.0,   # w_predicted_reuse
            0.2,   # theta_dram
            0.0,   # alpha_qos
            0.8    # target_p
        ]
        return cls(genes)

    @classmethod
    def create_lfu_seed(cls) -> "Chromosome":
        """Seed representing frequency-dominated behavior (LFU-like)."""
        genes = [
            0.0,   # w_recency
            1.0,   # w_frequency
            0.01,  # w_freq_decay
            0.0,   # w_size
            0.0,   # w_priority
            0.0,   # w_penalty
            0.0,   # w_predicted_reuse
            0.5,   # theta_dram
            0.0,   # alpha_qos
            0.2    # target_p
        ]
        return cls(genes)

    @classmethod
    def create_qm_arc_seed(cls) -> "Chromosome":
        """Seed representing QoS-aware multi-tier ARC behavior."""
        genes = [
            0.5,   # w_recency
            0.5,   # w_frequency
            0.05,  # w_freq_decay
            -0.1,  # w_size
            1.5,   # w_priority
            1.0,   # w_penalty
            0.3,   # w_predicted_reuse
            0.4,   # theta_dram
            0.2,   # alpha_qos (paper: alpha = 15/75 = 0.2)
            0.5    # target_p
        ]
        return cls(genes)

    @staticmethod
    def crossover(p1: "Chromosome", p2: "Chromosome", method: str = "sbx", eta: float = 2.0) -> Tuple["Chromosome", "Chromosome"]:
        """
        Produce two offspring from two parents using specified crossover operator.
        Methods: 'sbx' (Simulated Binary Crossover), 'uniform', 'blend' (BLX-alpha).
        """
        c1_genes = list(p1.genes)
        c2_genes = list(p2.genes)

        if method == "sbx":
            for i in range(len(GENE_SPECS)):
                if random.random() <= 0.5:
                    y1 = min(p1.genes[i], p2.genes[i])
                    y2 = max(p1.genes[i], p2.genes[i])
                    if abs(y2 - y1) > 1e-9:
                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1 - GENE_BOUNDS[i][0]) / (y2 - y1))
                        alpha = 2.0 - math.pow(beta, -(eta + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = math.pow(rand * alpha, 1.0 / (eta + 1.0))
                        else:
                            betaq = math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0))

                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        c1_genes[i] = c1
                        c2_genes[i] = c2

        elif method == "blend":
            blx_alpha = 0.3
            for i in range(len(GENE_SPECS)):
                y1 = min(p1.genes[i], p2.genes[i])
                y2 = max(p1.genes[i], p2.genes[i])
                diff = y2 - y1
                low = y1 - blx_alpha * diff
                high = y2 + blx_alpha * diff
                c1_genes[i] = random.uniform(low, high)
                c2_genes[i] = random.uniform(low, high)

        else:  # 'uniform'
            for i in range(len(GENE_SPECS)):
                if random.random() < 0.5:
                    c1_genes[i], c2_genes[i] = c2_genes[i], c1_genes[i]

        return Chromosome(c1_genes), Chromosome(c2_genes)

    def mutate(self, mutation_rate: float = 0.15, sigma: float = 0.15, method: str = "gaussian"):
        """
        In-place mutation of genes.
        Methods: 'gaussian', 'polynomial'.
        """
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                min_v, max_v = GENE_BOUNDS[i]
                range_span = max_v - min_v

                if method == "gaussian":
                    noise = random.gauss(0.0, sigma * range_span)
                    self.genes[i] += noise
                elif method == "polynomial":
                    eta_m = 20.0
                    rand = random.random()
                    if rand < 0.5:
                        delta = math.pow(2.0 * rand, 1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - math.pow(2.0 * (1.0 - rand), 1.0 / (eta_m + 1.0))
                    self.genes[i] += delta * range_span
                else:
                    # Random reset
                    self.genes[i] = random.uniform(min_v, max_v)

        self._clamp_genes()
        self.evaluated = False

    def __repr__(self) -> str:
        s = ", ".join(f"{name}={val:.3f}" for name, val in zip(GENE_NAMES, self.genes))
        return f"Chromosome(fitness={self.fitness:.4f}, [{s}])"
