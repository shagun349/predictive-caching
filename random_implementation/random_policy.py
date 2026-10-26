"""
Multi-Tier Random Eviction Policy
=================================
Selects items uniformly at random for eviction when capacity is exceeded.
"""

import random
from typing import Dict, List, Tuple, Any, Optional
from genetic_algorithm_implementation.baseline_policies import BasePolicy


class MultiTierRandomPolicy(BasePolicy):
    """
    Standard 2-tier Random Eviction Policy:
    - DRAM evicts a randomly chosen block to SSD.
    - SSD evicts a randomly chosen block to external repository.
    """
    def __init__(self, seed: Optional[int] = 42):
        super().__init__("Random")
        self.dram: Dict[str, Dict[str, Any]] = {}
        self.ssd: Dict[str, Dict[str, Any]] = {}
        self.dram_used = 0
        self.ssd_used = 0
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def reset(self, dram_max_bytes: int, ssd_max_bytes: int, slot_size: int = 4567):
        super().reset(dram_max_bytes, ssd_max_bytes, slot_size)
        self.dram.clear()
        self.ssd.clear()
        self.dram_used = 0
        self.ssd_used = 0
        if self.seed is not None:
            random.seed(self.seed)

    def lookup(self, key: str) -> Optional[str]:
        if key in self.dram:
            return "DRAM"
        if key in self.ssd:
            return "SSD"
        return None

    def on_hit(self, key: str, tier: str, record: Dict[str, Any], current_time: float):
        pass

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        key = record["key"]
        size = record["size"]
        evictions: List[Tuple[str, str]] = []

        while (self.dram_used + size > self.dram_max_bytes) and self.dram:
            rnd_k = random.choice(list(self.dram.keys()))
            rnd_r = self.dram.pop(rnd_k)
            self.dram_used -= rnd_r["size"]

            while (self.ssd_used + rnd_r["size"] > self.ssd_max_bytes) and self.ssd:
                ssd_rnd_k = random.choice(list(self.ssd.keys()))
                ssd_drop = self.ssd.pop(ssd_rnd_k)
                self.ssd_used -= ssd_drop["size"]
                evictions.append(("SSD", ssd_rnd_k))

            self.ssd[rnd_k] = rnd_r
            self.ssd_used += rnd_r["size"]

        self.dram[key] = record
        self.dram_used += size
        return "DRAM", evictions

    def get_used_capacities(self) -> Tuple[int, int]:
        return self.dram_used, self.ssd_used
