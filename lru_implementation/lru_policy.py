"""
Multi-Tier LRU (Least Recently Used) Policy
===========================================
Implements standard recency-based eviction across DRAM and SSD tiers.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional
from genetic_algorithm_implementation.baseline_policies import BasePolicy


class MultiTierLRUPolicy(BasePolicy):
    """
    Standard 2-tier LRU Policy:
    - DRAM is the primary high-speed tier.
    - SSD is the secondary high-capacity tier.
    - On DRAM eviction, the least recently used block is demoted to SSD.
    - On SSD eviction, the least recently used block in SSD is discarded.
    - On hit in SSD, the block is promoted back to DRAM.
    """
    def __init__(self):
        super().__init__("LRU")
        self.dram: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.ssd: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.dram_used = 0
        self.ssd_used = 0

    def reset(self, dram_max_bytes: int, ssd_max_bytes: int, slot_size: int = 4567):
        super().reset(dram_max_bytes, ssd_max_bytes, slot_size)
        self.dram.clear()
        self.ssd.clear()
        self.dram_used = 0
        self.ssd_used = 0

    def lookup(self, key: str) -> Optional[str]:
        if key in self.dram:
            return "DRAM"
        if key in self.ssd:
            return "SSD"
        return None

    def on_hit(self, key: str, tier: str, record: Dict[str, Any], current_time: float):
        if tier == "DRAM":
            self.dram.move_to_end(key)
        else:
            # Promote from SSD to DRAM
            rec = self.ssd.pop(key)
            self.ssd_used -= rec["size"]

            # Evict from DRAM to SSD if necessary
            while (self.dram_used + rec["size"] > self.dram_max_bytes) and self.dram:
                k_ev, r_ev = self.dram.popitem(last=False)
                self.dram_used -= r_ev["size"]

                # Demote evicted DRAM block to SSD
                while (self.ssd_used + r_ev["size"] > self.ssd_max_bytes) and self.ssd:
                    _, drop_r = self.ssd.popitem(last=False)
                    self.ssd_used -= drop_r["size"]

                self.ssd[k_ev] = r_ev
                self.ssd_used += r_ev["size"]

            self.dram[key] = rec
            self.dram_used += rec["size"]

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        key = record["key"]
        size = record["size"]
        evictions: List[Tuple[str, str]] = []

        # Make room in DRAM by demoting least recently used to SSD
        while (self.dram_used + size > self.dram_max_bytes) and self.dram:
            k_ev, r_ev = self.dram.popitem(last=False)
            self.dram_used -= r_ev["size"]

            while (self.ssd_used + r_ev["size"] > self.ssd_max_bytes) and self.ssd:
                k_drop, r_drop = self.ssd.popitem(last=False)
                self.ssd_used -= r_drop["size"]
                evictions.append(("SSD", k_drop))

            self.ssd[k_ev] = r_ev
            self.ssd_used += r_ev["size"]

        self.dram[key] = record
        self.dram_used += size
        return "DRAM", evictions

    def get_used_capacities(self) -> Tuple[int, int]:
        return self.dram_used, self.ssd_used
