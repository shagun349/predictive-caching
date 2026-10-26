"""
Multi-Tier LFU (Least Frequently Used) Policy
=============================================
Tracks item reference counts and evicts items with the lowest access frequency.
"""

from typing import Dict, List, Tuple, Any, Optional
from genetic_algorithm_implementation.baseline_policies import BasePolicy


class MultiTierLFUPolicy(BasePolicy):
    """
    Standard 2-tier LFU Policy:
    - Tracks frequency of accesses across all requested keys.
    - DRAM evicts item with min frequency down to SSD.
    - SSD evicts item with min frequency.
    - On SSD hit, frequency increments and item is promoted to DRAM.
    """
    def __init__(self):
        super().__init__("LFU")
        self.dram: Dict[str, Dict[str, Any]] = {}
        self.ssd: Dict[str, Dict[str, Any]] = {}
        self.freq: Dict[str, int] = {}
        self.dram_used = 0
        self.ssd_used = 0

    def reset(self, dram_max_bytes: int, ssd_max_bytes: int, slot_size: int = 4567):
        super().reset(dram_max_bytes, ssd_max_bytes, slot_size)
        self.dram.clear()
        self.ssd.clear()
        self.freq.clear()
        self.dram_used = 0
        self.ssd_used = 0

    def lookup(self, key: str) -> Optional[str]:
        if key in self.dram:
            return "DRAM"
        if key in self.ssd:
            return "SSD"
        return None

    def on_hit(self, key: str, tier: str, record: Dict[str, Any], current_time: float):
        self.freq[key] = self.freq.get(key, 0) + 1
        if tier == "SSD":
            rec = self.ssd.pop(key)
            self.ssd_used -= rec["size"]

            while (self.dram_used + rec["size"] > self.dram_max_bytes) and self.dram:
                # Evict lowest frequency item from DRAM to SSD
                lfu_k = min(self.dram.keys(), key=lambda k: self.freq.get(k, 0))
                lfu_r = self.dram.pop(lfu_k)
                self.dram_used -= lfu_r["size"]

                while (self.ssd_used + lfu_r["size"] > self.ssd_max_bytes) and self.ssd:
                    ssd_lfu_k = min(self.ssd.keys(), key=lambda k: self.freq.get(k, 0))
                    ssd_drop = self.ssd.pop(ssd_lfu_k)
                    self.ssd_used -= ssd_drop["size"]

                self.ssd[lfu_k] = lfu_r
                self.ssd_used += lfu_r["size"]

            self.dram[key] = rec
            self.dram_used += rec["size"]

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        key = record["key"]
        size = record["size"]
        self.freq[key] = self.freq.get(key, 0) + 1
        evictions: List[Tuple[str, str]] = []

        while (self.dram_used + size > self.dram_max_bytes) and self.dram:
            lfu_k = min(self.dram.keys(), key=lambda k: self.freq.get(k, 0))
            lfu_r = self.dram.pop(lfu_k)
            self.dram_used -= lfu_r["size"]

            while (self.ssd_used + lfu_r["size"] > self.ssd_max_bytes) and self.ssd:
                ssd_lfu_k = min(self.ssd.keys(), key=lambda k: self.freq.get(k, 0))
                ssd_drop = self.ssd.pop(ssd_lfu_k)
                self.ssd_used -= ssd_drop["size"]
                evictions.append(("SSD", ssd_lfu_k))

            self.ssd[lfu_k] = lfu_r
            self.ssd_used += lfu_r["size"]

        self.dram[key] = record
        self.dram_used += size
        return "DRAM", evictions

    def get_used_capacities(self) -> Tuple[int, int]:
        return self.dram_used, self.ssd_used
