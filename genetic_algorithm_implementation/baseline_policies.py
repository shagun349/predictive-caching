"""
Baseline Multi-Tier Caching Policies
====================================
Faithful implementations of the standard baseline caching policies:
  1. LRU         - Least Recently Used
  2. LFU         - Least Frequently Used
  3. Random      - Random Eviction
"""

from collections import OrderedDict
import random
from typing import Dict, List, Tuple, Any, Optional


class BasePolicy:
    def __init__(self, name: str):
        self.name = name
        self.dram_max_bytes = 0
        self.ssd_max_bytes = 0
        self.slot_size = 4567

    def reset(self, dram_max_bytes: int, ssd_max_bytes: int, slot_size: int = 4567):
        self.dram_max_bytes = dram_max_bytes
        self.ssd_max_bytes = ssd_max_bytes
        self.slot_size = slot_size

    def lookup(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def on_hit(self, key: str, tier: str, record: Dict[str, Any], current_time: float):
        raise NotImplementedError

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        raise NotImplementedError

    def get_used_capacities(self) -> Tuple[int, int]:
        raise NotImplementedError


# ==============================================================================
# 1. LRU Policy
# ==============================================================================
class LRUPolicy(BasePolicy):
    def __init__(self):
        super().__init__("LRU")
        self.dram = OrderedDict()  # key -> record
        self.ssd = OrderedDict()
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
            # Promote SSD to DRAM
            rec = self.ssd.pop(key)
            self.ssd_used -= rec["size"]

            # Cascade evictions from DRAM to SSD
            while (self.dram_used + rec["size"] > self.dram_max_bytes) and self.dram:
                k_ev, r_ev = self.dram.popitem(last=False)
                self.dram_used -= r_ev["size"]
                # Insert evicted into SSD
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
        evictions = []

        # Make room in DRAM, demote to SSD
        while (self.dram_used + size > self.dram_max_bytes) and self.dram:
            k_ev, r_ev = self.dram.popitem(last=False)
            self.dram_used -= r_ev["size"]

            # Demote to SSD
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


# ==============================================================================
# 2. LFU Policy
# ==============================================================================
class LFUPolicy(BasePolicy):
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
                # Evict lowest frequency from DRAM to SSD
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
        evictions = []

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


# ==============================================================================
# 3. Random Policy
# ==============================================================================
class RandPolicy(BasePolicy):
    def __init__(self):
        super().__init__("Random")
        self.dram: Dict[str, Dict[str, Any]] = {}
        self.ssd: Dict[str, Dict[str, Any]] = {}
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
        pass

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        key = record["key"]
        size = record["size"]
        evictions = []

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
