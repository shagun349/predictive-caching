"""
Whale Optimization Algorithm (WOA) Predictive Caching Policy
============================================================
Uses multi-factor predictive utility scores calculated with weights evolved
by the Whale Optimization Algorithm (WOA).
"""

import math
import random
from typing import Dict, List, Tuple, Any, Optional
from genetic_algorithm_implementation.cache_simulator import CacheItem
from woa_implementation.woa_whale import Whale


class WOAPredictivePolicy:
    """
    Multi-Tier predictive caching policy driven by WOA-evolved weights.
    """

    def __init__(self, whale: Optional[Whale] = None, tournament_sample: int = 24):
        self.whale = whale if whale is not None else Whale()
        self.weights = self.whale.position.copy()
        self.tournament_sample = tournament_sample

        self.dram_max_bytes = 0
        self.ssd_max_bytes = 0
        self.slot_size = 255

        # Storage dictionaries: key -> CacheItem
        self.dram_items: Dict[str, CacheItem] = {}
        self.ssd_items: Dict[str, CacheItem] = {}

        # Running used bytes
        self.dram_used_bytes = 0
        self.ssd_used_bytes = 0

    def reset(self, dram_max_bytes: int, ssd_max_bytes: int, slot_size: int = 255):
        """Reset internal state for a fresh simulation run."""
        self.dram_max_bytes = dram_max_bytes
        self.ssd_max_bytes = ssd_max_bytes
        self.slot_size = slot_size
        self.dram_items.clear()
        self.ssd_items.clear()
        self.dram_used_bytes = 0
        self.ssd_used_bytes = 0
        self.weights = self.whale.position.copy()

    def set_whale(self, whale: Whale):
        """Assign a new whale agent and refresh active weights."""
        self.whale = whale
        self.weights = whale.position.copy()

    def lookup(self, key: str) -> Optional[str]:
        """Check if key exists in Content Store and return tier ('DRAM' or 'SSD')."""
        if key in self.dram_items:
            return "DRAM"
        if key in self.ssd_items:
            return "SSD"
        return None

    def compute_utility(self, item: CacheItem, current_time: float) -> float:
        """
        Compute real-time predictive utility score using evolved WOA weights.
        Priority boost modulates recency so active HPC is preserved against eviction,
        while stale HPC smoothly ages out.
        """
        delta_t = max(0.0, current_time - item.last_access_time)

        # 1. Recency feature: temporal decay
        decay_rate = max(0.001, self.weights["w_freq_decay"])
        f_rec = 1.0 / (1.0 + decay_rate * delta_t)

        # 2. Frequency feature (log scale)
        f_freq = math.log1p(item.access_count)

        # 3. QoS Priority feature (HPC vs LPC)
        f_prio = 1.0 if item.priority == "h" else 0.0

        # 4. Miss penalty weight
        f_pen = (75.0 / 75.0) if item.priority == "h" else (15.0 / 75.0)

        # 5. Size penalty factor
        f_size = item.size / max(self.slot_size, 1)

        # 6. Predictive reuse likelihood based on inter-arrival history
        if item.inter_arrivals:
            mean_ia = sum(item.inter_arrivals) / len(item.inter_arrivals)
            f_reuse = math.exp(-delta_t / max(mean_ia, 0.005))
        else:
            f_reuse = 0.5 if item.access_count > 1 else 0.2

        # Combined core value (modulated by recency so stale items decay)
        core_val = (
            self.weights["w_recency"] +
            self.weights["w_frequency"] * f_freq +
            self.weights["w_priority"] * f_prio +
            self.weights["w_penalty"] * f_pen +
            self.weights["w_predicted_reuse"] * f_reuse
        )

        utility = (f_rec * core_val) - (self.weights["w_size"] * f_size)
        return utility

    def on_hit(self, key: str, tier: str, record: Dict[str, Any], current_time: float):
        """Update item access statistics and consider promotion from SSD to DRAM."""
        if tier == "DRAM":
            self.dram_items[key].record_access(current_time)
        else:  # "SSD"
            item = self.ssd_items[key]
            item.record_access(current_time)

            # Consider promoting to DRAM if utility meets threshold or priority is high
            u = self.compute_utility(item, current_time)
            promote = (item.priority == "h") or (u >= self.weights["theta_dram"])

            if promote:
                del self.ssd_items[key]
                self.ssd_used_bytes -= item.size

                self._ensure_dram_capacity(item.size, current_time)
                item.tier = "DRAM"
                self.dram_items[key] = item
                self.dram_used_bytes += item.size

    def on_miss(self, record: Dict[str, Any], current_time: float) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Admit incoming data packet into DRAM or SSD based on predictive scoring.
        Returns (destination_tier, list_of_evictions).
        """
        key = record["key"]
        size = record["size"]
        priority = record["priority"]
        evictions: List[Tuple[str, str]] = []

        new_item = CacheItem(key, size, priority, current_time)
        u = self.compute_utility(new_item, current_time)

        # QoS-aware placement decision
        admit_to_dram = (priority == "h" and u >= (self.weights["theta_dram"] * 0.5)) or (u >= self.weights["theta_dram"])

        if admit_to_dram:
            dest_tier = "DRAM"
            self._ensure_dram_capacity(size, current_time)
            new_item.tier = "DRAM"
            self.dram_items[key] = new_item
            self.dram_used_bytes += size
        else:
            dest_tier = "SSD"
            self._ensure_ssd_capacity(size, current_time)
            new_item.tier = "SSD"
            self.ssd_items[key] = new_item
            self.ssd_used_bytes += size

        return dest_tier, evictions

    def _ensure_dram_capacity(self, needed_bytes: int, current_time: float):
        """Demote lowest utility DRAM items to SSD until enough space is available."""
        while (self.dram_used_bytes + needed_bytes > self.dram_max_bytes) and self.dram_items:
            keys = list(self.dram_items.keys())
            sample_k = min(self.tournament_sample, len(keys))
            sample = random.sample(keys, sample_k)
            worst_key = min(sample, key=lambda k: self.compute_utility(self.dram_items[k], current_time))

            demoted = self.dram_items.pop(worst_key)
            self.dram_used_bytes -= demoted.size

            # Cascade demotion into SSD
            self._ensure_ssd_capacity(demoted.size, current_time)
            demoted.tier = "SSD"
            self.ssd_items[worst_key] = demoted
            self.ssd_used_bytes += demoted.size

    def _ensure_ssd_capacity(self, needed_bytes: int, current_time: float):
        """Evict lowest utility SSD items until enough space is available."""
        while (self.ssd_used_bytes + needed_bytes > self.ssd_max_bytes) and self.ssd_items:
            keys = list(self.ssd_items.keys())
            sample_k = min(self.tournament_sample, len(keys))
            sample = random.sample(keys, sample_k)
            worst_key = min(sample, key=lambda k: self.compute_utility(self.ssd_items[k], current_time))

            evicted = self.ssd_items.pop(worst_key)
            self.ssd_used_bytes -= evicted.size

    def get_used_capacities(self) -> Tuple[int, int]:
        """Return (dram_used_bytes, ssd_used_bytes)."""
        return self.dram_used_bytes, self.ssd_used_bytes
