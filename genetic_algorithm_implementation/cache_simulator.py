"""
Multi-Tier Cache Simulator Engine
=================================
Faithfully simulates a two-tier Content Store (DRAM + SSD) with Pending Interest Table (PIT)
aggregation, realistic hardware latencies/throughputs, and QoS SLA miss penalty modeling
matching the research paper:
  "QM-ARC: QoS-aware Multi-tier Adaptive Cache Replacement Strategy"
"""

import math
from typing import Dict, List, Tuple, Any, Optional


def get_penalty(response_time: float, priority: str) -> int:
    """Exact penalty function matching research paper common/penalty.py."""
    if priority == "h":
        if response_time == 0.0:
            return 5  # high_prio_disk_penalty
        if response_time < 0.02:
            return 0
        if response_time < 0.15:
            return 50
        else:
            return 75
    elif priority == "l":
        if response_time == 0.0:
            return 0  # low_prio_disk_penalty
        if response_time < 0.02:
            return 0
        if response_time < 0.15:
            return 10
        else:
            return 15
    return 0


class CacheItem:
    """Metadata tracked per cached item."""
    def __init__(self, key: str, size: int, priority: str, timestamp: float):
        self.key = key
        self.size = size
        self.priority = priority
        self.first_access_time = timestamp
        self.last_access_time = timestamp
        self.access_count = 1
        self.tier = "DRAM"  # "DRAM" or "SSD"
        self.inter_arrivals: List[float] = []

    def record_access(self, current_time: float):
        delta = current_time - self.last_access_time
        if delta > 0:
            self.inter_arrivals.append(delta)
            if len(self.inter_arrivals) > 10:
                self.inter_arrivals.pop(0)
        self.last_access_time = current_time
        self.access_count += 1


class MultiTierCacheSimulator:
    """
    Simulates DRAM + SSD multi-tier storage with NDN packet semantics.
    """

    def __init__(
        self,
        total_cache_size: int,
        dram_fraction: float = 0.20,
        ssd_throughput_gb: float = 2.0,
        slot_size: int = 4567
    ):
        self.total_cache_size = total_cache_size
        self.dram_fraction = dram_fraction
        self.slot_size = slot_size

        # Tier capacities
        self.dram_max_size = int(total_cache_size * dram_fraction)
        self.ssd_max_size = int(total_cache_size * (1.0 - dram_fraction))

        # Hardware parameters (paper values)
        # DRAM: 100ns latency, 40 GB/s read, 20 GB/s write
        self.dram_latency = 1e-7
        self.dram_read_bw = 40.0 * 1e9
        self.dram_write_bw = 20.0 * 1e9

        # SSD: 10μs latency, j GB/s read, j/2 GB/s write
        self.ssd_latency = 1e-5
        self.ssd_read_bw = ssd_throughput_gb * 1e9
        self.ssd_write_bw = (ssd_throughput_gb / 2.0) * 1e9

    def run(self, policy: Any, trace_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute simulation for given policy and trace.
        Returns comprehensive performance metrics.
        """
        policy.reset(self.dram_max_size, self.ssd_max_size, self.slot_size)

        # Counters
        hits_dram = 0
        hits_ssd = 0
        hits_hpc_dram = 0
        hits_hpc_ssd = 0
        hits_lpc_dram = 0
        hits_lpc_ssd = 0

        misses = 0
        misses_hpc = 0
        misses_lpc = 0

        penalty_hpc = 0
        penalty_lpc = 0

        reads_dram = 0
        reads_ssd = 0
        writes_dram = 0
        writes_ssd = 0

        high_p_retrieval_time = 0.0
        low_p_retrieval_time = 0.0

        total_requests = len(trace_records)
        nb_hpc = sum(1 for r in trace_records if r["priority"] == "h")
        nb_lpc = total_requests - nb_hpc

        # PIT: maps key -> expiration_time
        pit: Dict[str, float] = {}
        pit_aggregations = 0

        for record in trace_records:
            now = record["timestamp"]
            key = record["key"]
            size = record["size"]
            priority = record["priority"]
            rtt = record["response_time"]
            lifetime = record["interest_lifetime"]

            # Expire old PIT entries
            expired_keys = [k for k, exp in pit.items() if exp < now]
            for exp_k in expired_keys:
                del pit[exp_k]

            # 1. Check if packet is in Content Store
            tier = policy.lookup(key)

            if tier is not None:
                # --- Cache Hit ---
                if tier == "DRAM":
                    hits_dram += 1
                    reads_dram += 1
                    writes_dram += 1  # Hit write/update overhead
                    latency = self.dram_latency + (size / self.dram_read_bw)
                    if priority == "h":
                        hits_hpc_dram += 1
                    else:
                        hits_lpc_dram += 1
                else:  # "SSD"
                    hits_ssd += 1
                    reads_ssd += 1
                    writes_ssd += 1
                    # Accessing from SSD incurs disk penalty for HPC in paper (high_prio_disk_penalty = 5)
                    if priority == "h":
                        hits_hpc_ssd += 1
                        penalty_hpc += get_penalty(0.0, "h")
                    else:
                        hits_lpc_ssd += 1
                        penalty_lpc += get_penalty(0.0, "l")
                    latency = self.ssd_latency + (size / self.ssd_read_bw)

                policy.on_hit(key, tier, record, now)

                if priority == "h":
                    high_p_retrieval_time += latency
                else:
                    low_p_retrieval_time += latency

            else:
                # --- Cache Miss ---
                misses += 1
                if priority == "h":
                    misses_hpc += 1
                else:
                    misses_lpc += 1

                # Check PIT for aggregation
                if key in pit:
                    # PIT Hit: aggregate interest
                    pit_aggregations += 1
                    # Remaining time penalty
                    rem_time = max(0.001, rtt * 0.5)
                    if priority == "h":
                        penalty_hpc += get_penalty(rem_time, "h")
                        high_p_retrieval_time += rem_time
                    else:
                        penalty_lpc += get_penalty(rem_time, "l")
                        low_p_retrieval_time += rem_time
                    pit[key] = now + lifetime
                else:
                    # PIT Miss: forward Interest to upstream provider
                    pit[key] = now + lifetime
                    if priority == "h":
                        penalty_hpc += get_penalty(rtt, "h")
                        high_p_retrieval_time += rtt
                    else:
                        penalty_lpc += get_penalty(rtt, "l")
                        low_p_retrieval_time += rtt

                    # When data returns (after RTT), policy decides admission and tier placement
                    dest_tier, evictions = policy.on_miss(record, now)
                    if dest_tier == "DRAM":
                        writes_dram += 1
                    elif dest_tier == "SSD":
                        writes_ssd += 1

                    # Count migration/demotion writes from evictions
                    for ev_tier, ev_key in evictions:
                        if ev_tier == "SSD":
                            writes_ssd += 1

        # Calculate final normalized metrics
        chr_dram = hits_dram / max(total_requests, 1)
        chr_ssd = hits_ssd / max(total_requests, 1)
        chr_total = (hits_dram + hits_ssd) / max(total_requests, 1)

        chr_hpc = (hits_hpc_dram + hits_hpc_ssd) / max(nb_hpc, 1)
        chr_hpc_dram = hits_hpc_dram / max(nb_hpc, 1)
        chr_hpc_ssd = hits_hpc_ssd / max(nb_hpc, 1)

        chr_lpc = (hits_lpc_dram + hits_lpc_ssd) / max(nb_lpc, 1)
        chr_lpc_dram = hits_lpc_dram / max(nb_lpc, 1)
        chr_lpc_ssd = hits_lpc_ssd / max(nb_lpc, 1)

        avg_hpc_retrieval = high_p_retrieval_time / max(nb_hpc, 1)
        avg_lpc_retrieval = low_p_retrieval_time / max(nb_lpc, 1)
        avg_retrieval = (high_p_retrieval_time + low_p_retrieval_time) / max(total_requests, 1)

        dram_used, ssd_used = policy.get_used_capacities()

        return {
            "total_requests": total_requests,
            "nb_hpc": nb_hpc,
            "nb_lpc": nb_lpc,
            "chr_total": chr_total,
            "chr_dram": chr_dram,
            "chr_ssd": chr_ssd,
            "chr_hpc": chr_hpc,
            "chr_hpc_dram": chr_hpc_dram,
            "chr_hpc_ssd": chr_hpc_ssd,
            "chr_lpc": chr_lpc,
            "chr_lpc_dram": chr_lpc_dram,
            "chr_lpc_ssd": chr_lpc_ssd,
            "penalty_hpc": penalty_hpc,
            "penalty_lpc": penalty_lpc,
            "penalty_total": penalty_hpc + penalty_lpc,
            "avg_hpc_retrieval_time": avg_hpc_retrieval,
            "avg_lpc_retrieval_time": avg_lpc_retrieval,
            "avg_retrieval_time": avg_retrieval,
            "reads_dram": reads_dram,
            "reads_ssd": reads_ssd,
            "writes_dram": writes_dram,
            "writes_ssd": writes_ssd,
            "used_size_dram": dram_used,
            "used_size_ssd": ssd_used,
            "pit_aggregations": pit_aggregations
        }
