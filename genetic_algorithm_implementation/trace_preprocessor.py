"""
Trace Preprocessor for Memcached Dataset
========================================
Reads the raw memcached trace from 'Dataset/memcached-sample' and converts it into
a structured NDN/multi-tier caching trace with QoS priority tags, realistic timestamps,
content sizes, and network response times.

Format:
  timestamp, key, size, priority ('h'/'l'), response_time, operation
"""

import os
import hashlib
import random
from typing import List, Dict, Tuple, Any


class TracePreprocessor:
    def __init__(
        self,
        raw_dataset_path: str,
        high_priority_ratio: float = 0.30,
        req_rate_per_sec: float = 50.0,
        min_rtt: float = 0.01,
        max_rtt: float = 0.20,
        seed: int = 42
    ):
        self.raw_dataset_path = raw_dataset_path
        self.high_priority_ratio = high_priority_ratio
        self.req_rate_per_sec = req_rate_per_sec
        self.min_rtt = min_rtt
        self.max_rtt = max_rtt
        self.seed = seed
        self.rng = random.Random(seed)

    def _determine_priority(self, key: str) -> str:
        """Deterministically assign priority based on key hash for reproducibility."""
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
        return "h" if (h / 1000.0) < self.high_priority_ratio else "l"

    def _determine_rtt(self, key: str) -> float:
        """Deterministically assign origin response time (RTT) per key."""
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10000
        ratio = h / 10000.0
        return round(self.min_rtt + ratio * (self.max_rtt - self.min_rtt), 4)

    def process(self, max_records: int = -1) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parses raw memcached trace and returns list of request records and dataset statistics.
        """
        if not os.path.exists(self.raw_dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.raw_dataset_path}")

        records = []
        key_frequencies: Dict[str, int] = {}
        unique_keys = set()
        current_time = 0.0

        with open(self.raw_dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue

                raw_ts, key, key_size_s, val_size_s, client_id, op, ttl_s = parts[:7]

                key_size = int(key_size_s) if key_size_s.isdigit() else 16
                val_size = int(val_size_s) if val_size_s.isdigit() else 256
                # Object size is value_size or fallback to minimum 64 bytes
                obj_size = max(val_size, key_size, 64)

                priority = self._determine_priority(key)
                rtt = self._determine_rtt(key)

                # Synthetic arrival time spacing: Poisson inter-arrival
                # E[inter_arrival] = 1 / req_rate
                inter_arrival = self.rng.expovariate(self.req_rate_per_sec)
                current_time += inter_arrival

                record = {
                    "request_id": len(records),
                    "timestamp": round(current_time, 5),
                    "key": key,
                    "size": obj_size,
                    "priority": priority,
                    "response_time": rtt,
                    "operation": op.lower(),
                    "interest_lifetime": 4.0
                }
                records.append(record)

                unique_keys.add(key)
                key_frequencies[key] = key_frequencies.get(key, 0) + 1

                if 0 < max_records <= len(records):
                    break

        # Compute summary statistics
        total_requests = len(records)
        hpc_requests = sum(1 for r in records if r["priority"] == "h")
        lpc_requests = total_requests - hpc_requests

        hpc_keys = sum(1 for k in unique_keys if self._determine_priority(k) == "h")
        lpc_keys = len(unique_keys) - hpc_keys

        sizes = [r["size"] for r in records]
        slot_size = max(sizes) if sizes else 4096
        catalog_bytes = sum(max(int(parts[3]) if parts[3].isdigit() else 256, 64) for parts in [r["key"] for r in records])

        stats = {
            "total_requests": total_requests,
            "unique_objects": len(unique_keys),
            "hpc_requests": hpc_requests,
            "lpc_requests": lpc_requests,
            "hpc_ratio": hpc_requests / max(total_requests, 1),
            "lpc_ratio": lpc_requests / max(total_requests, 1),
            "hpc_unique_keys": hpc_keys,
            "lpc_unique_keys": lpc_keys,
            "min_object_size": min(sizes) if sizes else 0,
            "max_object_size": max(sizes) if sizes else 0,
            "mean_object_size": sum(sizes) / len(sizes) if sizes else 0,
            "slot_size": slot_size,
            "simulation_duration_sec": current_time,
            "read_operations": sum(1 for r in records if r["operation"] in ("get", "gets")),
            "write_operations": sum(1 for r in records if r["operation"] not in ("get", "gets"))
        }

        return records, stats

    def save_ndn_csv(self, records: List[Dict[str, Any]], output_path: str):
        """Save records in NDN-compatible CSV format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                # Format: data_back, timestamp, name, size, priority, InterestLifetime, responseTime
                f.write(f"d,{r['timestamp']},{r['key']},{r['size']},{r['priority']},{r['interest_lifetime']},{r['response_time']}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = os.path.join(base_dir, "Dataset", "memcached-sample")
    preprocessor = TracePreprocessor(dataset_file)
    records, stats = preprocessor.process()
    print("Trace Preprocessor Test:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
