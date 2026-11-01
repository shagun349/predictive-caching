"""Synthetic workload generation for RQ2 robustness comparisons."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


GENE_KEYS = [
    "w_recency",
    "w_frequency",
    "w_freq_decay",
    "w_size",
    "w_priority",
    "w_penalty",
    "w_predicted_reuse",
    "theta_dram",
    "alpha_qos",
    "target_p",
]


def deterministic_priority(key: str, high_priority_ratio: float = 0.30) -> str:
    digest = abs(hash(key)) % 1000
    return "h" if (digest / 1000.0) < high_priority_ratio else "l"


def deterministic_response_time(key: str, priority: str, min_rtt: float = 0.01, max_rtt: float = 0.20) -> float:
    digest = abs(hash(key + priority)) % 10000
    ratio = digest / 10000.0
    return round(min_rtt + ratio * (max_rtt - min_rtt), 4)


def build_object_catalog(num_objects: int, seed: int = 42) -> Dict[str, int]:
    rng = random.Random(seed)
    catalog: Dict[str, int] = {}
    for i in range(1, num_objects + 1):
        catalog[f"obj_{i:06d}"] = rng.randint(64, 2048)
    return catalog


def zipf_probabilities(alpha: float, n: int) -> List[float]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    weights = [1.0 / (float(i) ** alpha) for i in range(1, n + 1)]
    total = sum(weights)
    return [w / total for w in weights]


def generate_zipf_trace(
    alpha: float,
    num_objects: int,
    num_requests: int,
    high_priority_ratio: float = 0.30,
    seed: int = 42,
    start_timestamp: float = 0.0,
) -> List[Dict[str, object]]:
    if num_objects <= 0 or num_requests <= 0:
        raise ValueError("num_objects and num_requests must be positive")

    rng = random.Random(seed)
    catalog = build_object_catalog(num_objects, seed=seed)
    probs = zipf_probabilities(alpha, num_objects)
    records: List[Dict[str, object]] = []

    for req_id in range(num_requests):
        key = rng.choices(list(catalog.keys()), weights=probs, k=1)[0]
        size = catalog[key]
        priority = deterministic_priority(key, high_priority_ratio)
        response_time = deterministic_response_time(key, priority)
        timestamp = start_timestamp + req_id * 0.01 + rng.random() * 0.005
        records.append(
            {
                "request_id": req_id,
                "timestamp": round(timestamp, 6),
                "key": key,
                "size": size,
                "priority": priority,
                "response_time": response_time,
                "operation": "get",
                "interest_lifetime": 4.0,
            }
        )

    return records


def generate_bursty_trace(
    alpha: float,
    num_objects: int,
    num_requests: int,
    burst_windows: int = 10,
    hot_fraction: float = 0.10,
    burst_multiplier: float = 5.0,
    high_priority_ratio: float = 0.30,
    seed: int = 42,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    catalog = build_object_catalog(num_objects, seed=seed)
    keys = list(catalog.keys())
    base_probs = zipf_probabilities(alpha, num_objects)
    records: List[Dict[str, object]] = []
    window_size = max(1, num_requests // max(1, burst_windows))

    for req_id in range(num_requests):
        window_index = min(burst_windows - 1, req_id // window_size)
        if rng.random() < (0.35 + 0.05 * (window_index % 3)):
            hot_count = max(1, int(len(keys) * hot_fraction))
            hot_keys = set(rng.sample(keys, hot_count))
            weights = []
            for key in keys:
                w = base_probs[keys.index(key)]
                if key in hot_keys and (window_index % 2) == 0:
                    w *= burst_multiplier
                weights.append(w)
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            key = rng.choices(keys, weights=probs, k=1)[0]
        else:
            key = rng.choices(keys, weights=base_probs, k=1)[0]

        size = catalog[key]
        priority = deterministic_priority(key, high_priority_ratio)
        response_time = deterministic_response_time(key, priority)
        timestamp = req_id * 0.01 + (window_index * 0.25)
        records.append(
            {
                "request_id": req_id,
                "timestamp": round(timestamp, 6),
                "key": key,
                "size": size,
                "priority": priority,
                "response_time": response_time,
                "operation": "get",
                "interest_lifetime": 4.0,
            }
        )

    return records


def generate_trending_trace(
    alpha: float,
    num_objects: int,
    num_requests: int,
    trend_windows: int = 8,
    hot_fraction: float = 0.08,
    trend_multiplier: float = 12.0,
    high_priority_ratio: float = 0.30,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Generate a workload with a temporary rise in access intensity to a small object set."""
    rng = random.Random(seed)
    catalog = build_object_catalog(num_objects, seed=seed)
    keys = list(catalog.keys())
    base_probs = zipf_probabilities(alpha, num_objects)
    records: List[Dict[str, object]] = []
    window_size = max(1, num_requests // max(1, trend_windows))

    for req_id in range(num_requests):
        window_index = min(trend_windows - 1, req_id // window_size)
        if req_id % window_size < (0.35 * window_size):
            hot_count = max(1, int(len(keys) * hot_fraction))
            hot_keys = set(rng.sample(keys, hot_count))
            weights = []
            for key in keys:
                w = base_probs[keys.index(key)]
                if key in hot_keys:
                    w *= trend_multiplier * (1.0 + 0.25 * window_index)
                weights.append(w)
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            key = rng.choices(keys, weights=probs, k=1)[0]
        else:
            key = rng.choices(keys, weights=base_probs, k=1)[0]

        size = catalog[key]
        priority = deterministic_priority(key, high_priority_ratio)
        response_time = deterministic_response_time(key, priority)
        timestamp = req_id * 0.01 + (window_index * 0.5)
        records.append(
            {
                "request_id": req_id,
                "timestamp": round(timestamp, 6),
                "key": key,
                "size": size,
                "priority": priority,
                "response_time": response_time,
                "operation": "get",
                "interest_lifetime": 4.0,
            }
        )

    return records


def partition_requests(trace: List[Dict[str, object]], train_ratio: float = 0.5, validation_ratio: float = 0.2) -> Dict[str, List[Dict[str, object]]]:
    total = len(trace)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)
    return {
        "train": trace[:train_end],
        "validation": trace[train_end:validation_end],
        "test": trace[validation_end:],
    }


def export_trace_csv(trace: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["request_id", "timestamp", "key", "size", "priority", "response_time", "operation", "interest_lifetime"])
        for record in trace:
            writer.writerow([
                record["request_id"],
                record["timestamp"],
                record["key"],
                record["size"],
                record["priority"],
                record["response_time"],
                record["operation"],
                record["interest_lifetime"],
            ])


def export_trace_json(trace: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)


def summarize_trace(trace: List[Dict[str, object]]) -> Dict[str, float]:
    total = len(trace)
    unique = len({record["key"] for record in trace})
    hpc = sum(1 for record in trace if record["priority"] == "h")
    return {
        "total_requests": total,
        "unique_objects": unique,
        "hpc_ratio": hpc / max(1, total),
        "lpc_ratio": 1.0 - (hpc / max(1, total)),
        "mean_object_size": float(np.mean([record["size"] for record in trace])),
        "catalog_bytes": int(sum(record["size"] for record in trace if record["request_id"] == 0)),
    }
