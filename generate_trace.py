#!/usr/bin/env python3
"""
Generate a synthetic trace dataset for the Multi-Tier Cache Simulator.

Uses the paper's parameters with a larger dataset:
- 100 unique content items
- Zipf(α=0.8) popularity distribution
- Poisson arrival process (λ=5 requests/sec)
- 30% high-priority, 70% low-priority content
- Variable data sizes (200–10000 bytes)
- Response times (0.01s–0.3s), Interest lifetime = 10s
- Traffic period: 15 minutes (~4500 requests for larger dataset)
"""
import os
import sys

# Add the simulator to the Python path
SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "MultiTierCacheSimulator-main", "MultiTierCacheSimulator-main")
sys.path.insert(0, SIM_DIR)

from traces.trace_creating_and_parsing.synthetic_trace import TraceCreator

# Ensure the output directory exists
output_dir = os.path.join(SIM_DIR, "resources", "dataset_synthetic")
os.makedirs(output_dir, exist_ok=True)

# Change working directory so TraceCreator's relative '../resources/...' path resolves correctly
# TraceCreator opens '../resources/dataset_synthetic/...' so CWD must be traces/
trace_dir = os.path.join(SIM_DIR, "traces")
os.chdir(trace_dir)
print(f"Working directory: {os.getcwd()}")
print(f"Target output: {os.path.abspath('../resources/dataset_synthetic/')}")

print("=" * 60)
print("Generating synthetic trace dataset for base paper evaluation")
print("=" * 60)

# Paper parameters (with larger traffic period for more data)
creator = TraceCreator(
    n_unique_items=100,                    # 100 unique content items
    high_priority_content_percentage=0.3,  # 30% high priority
    pareto_alpha=1.5,                      # Pareto shape (unused, Zipf used instead)
    zipf_alpha=0.8,                        # Zipf popularity skew
    poisson_lambda=5.0,                    # ~5 requests per second
    min_data_size=200,                     # 200 bytes minimum
    max_data_size=10000,                   # 10000 bytes maximum
    min_data_rtt=0.01,                     # 10ms minimum RTT
    max_data_rtt=0.3,                      # 300ms maximum RTT
    interest_lifetime=10.0,               # 10 second interest lifetime
    traffic_period=15                      # 15 minutes of traffic
)

# Verify the trace was created
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    with open(fpath) as fp:
        line_count = sum(1 for _ in fp)
    print(f"  Created: {f} ({line_count} requests)")

print("\nDone! Trace dataset ready for simulation.")
