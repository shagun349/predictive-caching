import os
import sys
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from genetic_algorithm_implementation.trace_preprocessor import TracePreprocessor
from genetic_algorithm_implementation.cache_simulator import MultiTierCacheSimulator
from genetic_algorithm_implementation.baseline_policies import (
    LRUPolicy, LFUPolicy, RandPolicy
)
from genetic_algorithm_implementation.ga_predictive_policy import GAPredictivePolicy
from genetic_algorithm_implementation.ga_chromosome import Chromosome

from pso_implementation.pso_engine import PSOEngine
from pso_implementation.pso_predictive_policy import PSOPredictivePolicy

# Setup directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_bar_chart(policies: List[str], values: List[float], title: str, ylabel: str, filename: str, color: str = 'skyblue'):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(policies, values, color=color)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        # Format based on whether it's a percentage or raw number
        text_format = f"{yval:.2f}%" if "%" in ylabel or "Ratio" in title else f"{int(yval):,}"
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, text_format, va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
    plt.close()

def main():
    print("=" * 70)
    print("  PARTICLE SWARM OPTIMIZATION PREDICTIVE CACHING")
    print("  Dataset: 'Dataset/memcached-sample'")
    print("  Output Directory: pso_implementation/results/")
    print("=" * 70)

    # 1. Preprocess Dataset
    raw_dataset = os.path.join(PROJECT_ROOT, "Dataset", "memcached-sample")
    print(f"\n[1/5] Preprocessing dataset from: {raw_dataset} ...")
    preprocessor = TracePreprocessor(raw_dataset, high_priority_ratio=0.30, seed=42)
    records, stats = preprocessor.process()

    # 2. Configure Cache Sizes
    unique_key_sizes = {}
    for r in records:
        unique_key_sizes[r["key"]] = r["size"]
    total_catalog_bytes = sum(unique_key_sizes.values())
    total_cache_size = int(total_catalog_bytes * 0.10)  # 10% catalog
    dram_fraction = 0.20
    mean_obj_size = int(stats["mean_object_size"])

    simulator = MultiTierCacheSimulator(
        total_cache_size=total_cache_size,
        dram_fraction=dram_fraction,
        ssd_throughput_gb=2.0,
        slot_size=mean_obj_size
    )

    # 3. Train PSO
    train_size = min(5000, len(records))
    training_trace = records[:train_size]

    print(f"\n[3/5] Training PSO Engine on {train_size} requests...")
    pso_engine = PSOEngine(num_particles=30, iterations=25, seed=42)
    
    t0 = time.time()
    best_particle = pso_engine.optimize(simulator, training_trace, verbose=True)
    print(f"PSO Training completed in {time.time() - t0:.2f}s")
    
    # Save PSO history
    history_file = os.path.join(RESULTS_DIR, "pso_training_history.json")
    with open(history_file, "w") as f:
        json.dump(pso_engine.history, f, indent=2)

    # 4. Evaluate on unseen data
    test_trace = records[train_size:]
    print(f"\n[4/5] Evaluating policies on {len(test_trace)} UNSEEN test requests...")
    
    # Load GA Best Genes for fair comparison
    ga_best_genes = {
        "w_recency": 0.5478, "w_frequency": 0.4451, "w_freq_decay": 0.7075,
        "w_size": 0.0614, "w_priority": 2.1160, "w_penalty": 1.2894,
        "w_predicted_reuse": -0.8840, "theta_dram": 0.8938,
        "alpha_qos": 0.7815, "target_p": 0.7199
    }
    ga_chromosome = Chromosome()
    ga_chromosome.gene = list(ga_best_genes.values()) # order matters based on ga_chromosome.py keys

    policies = {
        "PSO-Predictive": PSOPredictivePolicy(best_particle),
        "GA-Predictive": GAPredictivePolicy(ga_chromosome),
        "LRU": LRUPolicy(),
        "LFU": LFUPolicy(),
        "Random": RandPolicy()
    }

    results = {}
    for pol_name, pol_inst in policies.items():
        res = simulator.run(pol_inst, test_trace)
        results[pol_name] = res
        print(f"  [{pol_name:14s}] Total CHR: {res['chr_total']*100:5.2f}% | HPC CHR: {res['chr_hpc']*100:5.2f}% | Penalty: {res['penalty_total']:7d}")

    # 5. Save and Plot
    print(f"\n[5/5] Generating figures and results...")
    
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    names = list(results.keys())
    
    # Total CHR
    generate_bar_chart(
        names, [r['chr_total']*100 for r in results.values()],
        "Total Cache Hit Ratio (CHR)", "Hit Ratio (%)", "1_total_chr.png", "steelblue"
    )
    
    # HPC CHR
    generate_bar_chart(
        names, [r['chr_hpc']*100 for r in results.values()],
        "High Priority (HPC) Cache Hit Ratio", "Hit Ratio (%)", "2_hpc_chr.png", "darkorange"
    )
    
    # LPC CHR
    generate_bar_chart(
        names, [r['chr_lpc']*100 for r in results.values()],
        "Low Priority (LPC) Cache Hit Ratio", "Hit Ratio (%)", "3_lpc_chr.png", "mediumseagreen"
    )
    
    # Total Penalty
    generate_bar_chart(
        names, [r['penalty_total'] for r in results.values()],
        "Total SLA Penalty Cost", "Penalty Cost", "4_total_penalty.png", "crimson"
    )
    
    print(f"Done! Results and figures saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
