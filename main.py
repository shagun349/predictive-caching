import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# --- Baseline Standard ARC Implementation ---
class ARC:
    def __init__(self, capacity):
        self.c = capacity
        self.p = 0  # Dynamic target size for T1
        self.T1, self.T2 = [], []
        self.B1, self.B2 = [], []
        self.hits = 0
        self.misses = 0
        self.p_history = []  # Track p adaptation over time

    def _replace(self, key):
        if self.T1 and (len(self.T1) > self.p or (key in self.B2 and len(self.T1) == self.p)):
            evicted = self.T1.pop()  # Evict LRU from T1 to B1
            self.B1.insert(0, evicted)
            if len(self.B1) > self.c:
                self.B1.pop()
        else:
            if self.T2:
                evicted = self.T2.pop()  # Evict LRU from T2 to B2
                self.B2.insert(0, evicted)
                if len(self.B2) > self.c:
                    self.B2.pop()

    def access(self, key):
        # Case 1: Cache Hit in T1 or T2
        if key in self.T1:
            self.T1.remove(key)
            self.T2.insert(0, key)
            self.hits += 1
            self.p_history.append(self.p)
            return True
        elif key in self.T2:
            self.T2.remove(key)
            self.T2.insert(0, key)
            self.hits += 1
            self.p_history.append(self.p)
            return True

        self.misses += 1

        # Case 2: Hit in Ghost List B1 (Increase recency target p)
        if key in self.B1:
            step = 1 if len(self.B1) == 0 else max(1, len(self.B2) / len(self.B1))
            self.p = min(self.c, self.p + step)
            self.B1.remove(key)
            self._replace(key)
            self.T2.insert(0, key)

        # Case 3: Hit in Ghost List B2 (Decrease recency target p)
        elif key in self.B2:
            step = 1 if len(self.B2) == 0 else max(1, len(self.B1) / len(self.B2))
            self.p = max(0, self.p - step)
            self.B2.remove(key)
            self._replace(key)
            self.T2.insert(0, key)

        # Case 4: Complete Miss
        else:
            if len(self.T1) + len(self.B1) == self.c:
                if len(self.T1) < self.c:
                    if self.B1: self.B1.pop()
                    self._replace(key)
                else:
                    if self.T1: self.T1.pop()
            else:
                total = len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2)
                if total >= self.c:
                    if total == 2 * self.c and self.B2:
                        self.B2.pop()
                    self._replace(key)
            self.T1.insert(0, key)

        self.p_history.append(self.p)
        return False

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# --- Baseline LRU Implementation ---
class LRU:
    def __init__(self, capacity):
        self.c = capacity
        self.cache = []
        self.hits = 0
        self.misses = 0

    def access(self, key):
        if key in self.cache:
            self.cache.remove(key)
            self.cache.insert(0, key)
            self.hits += 1
            return True
        else:
            self.misses += 1
            if len(self.cache) >= self.c:
                self.cache.pop()
            self.cache.insert(0, key)
            return False

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# --- Load Dataset from MultiTierCacheSimulator CSV ---
def load_dataset(csv_path):
    """
    Load a CSV dataset from MultiTierCacheSimulator.
    CSV format: data_back, timestamp, name(object_id), size, priority, InterestLifetime, responseTime
    Returns the list of object name/id accesses as strings.
    """
    accesses = []
    with open(csv_path, encoding='utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) >= 3:
                accesses.append(row[2].strip())  # column index 2 = object name/id
    return accesses


def augment_workload(base_accesses, target_length=5000, seed=42):
    """
    Augment a small dataset by repeating and shuffling access patterns
    with Zipf-like frequency bias to create a realistic workload.
    """
    np.random.seed(seed)
    unique_keys = list(set(base_accesses))
    num_keys = len(unique_keys)

    # Build frequency distribution from the original data
    freq = {}
    for k in base_accesses:
        freq[k] = freq.get(k, 0) + 1

    # Create probability weights biased toward frequent items
    weights = np.array([freq.get(k, 1) for k in unique_keys], dtype=float)
    weights = weights ** 1.5  # amplify frequency skew
    weights /= weights.sum()

    # Generate augmented workload: mix original pattern repeats with frequency-biased random accesses
    augmented = []
    # Repeat original trace several times
    repeats = max(1, target_length // (len(base_accesses) * 3))
    for _ in range(repeats):
        augmented.extend(base_accesses)
        # Insert random accesses between repeats
        rand_count = len(base_accesses) * 2
        rand_indices = np.random.choice(len(unique_keys), size=rand_count, p=weights)
        augmented.extend([unique_keys[i] for i in rand_indices])

    # Trim or pad to target length
    if len(augmented) >= target_length:
        augmented = augmented[:target_length]
    else:
        remaining = target_length - len(augmented)
        rand_indices = np.random.choice(len(unique_keys), size=remaining, p=weights)
        augmented.extend([unique_keys[i] for i in rand_indices])

    return augmented


# --- Benchmarking and Plotting ---
def run_experiments():
    # Paths to the datasets inside MultiTierCacheSimulator
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(base_dir, "MultiTierCacheSimulator-main", "MultiTierCacheSimulator-main", "resources")

    dataset_paths = {
        "lru_better (datasets)": os.path.join(sim_dir, "datasets", "lru_better.csv"),
        "lfu_better (easy)": os.path.join(sim_dir, "easy", "lfu_better.csv"),
    }

    # Verify datasets exist
    for name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"ERROR: Dataset not found: {path}")
            return
        print(f"Found dataset: {name} -> {path}")

    # Load and augment datasets
    workloads = {}
    for name, path in dataset_paths.items():
        raw = load_dataset(path)
        print(f"  {name}: {len(raw)} raw accesses, {len(set(raw))} unique objects")
        augmented = augment_workload(raw, target_length=8000)
        workloads[name] = augmented
        print(f"  Augmented to {len(augmented)} accesses, {len(set(augmented))} unique objects")

    capacities = [2, 3, 4, 5, 6, 7, 8, 10]

    # --- Create a 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ARC vs LRU Cache Performance on MultiTierCacheSimulator Datasets',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, (ds_name, workload) in enumerate(workloads.items()):
        arc_hit_rates = []
        lru_hit_rates = []

        # Experiment 1: Hit Rate vs Cache Size
        for cap in capacities:
            arc_cache = ARC(capacity=cap)
            lru_cache = LRU(capacity=cap)

            for key in workload:
                arc_cache.access(key)
                lru_cache.access(key)

            arc_hit_rates.append(arc_cache.hit_rate * 100)
            lru_hit_rates.append(lru_cache.hit_rate * 100)

        # Experiment 2: Parameter p adaptation
        sample_arc = ARC(capacity=5)
        for key in workload:
            sample_arc.access(key)

        # Plot 1: Hit Rate Comparison
        ax1 = axes[idx, 0]
        ax1.plot(capacities, arc_hit_rates, marker='o', linewidth=2, label='ARC', color='#1f77b4')
        ax1.plot(capacities, lru_hit_rates, marker='s', linestyle='--', linewidth=2, label='LRU', color='#ff7f0e')
        ax1.set_title(f'Hit Rate vs Capacity — {ds_name}', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Cache Capacity (C)', fontsize=10)
        ax1.set_ylabel('Hit Rate (%)', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # Print summary
        print(f"\n--- {ds_name} ---")
        print(f"{'Capacity':>10} | {'ARC Hit%':>10} | {'LRU Hit%':>10} | {'Diff(ARC-LRU)':>14}")
        print("-" * 50)
        for i, cap in enumerate(capacities):
            delta = arc_hit_rates[i] - lru_hit_rates[i]
            print(f"{cap:>10} | {arc_hit_rates[i]:>9.2f}% | {lru_hit_rates[i]:>9.2f}% | {delta:>+10.2f}%")

        # Plot 2: Dynamic Parameter p
        ax2 = axes[idx, 1]
        p_data = sample_arc.p_history[:2000]
        ax2.plot(p_data, linewidth=1.2, color='#2ca02c')
        ax2.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Max Capacity (C=5)')
        ax2.set_title(f'ARC Parameter p Adaptation — {ds_name}', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Request Number', fontsize=10)
        ax2.set_ylabel('Recency Target (p)', fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'results.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {os.path.join(base_dir, 'results.png')}")
    plt.show()


if __name__ == "__main__":
    run_experiments()