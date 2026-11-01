# 10k Matched Holdout Comparison (PSO, SFO, GA, DSOA, ABC, HO, WOA)

All values below are from the same 40,000-request holdout block (requests 50,001–90,000), after each policy was selected using validation only.

| Policy | Total CHR | HPC CHR | LPC CHR | Total Penalty | Avg Retrieval Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSO** | **67.62%** | 69.96% | **66.59%** | **309,275** | **34.37** |
| **SFO** | 66.99% | **70.11%** | 65.61% | 311,155 | 34.90 |
| **GA** | 67.38% | 69.96% | 66.24% | 310,920 | 34.61 |
| **DSOA** | 67.13% | 70.05% | 65.84% | 310,710 | 34.93 |
| **ABC** | 67.26% | 69.69% | 66.19% | 310,965 | 34.74 |
| **HO** | **67.62%** | 69.61% | 66.75% | 312,470 | 34.41 |
| **WOA** | 67.41% | 69.76% | 66.38% | 311,530 | 34.52 |

## Selected HO result

The HO configuration selected by validation was `levy_10`, using `p_defense=0.50` and `beta_levy=1.00`. Its untouched holdout result was 67.62% total CHR, 69.61% HPC CHR, 66.75% LPC CHR, 312,470 total penalty, and 34.41 ms average retrieval time.

## Selected WOA result

The WOA configuration selected by validation was `spiral_15`, using `b=1.50`. Its untouched holdout result was 67.41% total CHR, 69.76% HPC CHR, 66.38% LPC CHR, 311,530 total penalty, and 34.52 ms average retrieval time.

## Shared protocol

The comparison uses the same shared 10k protocol across all tuned algorithms:

- Trace: `Dataset/cluster07(00).csv`
- Preprocessing: high-priority ratio 30% with seed 42
- Request split: 10,000 train / 40,000 validation / 40,000 holdout
- Cache sizing: 10% of the training catalog, with 20% DRAM and 80% SSD
- Selection objective: `0.40 * total_CHR + 0.35 * HPC_CHR - 0.15 * normalized_penalty - 0.10 * normalized_latency`
- Search budget: 30 agents, 25 iterations, same seed 42
- Selection rule: choose the best validation score and evaluate that configuration once on the holdout split

This is the comparison that should be used when discussing algorithm performance under the shared protocol. Older benchmark numbers from different request splits are not directly comparable to this matched 10k/validation/holdout sweep.
