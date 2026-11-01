# RQ1: Matched real-trace benchmark

This section covers the fixed-protocol evaluation across the same 10k training / 40k validation / 40k holdout trace.

## Key comparison file
- `PSO_SFO_10K_COMPARISON.md`

## Summary table

| Policy | Total CHR | HPC CHR | LPC CHR | Total Penalty | Avg Retrieval Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PSO | 67.62% | 69.96% | 66.59% | 309,275 | 34.37 |
| SFO | 66.99% | 70.11% | 65.61% | 311,155 | 34.90 |
| GA | 67.38% | 69.96% | 66.24% | 310,920 | 34.61 |
| DSOA | 67.13% | 70.05% | 65.84% | 310,710 | 34.93 |
| ABC | 67.26% | 69.69% | 66.19% | 310,965 | 34.74 |
| HO | 67.62% | 69.61% | 66.75% | 312,470 | 34.41 |
| WOA | 67.41% | 69.76% | 66.38% | 311,530 | 34.52 |

## Interpretation
- PSO and HO are the strongest under the matched real-trace benchmark in total CHR.
- SFO performs best on HPC CHR but trails in LPC CHR and overall balance.
- This section is the cleanest foundation for the paper’s main benchmark claim.
