# RQ3: Drift robustness benchmark

This section reports the compact drift benchmark that evaluates the algorithms under train/test distribution shifts.

## Current results

The results below are taken from the generated summary file `rq3_summary.json`.

| Workload | Family | Best Algorithm | Best Score |
| :--- | :--- | :--- | :---: |
| static_zipf_alpha_12 | static_zipf | PSO | 0.3335378188529991 |
| bursty_5x | bursty_trending | WOA | 0.32951032445977346 |
| trending_8x | bursty_trending | DSOA | 0.2838288739023543 |
| drift_08_to_16 | drift | ABC | 0.5256244395181658 |

## Interpretation
- PSO leads on static Zipf behavior.
- WOA and DSOA are more competitive on bursty/trending workloads.
- ABC is strongest on the explicit drift scenario, suggesting better robustness when the popularity distribution changes after training.

## Figures
- `figures/static_zipf_alpha_12_scores.png`
- `figures/bursty_5x_scores.png`
- `figures/trending_8x_scores.png`
- `figures/drift_08_to_16_scores.png`

These figures are the core visual evidence for the RQ3 discussion.
