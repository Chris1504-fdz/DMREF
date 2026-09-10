# experiment02_update_all_sobol — PdO on Al₂O₃: full analysis + batch 2 + MOBO

**Study:** post-experiment analysis of the completed PdO-on-**Al₂O₃** delivery, design of its
Batch 2, and the first GP/MOBO pass.
Notebook: `analysis/received_data_analysis/experiment02_pdo_analysis.ipynb` (current outputs).

**DoE:** Sobol sequence, 23-condition batches × 3 replicates. Batch 2 continued the sequence with
**Vac Ann capped at 550 °C** (Ox Temp uncapped to 800 °C) — the precedent for capping only the
vacuum-anneal, since PdO decomposes at lower temperature in vacuum.
**Inputs:** same 9-factor family as experiment03, but Al₂O₃ bounds (thermal steps up to 800 °C).
**Outputs:** Hall mobility, carrier concentration.

**Contents:**
- `PdO_DoE_Batch2_capped.xlsx`, `sobol_batch2_metrics.xlsx`, `batch2_*` — Batch 2 design + diagnostics
- `anova_*`, `cv_*`, `linear_regression_*`, `per_factor_effects*`, `condition_*` — statistical analysis of the delivery
- `botorch_*` — GP training data, ARD sensitivity, Sobol indices, Pareto candidates (qLogNEHVI)
- `gp_mobo_slides.pdf/pptx`, `gp_architecture_diagram.png` — presentation material

**Vs the others:** the completed reference campaign — its workflow (ANOVA → GP → MOBO) is the
template experiment03 will mirror once HfO₂ data returns. Supersedes `../experiment02/`.
