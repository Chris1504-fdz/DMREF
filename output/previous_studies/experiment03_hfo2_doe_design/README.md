# experiment03_hfo2_doe_design — original PdO-on-HfO₂ DoE (superseded)

**Study:** first fabrication design for PdO on HfO₂ + the sampling-method selection study.
Notebook: `analysis/doe/experiment03_hfo2_doe_design.ipynb`.

**DoE:** LHS + simulated annealing, 23 conditions × 3 replicates = 69 rows, seed 20.
**Inputs (9 swept + 1 pinned):** Dep Power, O₂ Flow, Dep Press, Ox Temp 200–500 °C, Ox Press {0.01, 0.02} atm, Ox Time, **Vac Ann Temp 25–500 °C**, Vac Ann Time, Dep Thickness; Dep Temp pinned 425 °C.
**Outputs (measured downstream):** Hall mobility, carrier concentration.

**Contents:**
- `PdO_HfO2_LHS_SA_design.xlsx` — the run sheet sent to fab (batch-1 runs 1–5 came from it)
- `direct_method_comparison.*`, `method_comparison.*`, `metric_meaning_2d_demo.png` — why LHS+SA beat random / Sobol / plain LHS / optimized LHS / LHS+greedy (as-fabricated benchmark)
- `factor_distributions.png`, `pairplot_oxpress.png`, `correlation_matrix.png` (max |r| = 0.098), `temperature_plane.png` — design diagnostics

**Vs the others:** superseded by the anneal-capped v2 design in `output/experiment03_hfo2_batch1_analysis/` after batch-1 contact failures showed Vac Ann must stay ≤ 300 °C. The method-comparison figures remain the reference justification for LHS+SA.
