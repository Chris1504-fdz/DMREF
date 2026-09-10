# experiment03_hfo2_sobol_design — exploratory Sobol draft (superseded)

**Study:** first draft of the PdO-on-HfO₂ design using a scrambled Sobol sequence, plus the
exploration that led to choosing LHS+SA instead.
Notebook: `analysis/temp/experiment03_hfo2_sobol_design.ipynb` (archived exploratory version).

**DoE:** Sobol, 23 conditions × 3 replicates; same factors/bounds as the original design
(Vac Ann up to 500 °C — pre-cap).

**Contents:**
- `PdO_HfO2_Sobol_design.xlsx` / `PdO_HfO2_SimulatedAnnealing_design.xlsx` — draft run sheets (never sent to fab)
- `hfo2_design_quality_comparison.*`, `hfo2_5batch_*`, `hfo2_second_batch_*`, `annealing_vs_greedy.png`, `hfo2_2batch_correlation_compare.png` — method studies: Sobol vs LHS variants, batch-extension strategies, greedy vs simulated annealing
- `hfo2_*pairplot*`, `hfo2_factor_correlation.png`, `hfo2_temperature_plane.png`, `hfo2_all_factor_distributions.png` — draft-design diagnostics

**Vs the others:** the method-development sandbox. Its conclusions were consolidated into
`analysis/doe/experiment03_hfo2_doe_design.ipynb` (whose outputs are in `../experiment03_hfo2_doe_design/`); nothing here is a design of record.
