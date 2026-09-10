# output/ — generated artifacts

## Active study

### `experiment03_hfo2_batch1_analysis/` — PdO on HfO₂ (current)

- **DoE:** brand-new LHS+SA, 23 conditions × 3 reps, seed 20, with the new fab constraint **Vac Ann ≤ 300 °C** (from batch-1 contact failures).
- **Inputs (9 swept + 1 pinned):** Dep Power 2–40 W · O₂ Flow {0}∪0.3–20 sccm · Dep Press 2–10 mTorr · Ox Temp 200–500 °C · Ox Press {0.01, 0.02} atm · Ox Time 0–120 min · **Vac Ann Temp 25–300 °C** · Vac Ann Time 0–120 min · Dep Thickness 2–6 nm · Dep Temp pinned 425 °C.
- **Outputs (to measure):** Hall mobility ↑ · Carrier concentration ↓ · XRD 33–35° AUC ↓ (amorphousness proxy).
- **Key files:** `PdO_HfO2_LHS_SA_design_v2_anncap300.xlsx` (**current run sheet**), `received_batch1_transcribed.xlsx` (batch-1 data), feasibility + design-diagnostic figures.
- Notebook: `analysis/received_data_analysis/experiment03_hfo2_batch1_analysis_doe_n_caped_300.ipynb`

### `experiment02_xrd_crystallinity/` — PdO on Al₂O₃, XRD crystallinity of the Batch-1 Sobol films (delivered 2026-08-27)

- **Input:** `raw_data_received/08_27_26/PdO_data_0827.xlsx` — θ-2θ patterns for the 14 valid Batch-1 conditions × 3 replicates (`sobol` sheet, IDs `replicate_run`), the #14 recipe at 9/11/13 nm on HfO₂/Si (`scaling thickness`), and the old 5 nm recipe (`old_recipe`); Hall data in `benchmark`.
- **Metric:** relative PdO(101) crystallinity index from `utils/xrd_crystallinity.py` — unclipped 33–35° area above a linear baseline (the campaign's AUC objective), as % of the mean of the three largest detected areas (`ref_area = 333.93`); `crystallinity_pct_thk` divides by final thickness first (`ref = 23.88 / nm`). Constrained Gaussian fit gives detection (SNR ≥ 3), FWHM and apparent Scherrer size. The classic crystalline/(crystalline+amorphous) ratio is **not** computable on these thin films (the sapphire-subtracted halo is a scan artefact, identical for amorphous and crystalline films).
- **Result:** 19/42 films show a resolved PdO(101) peak. Every 25 °C film ≤ 12 nm final and every 425 °C film ≤ 4 nm is amorphous (index 11–26 %, no peak); 425 °C films ≥ 13 nm show a clear peak (68–92 %), #7 (25 °C, 14 nm) 58 %, #21 (425 °C, 9 nm) 47 % — exactly the fab team's slide claims. Replicate SEM² is exported for the GP. Films without a peak still read ~10–25 % (weak nanocrystalline signal + noise floor): treat as "below detection", not a fraction.
- **Key files:** `xrd_crystallinity_0827.xlsx` (per_sample / per_run / references / config), `pdo101_fits_sobol_0827.png` (fit QA per film), `diag_amorphous_halo_check_0827.png` (why the amorphous halo is not measurable), `crystallinity_by_run_0827.png`, `crystallinity_thk_by_run_0827.png`, `crystallinity_vs_thickness_0827.png`; `*_hfo2_thickness*` (HfO₂ pieces, Sobol reference pinned) and `*_old_recipe*` (absolute metrics only).
- **Crystallinity vs Hall:** condition-level Spearman ρ = −0.70 with log carrier concentration, +0.52 with mobility (`xrd_crystallinity_vs_hall_0827.xlsx`, `crystallinity_vs_hall_0827.png`); #15 is the low-crystallinity / high-mobility exception.
- Notebook (study of record, all steps and the why-not-the-classic-ratio diagnostic): `analysis/received_data_analysis/experiment02_xrd_crystallinity_0827.ipynb`
- Reproduce tables only: `python utils/xrd_crystallinity.py --xlsx raw_data_received/08_27_26/PdO_data_0827.xlsx --sheet sobol --run-table Processed_data_inputs/experiment02/raw_data_update_05_22_2026.xlsx --out-dir output/experiment02_xrd_crystallinity --tag 0827`

### `sampler_bo_1d_test/` — Sobol+BO vs LHS+SA+BO end-to-end check

- 1 continuous + 1 two-level categorical toy problem, identical GP (`utils/mobo.fit_gp`) + LogEI, 30 seeds; only the initializer differs. Result: statistical tie with a slight LHS+SA edge — no reason to prefer Sobol. Notebook: `analysis/doe/sampler_bo_1d_test.ipynb`; shared code: `utils/doe_bo_test.py`.
- Also holds the `hd_*` files from the 3-problem, 6-arm follow-up (`analysis/doe/sampler_bo_hd_batches_test.ipynb`; Hartmann-6 smooth / Shekel-4 deceptive / Michalewicz-5 rugged; pure BO, ± space-filling batches, and small-init + qNEI batches; fixed 60-eval budget, arms precomputed by parallel workers via `utils/run_doe_bo_arm.py`). Results: pure BO wins or ties on average; space-filling batches cut tail risk where BO locks in; batch-mode qNEI loses little vs sequential BO — but with a small init the initializer matters (LHS+SA(8)+qNEI ≫ Sobol(8)+qNEI on Hartmann); on deceptive/rugged problems total budget binds. Batch-policy isolation (§4, matched 12-pt init, all-batch delivery): qNEI batches ≫ SF-only batches everywhere (SF-only is the worst arm — quantifies the value of BO); a mixed 6 qNEI + 2 SF batch costs little and never wins — acceptable insurance, not a performance play.

## `previous_studies/` — superseded / historical

| Directory | Study | What sets it apart |
|---|---|---|
| `experiment03_hfo2_doe_design/` | Original HfO₂ DoE (LHS+SA, Vac Ann up to 500 °C) + full method comparison (Sobol / LHS / oLHS / greedy / SA) | Superseded by the anneal-capped v2 design; batch-1 runs 1–5 came from this sheet. Figures still valid for method justification. |
| `experiment03_hfo2_sobol_design/` | Exploratory Sobol draft of the HfO₂ design | First attempt before the LHS+SA method was chosen; kept for the extensibility/metric studies. |
| `experiment02_update_all_sobol/` | PdO on **Al₂O₃**, batch 2 (Sobol continuation, Vac Ann capped 550 °C) | Different substrate; the precedent for capping only the vac-anneal. |
| `experiment02/` | PdO on Al₂O₃, batch 1 post-experiment analysis (ANOVA, CV) | First PdO campaign; outputs were Hall mobility + carrier concentration only. |
| `previous_temp/` | Scratch ANOVA / outlier-sensitivity runs for experiment02 | Working files, not a study of record. |
| `DoE/` | (empty) old InOx Taguchi/Sobol DoE folder | Pre-PdO campaign (`doe/doe_design.ipynb`). |
| `mobo_1d_test/` | 1-D MOBO sanity test (`mobo_1d_loop.gif`) | Method validation for `utils/mobo.py`, not experiment data. |
