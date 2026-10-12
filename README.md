# DMREF — data-driven thin-film process optimisation (two sub-projects)

This repository holds the design-of-experiments, data pipeline, analysis and modelling code for two DMREF
campaigns that share the same **DoE → fabrication → characterisation → modelling** loop and the same code in
`utils/`, but are otherwise independent. Every data / notebook / output category therefore has one sub-folder
per project:

| sub-folder | project | material & objective | status (2026-10-08) |
|---|---|---|---|
| `inox/` | **InOx thin-film transistors** (first campaign, with the fab team's "NW collaboration") | sputtered indium oxide channel (XPS also shows a Ta 4f signal); maximise Hall mobility over 6 sputter/anneal knobs | paused Nov 2025 → Apr 2026 while the DoE was reworked; the constrained-Sobol batch-1 run sheet came back 2026-10 (`raw_data_received/inox/06_10_26/`) |
| `pdo/` | **PdO films** (second campaign, with the other student) | sputtered Pd oxidised to PdO on Al₂O₃ / HfO₂ / Si stacks; mobility ↑, carrier concentration ↓, crystallinity ↓ over 10 knobs | see `PROJECT_STATUS.md` and `output/README.md`; next design = four substrates as one sliced design (`doe/pdo/pdo_four_substrates_doe_compare.ipynb`, 2026-10-10) |

## Repository layout

```
repository_dmref/
├── raw_data_received/            # exactly what the fab team sent, never edited
│   ├── inox/   XPS & XRD.xlsx (iteration 0, L8), XPS & XRD_11_06_25.xlsx (iteration 1, 24 exp.),
│   │           data_09_17_25/ data_11_05_25/ condition09_17/ (slide decks + raw XPS/XRD),
│   │           06_10_26/ (constrained-Sobol batch-1 run sheet, 30 × 3, + the substrate-temperature question)
│   └── pdo/    study_02/ noise_analysis/ 08_27_26/
├── Processed_data_inputs/        # cleaned / merged tables used by the notebooks
│   ├── inox/   second_iteration_orthogonal/XPS_XRD_with_newdata_clean_v02.xlsx (51-row master), first_taguchi/
│   └── pdo/    experiment02/ (run tables with the delivered Hall data)
├── doe/                          # design-of-experiments notebooks
│   ├── inox/   doe_design.ipynb            L8 → L32 Taguchi, Sobol, constrained Sobol v03 (batch 1 = the 06_10_26 sheet)
│   │           inox_batch1_sampler_comparison.ipynb   Sobol vs Halton / LHS / optimised LHS / CD + w·|r| LHS on this space; builds the batch-1 design (435-600 °C, no RT)
│   │           inox_batch1_doe_compare.ipynb + .yaml  the same study run with the shared doe_compare package (sibling repo ../doe_compare); SAVE_* flag per cell
│   └── pdo/    pdo_four_substrates_doe_compare.ipynb + .yaml  the four-substrate design (substrate = 4-level slicing factor, sliced LHS placed around the recipes in hand) with doe_compare
│               experiment03_hfo2_doe_design.ipynb, experiment03_hfo2_batch1_analysis_doe_n_caped_300.ipynb,
│               experiment04_si_stacks_doe_design.ipynb, sampler_bo_1d_test.ipynb, sampler_bo_hd_batches_test.ipynb
├── data_pipeline/inox/data_pipeline.ipynb   builds the InOx master table + mixed-Taguchi candidates
├── analysis/
│   ├── inox/   inox_sobol_batch1_provenance.ipynb (what the 06_10_26 sheet is and where it comes from),
│   │           analysis.ipynb (regression / ANOVA / GP / BO on the 51-row master), tab_dags.ipynb, pc_with_prior_knowledge.ipynb
│   ├── pdo/    experiment02_pdo_analysis.ipynb, experiment02_anova_cv_gp.ipynb, experiment02_xrd_crystallinity_0827.ipynb,
│   │           pdo_thickness_noise_analysis.ipynb, temp/ (drafts)
│   └── tutorial/lhs_sa_build_it_yourself.ipynb   (method tutorial, shared)
├── output/
│   ├── inox/   master_df.pkl, new_design_candidates.xlsx, DoE/, previous_studies/previous_temp/,
│   │           batch1_design/, sobol_batch1_provenance/, batch1_sampler_comparison/ (all empty until a notebook's SAVE_* flag is set)
│   ├── pdo/    experiment02_xrd_crystallinity/ experiment03_hfo2_batch1_analysis/ experiment04_si_stacks_doe/
│   │           pdo_thickness_noise_analysis/ sampler_bo_1d_test/ project_status/ previous_studies/
│   └── README.md                 # what every output folder contains and which notebook wrote it
├── archive/inox/data_analysis_original.ipynb   # original monolithic InOx notebook (backup)
├── utils/                        # shared code: mobo.py (GP + MOBO), xrd_crystallinity.py, doe_bo_test.py,
│                                 #   run_doe_bo_arm.py, pdf_card_to_readable.py
├── PROJECT_STATUS.md             # PdO campaign status write-up (2026-07-31)
└── README.md
```

Conventions:

- Notebooks write **nothing** to `output/` by default. The PdO notebooks use a `SAVE_OUTPUTS` switch with an `OUT_FILES` dict in the
  setup cell; the InOx sampler-comparison notebook puts a `SAVE_* = False` flag and a `*_FILE` name at the top of every figure or
  table cell (see `output/README.md`). Older InOx notebooks still save unconditionally, into `output/inox/`.
- Every notebook `os.chdir()`s to the repository root in its first code cell, so relative paths work from anywhere.
- Received data goes under `raw_data_received/<project>/<date or topic>/` with a `note.txt` / `info.txt` saying how it arrived.
- The `*_doe_compare.ipynb` notebooks import the shared sampler-comparison package from the sibling repository
  `/data/zhq7531/IDEAL/DMREF/doe_compare` (GitHub: Chris1504-fdz/optimized_lhs); rebuild them with its `notebooks/build_notebook.py
  <config> <out> --package-dir ../doe_compare --repo-root . --out-dir output/<project>/<study>` after editing the YAML.

## InOx sub-project (`inox/`)

**Process factors (6 knobs).** Taguchi phase (iterations 0–1): two levels each. Constrained-Sobol phase (batch 1,
2026): continuous ranges with the physical ordering constraint *substrate temperature < annealing temperature*.

| Factor | Taguchi levels | Sobol batch-1 range |
|--------|----------------|---------------------|
| Power (W) | 10, 30 (50 in iteration 0, too high: insulating films) | 10 – 30, integer |
| Substrate Temp (°C) | RT (25), 400 | 25 (≈14 % of runs) or 400 – 600 |
| O₂ Flow (sccm) | 0.3, 3.0 (2.0 added in runs 9–17) | 0.3 – 3.0 |
| Pressure (mTorr) | 2, 4 | 2 – 4 (header in the sheet says "Torr") |
| Anneal Condition | O₂, N₂ | O / N, 50 : 50 |
| Anneal Temp (°C) | 150, 400 (600 considered) | 150 – 600, always above the substrate temperature |

**Characterisation columns.** XPS: `In3d_p1/p2`, `O1s_p1`, `Ta4f_p1/p2` (positions), `Area_In3d`, `Area_O1s`,
`Area_Ta4f`; XRD: `Peak01–03`, `Height01–03`, `FWHM01–03`; response: `Hall Mobility (cm^2/V·s)`.

**Data versions.**

| File | Experiments | Description |
|------|-------------|-------------|
| `raw_data_received/inox/XPS & XRD.xlsx` | 8 (iteration 0, L8 Taguchi) | first round, Sept 2025; three of the 50 W films insulating |
| `raw_data_received/inox/XPS & XRD_11_06_25.xlsx` | 8 × 3 (iteration 1 repeat at ≤ 30 W) | Nov 2025 repeat with replicates; sheets `Experiments_v00/v01/v01_02`, `XPS`, `XRD` |
| `Processed_data_inputs/inox/second_iteration_orthogonal/XPS_XRD_with_newdata_clean_v02.xlsx` | 17 conditions, 51 rows | master table: iteration 1 + the runs 9–17 augmentation (Feb 2026 "Data Update") |
| `raw_data_received/inox/06_10_26/sobol_constrained_batch_1_repeated (1).xlsx` | 30 × 3 (design only) | the batch-1 constrained-Sobol run sheet from `doe/inox/doe_design.ipynb` (seed 42), returned with the question "integer substrate temperatures 435–600 °C?" |

**Why Taguchi, then Sobol.** Six two-level factors need 64 runs for a full factorial; the L8 orthogonal array screens
the main effects in 8 runs and the L32 extension (which contains the L8) adds two-factor interactions. The screening
found anneal temperature, power, O₂ flow and substrate temperature significant and pressure / anneal gas not, but two
levels cannot show curvature and a GP trained on the 17 conditions did not generalise (leave-one-experiment-out
R² ≈ 0.2). The campaign therefore moved to a space-filling Sobol design over continuous ranges with the substrate <
anneal constraint, which is what batch 1 is. Full story: `analysis/inox/inox_sobol_batch1_provenance.ipynb`.

**Batch 1 recomputed (2026-10-08).** The fab team asked for integer substrate temperatures in 435–600 °C before fabricating; with no
measurements yet the batch is regenerated from that range with no room-temperature runs. `doe/inox/inox_batch1_sampler_comparison.ipynb`
compares the generators on this constrained space (the UIC-style LHS optimised on discrepancy + w × |r| is best or tied on every single-batch
metric, Sobol keeps spacing and holes best as batches accumulate, plain LHS fails the |r| < 0.3 guard) and builds the chosen design — the
CD + 0.1 × |r| Latin hypercube, seed 8 — written to `output/inox/batch1_design/` when the flags in its §6 are set.

**Execution order (InOx).**

```
1. doe/inox/doe_design.ipynb                 # standalone — generates the design tables (output/inox/DoE/)
2. data_pipeline/inox/data_pipeline.ipynb    # produces output/inox/master_df.pkl
3. analysis/inox/analysis.ipynb              # consumes the master table
4. analysis/inox/tab_dags.ipynb              # causal discovery on the master table
```

## PdO sub-project (`pdo/`)

Ten process knobs (deposition power / O₂ flow / pressure / temperature / thickness, oxidation temperature / pressure /
time, vacuum-anneal temperature / time), three objectives (Hall mobility ↑, carrier concentration ↓, PdO(101) relative
crystallinity ↓). Experiment 02 = Sobol batch on Al₂O₃ (analysed), experiment 03 = HfO₂ (dropped), experiment 04 =
Al₂O₃-on-Si and TiO₂-on-Si sliced-LHS design (fabricated, waiting for data). **Next design (2026-10-10):** the four substrates
(Al₂O₃/sapphire, Al₂O₃/Si, TiO₂/Si, HfO₂/Si) as one categorical slicing factor; a sliced optimised LHS of 23 new recipes per
substrate × 5 replicates on the experiment-04 bounds (vacuum anneal ≤ 300 °C), placed around the 56 recipes already in hand
(9 sapphire recipes inside the bounds, the 46 fabricated Si recipes, 1 HfO₂ trial recipe) — `doe/pdo/pdo_four_substrates_doe_compare.ipynb`.
Narrative and figures: `PROJECT_STATUS.md`; per-folder detail: `output/README.md`; reusable GP + MOBO code: `utils/mobo.py`.

## How to run

Use the **`ml_gp_env`** conda environment (the notebooks' kernel):

```bash
conda activate /data/zhq7531/envs/ml_gp_env
cd repository_dmref
jupyter nbconvert --to notebook --execute --inplace <path/to/notebook.ipynb>
```

Required packages: `torch`, `botorch`, `statsmodels`, `scikit-learn`, `seaborn`, `causal-learn`, `pandas`, `numpy`,
`matplotlib`, `openpyxl`, `scipy`. The shell's default `python3` (anaconda base) has a statsmodels build that does not
import against its scipy; use the environment above.
