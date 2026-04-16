# DMREF — Data-Driven Discovery of InOx Thin-Film Transistors

## Overview

This repository contains the data pipeline, analysis, and modeling code for a DMREF project on sputtered Indium Oxide (InOx) thin-film transistors.
The workflow follows a **Design of Experiments (DOE) -> Fabrication -> Characterization -> Modeling** loop, iterating over sputtering and annealing conditions to optimize Hall mobility.

## Repository structure

```
repository_dmref/
├── doe/
│   └── doe_design.ipynb              # DOE: L32 Taguchi, Sobol sequences
├── data_pipeline/
│   └── data_pipeline.ipynb           # Builds the master DataFrame from raw Excel data
├── analysis/
│   ├── analysis.ipynb                # Visualization, regression, BO, LVGP, trees
│   └── tab_dags.ipynb                # Causal discovery (tabular data -> DAGs)
├── output/                           # Generated artifacts (Excel, pickle files)
├── archive/
│   └── data_analysis_original.ipynb  # Original monolithic notebook (backup)
├── XPS & XRD.xlsx                    # Iteration 0 raw data (8 experiments)
├── XPS & XRD_11_06_25.xlsx           # Iteration 1 raw data (24 experiments)
└── README.md
```

## Data

### Raw input files

| File | Experiments | Description |
|------|-------------|-------------|
| `XPS & XRD.xlsx` | 8 (iteration 0) | First round of experiments. Sheets: **Experiments**, **XPS**, **XRD**. Column names use underscores (e.g. `Power_W`). |
| `XPS & XRD_11_06_25.xlsx` | 24 (iteration 1) | Expanded dataset after Taguchi DOE. Sheets: **Experiments_v00/v01/v01_02**, **XPS**, **XRD**. Column names use descriptive labels (e.g. `Power (W)`). |

### Process factors (6 knobs)

| Factor | Levels |
|--------|--------|
| Power (W) | 10, 30 |
| Substrate Temp (C) | RT (25), 400 |
| O2 Flow (sccm) | 0.3, 3.0 |
| Pressure (mTorr) | 2, 4 |
| Anneal Condition | O2, N2 |
| Anneal Temp (C) | 150, 400 |

### Characterization columns

- **XPS** (X-ray Photoelectron Spectroscopy): `In3d_p1`, `In3d_p2`, `O1s_p1`, `Ta4f_p1`, `Ta4f_p2`, `Area_In3d`, `Area_O1s`, `Area_Ta4f`
- **XRD** (X-ray Diffraction): `Peak01`-`Peak03` positions, `Height01`-`Height03`, `FWHM01`-`FWHM03`
- **Response**: `Hall Mobility (cm^2/V·s)`

### How the master file is built

The `data_pipeline/data_pipeline.ipynb` notebook:

1. Reads `XPS & XRD_11_06_25.xlsx` (iteration 1 data)
2. Merges the Experiments, XPS, and XRD sheets on `(Experiment, Trial)`
3. Encodes categorical variables (anneal condition -> numeric)
4. Extends with a Mixed-Taguchi DOE for the next iteration
5. Saves the result to **`output/master_df.pkl`** (51 rows x 42 columns)

### Old vs. new data versions

- **Old** (`XPS & XRD.xlsx`): 8 unique sputtering conditions, ~15 XPS/XRD measurements. This was the initial L8 Taguchi screening round.
- **New** (`XPS & XRD_11_06_25.xlsx`): 24 experiments (8 original + 16 new from L32/Sobol augmentation), ~24 XPS/XRD measurements. Adds more conditions, 3 experiment version sheets, and an additional XRD peak column.

The master file (`output/master_df.pkl`) merges all iterations and is the single source of truth for downstream analysis.

## How to run

### Prerequisites

Use the **`ml_gp_env`** conda environment:

```bash
conda activate /data/zhq7531/envs/ml_gp_env
```

Required packages: `torch`, `botorch`, `statsmodels`, `scikit-learn`, `seaborn`, `causal-learn`, `pandas`, `numpy`, `matplotlib`, `openpyxl`.

### Execution order

The notebooks must be run in this order (each persists state for the next):

```
1. doe/doe_design.ipynb              # standalone — generates DOE tables
2. data_pipeline/data_pipeline.ipynb # produces output/master_df.pkl
3. analysis/analysis.ipynb           # consumes master_df.pkl
4. analysis/tab_dags.ipynb           # consumes master_df.pkl + both Excel files
```

To execute a notebook from the command line:

```bash
cd repository_dmref
jupyter nbconvert --to notebook --execute --inplace <path/to/notebook.ipynb>
```

Each notebook has a preamble cell that sets `os.chdir()` to the repo root so relative paths work regardless of where Jupyter opens the file.