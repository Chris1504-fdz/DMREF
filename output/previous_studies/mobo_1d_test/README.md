# mobo_1d_test — 1-D MOBO sanity check (method validation)

**What it is:** a toy 1-D validation of the multi-objective Bayesian-optimization loop
(GP + qLogNEHVI) used by the campaign — verifying the machinery in `utils/mobo.py` behaves
correctly before applying it to real fabrication data.
Notebook: `analysis/temp/mobo_1d_test.ipynb`.

**Contents:** `mobo_1d_loop.gif` — animation of the acquisition loop iterating on a known 1-D
test problem.

**Vs the others:** no experimental data and no design — purely method validation.
