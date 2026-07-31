"""Multi-objective Bayesian-optimization building blocks.

These helpers factor the model + BO logic out of the analysis notebook so it can
be reused verbatim on a reduced (e.g. 1-D, single-objective) test problem. Every
function works on plain numpy/torch arrays and never assumes a particular input
dimension ``d`` or number of objectives ``m``:

    model = fit_gp(X, Y, Yvar=Yvar)                 # FixedNoise GP if Yvar given
    mask, hv, ref = pareto_front(Y)                 # observed front + hypervolume
    cand, joint, indiv, _ = propose_candidates(     # qLogNEHVI batch proposal
        model, ref, X_baseline=X, bounds=bounds, q=3)

The encoding helpers (:func:`encode_unit` / :func:`decode_unit`) and the
replicate-noise helper (:func:`variance_of_mean_by_group`) are the only pieces
that know about engineering-unit DataFrames; the GP/BO core is config-free.
"""

from __future__ import annotations

from typing import Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

DTYPE = torch.double


def _as_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=DTYPE)


# --------------------------------------------------------------------------- #
# Replicate noise -> observation variance (train_Yvar / "y_var")
# --------------------------------------------------------------------------- #
def variance_of_mean_by_group(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    transform=None,
    floor: str | float = "median",
) -> pd.Series:
    """Per-group variance **of the group mean** (SEM^2) for a response column.

    Each BO training point is a per-condition mean over its replicates, so the
    statistically correct observation noise for that target is the variance of
    that mean: ``var(replicates, ddof=1) / n``. Groups with fewer than two
    replicates (or an exactly-zero spread) have no usable estimate and are
    filled with ``floor``:

    * ``"median"`` -> median of the available positive SEM^2 values, or
    * a float     -> that fixed value.

    ``transform`` (e.g. :func:`numpy.log10`) is applied to the raw values before
    the variance, matching whatever space the objective is modeled in.

    Returns a Series indexed by ``group_col``; reindex it onto your training
    table's group order before stacking into ``Yvar``.
    """
    sem2 = {}
    for g, sub in data.dropna(subset=[value_col]).groupby(group_col):
        v = sub[value_col].astype(float).to_numpy()
        if transform is not None:
            v = transform(v)
        n = len(v)
        sem2[g] = float(np.var(v, ddof=1) / n) if n >= 2 else np.nan
    out = pd.Series(sem2, name=f"sem2_{value_col}")
    out = out.where(out > 0.0)  # treat zero spread as "no estimate"
    if floor == "median":
        positive = out.dropna()
        fill = float(np.median(positive.to_numpy())) if len(positive) else 1.0
    else:
        fill = float(floor)
    return out.fillna(fill)


# --------------------------------------------------------------------------- #
# Unit-cube encoding for mixed continuous / categorical factors
# --------------------------------------------------------------------------- #
def encode_unit(
    df: pd.DataFrame,
    factors: Sequence[str],
    cont_bounds: Mapping[str, tuple],
    cat_levels: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Encode engineering-unit rows into the unit cube ``[0, 1]^d``.

    Continuous factors are min-max scaled by ``cont_bounds`` (clipped to the
    box); categorical factors are mapped to evenly spaced points by their level
    index in ``cat_levels``.
    """
    df = df.reset_index(drop=True)
    U = np.zeros((len(df), len(factors)))
    for j, f in enumerate(factors):
        if f in cont_bounds:
            lo, hi = cont_bounds[f]
            U[:, j] = np.clip((df[f].astype(float).to_numpy() - lo) / (hi - lo), 0.0, 1.0)
        else:
            lv = cat_levels[f]
            U[:, j] = df[f].map({v: i / (len(lv) - 1) for i, v in enumerate(lv)}).to_numpy()
    return U


def decode_unit(
    U,
    factors: Sequence[str],
    cont_bounds: Mapping[str, tuple],
    cat_levels: Mapping[str, Sequence[float]],
    decimals: Optional[Mapping[str, int]] = None,
) -> pd.DataFrame:
    """Inverse of :func:`encode_unit`: unit cube -> engineering-unit DataFrame."""
    rows = []
    for u in np.atleast_2d(np.asarray(U)):
        r = {}
        for j, f in enumerate(factors):
            if f in cont_bounds:
                lo, hi = cont_bounds[f]
                val = lo + float(u[j]) * (hi - lo)
                if decimals is not None:
                    val = round(val, decimals[f])
                r[f] = val
            else:
                lv = cat_levels[f]
                r[f] = lv[int(round(float(u[j]) * (len(lv) - 1)))]
        rows.append(r)
    return pd.DataFrame(rows)[list(factors)]


# --------------------------------------------------------------------------- #
# GP surrogate
# --------------------------------------------------------------------------- #
def fit_gp(X, Y, Yvar=None, standardize: bool = True) -> SingleTaskGP:
    """Fit a ``SingleTaskGP`` surrogate (any input dim ``d``, any #outputs ``m``).

    Parameters
    ----------
    X : (n, d) inputs, assumed already normalized to ``[0, 1]^d``.
    Y : (n, m) outcomes in **maximization** convention.
    Yvar : (n, m) or None
        Per-observation noise variance ("y_var"). When provided the model uses a
        fixed-noise likelihood (``train_Yvar``) instead of inferring a single
        homoskedastic noise level -- this is the SEM^2 of each averaged training
        point. ``Standardize`` rescales ``Yvar`` consistently with ``Y``.
    standardize : bool
        Apply an outcome ``Standardize(m)`` transform (recommended).
    """
    X = _as_tensor(X)
    Y = _as_tensor(Y)
    m = Y.shape[-1]
    kwargs = {}
    if standardize:
        kwargs["outcome_transform"] = Standardize(m=m)
    if Yvar is not None:
        kwargs["train_Yvar"] = _as_tensor(Yvar)
    model = SingleTaskGP(X, Y, **kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# --------------------------------------------------------------------------- #
# Pareto front / hypervolume
# --------------------------------------------------------------------------- #
def infer_reference_point(Y, margin: float = 0.1) -> torch.Tensor:
    """Reference point a fraction ``margin`` below the worst value per objective
    (in maximization space)."""
    Y = _as_tensor(Y)
    span = Y.max(dim=0).values - Y.min(dim=0).values
    return Y.min(dim=0).values - margin * span


def pareto_front(Y, ref_point=None, margin: float = 0.1):
    """Observed Pareto mask, dominated hypervolume, and the reference point used.

    ``ref_point`` defaults to :func:`infer_reference_point`. Returns
    ``(mask, hypervolume, ref_point)``.
    """
    Y = _as_tensor(Y)
    ref_point = infer_reference_point(Y, margin) if ref_point is None else _as_tensor(ref_point)
    mask = is_non_dominated(Y)
    hv = Hypervolume(ref_point=ref_point).compute(Y[mask])
    return mask, float(hv), ref_point


# --------------------------------------------------------------------------- #
# Acquisition: qLogNEHVI batch proposal
# --------------------------------------------------------------------------- #
def propose_candidates(
    model: SingleTaskGP,
    ref_point,
    X_baseline,
    bounds,
    q: int = 1,
    fixed_features_list: Optional[list] = None,
    num_restarts: int = 20,
    raw_samples: int = 256,
    mc_samples: int = 256,
    seed: Optional[int] = None,
):
    """Maximize qLogNEHVI to propose a ``q``-batch of candidates.

    ``bounds`` is a (2, d) tensor/array. When ``fixed_features_list`` is given
    (each entry a ``{col_index: value}`` dict) the search is *mixed* -- those
    columns are enumerated as fixed features while the rest are optimized on the
    box; otherwise a plain continuous ``optimize_acqf`` is used. The latter is
    what a 1-D / no-categorical test would call.

    Returns ``(candidates, joint_acq, indiv_acq, acq)`` where ``indiv_acq[i]`` is
    candidate ``i``'s own qLogNEHVI (its standalone expected HV improvement).
    """
    if seed is not None:
        torch.manual_seed(seed)
    bounds = _as_tensor(bounds)
    d = bounds.shape[1]
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=_as_tensor(ref_point),
        X_baseline=_as_tensor(X_baseline),
        prune_baseline=True,
        sampler=sampler,
    )
    if fixed_features_list:
        candidates, joint = optimize_acqf_mixed(
            acq, bounds=bounds, q=q, num_restarts=num_restarts,
            raw_samples=raw_samples, fixed_features_list=fixed_features_list,
        )
    else:
        candidates, joint = optimize_acqf(
            acq, bounds=bounds, q=q, num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    indiv = np.array([float(acq(candidates[i].reshape(1, 1, d))) for i in range(q)])
    return candidates, float(joint), indiv, acq


# --------------------------------------------------------------------------- #
# Parameter relevance from the fitted GP (ARD lengthscales)
# --------------------------------------------------------------------------- #
def ard_sensitivity(
    model: SingleTaskGP,
    factor_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    normalize: str = "sum",
) -> pd.DataFrame:
    """Per-factor relevance read off the GP's ARD lengthscales.

    The surrogate fits one lengthscale per input dimension (and per output). A
    *short* lengthscale means the posterior changes quickly along that factor, so
    the factor matters more -- hence **relevance = 1 / lengthscale**. This is only
    meaningful because the inputs are normalized to ``[0, 1]^d`` (the lengthscales
    are then on a common scale and directly comparable across factors).

    Parameters
    ----------
    factor_names : length-d labels for the rows (defaults to ``x0..x{d-1}``).
    output_names : length-m labels for the columns (defaults to ``out0..``).
    normalize : ``"sum"`` (each output's relevances sum to 1, i.e. share of
        relevance), ``"max"`` (peak factor = 1), or ``"none"`` (raw 1/lengthscale).

    Returns a DataFrame indexed by factor, one column per output. This reflects
    GP *relevance*, not a statistical p-value -- with few training points and a
    lengthscale prior the magnitudes are suggestive, not significance tests. For
    rigorous variance-based importance, run Sobol indices on the GP posterior.
    """
    kernel = model.covar_module
    base = getattr(kernel, "base_kernel", kernel)  # unwrap a ScaleKernel if present
    ls = base.lengthscale.detach().cpu().numpy()
    ls = ls.reshape(-1, ls.shape[-1])              # (m_outputs, d)
    imp = 1.0 / ls
    if normalize == "sum":
        imp = imp / imp.sum(axis=1, keepdims=True)
    elif normalize == "max":
        imp = imp / imp.max(axis=1, keepdims=True)
    elif normalize != "none":
        raise ValueError("normalize must be 'sum', 'max', or 'none'")
    m, d = imp.shape
    if factor_names is None:
        factor_names = [f"x{j}" for j in range(d)]
    if output_names is None:
        output_names = [f"out{i}" for i in range(m)]
    return pd.DataFrame(imp.T, index=list(factor_names), columns=list(output_names))


# --------------------------------------------------------------------------- #
# Variance-based global sensitivity (Sobol indices / functional ANOVA)
# --------------------------------------------------------------------------- #
def sobol_indices(
    model: SingleTaskGP,
    factor_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    bounds=None,
    cat_dims: Optional[Sequence[int]] = None,
    cat_levels: Optional[Mapping[int, Sequence[float]]] = None,
    n_samples: int = 4096,
    seed: int = 0,
    batch: int = 8192,
) -> pd.DataFrame:
    """First-order (``S1``) and total (``ST``) Sobol indices of the GP posterior mean.

    Variance-based / functional-ANOVA sensitivity: decompose the variance of each
    GP-predicted objective (taken over the input domain) into the share explained
    by each factor **alone** (``S1``) and by the factor **including all of its
    interactions** (``ST``). The gap ``ST - S1`` is that factor's interaction
    effect; ``1 - sum(S1)`` is the share of variance carried by interactions
    overall. Unlike :func:`ard_sensitivity` (a lengthscale ranking), these are
    honest variance shares that account for interactions.

    The fitted surrogate is treated as the (cheap) function to analyse, so the
    indices describe *what the GP learned* and are estimated by Saltelli
    Monte-Carlo on the GP posterior mean -- no extra wet-lab runs. Estimators:
    Saltelli (2010) for ``S1`` and Jansen (1999) for ``ST``.

    Parameters
    ----------
    bounds : ``(2, d)`` lower/upper sampling bounds in the ``[0, 1]`` encoding
        (default: the full unit cube). Pass the proposal bounds to analyse the
        same domain the optimizer searched (e.g. a capped Ox Temp).
    cat_dims : indices sampled at discrete levels rather than continuously
        (e.g. the binary factors). Samples on those axes are snapped to the
        nearest level in ``cat_levels`` (default ``{0.0, 1.0}``), i.e. each level
        gets equal weight.
    cat_levels : optional per-dim list of normalized levels for ``cat_dims``.
    n_samples : base sample size ``N``; total GP evaluations are ``N * (d + 2)``.

    Returns a DataFrame indexed by factor with a 2-level column
    ``(output, {"S1", "ST"})``. Monte-Carlo noise can push small indices slightly
    negative; clip at 0 for display.
    """
    from scipy.stats import qmc

    d = int(model.train_inputs[0].shape[-1])
    m = int(model.num_outputs)
    if bounds is None:
        lo, hi = np.zeros(d), np.ones(d)
    else:
        b = np.asarray(bounds, dtype=float)
        lo, hi = b[0], b[1]
    cat_dims = list(cat_dims) if cat_dims is not None else []
    cat_levels = dict(cat_levels) if cat_levels is not None else {}

    def _snap_cats(M):
        for j in cat_dims:
            levels = np.asarray(cat_levels.get(j, [0.0, 1.0]), dtype=float)
            idx = np.abs(M[:, j][:, None] - levels[None, :]).argmin(axis=1)
            M[:, j] = levels[idx]
        return M

    # Saltelli A / B design, drawn in the requested sampling box.
    sob = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    base = sob.random(n_samples)
    A = _snap_cats(lo + base[:, :d] * (hi - lo))
    B = _snap_cats(lo + base[:, d:] * (hi - lo))

    def _gp_mean(U):
        model.eval()
        out = np.empty((len(U), m))
        with torch.no_grad():
            for s in range(0, len(U), batch):
                Xt = _as_tensor(U[s:s + batch])
                out[s:s + batch] = (
                    model.posterior(Xt).mean.detach().cpu().numpy().reshape(-1, m)
                )
        return out

    fA, fB = _gp_mean(A), _gp_mean(B)
    varY = fA.var(axis=0, ddof=1)  # (m,)

    S1 = np.zeros((d, m))
    ST = np.zeros((d, m))
    for i in range(d):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = _gp_mean(AB)
        S1[i] = np.mean(fB * (fAB - fA), axis=0) / varY          # Saltelli 2010
        ST[i] = 0.5 * np.mean((fA - fAB) ** 2, axis=0) / varY    # Jansen 1999

    if factor_names is None:
        factor_names = [f"x{j}" for j in range(d)]
    if output_names is None:
        output_names = [f"out{i}" for i in range(m)]
    cols = pd.MultiIndex.from_product([list(output_names), ["S1", "ST"]])
    data = np.empty((d, 2 * m))
    for oi in range(m):
        data[:, 2 * oi] = S1[:, oi]
        data[:, 2 * oi + 1] = ST[:, oi]
    return pd.DataFrame(data, index=list(factor_names), columns=cols)
