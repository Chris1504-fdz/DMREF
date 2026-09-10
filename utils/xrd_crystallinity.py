"""PdO(101) crystallinity metrics from theta-2theta XRD patterns.

The fab team delivers XRD patterns as an Excel workbook in which every sample
occupies a *pair* of columns (2theta, intensity) and the sample name sits in the
header row above the intensity column (see ``raw_data_received/08_27_26``).
This module turns those patterns into per-sample crystallinity numbers and a
per-condition summary that the campaign notebooks / GP can consume.

What is measured
----------------
Only the **PdO(101) reflection at 2theta ~ 33.9 deg** carries film information in
these scans: the c-plane sapphire substrate dominates everything else
(0006 at 41.7 deg, its K-beta / W-L ghosts at 37.5 / 39.9 deg, ...), and a 3-16 nm
film is far too thin for its amorphous halo to be separated from the substrate
background (checked on the 2026-08-27 delivery: the residual "halo" after
sapphire subtraction is identical for crystalline and amorphous films and does
not scale with thickness, i.e. it is a scan-geometry artefact). The classic
``crystalline / (crystalline + amorphous)`` degree of crystallinity is therefore
*not* computable here; the metrics below are the honest alternatives.

Per sample (``analyze_sheet``)
    auc_101          integrated PdO(101) intensity in 33-35 deg above a linear
                     baseline through the flanking windows 31.5-32.5 / 35.5-36.5,
                     negative excess clipped to zero - the campaign objective as
                     defined in ``analysis/received_data_analysis/pdo_thickness_noise_analysis.ipynb``
                     (kept for continuity; clipping biases it upward by ~20-25
                     counts*deg at these noise levels).
    auc_101_net      same integral **without** clipping -> unbiased, zero-mean
                     for a peak-free window. This is what the % index uses.
    fit_*            constrained Gaussian + linear background fit on 31.5-36.5 deg
                     (center, FWHM, area, amplitude, amplitude sd, SNR) and the
                     apparent Scherrer crystallite size (K = 0.9, Cu K-alpha,
                     **no** instrumental-broadening correction -> lower bound).
    peak_detected    fit SNR >= ``snr_min`` (default 3) with a FWHM strictly
                     inside the allowed band (rejects noise spikes / runaway fits).
    crystallinity_pct
                     100 * auc_101_net / reference_area  -> **relative crystallinity
                     index** (0 = no detectable PdO(101), 100 = reference film).
                     The reference is, by default, the mean of the three largest
                     detected areas in the same sheet (``--ref top3``); pin it with
                     ``--ref <value>`` to compare deliveries measured with the same
                     instrument settings, or ``--ref <sample>``.
    crystallinity_pct_thk
                     same index after dividing the area by the *final* film
                     thickness (deposited thickness x 5/3 Pd->PdO expansion, or a
                     thickness parsed from the sample name such as ``#14_9nm``).
                     A thin, fully crystalline film and a thick, half-crystalline
                     film give the same ``auc_101_net``; only the thickness-
                     normalised index tells them apart. It is left NaN when the
                     peak is not detected (the noise floor divided by a 3 nm
                     thickness would otherwise read as 60-80 %).

Detection floor: with the 08_27 scan settings (~30 counts rms in the flanks)
films without a resolved peak still return ``crystallinity_pct`` of ~10-25 %.
Part of that is a genuine broad, weak (101) feature (FWHM 1.2-1.5 deg, growing
with thickness) from nano-crystalline PdO, part is the noise floor; values in
that band with ``peak_detected == False`` should be read as "amorphous / below
detection", not as a measured fraction.

Substrate handling
    If the sheet contains a bare-substrate scan (a column whose name contains
    "sapphire", "substrate", "blank" or "bare") it is interpolated onto each
    sample grid, scaled by least squares on the peak-free 22-60 deg region and
    subtracted - but with ``--substrate auto`` (default) **only when the bare
    scan itself shows a feature inside the 33-35 deg window** (that is the case
    for the ``old_recipe`` sheet of the 08_27 delivery, whose sapphire scan has
    sharp artefacts at 33.3 and 34.4 deg, and not for the ``sobol`` sheet, where
    subtracting the reference would only add its noise).

Usage
-----
Command line (run from the repository root)::

    python utils/xrd_crystallinity.py \
        --xlsx raw_data_received/08_27_26/PdO_data_0827.xlsx --sheet sobol \
        --run-table Processed_data_inputs/experiment02/raw_data_update_05_22_2026.xlsx \
        --out-dir output/experiment02_xrd_crystallinity

or from a notebook::

    from utils import xrd_crystallinity as xc
    df   = xc.analyze_sheet(path, "sobol")
    df   = xc.attach_run_table(df, run_table_path)
    df, refs = xc.add_crystallinity(df, ref="top3")
    runs = xc.summarize_by_run(df)

Sample names of the form ``RR_NN`` are parsed as replicate ``RR`` of design
condition ``NN`` (``Base_Run_ID``), matching the run tables in
``Processed_data_inputs``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

Pattern = Tuple[np.ndarray, np.ndarray]

CU_KALPHA_NM = 0.15406
SCHERRER_K = 0.9
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# 2theta windows (deg). ``window`` is the campaign objective window, ``bg1``/``bg2``
# the flanks used for the linear baseline, ``fit_window`` the Gaussian-fit range.
DEFAULT_CFG = dict(
    window=(33.0, 35.0),
    bg1=(31.5, 32.5),
    bg2=(35.5, 36.5),
    fit_window=(31.5, 36.5),
    center0=33.9,
    center_bounds=(33.2, 34.8),
    fwhm_bounds=(0.12, 2.5),
    snr_min=3.0,
    smooth_pts=9,
)

# Regions excluded when scaling a bare-substrate scan onto a sample pattern
# (PdO(101), sapphire (0006) family and its ghosts, other substrate lines).
SUBSTRATE_SCALE_RANGE = (22.0, 60.0)
SUBSTRATE_EXCLUDE = [(32.8, 35.4), (37.0, 38.0), (39.3, 42.8), (43.5, 44.6),
                     (45.0, 46.0), (57.5, 58.8)]
SUBSTRATE_NAME_HINTS = ("sapphire", "substrate", "blank", "bare")

_ID_RE = re.compile(r"^\s*(\d+)_(\d+)\s*$")           # RR_NN  -> replicate, run
_HASH_RUN_RE = re.compile(r"#\s*(\d+)")                 # '#14_9nm' -> run 14
_THICK_RE = re.compile(r"(\d+(?:\.\d+)?)\s*nm", re.I)   # '9nm'    -> 9.0


# --------------------------------------------------------------------------- I/O
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    f = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed it
    return float(f(y, x))


def read_xrd_sheet(path: str, sheet: str) -> Dict[str, Pattern]:
    """Return ``{sample_name: (two_theta, intensity)}`` for one paired-column sheet."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header = raw.iloc[0].tolist()
    body = raw.iloc[1:]
    patterns: Dict[str, Pattern] = {}
    for j in range(0, raw.shape[1] - 1, 2):
        name = header[j + 1]
        if name is None or (isinstance(name, float) and np.isnan(name)):
            continue
        tt = pd.to_numeric(body.iloc[:, j], errors="coerce").to_numpy(float)
        ii = pd.to_numeric(body.iloc[:, j + 1], errors="coerce").to_numpy(float)
        m = np.isfinite(tt) & np.isfinite(ii)
        if m.sum() < 10:
            continue
        order = np.argsort(tt[m])
        patterns[str(name).strip()] = (tt[m][order], ii[m][order])
    if not patterns:
        raise ValueError(f"sheet {sheet!r} of {path} holds no (2theta, intensity) column pairs")
    return patterns


def parse_sample_name(name: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """``'02_14' -> (run 14, replicate 2, None)``; ``'#14_9nm' -> (14, None, 9.0)``."""
    m = _ID_RE.match(name)
    if m:
        return int(m.group(2)), int(m.group(1)), None
    run = int(_HASH_RUN_RE.search(name).group(1)) if _HASH_RUN_RE.search(name) else None
    thick = float(_THICK_RE.search(name).group(1)) if _THICK_RE.search(name) else None
    return run, None, thick


def find_substrate_ref(names: Iterable[str]) -> Optional[str]:
    for n in names:
        if any(h in n.lower() for h in SUBSTRATE_NAME_HINTS):
            return n
    return None


# --------------------------------------------------------------- substrate scan
def scale_reference(tt: np.ndarray, ii: np.ndarray, ref: Pattern,
                    scale_range=SUBSTRATE_SCALE_RANGE,
                    exclude=SUBSTRATE_EXCLUDE) -> Tuple[float, np.ndarray]:
    """Least-squares scale of a bare-substrate scan onto ``(tt, ii)``.

    Returns ``(scale, ref_on_sample_grid)``; the fit uses only 2theta values
    covered by both scans, inside ``scale_range`` and outside every ``exclude``
    window. An additive offset is fitted too but not returned - the linear
    baseline of the peak metrics absorbs it.
    """
    tref, iref = ref
    ref_i = np.interp(tt, tref, iref, left=np.nan, right=np.nan)
    m = np.isfinite(ref_i) & (tt >= scale_range[0]) & (tt <= scale_range[1])
    for a, b in exclude:
        m &= ~((tt >= a) & (tt <= b))
    if m.sum() < 50:
        raise ValueError("not enough overlap between sample and substrate scan to scale it")
    A = np.column_stack([ref_i[m], np.ones(m.sum())])
    (scale, _offset), *_ = np.linalg.lstsq(A, ii[m], rcond=None)
    ref_i = np.where(np.isfinite(ref_i), ref_i, 0.0)
    return float(scale), ref_i


def despike(tt: np.ndarray, ii: np.ndarray, width_deg: float = 0.4,
            nsig: float = 5.0, passes: int = 2) -> Tuple[np.ndarray, int]:
    """Replace sharp single-crystal / instrument lines by the local median.

    A running median of ``width_deg`` (0.4 deg: removes features narrower than
    ~0.2 deg, leaves the 0.8-1.5 deg PdO(101) peak untouched) is subtracted;
    points deviating by more than ``nsig`` robust sigmas (1.4826 * MAD) are
    replaced by the median value, in ``passes`` rounds so that the core of a
    wide spike no longer biases the median in the second round. Needed for the
    HfO2/Si pieces, whose Si(200) forbidden reflection sits at 2theta = 33.0 deg,
    right inside the objective window, and for the ``old_recipe`` sapphire
    artefacts at 33.4 / 34.4 deg.
    """
    from scipy.ndimage import median_filter
    step = float(np.median(np.diff(tt)))
    k = max(3, int(round(width_deg / step)) | 1)
    out = ii.copy()
    total = 0
    for _ in range(max(1, passes)):
        med = median_filter(out, size=k, mode="nearest")
        dev = out - med
        sd = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
        if sd <= 0:
            break
        bad = np.abs(dev) > nsig * sd
        if not bad.any():
            break
        out[bad] = med[bad]
        total += int(bad.sum())
    return out, total


# ------------------------------------------------------------------- metrics
def _window(tt: np.ndarray, w: Sequence[float]) -> np.ndarray:
    return (tt >= w[0]) & (tt <= w[1])


def baseline_metrics(tt: np.ndarray, ii: np.ndarray, cfg: Mapping = DEFAULT_CFG) -> dict:
    """Campaign-convention PdO(101) metrics: linear baseline through the flanks."""
    def flank(w):
        m = _window(tt, w)
        return float(np.median(tt[m])), float(np.median(ii[m])), float(np.std(ii[m], ddof=1))
    (x1, y1, s1), (x2, y2, s2) = flank(cfg["bg1"]), flank(cfg["bg2"])
    m = _window(tt, cfg["window"])
    base = y1 + (y2 - y1) * (tt[m] - x1) / (x2 - x1)
    excess = ii[m] - base
    k = int(cfg.get("smooth_pts", 9))
    smooth = np.convolve(excess, np.ones(k) / k, mode="same") if k > 1 else excess
    noise_sd = 0.5 * (s1 + s2)
    width = float(tt[m][-1] - tt[m][0])
    return dict(
        auc_101=_trapz(np.clip(excess, 0, None), tt[m]),
        auc_101_net=_trapz(excess, tt[m]),
        height_101=float(smooth.max()),
        noise_sd=noise_sd,
        # expected clipped-noise contribution to auc_101 for a flat window
        auc_101_noise_floor=float(noise_sd / np.sqrt(2 * np.pi) * width),
        snr_simple=float(smooth.max() / noise_sd) if noise_sd > 0 else np.nan,
        argmax_2theta=float(tt[m][np.argmax(smooth)]),
    )


def _gauss_lin(x, b0, b1, amp, c, s):
    return b0 + b1 * (x - 34.0) + amp * np.exp(-0.5 * ((x - c) / s) ** 2)


def fit_peak(tt: np.ndarray, ii: np.ndarray, cfg: Mapping = DEFAULT_CFG) -> dict:
    """Gaussian + linear background fit of the PdO(101) region; NaNs if it fails."""
    m = _window(tt, cfg["fit_window"])
    x, y = tt[m], ii[m]
    out = dict(fit_center=np.nan, fit_fwhm=np.nan, fit_area=np.nan, fit_amp=np.nan,
               fit_amp_sd=np.nan, fit_resid_sd=np.nan, snr_fit=np.nan, scherrer_nm=np.nan,
               fit_ok=False)
    if x.size < 20:
        return out
    fl, fh = cfg["fwhm_bounds"]
    cl, ch = cfg["center_bounds"]
    k = int(cfg.get("smooth_pts", 9))
    ys = np.convolve(y, np.ones(k) / k, mode="same") if k > 1 else y
    b0 = float(np.median(y))
    amp0 = max(float(ys[_window(x, cfg["window"])].max() - b0), 1.0)
    p0 = [b0, 0.0, amp0, cfg["center0"], 0.6 / FWHM_PER_SIGMA]
    lo = [-np.inf, -np.inf, 0.0, cl, fl / FWHM_PER_SIGMA]
    hi = [np.inf, np.inf, np.inf, ch, fh / FWHM_PER_SIGMA]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(_gauss_lin, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    b0, b1, amp, c, s = popt
    resid = y - _gauss_lin(x, *popt)
    resid_sd = float(1.4826 * np.median(np.abs(resid - np.median(resid))))  # robust to spikes
    fwhm = float(s * FWHM_PER_SIGMA)
    amp_sd = float(np.sqrt(pcov[2, 2])) if np.isfinite(pcov[2, 2]) else np.nan
    theta = np.deg2rad(c / 2.0)
    out.update(
        fit_center=float(c), fit_fwhm=fwhm, fit_area=float(amp * s * np.sqrt(2 * np.pi)),
        fit_amp=float(amp), fit_amp_sd=amp_sd, fit_resid_sd=resid_sd,
        snr_fit=float(amp / resid_sd) if resid_sd > 0 else np.nan,
        scherrer_nm=float(SCHERRER_K * CU_KALPHA_NM / (np.deg2rad(fwhm) * np.cos(theta))),
        fit_ok=True,
    )
    return out


def analyze_pattern(tt: np.ndarray, ii: np.ndarray, cfg: Mapping = DEFAULT_CFG) -> dict:
    res = baseline_metrics(tt, ii, cfg)
    res.update(fit_peak(tt, ii, cfg))
    fl, fh = cfg["fwhm_bounds"]
    inside = np.isfinite(res["fit_fwhm"]) and (fl * 1.02 < res["fit_fwhm"] < fh * 0.98)
    res["peak_detected"] = bool(res["fit_ok"] and inside and res["snr_fit"] >= cfg["snr_min"])
    return res


def analyze_sheet(path: str, sheet: str, substrate: str = "auto",
                  cfg: Mapping = DEFAULT_CFG, keep_patterns: bool = False,
                  verbose: bool = True, despike_width: Optional[float] = 0.4,
                  despike_nsig: float = 5.0) -> pd.DataFrame:
    """Per-sample PdO(101) metrics for every pattern in ``sheet``.

    ``substrate``: ``'auto'`` (subtract the bare scan only if it has a feature in
    the objective window), ``'always'``, ``'never'``, or the name of the column
    to use as the bare-substrate scan (implies ``'always'``).
    ``despike_width``: running-median width (deg) for :func:`despike`; ``None``
    disables it.
    """
    patterns = read_xrd_sheet(path, sheet)
    n_spikes: Dict[str, int] = {}
    if despike_width:
        for name, (tt, ii) in list(patterns.items()):
            clean, n = despike(tt, ii, despike_width, despike_nsig)
            patterns[name] = (tt, clean)
            n_spikes[name] = n
    ref_name = None
    if substrate not in ("never",):
        ref_name = substrate if substrate not in ("auto", "always") else find_substrate_ref(patterns)
        if substrate not in ("auto", "always") and ref_name not in patterns:
            raise KeyError(f"substrate column {substrate!r} not in sheet {sheet!r}")
    subtract = False
    ref_pat = None
    if ref_name is not None:
        ref_pat = patterns[ref_name]
        ref_metrics = analyze_pattern(*ref_pat, cfg)
        subtract = substrate != "auto" or ref_metrics["peak_detected"] \
            or ref_metrics["snr_simple"] >= cfg["snr_min"]
        if verbose:
            state = "subtracted" if subtract else "not subtracted (flat in the objective window)"
            print(f"[{sheet}] substrate scan {ref_name!r}: window SNR {ref_metrics['snr_simple']:.1f} "
                  f"(fit SNR {ref_metrics['snr_fit']:.1f}) -> {state}")

    rows: List[dict] = []
    kept: Dict[str, dict] = {}
    for name, (tt, ii) in patterns.items():
        if name == ref_name:
            continue
        scale = np.nan
        corr = ii
        if subtract:
            scale, ref_i = scale_reference(tt, ii, ref_pat)
            corr = ii - scale * ref_i
        run, rep, thick = parse_sample_name(name)
        row = dict(sheet=sheet, sample=name, run=run, replicate=rep, thickness_nm_from_name=thick,
                   n_despiked=n_spikes.get(name, 0),
                   substrate_subtracted=subtract, substrate_scale=scale)
        row.update(analyze_pattern(tt, corr, cfg))
        rows.append(row)
        if keep_patterns:
            kept[name] = dict(two_theta=tt, intensity=corr, raw=ii)
    df = pd.DataFrame(rows)
    df.attrs["patterns"] = kept
    df.attrs["substrate_ref"] = ref_name if subtract else None
    df.attrs["cfg"] = dict(cfg)
    return df


# --------------------------------------------------------------- enrichment
FACTOR_COLS = ["Dep Power (W)", "O2 Flow (sccm)", "Dep Press (mTorr)", "Dep Temp (°C)",
               "Ox Temp (°C)", "Ox Press (atm)", "Ox Time (min)", "Vac Ann Temp (°C)",
               "Vac Ann Time (min)", "Dep Thickness (nm)"]


def attach_run_table(df: pd.DataFrame, run_table: str, expansion: float = 5.0 / 3.0,
                     sheet_name: Optional[str] = None, run_col: str = "Base_Run_ID") -> pd.DataFrame:
    """Merge design factors on ``run`` and derive the final film thickness.

    ``expansion`` is the Pd -> PdO thickness ratio (5/3 in this campaign). A
    thickness parsed from the sample name (``'#14_9nm'``) takes precedence.
    """
    tab = pd.read_excel(run_table, sheet_name=sheet_name or 0)
    if run_col not in tab.columns:
        raise KeyError(f"{run_col!r} not in {run_table}")
    cols = [c for c in FACTOR_COLS if c in tab.columns]
    tab = tab.drop_duplicates(run_col)[[run_col] + cols].rename(columns={run_col: "run"})
    out = df.merge(tab, on="run", how="left")
    t_final = pd.Series(np.nan, index=out.index, dtype=float)
    if "Dep Thickness (nm)" in out.columns:
        t_final = out["Dep Thickness (nm)"] * expansion
    if "thickness_nm_from_name" in out.columns:
        t_final = out["thickness_nm_from_name"].astype(float).fillna(t_final)
    out["thickness_final_nm"] = t_final
    out.attrs.update(df.attrs)
    return out


def _reference_value(values: pd.Series, detected: pd.Series, names: pd.Series, ref) -> float:
    """Resolve ``ref`` (``'max'``, ``'topN'``, a sample name or a number) to a number."""
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        return float(ref)
    ref = str(ref)
    try:
        return float(ref)
    except ValueError:
        pass
    if ref in set(names):
        return float(values[names == ref].iloc[0])
    pool = values[detected] if detected.any() else values
    pool = pool.sort_values(ascending=False)
    if ref == "max":
        return float(pool.iloc[0])
    m = re.match(r"top(\d+)$", ref)
    if m:
        n = int(m.group(1))
        return float(pool.iloc[:n].mean())
    raise ValueError(f"unknown reference spec {ref!r}")


def add_crystallinity(df: pd.DataFrame, ref="top3", area_col: str = "auc_101_net",
                      group_col: Optional[str] = "sheet", ref_per_nm=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``crystallinity_pct`` and ``crystallinity_pct_thk`` (when thickness is known).

    ``ref`` fixes the area that counts as 100 % (``'max'``, ``'topN'``, a sample
    name or a number); ``ref_per_nm`` does the same for the thickness-normalised
    index (a number, or None -> same spec as ``ref`` when that is not numeric,
    else ``'top3'``). Returns the enriched frame and a small table of the
    reference values used per group (write it next to the results - the index
    is only comparable across deliveries when the reference is the same).
    """
    out = df.copy()
    out["crystallinity_pct"] = np.nan
    has_t = "thickness_final_nm" in out.columns and out["thickness_final_nm"].notna().any()
    if has_t:
        out["area_per_nm"] = out[area_col] / out["thickness_final_nm"]
        out["crystallinity_pct_thk"] = np.nan
    groups = [(None, out.index)] if group_col is None else \
        [(g, idx) for g, idx in out.groupby(group_col).groups.items()]
    refs = []
    for g, idx in groups:
        sub = out.loc[idx]
        if len(sub) < 3 and not _is_number(ref) and ref not in set(sub["sample"]):
            warnings.warn(f"group {g!r} has {len(sub)} sample(s): a within-group reference is "
                          f"meaningless, crystallinity_pct left NaN (pass --ref <value>)")
            refs.append(dict(group=g, area_col=area_col, ref_spec=str(ref), ref_area=np.nan,
                             ref_area_per_nm=np.nan))
            continue
        r_area = _reference_value(sub[area_col], sub["peak_detected"], sub["sample"], ref)
        out.loc[idx, "crystallinity_pct"] = 100.0 * sub[area_col].clip(lower=0) / r_area
        r_pnm = np.nan
        if has_t:
            ok = sub["area_per_nm"].notna()
            if ok.any():
                if ref_per_nm is not None:
                    spec = ref_per_nm
                else:  # a numeric --ref pins the *area*; the per-nm reference then follows top3
                    spec = ref if not _is_number(ref) else "top3"
                r_pnm = _reference_value(sub.loc[ok, "area_per_nm"], sub.loc[ok, "peak_detected"],
                                         sub.loc[ok, "sample"], spec)
                thk = 100.0 * sub["area_per_nm"].clip(lower=0) / r_pnm
                out.loc[idx, "crystallinity_pct_thk"] = thk.where(sub["peak_detected"])
        refs.append(dict(group=g, area_col=area_col, ref_spec=str(ref), ref_area=r_area,
                         ref_area_per_nm=r_pnm))
    out.attrs.update(df.attrs)
    return out, pd.DataFrame(refs)


def _is_number(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    try:
        float(str(x))
        return True
    except ValueError:
        return False


def summarize_by_run(df: pd.DataFrame) -> pd.DataFrame:
    """Condition-level summary (n, detected count, mean / sd / SEM^2 per metric).

    FWHM / Scherrer statistics use detected peaks only.
    """
    metrics = [c for c in ["auc_101_net", "auc_101", "crystallinity_pct", "crystallinity_pct_thk",
                           "fit_fwhm", "scherrer_nm"] if c in df.columns]
    keys = [k for k in ["sheet", "run"] if k in df.columns]
    if "thickness_nm_from_name" in df.columns and df["thickness_nm_from_name"].notna().any():
        keys.append("thickness_nm_from_name")   # '#14_9nm' pieces are thicknesses, not replicates
    d = df.dropna(subset=["run"]) if "run" in df.columns else df
    if d.empty:
        return pd.DataFrame()
    agg = d.groupby(keys).agg(n=("sample", "size"), n_detected=("peak_detected", "sum"))
    det_only = {"fit_fwhm", "scherrer_nm"}          # shape metrics mean nothing without a peak
    for m in metrics:
        col = d[m].where(d["peak_detected"]) if m in det_only else d[m]
        g = col.groupby([d[k] for k in keys])
        agg[f"{m}_mean"] = g.mean()
        agg[f"{m}_sd"] = g.std(ddof=1)
        agg[f"{m}_sem2"] = g.var(ddof=1) / g.count()
    firsts = [c for c in FACTOR_COLS + ["thickness_final_nm"] if c in d.columns]
    if firsts:
        agg = agg.join(d.groupby(keys)[firsts].first())
    return agg.reset_index()


# ------------------------------------------------------------------ figures
PALETTE = dict(series=["#2a78d6", "#eb6834", "#1baf7a"], muted="#898781", grid="#e1e0d9",
               axis="#c3c2b7", ink="#0b0b0b", ink2="#52514e", surface="#fcfcfb", window="#f0efec")


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["surface"], "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["axis"], "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": PALETTE["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": PALETTE["ink"], "axes.labelcolor": PALETTE["ink2"],
        "xtick.color": PALETTE["ink2"], "ytick.color": PALETTE["ink2"],
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "sans-serif",
    })


YLABEL = {"crystallinity_pct": "relative crystallinity (% of reference PdO(101) area)",
          "crystallinity_pct_thk": "thickness-normalised crystallinity (% of reference area / nm)"}
TITLE = {"crystallinity_pct": "PdO(101) relative crystallinity per condition (dots = replicate films)",
         "crystallinity_pct_thk": "Thickness-normalised crystallinity per condition (detected peaks only)"}


def _condition_label(df: pd.DataFrame) -> pd.Series:
    """'#14' for design replicates, '#14 9 nm' for thickness pieces named like '#14_9nm'."""
    run = df["run"] if "run" in df.columns else pd.Series(np.nan, index=df.index)
    lab = run.map(lambda r: f"#{int(r)}" if pd.notna(r) else "")
    if "thickness_nm_from_name" in df.columns:
        t = df["thickness_nm_from_name"]
        lab = lab.where(t.isna(), lab + t.map(lambda v: f" {v:g} nm" if pd.notna(v) else ""))
    lab = lab.where(lab != "", df["sample"].astype(str))
    return lab


def _series_color(df: pd.DataFrame) -> Tuple[pd.Series, Dict[str, str]]:
    """Colour by deposition temperature when known (slot order = sorted levels)."""
    if "Dep Temp (°C)" in df.columns and df["Dep Temp (°C)"].notna().any():
        levels = sorted(df["Dep Temp (°C)"].dropna().unique())[:3]
        cmap = {f"Dep Temp {int(t)} °C": PALETTE["series"][i] for i, t in enumerate(levels)}
        col = df["Dep Temp (°C)"].map({t: PALETTE["series"][i] for i, t in enumerate(levels)})
        return col.fillna(PALETTE["muted"]), cmap
    return pd.Series(PALETTE["series"][0], index=df.index), {"samples": PALETTE["series"][0]}


def plot_fits(df: pd.DataFrame, out_png: str, cfg: Mapping = DEFAULT_CFG, ncols: int = 6) -> None:
    """One small panel per sample: corrected data, fitted baseline + Gaussian."""
    import matplotlib.pyplot as plt
    _style()
    pats = df.attrs.get("patterns", {})
    if not pats:
        raise ValueError("analyze_sheet(..., keep_patterns=True) is required for plot_fits")
    d = df.sort_values(["run", "replicate"], na_position="last").reset_index(drop=True)
    colors, _ = _series_color(d)
    n = len(d)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.1 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    lo, hi = cfg["fit_window"][0] - 0.5, cfg["fit_window"][1] + 0.5
    for ax, (idx, r) in zip(axes, d.iterrows()):
        p = pats[r["sample"]]
        tt, ii = p["two_theta"], p["intensity"]
        m = (tt >= lo) & (tt <= hi)
        base_lvl = np.median(ii[_window(tt, cfg["bg1"]) | _window(tt, cfg["bg2"])])
        ax.axvspan(*cfg["window"], color=PALETTE["window"], lw=0, zorder=0)
        ax.plot(tt[m], ii[m] - base_lvl, lw=0.7, color=PALETTE["muted"], zorder=2)
        if r["fit_ok"]:
            x = np.linspace(*cfg["fit_window"], 300)
            mm = _window(tt, cfg["fit_window"])
            # re-derive the background line from the fitted centre/width and the data
            s = r["fit_fwhm"] / FWHM_PER_SIGMA
            g = np.exp(-0.5 * ((tt[mm] - r["fit_center"]) / s) ** 2)
            A = np.column_stack([np.ones(mm.sum()), tt[mm] - 34.0, g])
            coef, *_ = np.linalg.lstsq(A, ii[mm], rcond=None)
            bg = coef[0] + coef[1] * (x - 34.0)
            ax.plot(x, bg - base_lvl, lw=0.8, color=PALETTE["axis"], zorder=3)
            ax.plot(x, bg + coef[2] * np.exp(-0.5 * ((x - r["fit_center"]) / s) ** 2) - base_lvl,
                    lw=1.6, color=colors[idx] if r["peak_detected"] else PALETTE["muted"],
                    ls="-" if r["peak_detected"] else (0, (3, 2)), zorder=4)
        pct = r.get("crystallinity_pct", np.nan)
        tag = f"{pct:.0f} %" if np.isfinite(pct) else "n/a"
        ax.set_title(f"{r['sample']}   {tag}   SNR {r['snr_fit']:.1f}", loc="left", fontsize=8,
                     color=PALETTE["ink"] if r["peak_detected"] else PALETTE["ink2"])
        ax.set_xlim(lo, hi)
        ax.tick_params(labelleft=False, length=2)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel("2θ (deg)")
    fig.suptitle("PdO(101) region per sample - data (grey), fitted background (thin) and Gaussian "
                 "(solid = detected, dashed = below SNR threshold); shaded = 33-35° objective window",
                 x=0.01, ha="left", fontsize=9, color=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_by_run(df: pd.DataFrame, out_png: str, ycol: str = "crystallinity_pct") -> None:
    """Replicates as dots per design condition, ordered by Dep Temp then thickness."""
    import matplotlib.pyplot as plt
    _style()
    if "run" not in df.columns or ycol not in df.columns:
        return
    d = df.dropna(subset=["run", ycol]).copy()
    if d.empty:
        return
    d["cond"] = _condition_label(d)
    order_cols = [c for c in ["Dep Temp (°C)", "thickness_final_nm", "run"] if c in d.columns]
    conds = d.sort_values(order_cols).drop_duplicates("cond")["cond"].tolist()
    xpos = {c: i for i, c in enumerate(conds)}
    colors, cmap = _series_color(d)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(conds) + 2), 4.2))
    rng = np.random.default_rng(0)
    for idx, r in d.iterrows():
        x = xpos[r["cond"]] + rng.uniform(-0.12, 0.12)
        det = bool(r["peak_detected"])
        ax.scatter(x, r[ycol], s=34, color=colors[idx] if det else "none",
                   edgecolor=colors[idx], linewidth=1.2, zorder=3)
    means = d.groupby("cond")[ycol].mean()
    for c, mval in means.items():
        ax.hlines(mval, xpos[c] - 0.28, xpos[c] + 0.28, color=PALETTE["ink2"], lw=1.4, zorder=4)
    ax.set_xticks(range(len(conds)))
    labels = []
    for c in conds:
        row = d[d["cond"] == c].iloc[0]
        t = row.get("thickness_final_nm", np.nan)
        labels.append(f"{c}\n{t:.0f} nm" if (np.isfinite(t) and "nm" not in c) else c)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("design condition (final PdO thickness)")
    ax.set_ylabel(YLABEL.get(ycol, ycol))
    ax.set_ylim(bottom=min(0, np.nanmin(d[ycol]) - 2))
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cmap.items()]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", color=PALETTE["ink2"],
                       label="PdO(101) not detected (SNR < threshold)"),
                Line2D([], [], color=PALETTE["ink2"], lw=1.4, label="condition mean")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
              handletextpad=0.4, columnspacing=1.2)
    ax.set_title(TITLE.get(ycol, ycol), loc="left", color=PALETTE["ink2"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_vs_thickness(df: pd.DataFrame, out_png: str, ycol: str = "crystallinity_pct") -> None:
    """Crystallinity index against final film thickness, coloured by deposition temperature."""
    import matplotlib.pyplot as plt
    _style()
    if "thickness_final_nm" not in df.columns or ycol not in df.columns:
        return
    d = df.dropna(subset=["thickness_final_nm", ycol]).copy()
    if d.empty:
        return
    colors, cmap = _series_color(d)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for idx, r in d.iterrows():
        det = bool(r["peak_detected"])
        ax.scatter(r["thickness_final_nm"], r[ycol], s=36, color=colors[idx] if det else "none",
                   edgecolor=colors[idx], linewidth=1.2, zorder=3)
    if "run" in d.columns:
        d["cond"] = _condition_label(d)
        g = d.groupby("cond").agg(t=("thickness_final_nm", "first"), y=(ycol, "mean"))
        for c, row in g.iterrows():
            ax.annotate(c, (row["t"], row["y"]), xytext=(5, 3), textcoords="offset points",
                        fontsize=7, color=PALETTE["ink2"])
    ax.set_xlabel("final PdO thickness (nm)")
    ax.set_ylabel(YLABEL.get(ycol, ycol))
    ax.set_ylim(bottom=min(0, np.nanmin(d[ycol]) - 2))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in cmap.items()]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="none", color=PALETTE["ink2"],
                       label="PdO(101) not detected")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              handletextpad=0.4, columnspacing=1.2)
    ax.set_title("Crystallinity grows with thickness; labels = design condition (mean position)",
                 loc="left", color=PALETTE["ink2"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- CLI
def _cfg_from_args(a) -> dict:
    cfg = dict(DEFAULT_CFG)
    cfg.update(window=tuple(a.window), bg1=tuple(a.bg1), bg2=tuple(a.bg2),
               fit_window=tuple(a.fit_window), snr_min=a.snr_min)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--xlsx", required=True, help="workbook with paired (2theta, intensity) columns")
    p.add_argument("--sheet", action="append", default=None,
                   help="sheet(s) to process (repeatable); default: 'sobol'")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: output/xrd_crystallinity_<workbook stem>)")
    p.add_argument("--run-table", default=None,
                   help="design run sheet with Base_Run_ID + factor columns (adds Dep Temp, thickness, ...)")
    p.add_argument("--expansion", type=float, default=5.0 / 3.0,
                   help="final / deposited thickness ratio (Pd -> PdO)")
    p.add_argument("--substrate", default="auto",
                   help="'auto' | 'always' | 'never' | <column name of the bare-substrate scan>")
    p.add_argument("--ref", default="top3",
                   help="reference area for 100 %%: 'max', 'topN', a sample name, or a number")
    p.add_argument("--ref-per-nm", type=float, default=None,
                   help="pin the reference for the thickness-normalised index (area per nm)")
    p.add_argument("--snr-min", type=float, default=DEFAULT_CFG["snr_min"],
                   help="fit SNR needed to call the PdO(101) peak detected")
    p.add_argument("--window", nargs=2, type=float, default=DEFAULT_CFG["window"], metavar=("LO", "HI"))
    p.add_argument("--bg1", nargs=2, type=float, default=DEFAULT_CFG["bg1"], metavar=("LO", "HI"))
    p.add_argument("--bg2", nargs=2, type=float, default=DEFAULT_CFG["bg2"], metavar=("LO", "HI"))
    p.add_argument("--fit-window", nargs=2, type=float, default=DEFAULT_CFG["fit_window"], metavar=("LO", "HI"))
    p.add_argument("--despike-width", type=float, default=0.4,
                   help="running-median width (deg) used to remove sharp substrate/instrument lines")
    p.add_argument("--despike-nsig", type=float, default=5.0, help="despike threshold in robust sigmas")
    p.add_argument("--no-despike", action="store_true")
    p.add_argument("--tag", default="", help="suffix for output file names (e.g. the delivery date)")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)

    sheets = a.sheet or ["sobol"]
    cfg = _cfg_from_args(a)
    stem = os.path.splitext(os.path.basename(a.xlsx))[0]
    out_dir = a.out_dir or os.path.join("output", f"xrd_crystallinity_{stem}")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""

    frames = []
    for sh in sheets:
        df = analyze_sheet(a.xlsx, sh, substrate=a.substrate, cfg=cfg, keep_patterns=True,
                           despike_width=None if a.no_despike else a.despike_width,
                           despike_nsig=a.despike_nsig)
        frames.append(df)
    pats = {k: v for f in frames for k, v in f.attrs["patterns"].items()}
    subs = {f["sheet"].iloc[0]: f.attrs["substrate_ref"] for f in frames}
    df = pd.concat(frames, ignore_index=True)
    df.attrs["patterns"] = pats
    if a.run_table:
        df = attach_run_table(df, a.run_table, expansion=a.expansion)
        df.attrs["patterns"] = pats
    ref_spec = float(a.ref) if _is_number(a.ref) else a.ref
    df, refs = add_crystallinity(df, ref=ref_spec, ref_per_nm=a.ref_per_nm)
    df.attrs["patterns"] = pats
    runs = summarize_by_run(df)

    config = pd.DataFrame(
        [("workbook", a.xlsx), ("sheets", ", ".join(sheets)), ("run_table", a.run_table or ""),
         ("expansion_final_over_deposited", a.expansion), ("substrate_policy", a.substrate),
         ("substrate_ref_used", "; ".join(f"{k}: {v}" for k, v in subs.items())),
         ("reference_spec", str(a.ref)), ("reference_per_nm", "" if a.ref_per_nm is None else a.ref_per_nm),
         ("despike", "off" if a.no_despike else f"running median {a.despike_width} deg, {a.despike_nsig} robust sigma"),
         ("snr_min", a.snr_min),
         ("objective_window_deg", f"{cfg['window'][0]}-{cfg['window'][1]}"),
         ("baseline_flanks_deg", f"{cfg['bg1']} & {cfg['bg2']}"),
         ("fit_window_deg", f"{cfg['fit_window'][0]}-{cfg['fit_window'][1]}"),
         ("fit_center_bounds_deg", str(cfg["center_bounds"])), ("fit_fwhm_bounds_deg", str(cfg["fwhm_bounds"])),
         ("scherrer", f"K={SCHERRER_K}, lambda={CU_KALPHA_NM} nm, no instrumental correction"),
         ("crystallinity_pct", "100 * max(auc_101_net, 0) / ref_area  (ref per sheet, see 'references')"),
         ("crystallinity_pct_thk", "100 * max(auc_101_net / thickness_final_nm, 0) / ref_area_per_nm")],
        columns=["item", "value"])

    xlsx_out = os.path.join(out_dir, f"xrd_crystallinity{suffix}.xlsx")
    with pd.ExcelWriter(xlsx_out) as w:
        df.drop(columns=[], errors="ignore").to_excel(w, sheet_name="per_sample", index=False)
        if not runs.empty:
            runs.to_excel(w, sheet_name="per_run", index=False)
        refs.to_excel(w, sheet_name="references", index=False)
        config.to_excel(w, sheet_name="config", index=False)
    df.to_csv(os.path.join(out_dir, f"xrd_crystallinity{suffix}_per_sample.csv"), index=False)

    if not a.no_figures:
        for sh, f in zip(sheets, frames):
            sub = df[df["sheet"] == sh].copy()
            sub.attrs["patterns"] = pats
            plot_fits(sub, os.path.join(out_dir, f"pdo101_fits_{_slug(sh)}{suffix}.png"), cfg)
        plot_by_run(df, os.path.join(out_dir, f"crystallinity_by_run{suffix}.png"))
        if "crystallinity_pct_thk" in df.columns and df["crystallinity_pct_thk"].notna().any():
            plot_by_run(df, os.path.join(out_dir, f"crystallinity_thk_by_run{suffix}.png"),
                        ycol="crystallinity_pct_thk")
        plot_vs_thickness(df, os.path.join(out_dir, f"crystallinity_vs_thickness{suffix}.png"))

    show = [c for c in ["sheet", "sample", "run", "replicate", "thickness_final_nm", "auc_101_net",
                        "snr_fit", "fit_fwhm", "peak_detected", "crystallinity_pct",
                        "crystallinity_pct_thk"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df[show].round(2).to_string(index=False))
        print("\nreference areas:\n" + refs.round(2).to_string(index=False))
    print(f"\nwrote {xlsx_out}")
    return 0


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


if __name__ == "__main__":
    sys.exit(main())
