"""End-to-end initializer benchmark: Sobol + BO vs LHS+SA + BO.

Shared machinery for the sampler-comparison test notebooks
(``analysis/doe/sampler_bo_1d_test.ipynb``): a mixed test problem with **one
continuous input and one 2-level categorical input** (mimicking Ox Press), the
same sampler code as the DoE study notebooks, and a single-objective BO loop
built on the campaign's own GP (:func:`utils.mobo.fit_gp`) so the *only*
difference between the two arms is the initial design.

    hist = run_bo('sobol', seed=0)          # one arm, one seed
    results = run_benchmark(n_seeds=20)     # both arms, all seeds

Minimization convention: the test functions are minimized; internally the GP
maximizes ``-f``. ``regret`` is best-observed true f minus the achievable
optimum on the candidate grid, so it can reach exactly 0.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc
from scipy.spatial.distance import pdist

from botorch.acquisition.analytic import LogExpectedImprovement

from .mobo import fit_gp, DTYPE

# --------------------------------------------------------------------------- #
# Samplers — same machinery as the DoE study notebooks
# --------------------------------------------------------------------------- #
def sobol_design(n, dim, seed):
    """Scrambled Sobol sequence in the unit cube."""
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)


def optimized_lhs(n, dim, seed):
    """Discrepancy-optimized Latin Hypercube (scipy random-cd) — the SA start."""
    return qmc.LatinHypercube(d=dim, optimization='random-cd', seed=seed).random(n)


def _swap(P, col, a, b):
    P[[a, b], col] = P[[b, a], col]            # within-column swap -> stays a valid LHS


def conditional_sa_lhs(fixed, n_new, dim, seed, n_iter=3000, T_end=1e-3):
    """LHS + simulated annealing minimizing discrepancy of [fixed; new], `fixed` frozen.

    Verbatim port of the DoE study sampler; with empty ``fixed`` it reduces to a
    plain discrepancy-optimized LHS refined by SA.
    """
    rng = np.random.default_rng(seed)
    P = optimized_lhs(n_new, dim, seed)
    score = lambda: qmc.discrepancy(np.vstack([fixed, P]) if len(fixed) else P)
    cur = best = score(); best_P = P.copy()
    ups = []                                   # auto-scale start temperature
    for _ in range(200):
        c = rng.integers(dim); a, b = rng.choice(n_new, 2, replace=False)
        _swap(P, c, a, b); delta = score() - cur; _swap(P, c, a, b)
        if delta > 0: ups.append(delta)
    T = float(np.mean(ups)) if ups else 1e-3
    alpha = T_end ** (1.0 / n_iter)
    for _ in range(n_iter):
        c = rng.integers(dim); a, b = rng.choice(n_new, 2, replace=False)
        _swap(P, c, a, b); cand = score(); delta = cand - cur
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            cur = cand
            if cur < best: best, best_P = cur, P.copy()
        else:
            _swap(P, c, a, b)
        T *= alpha
    return best_P


# --------------------------------------------------------------------------- #
# Test problem: Forrester pair — 1 continuous x + 1 categorical level (2 levels)
# --------------------------------------------------------------------------- #
def forrester(x):
    """Forrester (2008) function, multimodal on [0, 1]. Minimize."""
    x = np.asarray(x, float)
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def forrester_shift(x):
    """Correlated variant (the classic 'low-fidelity' Forrester): same shape
    family, different optimum — plays the role of the second categorical level."""
    x = np.asarray(x, float)
    return 0.5 * forrester(x) + 10.0 * (x - 0.5) - 5.0


def true_f(x, level):
    """f(x, level): level 0 -> forrester, level 1 -> forrester_shift."""
    return np.where(np.asarray(level) < 0.5, forrester(x), forrester_shift(x))


X_STEP = 0.005                                            # continuous resolution grid
_XGRID = np.round(np.arange(0.0, 1.0 + 1e-9, X_STEP), 6)  # 201 points
CAND = np.column_stack([np.tile(_XGRID, 2),               # candidate set: 201 x 2 levels
                        np.repeat([0.0, 1.0], len(_XGRID))])
TRUE_MIN = float(true_f(CAND[:, 0], CAND[:, 1]).min())    # achievable optimum on the grid


def snap_design(U):
    """Unit-square design -> (x, level): x on the X_STEP grid, dim 1 -> 2 levels
    (same list-mapping as `map_parameter` with 2 discrete levels)."""
    x = np.clip(np.round(U[:, 0] / X_STEP) * X_STEP, 0.0, 1.0)
    level = (U[:, 1] >= 0.5).astype(float)
    return np.column_stack([x, level])


def make_initial(sampler, n_init, seed, sa_iters=3000):
    """Initial design for one arm ('sobol' | 'lhs_sa'), as fabricated."""
    if sampler == 'sobol':
        U = sobol_design(n_init, 2, seed)
    elif sampler == 'lhs_sa':
        U = conditional_sa_lhs(np.empty((0, 2)), n_init, 2, seed, n_iter=sa_iters)
    else:
        raise ValueError(f'unknown sampler {sampler!r}')
    return snap_design(U)


def design_metrics(X):
    """(discrepancy, min pairwise distance) of a snapped design."""
    return float(qmc.discrepancy(X)), float(pdist(X).min()) if len(X) > 1 else np.nan


def observe(X, sigma, rng):
    """Noisy evaluation of the true function."""
    y = true_f(X[:, 0], X[:, 1])
    return y + rng.normal(0.0, sigma, size=y.shape)


# --------------------------------------------------------------------------- #
# BO loop — identical GP (utils.mobo.fit_gp) for both arms
# --------------------------------------------------------------------------- #
def run_bo(sampler, seed, n_init=8, n_iter=15, sigma=0.5, sa_iters=3000):
    """One full run: initial design -> LogEI BO loop on the campaign GP.

    Returns a dict with the simple-regret history (length ``n_iter + 1``:
    after the initial design, then after each BO evaluation), the queried
    points, and the initial design's space-filling metrics.
    """
    rng = np.random.default_rng(100_000 * seed + (0 if sampler == 'sobol' else 1))
    X = make_initial(sampler, n_init, seed, sa_iters)
    Y = observe(X, sigma, rng)
    best_true = [float(true_f(X[:, 0], X[:, 1]).min())]

    for _ in range(n_iter):
        model = fit_gp(X, -Y[:, None])                      # maximize -f
        acq = LogExpectedImprovement(model, best_f=float((-Y).max()))
        with torch.no_grad():
            a = acq(torch.as_tensor(CAND, dtype=DTYPE).unsqueeze(1))
        x_next = CAND[int(a.argmax())][None, :]
        Y = np.concatenate([Y, observe(x_next, sigma, rng)])
        X = np.vstack([X, x_next])
        best_true.append(float(true_f(X[:, 0], X[:, 1]).min()))

    disc, mind = design_metrics(X[:n_init])
    return {'sampler': sampler, 'seed': seed,
            'regret': np.array(best_true) - TRUE_MIN,
            'X': X, 'Y': Y, 'init_disc': disc, 'init_mind': mind}


def run_benchmark(samplers=('sobol', 'lhs_sa'), n_seeds=20, verbose=True, **kw):
    """Both arms x all seeds. Returns {sampler: [run_bo dict, ...]}."""
    out = {}
    for s in samplers:
        runs = []
        for seed in range(n_seeds):
            runs.append(run_bo(s, seed, **kw))
            if verbose and (seed + 1) % 5 == 0:
                print(f'  {s}: {seed + 1}/{n_seeds} seeds done')
        out[s] = runs
    return out


def regret_matrix(runs):
    """Stack a list of run dicts into a (n_seeds, n_iter+1) regret array."""
    return np.vstack([r['regret'] for r in runs])


# =========================================================================== #
# Higher-dimensional campaign benchmark: sliced Hartmann-6
# (5 continuous inputs + 1 two-level categorical), with optional mid-campaign
# space-filling batches — used by analysis/doe/sampler_bo_hd_batches_test.ipynb
# =========================================================================== #
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                  [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

D_CONT = 5                       # continuous inputs
D_ALL = D_CONT + 1               # + 1 categorical (2 levels)
LEVEL_SLICE = {0.0: 0.35, 1.0: 0.65}   # categorical level -> 6th Hartmann coordinate


def hartmann6(X):
    """Standard 6-D Hartmann (minimize; global min -3.32237). X in [0,1]^6, (n, 6)."""
    X = np.atleast_2d(np.asarray(X, float))
    inner = ((X[:, None, :] - _H6_P[None, :, :]) ** 2 * _H6_A[None, :, :]).sum(-1)
    return -(np.exp(-inner) * _H6_ALPHA[None, :]).sum(-1)


def true_f6(X):
    """f for the sliced problem. X (n, 6): 5 continuous cols + level (0/1) col."""
    X = np.atleast_2d(np.asarray(X, float))
    x6 = np.where(X[:, -1] < 0.5, LEVEL_SLICE[0.0], LEVEL_SLICE[1.0])
    return hartmann6(np.column_stack([X[:, :D_CONT], x6]))


def _slice_min(level, n_scan=2 ** 13, n_polish=5):
    """Deterministic minimum of a level's 5-D slice: Sobol scan + L-BFGS polish."""
    from scipy.optimize import minimize
    S = qmc.Sobol(d=D_CONT, scramble=True, seed=7).random(n_scan)
    x6 = LEVEL_SLICE[level]
    f = lambda xc: float(hartmann6(np.append(xc, x6)[None, :])[0])
    vals = hartmann6(np.column_stack([S, np.full(len(S), x6)]))
    best = np.inf
    for i in np.argsort(vals)[:n_polish]:
        r = minimize(f, S[i], bounds=[(0, 1)] * D_CONT, method='L-BFGS-B')
        best = min(best, float(r.fun))
    return best


SLICE_MIN = {lv: _slice_min(lv) for lv in (0.0, 1.0)}
TRUE_MIN6 = min(SLICE_MIN.values())


def snap6(U):
    """Unit-cube design (n, 6) -> model space: 5 continuous cols kept, last col -> level 0/1."""
    U = np.atleast_2d(U)
    return np.column_stack([np.clip(U[:, :D_CONT], 0.0, 1.0), (U[:, -1] >= 0.5).astype(float)])


def observe6(X, sigma, rng):
    return true_f6(X) + rng.normal(0.0, sigma, size=len(X))


def _propose_ei6(X, Y, it_seed, n_pool=4096):
    """One LogEI proposal on a fresh Sobol candidate pool per level (identical for all arms)."""
    model = fit_gp(X, -Y[:, None])
    acq = LogExpectedImprovement(model, best_f=float((-Y).max()))
    pool_c = qmc.Sobol(d=D_CONT, scramble=True, seed=int(it_seed)).random(n_pool)
    pool = np.column_stack([np.tile(pool_c, (2, 1)),
                            np.repeat([0.0, 1.0], n_pool)])
    with torch.no_grad():
        a = acq(torch.as_tensor(pool, dtype=DTYPE).unsqueeze(1))
    return pool[int(a.argmax())][None, :]


def run_campaign(strategy, seed, n_init=12, total=60, bo_block=8, batch_size=8,
                 sigma=0.2, sa_iters=2000):
    """One campaign on the sliced Hartmann-6 problem.

    strategy: 'sobol_bo' | 'lhs_sa_bo' | 'sobol_bo_batch' | 'lhs_sa_bo_batch'.
    Pure-BO arms spend the whole post-init budget on LogEI proposals; '+batch'
    arms alternate ``bo_block`` BO evaluations with a ``batch_size`` space-filling
    batch (Sobol: continue the same engine; LHS+SA: ``conditional_sa_lhs``
    freezing every point evaluated so far) until ``total`` evaluations.
    Returns per-evaluation simple-regret history (length ``total - n_init + 1``).
    """
    sampler = 'sobol' if strategy.startswith('sobol') else 'lhs_sa'
    with_batches = strategy.endswith('_batch')
    rng = np.random.default_rng(1_000_000 * seed + hash(strategy) % 9973)

    eng = qmc.Sobol(d=D_ALL, scramble=True, seed=seed)          # kept for continuation
    if sampler == 'sobol':
        U0 = eng.random(n_init)
    else:
        U0 = conditional_sa_lhs(np.empty((0, D_ALL)), n_init, D_ALL, seed, n_iter=sa_iters)
    X = snap6(U0)
    Y = observe6(X, sigma, rng)
    best = [float(true_f6(X).min())]
    kinds = ['init'] * n_init                                   # provenance of each point

    it = 0
    while len(X) < total:
        if with_batches and len(X) > n_init and (len(X) - n_init) % (bo_block + batch_size) == bo_block:
            n_b = min(batch_size, total - len(X))               # space-filling batch
            if sampler == 'sobol':
                U_b = eng.random(n_b)
            else:
                U_b = conditional_sa_lhs(X.copy(), n_b, D_ALL, 10_000 + 100 * seed + it,
                                         n_iter=sa_iters)
            X_b = snap6(U_b)
            Y = np.concatenate([Y, observe6(X_b, sigma, rng)])
            for r in X_b:
                X = np.vstack([X, r]); best.append(float(true_f6(X).min()))
            kinds += ['batch'] * n_b
        else:                                                   # one BO proposal
            x_next = _propose_ei6(X, Y, it_seed=100_000 * seed + it)
            Y = np.concatenate([Y, observe6(x_next, sigma, rng)])
            X = np.vstack([X, x_next]); best.append(float(true_f6(X).min()))
            kinds.append('bo')
        it += 1

    return {'strategy': strategy, 'seed': seed, 'X': X, 'Y': Y, 'kinds': kinds,
            'regret': np.array(best) - TRUE_MIN6}


def run_campaign_benchmark(strategies, n_seeds=10, verbose=True, **kw):
    """All strategies x all seeds -> {strategy: [run dicts]}."""
    out = {}
    for s in strategies:
        runs = []
        for seed in range(n_seeds):
            runs.append(run_campaign(s, seed, **kw))
            if verbose:
                print(f'  {s}: seed {seed + 1}/{n_seeds} done '
                      f'(final regret {runs[-1]["regret"][-1]:.3f})', flush=True)
        out[s] = runs
    return out


# =========================================================================== #
# Generalized multi-problem campaign benchmark (3 sliced mixed-input problems)
# used by analysis/doe/sampler_bo_hd_batches_test.ipynb
# =========================================================================== #
_SHEKEL_BETA = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
_SHEKEL_C = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                      [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                      [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                      [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
_SHEKEL_SLICE = {0.0: 0.80, 1.0: 0.40}      # u4 per level; global basin sits at u = 0.4

def shekel_sliced(X):
    """Sliced Shekel-4 (m=10): 3 continuous inputs + level -> 4th coordinate.
    Deceptive — ten competing basins of different depth. Minimize."""
    X = np.atleast_2d(np.asarray(X, float))
    u4 = np.where(X[:, -1] < 0.5, _SHEKEL_SLICE[0.0], _SHEKEL_SLICE[1.0])
    U = 10.0 * np.column_stack([X[:, :3], u4])          # scale to [0, 10]^4
    d2 = ((U[:, None, :] - _SHEKEL_C.T[None, :, :]) ** 2).sum(-1)
    return -(1.0 / (d2 + _SHEKEL_BETA[None, :])).sum(-1)


_MICH_M = {0.0: 3, 1.0: 10}                  # level -> steepness (needle sharpness)

def michalewicz_sliced(X):
    """Sliced Michalewicz-5: 5 continuous inputs + level -> steepness m.
    Rugged, needle-like valleys in a mostly flat landscape. Minimize."""
    X = np.atleast_2d(np.asarray(X, float))
    xs = np.pi * X[:, :5]
    i = np.arange(1, 6)[None, :]
    base = np.sin(xs)
    ridge = np.sin(i * xs ** 2 / np.pi)
    out = np.empty(len(X))
    for lv, m in _MICH_M.items():
        mask = (X[:, -1] < 0.5) if lv == 0.0 else (X[:, -1] >= 0.5)
        if mask.any():
            out[mask] = -(base[mask] * ridge[mask] ** (2 * m)).sum(-1)
    return out


def _polish_slice_min(f, d_cont, level, n_scan=2 ** 15, n_polish=20, seed=7):
    """Deterministic minimum of one level's continuous slice: Sobol scan + L-BFGS."""
    from scipy.optimize import minimize
    S = qmc.Sobol(d=d_cont, scramble=True, seed=seed).random(n_scan)
    F = f(np.column_stack([S, np.full(len(S), level)]))
    g = lambda xc: float(f(np.append(xc, level)[None, :])[0])
    best = np.inf
    for i in np.argsort(F)[:n_polish]:
        r = minimize(g, S[i], bounds=[(0, 1)] * d_cont, method='L-BFGS-B')
        best = min(best, float(r.fun))
    return best


def _make_problem(name, f, d_cont):
    sm = {lv: _polish_slice_min(f, d_cont, lv) for lv in (0.0, 1.0)}
    return {'name': name, 'f': f, 'd_cont': d_cont,
            'slice_min': sm, 'true_min': min(sm.values())}


PROBLEMS = {
    'hartmann6':   {'name': 'hartmann6', 'f': true_f6, 'd_cont': D_CONT,
                    'slice_min': SLICE_MIN, 'true_min': TRUE_MIN6},
    'shekel4':     _make_problem('shekel4', shekel_sliced, 3),
    'michalewicz5': _make_problem('michalewicz5', michalewicz_sliced, 5),
}


def _propose_ei_p(problem, X, Y, it_seed, n_pool=4096):
    """LogEI proposal on a per-iteration Sobol candidate pool (identical across arms)."""
    model = fit_gp(X, -Y[:, None])
    acq = LogExpectedImprovement(model, best_f=float((-Y).max()))
    pool_c = qmc.Sobol(d=problem['d_cont'], scramble=True, seed=int(it_seed)).random(n_pool)
    pool = np.column_stack([np.tile(pool_c, (2, 1)), np.repeat([0.0, 1.0], n_pool)])
    with torch.no_grad():
        a = acq(torch.as_tensor(pool, dtype=DTYPE).unsqueeze(1))
    return pool[int(a.argmax())][None, :]


_STRATEGY_ID = {'sobol_bo': 0, 'lhs_sa_bo': 1, 'sobol_bo_batch': 2, 'lhs_sa_bo_batch': 3,
                'sobol_qnei': 4, 'lhs_sa_qnei': 5,
                'sobol_qnei12': 6, 'lhs_sa_qnei12': 7,
                'sobol_sf_only': 8, 'lhs_sa_sf_only': 9,
                'sobol_mixed': 10, 'lhs_sa_mixed': 11}


def run_campaign_p(problem_name, strategy, seed, n_init=12, total=60, bo_block=8,
                   batch_size=8, sigma=0.2, sa_iters=2000):
    """Generalized `run_campaign` over any problem in PROBLEMS.

    Strategies: the four BO/±batch arms, plus '<sampler>_qnei' — a *small* initial
    design followed by qLogNEI-proposed q-batches (see `_run_campaign_qnei`).
    """
    if strategy.endswith('_qnei'):
        return _run_campaign_qnei(problem_name, strategy, seed, total=total,
                                  sigma=sigma, sa_iters=sa_iters)
    if strategy.endswith('_qnei12'):                 # qNEI batches from the full 12-pt init
        return _run_campaign_qnei(problem_name, strategy, seed, small_init=n_init,
                                  total=total, sigma=sigma, sa_iters=sa_iters)
    if strategy.endswith('_sf_only'):                # space-filling batches only, no BO at all
        return _run_campaign_sf_only(problem_name, strategy, seed, n_init=n_init,
                                     total=total, batch_size=batch_size,
                                     sigma=sigma, sa_iters=sa_iters)
    if strategy.endswith('_mixed'):                  # mixed batches: 6 qNEI + 2 SF per 8
        return _run_campaign_mixed(problem_name, strategy, seed, n_init=n_init,
                                   total=total, sigma=sigma, sa_iters=sa_iters)
    pb = PROBLEMS[problem_name]
    f, d_all = pb['f'], pb['d_cont'] + 1
    sampler = 'sobol' if strategy.startswith('sobol') else 'lhs_sa'
    with_batches = strategy.endswith('_batch')
    rng = np.random.default_rng(1_000_000 * seed + 1013 * _STRATEGY_ID[strategy])

    def _snap(U):
        U = np.atleast_2d(U)
        return np.column_stack([np.clip(U[:, :-1], 0, 1), (U[:, -1] >= 0.5).astype(float)])

    eng = qmc.Sobol(d=d_all, scramble=True, seed=seed)
    if sampler == 'sobol':
        U0 = eng.random(n_init)
    else:
        U0 = conditional_sa_lhs(np.empty((0, d_all)), n_init, d_all, seed, n_iter=sa_iters)
    X = _snap(U0)
    Y = f(X) + rng.normal(0.0, sigma, size=len(X))
    best = [float(f(X).min())]
    kinds = ['init'] * n_init

    it = 0
    while len(X) < total:
        if with_batches and len(X) > n_init and (len(X) - n_init) % (bo_block + batch_size) == bo_block:
            n_b = min(batch_size, total - len(X))
            U_b = eng.random(n_b) if sampler == 'sobol' else conditional_sa_lhs(
                X.copy(), n_b, d_all, 10_000 + 100 * seed + it, n_iter=sa_iters)
            X_b = _snap(U_b)
            Y = np.concatenate([Y, f(X_b) + rng.normal(0.0, sigma, size=n_b)])
            for r in X_b:
                X = np.vstack([X, r]); best.append(float(f(X).min()))
            kinds += ['batch'] * n_b
        else:
            x_next = _propose_ei_p(pb, X, Y, it_seed=100_000 * seed + it)
            Y = np.concatenate([Y, f(x_next) + rng.normal(0.0, sigma, size=1)])
            X = np.vstack([X, x_next]); best.append(float(f(X).min()))
            kinds.append('bo')
        it += 1

    return {'problem': problem_name, 'strategy': strategy, 'seed': seed, 'X': X, 'Y': Y,
            'kinds': kinds, 'regret': np.array(best) - pb['true_min']}


def run_multi_benchmark(problems, strategies, n_seeds=10, verbose=True, **kw):
    """{problem: {strategy: [run dicts]}} over all combinations."""
    out = {}
    for p in problems:
        out[p] = {}
        for s in strategies:
            runs = []
            for seed in range(n_seeds):
                runs.append(run_campaign_p(p, s, seed, **kw))
                if verbose:
                    print(f'  {p} / {s}: seed {seed + 1}/{n_seeds} '
                          f'(final regret {runs[-1]["regret"][-1]:.3f})', flush=True)
            out[p][s] = runs
    return out


# =========================================================================== #
# Small-init + qLogNEI batch arm — the closest analogue of the real campaign
# (small screening design, then BO-proposed q-batches)
# =========================================================================== #
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler


def _propose_qnei_p(problem, X, Y, q, it_seed, n_pool=2048, mc=64, chunk=1024):
    """Sequential-greedy q-batch from a Sobol candidate pool using qLogNEI.

    Standard greedy batch construction: pick the pool argmax, add it to
    ``X_pending``, re-score, repeat q times — all under one fitted model (the
    batch is fabricated jointly, so no refit inside the batch)."""
    model = fit_gp(X, -Y[:, None])
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc]), seed=int(it_seed))
    acq = qLogNoisyExpectedImprovement(
        model, X_baseline=torch.as_tensor(X, dtype=DTYPE), sampler=sampler,
        prune_baseline=True)
    pool_c = qmc.Sobol(d=problem['d_cont'], scramble=True, seed=int(it_seed)).random(n_pool)
    pool = np.column_stack([np.tile(pool_c, (2, 1)), np.repeat([0.0, 1.0], n_pool)])
    pool_t = torch.as_tensor(pool, dtype=DTYPE)
    chosen = []
    for _ in range(q):
        vals = []
        with torch.no_grad():
            for j in range(0, len(pool_t), chunk):
                vals.append(acq(pool_t[j:j + chunk].unsqueeze(1)))
        i = int(torch.cat(vals).argmax())
        chosen.append(pool[i])
        acq.set_X_pending(torch.as_tensor(np.array(chosen), dtype=DTYPE))
    return np.array(chosen)


def _run_campaign_qnei(problem_name, strategy, seed, small_init=8, total=60,
                       q_batch=8, sigma=0.2, sa_iters=2000):
    """Small initial design (``small_init`` points) + qLogNEI batches of ``q_batch``."""
    pb = PROBLEMS[problem_name]
    f, d_all = pb['f'], pb['d_cont'] + 1
    sampler = 'sobol' if strategy.startswith('sobol') else 'lhs_sa'
    rng = np.random.default_rng(1_000_000 * seed + 1013 * _STRATEGY_ID[strategy])

    def _snap(U):
        U = np.atleast_2d(U)
        return np.column_stack([np.clip(U[:, :-1], 0, 1), (U[:, -1] >= 0.5).astype(float)])

    if sampler == 'sobol':
        U0 = qmc.Sobol(d=d_all, scramble=True, seed=seed).random(small_init)
    else:
        U0 = conditional_sa_lhs(np.empty((0, d_all)), small_init, d_all, seed, n_iter=sa_iters)
    X = _snap(U0)
    Y = f(X) + rng.normal(0.0, sigma, size=len(X))
    best = [float(f(X).min())]
    kinds = ['init'] * small_init

    it = 0
    while len(X) < total:
        q = min(q_batch, total - len(X))
        X_b = _propose_qnei_p(pb, X, Y, q, it_seed=100_000 * seed + it)
        Y = np.concatenate([Y, f(X_b) + rng.normal(0.0, sigma, size=len(X_b))])
        for r in X_b:
            X = np.vstack([X, r]); best.append(float(f(X).min()))
        kinds += ['qnei'] * len(X_b)
        it += 1

    return {'problem': problem_name, 'strategy': strategy, 'seed': seed, 'X': X, 'Y': Y,
            'kinds': kinds, 'regret': np.array(best) - pb['true_min']}


# =========================================================================== #
# Batch-policy isolation arms: SF-only (no BO), and mixed qNEI+SF batches
# =========================================================================== #
def _run_campaign_sf_only(problem_name, strategy, seed, n_init=12, total=60,
                          batch_size=8, sigma=0.2, sa_iters=2000):
    """Space-filling batches only — an incrementally built DoE, no BO at all."""
    pb = PROBLEMS[problem_name]
    f, d_all = pb['f'], pb['d_cont'] + 1
    sampler = 'sobol' if strategy.startswith('sobol') else 'lhs_sa'
    rng = np.random.default_rng(1_000_000 * seed + 1013 * _STRATEGY_ID[strategy])

    def _snap(U):
        U = np.atleast_2d(U)
        return np.column_stack([np.clip(U[:, :-1], 0, 1), (U[:, -1] >= 0.5).astype(float)])

    eng = qmc.Sobol(d=d_all, scramble=True, seed=seed)
    if sampler == 'sobol':
        U0 = eng.random(n_init)
    else:
        U0 = conditional_sa_lhs(np.empty((0, d_all)), n_init, d_all, seed, n_iter=sa_iters)
    X = _snap(U0)
    Y = f(X) + rng.normal(0.0, sigma, size=len(X))
    best = [float(f(X).min())]
    kinds = ['init'] * n_init

    it = 0
    while len(X) < total:
        n_b = min(batch_size, total - len(X))
        U_b = eng.random(n_b) if sampler == 'sobol' else conditional_sa_lhs(
            X.copy(), n_b, d_all, 10_000 + 100 * seed + it, n_iter=sa_iters)
        X_b = _snap(U_b)
        Y = np.concatenate([Y, f(X_b) + rng.normal(0.0, sigma, size=n_b)])
        for r in X_b:
            X = np.vstack([X, r]); best.append(float(f(X).min()))
        kinds += ['batch'] * n_b
        it += 1

    return {'problem': problem_name, 'strategy': strategy, 'seed': seed, 'X': X, 'Y': Y,
            'kinds': kinds, 'regret': np.array(best) - pb['true_min']}


def _run_campaign_mixed(problem_name, strategy, seed, n_init=12, total=60,
                        q_batch=8, n_sf=2, sigma=0.2, sa_iters=2000):
    """Mixed batches: each batch of ``q_batch`` = (q_batch - n_sf) qNEI-proposed points
    + ``n_sf`` space-filling points (conditioned on everything incl. the pending qNEI
    picks). Tests the 'mostly BO + fixed exploration ration' policy."""
    pb = PROBLEMS[problem_name]
    f, d_all = pb['f'], pb['d_cont'] + 1
    sampler = 'sobol' if strategy.startswith('sobol') else 'lhs_sa'
    rng = np.random.default_rng(1_000_000 * seed + 1013 * _STRATEGY_ID[strategy])

    def _snap(U):
        U = np.atleast_2d(U)
        return np.column_stack([np.clip(U[:, :-1], 0, 1), (U[:, -1] >= 0.5).astype(float)])

    eng = qmc.Sobol(d=d_all, scramble=True, seed=seed)
    if sampler == 'sobol':
        U0 = eng.random(n_init)
    else:
        U0 = conditional_sa_lhs(np.empty((0, d_all)), n_init, d_all, seed, n_iter=sa_iters)
    X = _snap(U0)
    Y = f(X) + rng.normal(0.0, sigma, size=len(X))
    best = [float(f(X).min())]
    kinds = ['init'] * n_init

    it = 0
    while len(X) < total:
        n_b = min(q_batch, total - len(X))
        n_sf_b = min(n_sf, n_b)
        q_bo = n_b - n_sf_b
        parts = []
        if q_bo > 0:
            parts.append(_propose_qnei_p(pb, X, Y, q_bo, it_seed=100_000 * seed + it))
        if n_sf_b > 0:
            pending = np.vstack([X] + parts)
            U_sf = eng.random(n_sf_b) if sampler == 'sobol' else conditional_sa_lhs(
                pending, n_sf_b, d_all, 10_000 + 100 * seed + it, n_iter=sa_iters)
            parts.append(_snap(U_sf))
        X_b = np.vstack(parts)
        Y = np.concatenate([Y, f(X_b) + rng.normal(0.0, sigma, size=len(X_b))])
        for r in X_b:
            X = np.vstack([X, r]); best.append(float(f(X).min()))
        kinds += ['bo'] * q_bo + ['batch'] * n_sf_b
        it += 1

    return {'problem': problem_name, 'strategy': strategy, 'seed': seed, 'X': X, 'Y': Y,
            'kinds': kinds, 'regret': np.array(best) - pb['true_min']}
