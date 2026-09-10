"""Worker: run one (problem, strategy) arm of the campaign benchmark and cache it.

The 3-problem x 4-arm x N-seed benchmark in
``analysis/doe/sampler_bo_hd_batches_test.ipynb`` is embarrassingly parallel, so
on a cluster the arms are precomputed by parallel workers:

    python -m utils.run_doe_bo_arm <problem> <strategy> [n_seeds]

e.g. launch all 12 arms with xargs (from the repo root):

    for p in hartmann6 shekel4 michalewicz5; do
      for s in sobol_bo lhs_sa_bo sobol_bo_batch lhs_sa_bo_batch; do echo "$p $s"; done
    done | OMP_NUM_THREADS=2 xargs -P 12 -n 2 python -m utils.run_doe_bo_arm

Results go to ``output/sampler_bo_1d_test/campaign_cache/<problem>__<strategy>.pkl``.
``run_campaign_p`` is deterministic per (problem, strategy, seed), so cached and
inline-computed results are identical — the notebook loads the cache when present.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, 'output', 'sampler_bo_1d_test', 'campaign_cache')


def main():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, REPO_ROOT)
    from utils.doe_bo_test import run_campaign_p

    problem, strategy = sys.argv[1], sys.argv[2]
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    runs = []
    for seed in range(n_seeds):
        runs.append(run_campaign_p(problem, strategy, seed))
        print(f'{problem}/{strategy}: seed {seed + 1}/{n_seeds} '
              f'(final regret {runs[-1]["regret"][-1]:.3f})', flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f'{problem}__{strategy}.pkl')
    with open(out, 'wb') as fh:
        pickle.dump(runs, fh)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
