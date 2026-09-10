"""Reusable utilities for the DMREF PdO analysis.

The :mod:`utils.mobo` module holds the multi-objective Bayesian-optimization
building blocks (GP surrogate, Pareto/hypervolume helpers, qLogNEHVI proposal
loop, and unit-cube encoding). Everything there is written to be dimension- and
objective-agnostic so the exact same model + BO structure can be exercised on a
small 1-D test as well as on the full 10-factor / 2-objective PdO problem.

The :mod:`utils.xrd_crystallinity` module turns the fab team's paired-column
XRD workbooks into PdO(101) peak metrics and a reference-normalised
crystallinity index per film / per design condition (also runnable as a CLI:
``python utils/xrd_crystallinity.py --help``).
"""

from . import mobo  # noqa: F401
from . import xrd_crystallinity  # noqa: F401
