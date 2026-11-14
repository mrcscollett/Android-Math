"""Step-by-step differential equation engine.

Pure Python + SymPy. Runs on-device through Chaquopy and can be exercised
directly on a desktop (see tests/python).
"""
# The per-tool entry points (api.linalg, api.calculus, ...) are not re-exported:
# their names would shadow the modules of the same name. The app calls `tool`.
# The physics scenarios live in the `scenarios` package, so that `physics` can be
# the name of the entry-point function.
from .api import (boundary, eigen_problem, fourier, numeric, pde, physics, physics_catalog,  # noqa: F401
                  preview, solve, system, tool, transform, warmup)
