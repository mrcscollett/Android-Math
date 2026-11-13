"""Step-by-step differential equation engine.

Pure Python + SymPy. Runs on-device through Chaquopy and can be exercised
directly on a desktop (see tests/python).
"""
from .api import (boundary, eigen_problem, fourier, numeric, preview, solve,  # noqa: F401
                  system, transform, warmup)
