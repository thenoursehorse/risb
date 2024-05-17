"""Optimizers for the root and minimization problems."""

from .diis import DIIS  # noqa: F401
from .linear_mixing import LinearMixing  # noqa: F401

# FIXME sort out doc inheritance from base class
# DIIS.__doc__ = NewtonSolver.__doc__
# LinearMixing.__doc__ = NewtonSolver.__doc__
