"""Optimizers for the root and minimization problems."""

from .diis import DIIS
from .linear_mixing import LinearMixing
from .solver_newton import NewtonSolver

__all__ = ["DIIS", "LinearMixing", "NewtonSolver"]

# FIXME sort out doc inheritance from base class
# DIIS.__doc__ = NewtonSolver.__doc__
# LinearMixing.__doc__ = NewtonSolver.__doc__
