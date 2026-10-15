"""Solvers for the impurity problem in RISB."""

from .dummy import DummyImpuritySolver
from .fockit import FockitImpuritySolver

__all__ = ["DummyImpuritySolver", "FockitImpuritySolver"]
