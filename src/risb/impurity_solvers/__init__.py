"""Solvers for the impurity problem in RISB."""

from .atom_diag import AtomDiagImpuritySolver
from .dummy import DummyImpuritySolver

__all__ = ["AtomDiagImpuritySolver", "DummyImpuritySolver"]
