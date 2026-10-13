"""Solvers for the impurity problem in RISB."""

from typing import TYPE_CHECKING

from .dummy import DummyImpuritySolver
from .fockit import FockitImpuritySolver

if TYPE_CHECKING:
    from .atom_diag import AtomDiagImpuritySolver

__all__ = ["AtomDiagImpuritySolver", "DummyImpuritySolver", "FockitImpuritySolver"]


def __getattr__(name: str) -> object:
    # AtomDiagImpuritySolver needs TRIQS, so it is only imported when used
    if name == "AtomDiagImpuritySolver":
        from .atom_diag import AtomDiagImpuritySolver  # noqa: PLC0415

        return AtomDiagImpuritySolver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
