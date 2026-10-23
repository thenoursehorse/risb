"""Solvers for the impurity problem in RISB."""

from typing import TYPE_CHECKING

from .dummy import DummyImpuritySolver
from .fockit import FockitImpuritySolver, FockitThermalImpuritySolver

if TYPE_CHECKING:
    from .block2 import Block2ImpuritySolver

__all__ = [
    "Block2ImpuritySolver",
    "DummyImpuritySolver",
    "FockitImpuritySolver",
    "FockitThermalImpuritySolver",
]


def __getattr__(name: str) -> object:
    # Block2ImpuritySolver needs the optional block2, so it is only imported when used
    if name == "Block2ImpuritySolver":
        from .block2 import Block2ImpuritySolver  # noqa: PLC0415

        return Block2ImpuritySolver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
