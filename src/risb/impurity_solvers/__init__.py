"""Solvers for the impurity problem in RISB."""

from typing import TYPE_CHECKING

from .dummy import DummyImpuritySolver
from .fockit import FockitImpuritySolver, FockitThermalImpuritySolver

if TYPE_CHECKING:
    from .block2 import Block2ImpuritySolver, Block2ThermalImpuritySolver

__all__ = [
    "Block2ImpuritySolver",
    "Block2ThermalImpuritySolver",
    "DummyImpuritySolver",
    "FockitImpuritySolver",
    "FockitThermalImpuritySolver",
]


def __getattr__(name: str) -> object:
    # The block2 solvers need the optional block2, so they are only imported
    # when used
    if name in ("Block2ImpuritySolver", "Block2ThermalImpuritySolver"):
        from . import block2  # noqa: PLC0415

        return getattr(block2, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__
