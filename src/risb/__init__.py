"""Library to use rotationally invariant slave-boson mean-field theory for lattice strongly correlated electrons."""

from importlib.metadata import PackageNotFoundError, version

from .solve_lattice import LatticeSolver
from .solve_lattice_base import BaseLatticeSolver
from .solve_lattice_thermal import ThermalLatticeSolver

__all__ = ["BaseLatticeSolver", "LatticeSolver", "ThermalLatticeSolver"]

try:
    __version__ = version("risb")
except PackageNotFoundError:
    __version__ = "unknown version"
