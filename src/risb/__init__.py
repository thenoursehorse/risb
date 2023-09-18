from .solve_lattice import LatticeSolver

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("risb")
except PackageNotFoundError:
    __version__ = "unknown version"