"""Type aliases and interfaces used across risb."""

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type GfStruct = list[tuple[str, int]]
"""Block structure of one cluster: (block name, block size) pairs."""

type BlockMatrix = dict[str, NDArray[np.inexact]]
"""Real or complex array for each block, e.g. R, Lambda, densities, h0_k."""

type BlockMatrixLike = Mapping[str, NDArray[np.inexact]]
"""Read-only input form of :data:`BlockMatrix`. Accepts any mapping of real or complex arrays, e.g. a dict of float64 arrays."""

type BlockRealArray = dict[str, NDArray[np.float64]]
"""Real array for each block, e.g. band energies, k-weights."""

type UpdateWeights = Callable[[BlockRealArray], BlockRealArray]
"""Takes band energies for each block and returns the k-space integration weights."""

type Symmetrize = Callable[[list[BlockMatrix]], list[BlockMatrix]]
"""Takes a block matrix for each cluster and returns the symmetrized ones."""

type SmearFunction = Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]
"""Smears band energies at inverse temperature beta and chemical potential mu into occupations."""


class ImpuritySolver(Protocol):
    """
    Solver for the impurity problem of one cluster.

    Subclass this to write your own solver, so that type checkers verify it
    implements everything below.
    """

    gf_struct: GfStruct
    """Block structure of the physical orbitals (c-electrons)."""

    rho_f: BlockMatrix
    """f-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_cf: BlockMatrix
    """c-f hybridization density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_c: BlockMatrix
    """c-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    entropy: float
    """Entropy of the states of the impurity problem, set by :meth:`solve`."""

    def set_h_imp(
        self,
        Lambda_c: BlockMatrixLike,
        D: BlockMatrixLike,
        h0_loc_matrix: BlockMatrixLike | None = None,
        mu: float | None = None,
    ) -> None:
        """Set the impurity Hamiltonian, with the chemical potential `mu` if it is given."""

    def solve(self) -> None:
        """Solve the impurity problem and set :attr:`rho_f`, :attr:`rho_cf`, and :attr:`rho_c` for every block."""

    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        """Return the f-electron density matrix of block ``bl``."""

    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        """Return the c-f hybridization density matrix of block ``bl``."""

    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        """Return the c-electron density matrix of block ``bl``."""


class FixedPointProblem(Protocol):
    """
    A problem that :func:`risb.optimize.drive` can solve, e.g., :class:`risb.LatticeSolver`.

    One cycle maps an input vector ``x`` to a new guess, which is ``x`` at a
    solution.
    """

    def get_x(self) -> NDArray[np.float64]:
        """Return the current guess, where a solve starts."""

    def cycle(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the new guess and the residual of the root functions after one cycle from ``x``."""
