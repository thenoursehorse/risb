"""Type aliases and interfaces used across risb."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

type GfStruct = list[tuple[str, int]]
"""Block structure of one cluster: (block name, block size) pairs."""

type BlockMatrix = dict[str, NDArray[np.inexact]]
"""Real or complex array for each block, e.g. R, Lambda, densities, h0_k."""

type BlockRealArray = dict[str, NDArray[np.float64]]
"""Real array for each block, e.g. band energies, k-weights."""

type UpdateWeights = Callable[[BlockRealArray], BlockRealArray]
"""Takes band energies for each block and returns the k-space integration weights."""

type Symmetrize = Callable[[list[BlockMatrix]], list[BlockMatrix]]
"""Takes a block matrix for each cluster and returns the symmetrized ones."""

type SmearFunction = Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]
"""Smears band energies at inverse temperature beta and chemical potential mu into occupations."""

type RootFinder = Callable[..., NDArray[np.float64] | OptimizeResult]
"""Root finder called as ``root(fun, x0, args=..., **kwargs)``, e.g. DIIS.solve or scipy.optimize.root."""


class Embedding(Protocol):
    """
    Impurity solver for the embedding problem of one cluster.

    Subclass this to write your own solver, so that type checkers verify it
    implements everything below.
    """

    gf_struct: GfStruct
    """Block structure of the c-electrons."""

    rho_f: BlockMatrix
    """f-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_cf: BlockMatrix
    """c-f hybridization density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_c: BlockMatrix
    """c-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    def set_h_emb(
        self,
        Lambda_c: BlockMatrix,
        D: BlockMatrix,
        h0_loc_matrix: BlockMatrix | None = None,
    ) -> None:
        """Set the embedding Hamiltonian."""

    def solve(self) -> None:
        """Solve the embedding problem and set :attr:`rho_f`, :attr:`rho_cf`, and :attr:`rho_c` for every block."""

    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        """Return the f-electron density matrix of block ``bl``."""

    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        """Return the c-f hybridization density matrix of block ``bl``."""

    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        """Return the c-electron density matrix of block ``bl``."""
