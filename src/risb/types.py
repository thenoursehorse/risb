"""Type aliases and interfaces used across risb."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from fockit.algebra import OpSum
from numpy.typing import NDArray

type GfStruct = list[tuple[str, int]]
"""Block structure of one cluster: (block name, block size) pairs."""

type BlockMatrix = dict[str, NDArray[np.inexact]]
"""Real or complex array for each block, e.g. R, Lambda, densities, h0_k."""

type BlockMatrixLike = Mapping[str, NDArray[np.inexact]]
"""Read-only input form of :data:`BlockMatrix`. Accepts any mapping of real or complex arrays, e.g. a dict of float64 arrays."""

type BlockRealArray = dict[str, NDArray[np.float64]]
"""Real array for each block, e.g. band energies, k-weights."""

type DensityMatchingJacobian = Literal["2-point", "analytic"]
"""Derivative method for thermal density matching."""

type DensityMatchingRegularization = Literal["uniform", "response-weighted"]
"""How the anchor penalizes changes to the thermal matching parameters."""

type Symmetrize = Callable[[list[BlockMatrix]], list[BlockMatrix]]
"""Takes a block matrix for each cluster and returns the symmetrized ones."""

type GaugeFix = Callable[
    [NDArray[np.inexact], NDArray[np.inexact]],
    tuple[NDArray[np.inexact], NDArray[np.inexact]],
]
"""Takes the R and Lambda of a block and returns them rotated into a chosen quasiparticle basis."""

type SmearFunction = Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]
"""Smears band energies at inverse temperature beta and chemical potential mu into occupations."""


type FixedPointFunction = Callable[
    [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]
"""One evaluation of a fixed-point map and its root residual."""


@dataclass(frozen=True)
class Checkpoint:
    """
    Saved state and the function used to restore it.

    Parameters
    ----------
    restore : callable
        Restore saved data and recalculate outputs as needed.
    evaluations : int, optional
        Number of cycle evaluations needed to restore the saved state.

    """

    restore: Callable[[], None]
    evaluations: int = 0


@runtime_checkable
class Checkpointable(Protocol):
    """A calculation whose saved state can be restored."""

    def checkpoint(self) -> Checkpoint:
        """Return a checkpoint of the last completed calculation."""


class ImpuritySolver(Protocol):
    """
    Solver for the impurity problem of one cluster.

    Subclass this to write your own solver, so that type checkers verify it
    implements everything below.
    """

    gf_struct: GfStruct
    """Block structure of the physical orbitals (c-electrons)."""

    h_int: OpSum
    """Interaction of the physical orbitals."""

    rho_f: BlockMatrix
    """f-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_cf: BlockMatrix
    """c-f hybridization density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    rho_c: BlockMatrix
    """c-electron density matrix of each block in :attr:`gf_struct`, set by :meth:`solve`."""

    entropy: float
    """Entropy of the states of the impurity problem, set by :meth:`solve`."""

    def checkpoint(self) -> Checkpoint:
        """
        Save any state that affects the next impurity solve.

        Override this if solve uses results from earlier solves. The default
        saves nothing. When restoring its checkpoint, the lattice sets the
        Hamiltonian and solves the impurity again.
        """
        return Checkpoint(lambda: None)

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


class PinFullEmpty(Protocol):
    """Levels of the auxiliary orbitals that are full or empty, for :class:`risb.LatticeSolver`."""

    def bath_levels(
        self,
        Lambda_c: NDArray[np.inexact],
        rho_qp: NDArray[np.inexact],
        energy_scale: float,
    ) -> NDArray[np.inexact]:
        """Return the bath levels ``Lambda_c`` of one block, set where the quasiparticle orbitals of ``rho_qp`` are full or empty."""

    def qp_levels(
        self,
        Lambda: NDArray[np.inexact],
        Lambda_in: NDArray[np.inexact],
        rho_f: NDArray[np.inexact],
        energy_scale: float,
        half_bandwidth: float,
    ) -> NDArray[np.inexact]:
        """Return the new quasiparticle levels ``Lambda`` of one block, set where the orbitals of ``rho_f`` are full or empty, from the levels ``Lambda_in`` of the cycle."""


class DensityFeedback(Protocol):
    """Shift of the quasiparticle levels by the density mismatch, for :class:`risb.LatticeSolver`."""

    def qp_levels(
        self,
        Lambda: NDArray[np.inexact],
        R: NDArray[np.inexact],
        f2: NDArray[np.inexact],
        occupation_slope: float,
    ) -> NDArray[np.inexact]:
        """
        Return the quasiparticle levels ``Lambda`` of one block, shifted using ``f2 = rho_f - rho_qp`` and the renormalization matrix ``R``.

        ``occupation_slope`` is the largest slope of the occupation function
        of the quasiparticle bands, infinite for a step function.
        """


type ImpurityChargeSusceptibility = Callable[["ImpuritySolver"], float]
"""Charge susceptibility of the physical orbitals of a solved impurity problem, e.g., :func:`risb.impurity_solvers.fockit.charge_susceptibility`."""


class ChemicalPotentialStep(Protocol):
    """Step of the chemical potential in a cycle of :class:`risb.PhysicalMuLatticeSolver` or :class:`risb.ThermalLatticeSolver`."""

    def step(
        self,
        filling_error: float,
        slope: Callable[[], float],
        history: Sequence[tuple[float, float]],
    ) -> float:
        """
        Return the change of the chemical potential.

        ``filling_error`` is ``n_electrons - n_target`` at the current
        chemical potential. ``slope()`` returns ``dn_electrons/dmu`` from the
        bands of the orbitals in no cluster and the impurity charge
        susceptibilities given to the lattice solver; it costs a band
        integration and possibly impurity calculations. ``history`` holds the
        chemical potential and ``n_electrons`` of each impurity solution in
        this cycle, in order, the current one last.
        """


class FixedPointProblem(Protocol):
    """
    A problem for :func:`risb.optimize.solve_fixed_point`, e.g., :class:`risb.LatticeSolver`.

    One cycle maps an input vector ``x`` to a new guess, which is ``x`` at a
    solution.
    """

    def get_x(self) -> NDArray[np.float64]:
        """Return the current guess, where a solve starts."""

    def cycle(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the new guess and the residual of the root functions after one cycle from ``x``."""
