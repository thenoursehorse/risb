"""Levels of the auxiliary orbitals that are full or empty, for :class:`risb.LatticeSolver`."""

from typing import override

import numpy as np
from numpy.typing import NDArray

from risb.types import PinFullEmpty


class FullEmptyPinning(PinFullEmpty):
    """
    Move the levels of full or empty auxiliary orbitals far from the active levels.

    An orbital is full or empty when its density eigenvalue ``n`` satisfies
    ``n * (1 - n) < edge_tol``. Bath levels use the quasiparticle density
    :attr:`~risb.LatticeSolver.rho_qp`; quasiparticle levels use
    :attr:`~risb.LatticeSolver.rho_f` after the impurity solve. This enforces
    complementary bath occupations in the decoupled limit, but can affect
    convergence and which solution is found.

    Close to full or empty, the quasiparticle level grows as
    ``1 / sqrt(n * (1 - n))``, and one cycle can move it far away. So the
    change in a cycle of the levels of nearly full or empty quasiparticle
    orbitals is limited.

    Parameters
    ----------
    edge_tol : float, optional
        An orbital is full or empty when ``n * (1 - n)`` is below this.
        Default is ``1e-12``. The gradients in :mod:`risb.helpers` keep
        :data:`risb.helpers.EDGE_TOL`.
    margin : float, optional
        Distance of the pinned levels from the active levels, in units of the
        energy scale of the model. Default is ``10``.
    near_edge_tol : float, optional
        A quasiparticle orbital that is not full or empty is nearly full or
        empty when ``n * (1 - n)`` is below this. Default is ``1e-3``.
    step_cap : float or None, optional
        Largest change in a cycle (spectral norm) of the levels of the nearly
        full or empty quasiparticle orbitals, in units of the kinetic
        half-bandwidth. ``None`` does not limit it. Default is ``2``.

    """

    def __init__(
        self,
        edge_tol: float = 1e-12,
        margin: float = 10.0,
        near_edge_tol: float = 1e-3,
        step_cap: float | None = 2.0,
    ) -> None:
        self.edge_tol = edge_tol
        self.margin = margin
        self.near_edge_tol = near_edge_tol
        self.step_cap = step_cap

    @override
    def bath_levels(
        self,
        Lambda_c: NDArray[np.inexact],
        rho_qp: NDArray[np.inexact],
        energy_scale: float,
    ) -> NDArray[np.inexact]:
        """
        Return the bath levels with the bath partners of full or empty quasiparticle orbitals far away.

        A completely full (empty) quasiparticle orbital has no bath partner.
        Its D is zero, and the limit of the gradient term that would put its
        bath orbital empty (full) is not finite, so put the bath level far
        away instead. Otherwise the bath orbital competes for electrons and
        the impurity problem can end at R = 0 with the wrong filling.
        """
        eigenvalues, vectors = np.linalg.eigh(rho_qp)
        edge = eigenvalues * (1 - eigenvalues) < self.edge_tol
        scale = self.margin * energy_scale
        # The bath term is f Lambda_c f^dag, so filling the bath orbital costs
        # -Lambda_c: empty for a full quasiparticle orbital, full for an empty one
        for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
            level = -scale if delta > 0.5 else scale
            Lambda_c = Lambda_c + (level - np.real(v.conj() @ Lambda_c @ v)) * np.outer(
                v, v.conj()
            )
        return Lambda_c

    @override
    def qp_levels(
        self,
        Lambda: NDArray[np.inexact],
        Lambda_in: NDArray[np.inexact],
        rho_f: NDArray[np.inexact],
        energy_scale: float,
        half_bandwidth: float,
    ) -> NDArray[np.inexact]:
        """
        Return the quasiparticle levels with full or empty orbitals below or above every other band.

        The quasiparticle level of a full (empty) orbital is only fixed to be
        below (above) the chemical potential, so keep it below (above) every
        other quasiparticle band. This finds the solution where the orbital
        stays decoupled. A solution with a lower energy, where the
        quasiparticle orbital is not exactly full (empty) and is an extra
        channel for the other orbitals, can also exist.
        """
        eigenvalues, vectors = np.linalg.eigh(rho_f)
        spread = eigenvalues * (1 - eigenvalues)
        edge = spread < self.edge_tol
        # rho_qp is the transpose of n_F(lambda - mu), so these are the
        # eigenvectors of lambda
        vectors = vectors.conj()

        if self.step_cap is not None:
            near = vectors[:, ~edge & (spread < self.near_edge_tol)]
            P = near @ near.conj().T
            step = P @ (Lambda - Lambda_in) @ P
            norm = np.linalg.norm(step, 2)
            limit = self.step_cap * half_bandwidth
            if norm > limit:
                Lambda = Lambda - (1 - limit / norm) * step

        if not edge.any():
            return Lambda
        active = vectors[:, ~edge]
        others = (
            np.linalg.eigvalsh(active.conj().T @ Lambda @ active)
            if active.size
            else np.zeros(1)
        )
        margin = half_bandwidth + self.margin * energy_scale
        for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
            level = others.min() - margin if delta > 0.5 else others.max() + margin
            Lambda = Lambda + (level - np.real(v.conj() @ Lambda @ v)) * np.outer(
                v, v.conj()
            )
        return Lambda
