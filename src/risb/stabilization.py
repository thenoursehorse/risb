"""Stabilization of the level updates of :class:`risb.LatticeSolver` and of the chemical potential."""

from collections.abc import Callable, Sequence
from typing import Literal, override

import numpy as np
from numpy.typing import NDArray

from risb.helpers import _occupations
from risb.types import ChemicalPotentialStep, DensityFeedback, PinFullEmpty


def clip_singular_values(A: NDArray[np.inexact]) -> NDArray[np.inexact]:
    r"""
    Return the matrix with its singular values above one set to one.

    :math:`R^\dagger R` is the spectral weight of the quasiparticles in the
    physical orbitals. It is at most one, and reaches one only as the number of
    ghosts goes to infinity, so the singular values of :math:`R` are at most
    one.

    Parameters
    ----------
    A : numpy.ndarray
        A matrix, e.g., :math:`R` of a block.

    Returns
    -------
    numpy.ndarray
        `A` itself if its singular values are at most one.

    """
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s.max() <= 1:
        return A
    result: NDArray[np.inexact] = (U * np.minimum(s, 1)) @ Vh
    return result


class FullEmptyPinning(PinFullEmpty):
    """
    Decouple the auxiliary orbitals that are close to full or empty, and move their levels away from the others.

    An auxiliary orbital is close to full or empty when its occupation ``n``,
    an eigenvalue of :attr:`~risb.LatticeSolver.rho_qp` (for the bath) or of
    :attr:`~risb.LatticeSolver.rho_f` after the impurity solve (for the
    quasiparticles), has ``n * (1 - n)`` below ``onset_tol``. It is pinned
    with a weight ``w`` that rises smoothly, as a smoothstep in
    ``log10(n * (1 - n))``, from 0 at ``onset_tol`` to 1 at ``edge_tol``.
    Its row of D (for the bath) or R (for the quasiparticles) and its
    couplings to the other levels are scaled by ``1 - w``, and its level
    moves a fraction ``w`` of the way to a level far from the others:

    - the bath partner of a full (empty) quasiparticle orbital to
      ``-pinned_level_distance`` (``+pinned_level_distance``) times the
      energy scale of the model, so that the bath orbital is empty (full);
    - a full (empty) quasiparticle orbital below (above) every other level,
      by the kinetic half-bandwidth plus ``pinned_level_distance`` times the
      energy scale.

    An orbital that is full or empty to below ``edge_tol`` is decoupled, and
    the RISB equations do not fix its levels; the pinning chooses them. This
    finds the solution where the orbital stays decoupled. A solution with a
    lower energy, where it is weakly coupled as an extra channel for the
    other orbitals, can also exist. With ``onset_tol`` equal to ``edge_tol``
    the pinning switches on at once.

    Close to full or empty, the quasiparticle level changes fast with the
    other parameters, and one cycle can move it far away. So the change in a
    cycle of the levels of nearly full or empty quasiparticle orbitals is
    limited. The occupations come from the density matrices of the occupied
    and of the empty states separately, which keep their digits close to 0
    and 1.

    Parameters
    ----------
    onset_tol : float, optional
        Value of ``n * (1 - n)`` below which the pinning starts to act.
    edge_tol : float, optional
        Value of ``n * (1 - n)`` below which the pinning acts completely.
    pinned_level_distance : float, optional
        Distance of the pinned levels from the other levels, in units of the
        energy scale of the model.
    near_edge_tol : float, optional
        A quasiparticle orbital is nearly full or empty when ``n * (1 - n)``
        is below this.
    step_cap : float or None, optional
        Scale for limiting quasiparticle level changes, in units of the
        kinetic half-bandwidth. ``None`` disables limiting.
    step_cap_method : {"clip", "smooth"}, optional
        ``"clip"`` limits changes to the levels of nearly full or empty
        orbitals, excluding those below ``edge_tol``. It uses the spectral
        norm. ``"smooth"`` weights the change continuously as occupations
        approach 0 or 1 and limits its Frobenius norm.

    """

    def __init__(
        self,
        onset_tol: float = 1e-10,
        edge_tol: float = 1e-12,
        pinned_level_distance: float = 10.0,
        near_edge_tol: float = 1e-3,
        step_cap: float | None = 2.0,
        *,
        step_cap_method: Literal["clip", "smooth"] = "smooth",
    ) -> None:
        if onset_tol < edge_tol:
            msg = "onset_tol must not be below edge_tol."
            raise ValueError(msg)
        self.onset_tol = onset_tol
        self.edge_tol = edge_tol
        self.pinned_level_distance = pinned_level_distance
        self.near_edge_tol = near_edge_tol
        self.step_cap = step_cap
        methods = {"clip": self._clip_level_step, "smooth": self._smooth_level_step}
        if step_cap_method not in methods:
            msg = f"Unknown step-cap method {step_cap_method!r}."
            raise ValueError(msg)
        self._limit_level_step = methods[step_cap_method]

    @override
    def bath(
        self,
        D: NDArray[np.inexact],
        Lambda_c: NDArray[np.inexact],
        rho_qp: NDArray[np.inexact],
        hole_rho_qp: NDArray[np.inexact],
        energy_scale: float,
    ) -> tuple[NDArray[np.inexact], NDArray[np.inexact]]:
        """
        Return D and the bath levels, with the bath partners of quasiparticle orbitals close to full or empty decoupled.

        A completely full (empty) quasiparticle orbital has no bath partner:
        the bath orbital would compete for electrons, and the impurity problem
        can end at R = 0 with the wrong filling. So its row of D goes to zero
        and its bath level far away, empty (full).
        """
        # rho_qp, hole_rho_qp and Lambda_c are the transposes of the matrices
        # in the basis of the quasiparticles, in which D has its rows
        vectors, n, s = _occupations(rho_qp.T, hole_rho_qp.T)
        w = self._weights(n * s)
        if not w.any():
            return D, Lambda_c
        scale = self.pinned_level_distance * energy_scale
        # The bath term is f Lambda_c f^dag, so filling the bath orbital costs
        # -Lambda_c: empty for a full quasiparticle orbital, full for an empty one
        targets = np.where(n > 0.5, -scale, scale)
        levels, D = _decouple(Lambda_c.T, D, vectors, w, targets)
        return D, levels.T

    @override
    def qp(
        self,
        R: NDArray[np.inexact],
        Lambda: NDArray[np.inexact],
        Lambda_in: NDArray[np.inexact],
        rho_f: NDArray[np.inexact],
        hole_rho_f: NDArray[np.inexact],
        energy_scale: float,
        half_bandwidth: float,
    ) -> tuple[NDArray[np.inexact], NDArray[np.inexact]]:
        """
        Return R and the quasiparticle levels, with the orbitals close to full or empty decoupled below or above every other band.

        The quasiparticle level of a full (empty) orbital is only fixed to be
        below (above) the chemical potential, so keep it below (above) every
        other quasiparticle band.
        """
        vectors, n, s = _occupations(rho_f, hole_rho_f)
        # rho_qp is the transpose of n_F(lambda - mu), so these are the
        # eigenvectors of lambda, and the basis of the rows of R
        vectors = vectors.conj()
        spread = n * s
        if self.step_cap is not None:
            Lambda = self._limit_level_step(
                Lambda,
                Lambda_in,
                spread,
                vectors,
                self.step_cap * half_bandwidth,
            )
        w = self._weights(spread)
        if not w.any():
            return R, Lambda
        active = vectors[:, w == 0]
        others = (
            np.linalg.eigvalsh(active.conj().T @ Lambda @ active)
            if active.size
            else np.zeros(1)
        )
        distance = half_bandwidth + self.pinned_level_distance * energy_scale
        targets = np.where(n > 0.5, others.min() - distance, others.max() + distance)
        Lambda, R = _decouple(Lambda, R, vectors, w, targets)
        return R, Lambda

    def _weights(self, spread: NDArray[np.floating]) -> NDArray[np.float64]:
        # 0 above onset_tol, 1 below edge_tol, a smoothstep in log10 between
        if self.onset_tol == self.edge_tol:
            return (spread < self.edge_tol).astype(float)
        with np.errstate(divide="ignore"):
            x = (np.log10(self.onset_tol) - np.log10(spread)) / (
                np.log10(self.onset_tol) - np.log10(self.edge_tol)
            )
        x = np.clip(np.nan_to_num(x, posinf=1.0), 0, 1)
        weights: NDArray[np.float64] = x**2 * (3 - 2 * x)
        return weights

    def _clip_level_step(
        self,
        Lambda: NDArray[np.inexact],
        Lambda_in: NDArray[np.inexact],
        spread: NDArray[np.floating],
        vectors: NDArray[np.inexact],
        limit: float,
    ) -> NDArray[np.inexact]:
        near = vectors[:, (spread >= self.edge_tol) & (spread < self.near_edge_tol)]
        P = near @ near.conj().T
        step = P @ (Lambda - Lambda_in) @ P
        norm = np.linalg.norm(step, 2)
        if norm > limit:
            Lambda = Lambda - (1 - limit / norm) * step
        return Lambda

    def _smooth_level_step(
        self,
        Lambda: NDArray[np.inexact],
        Lambda_in: NDArray[np.inexact],
        spread: NDArray[np.floating],
        vectors: NDArray[np.inexact],
        limit: float,
    ) -> NDArray[np.inexact]:
        x = np.clip(spread / self.near_edge_tol, 0, 1)
        weights = 1 - x**3 * (10 - 15 * x + 6 * x**2)
        Q = (vectors * weights) @ vectors.conj().T
        step = Q @ (Lambda - Lambda_in) @ Q
        squared_norm = float(np.vdot(step, step).real)
        factor = limit / np.sqrt(limit**2 + squared_norm) if limit else 0.0
        result: NDArray[np.inexact] = Lambda - (1 - factor) * step
        return result


class LinearWeightedDensityFeedback(DensityFeedback):
    r"""
    Shift the quasiparticle levels against the density mismatch.

    Adds :math:`-c M f_2^T M` to :math:`\Lambda`, with
    :math:`f_2 = \rho_f - \rho_{qp}` and :math:`M = (1 - R R^\dagger)^{1/2}`.
    It vanishes at a solution, so the solutions are unchanged.

    A ghost orbital with zero rows of :math:`R` and :math:`D` has a flat band
    at the chemical potential. The update does not change its level, so its
    :math:`\rho_{qp}` never relaxes to :math:`\rho_f`. :math:`M` removes the
    term from quasiparticle orbitals that :math:`R` couples to the physical
    orbitals, whose levels the update already sets.

    The optimal :math:`c` is the inverse of the maximum slope of the occupation
    function: :math:`4 / \beta` for :func:`~risb.kintegration.fermi_function`
    and :math:`\sqrt{\pi} / \beta` for
    :func:`~risb.kintegration.gaussian_function`. A decoupled ghost then
    reaches :math:`\rho_f` in one step. A smaller :math:`c` converges more
    slowly; a larger one overshoots. By default :math:`c` is this inverse
    slope, which is zero for a step function, e.g. at :math:`\beta = \infty`.

    Parameters
    ----------
    strength : float, optional
        :math:`c`, in the energy units of the Hamiltonian. Default is the
        inverse of the largest slope of the band occupation function.

    """

    def __init__(self, strength: float | None = None) -> None:
        self.strength = strength

    @override
    def qp_levels(
        self,
        Lambda: NDArray[np.inexact],
        R: NDArray[np.inexact],
        f2: NDArray[np.inexact],
        occupation_slope: float,
    ) -> NDArray[np.inexact]:
        """Return the shifted quasiparticle levels."""
        strength = 1 / occupation_slope if self.strength is None else self.strength
        eigenvalues, vectors = np.linalg.eigh(R @ R.conj().T)
        M = (vectors * np.sqrt(np.clip(1 - eigenvalues, 0, 1))) @ vectors.conj().T
        # rho_qp is the transpose of n_F(Lambda - mu)
        result: NDArray[np.inexact] = Lambda - strength * M @ f2.T @ M
        return result


class MuUnitStep(ChemicalPotentialStep):
    """Step the chemical potential by the missing number of electrons, ``n_target - n_electrons``."""

    @override
    def step(
        self,
        filling_error: float,
        slope: Callable[[], float],
        history: Sequence[tuple[float, float]],
    ) -> float:
        """Return the change of the chemical potential."""
        return -filling_error


class MuNewtonStep(ChemicalPotentialStep):
    r"""
    Newton step of the chemical potential, :math:`(n_\mathrm{target} - n) / (dn/d\mu)`.

    :math:`dn/d\mu` is that of the bands of the orbitals in no cluster plus
    the charge susceptibilities of the impurities given to the lattice solver.
    Without them, only the bands count.

    Parameters
    ----------
    min_slope : float, optional
        Smallest :math:`dn/d\mu` used, so that a step is at most
        ``|n_target - n| / min_slope``. With ``min_slope = 1`` a step is never
        longer than that of :class:`MuUnitStep`.

    """

    def __init__(self, min_slope: float = 1.0) -> None:
        self.min_slope = min_slope

    @override
    def step(
        self,
        filling_error: float,
        slope: Callable[[], float],
        history: Sequence[tuple[float, float]],
    ) -> float:
        """Return the change of the chemical potential."""
        return -filling_error / max(slope(), self.min_slope)


class MuSecantStep(ChemicalPotentialStep):
    r"""
    Secant step of the chemical potential, :math:`(n_\mathrm{target} - n) / (dn/d\mu)`.

    :math:`dn/d\mu` is the slope through the last two entries of the history
    of the cycle. With only one entry, the step is that of :class:`MuUnitStep`.

    Parameters
    ----------
    min_slope : float, optional
        Smallest :math:`dn/d\mu` used, so that a step is at most
        ``|n_target - n| / min_slope``.

    """

    def __init__(self, min_slope: float = 1e-3) -> None:
        self.min_slope = min_slope

    @override
    def step(
        self,
        filling_error: float,
        slope: Callable[[], float],
        history: Sequence[tuple[float, float]],
    ) -> float:
        """Return the change of the chemical potential."""
        if len(history) < 2:
            return -filling_error
        (mu_0, n_0), (mu_1, n_1) = history[-2:]
        return -filling_error / max((n_1 - n_0) / (mu_1 - mu_0), self.min_slope)


def _decouple(
    levels: NDArray[np.inexact],
    rows: NDArray[np.inexact],
    vectors: NDArray[np.inexact],
    weights: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.inexact], NDArray[np.inexact]]:
    # For each column v of vectors with weight w > 0: the couplings of v to
    # the rest in levels, and the row of v in rows, scaled by 1 - w, and the
    # level of v moved a fraction w of the way to its target
    for k in np.flatnonzero(weights > 0):
        v = vectors[:, k : k + 1]
        P = v @ v.conj().T
        Q = np.eye(len(P)) - P
        level = float(np.real(v.conj().T @ levels @ v).item())
        w = weights[k]
        levels = (
            P @ levels @ P
            + Q @ levels @ Q
            + (1 - w) * (P @ levels @ Q + Q @ levels @ P)
            + w * (targets[k] - level) * P
        )
        rows = rows - w * P @ rows
    return levels, rows
