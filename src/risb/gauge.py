"""Choices of the quasiparticle basis, for the `gauge_fix` of :class:`risb.BaseLatticeSolver`."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def lambda_eigenbasis(
    R: NDArray[np.inexact], Lambda: NDArray[np.inexact], degenerate_tol: float = 1e-3
) -> tuple[NDArray[np.inexact], NDArray[np.inexact]]:
    """
    Rotate R and Lambda into the eigenbasis of Lambda closest to the current basis.

    Rotating the quasiparticles, ``R -> U^dag R`` and ``Lambda -> U^dag Lambda
    U``, does not change the physics, so an optimizer that mixes guesses from
    several cycles needs them in the same basis. Each eigenvector of Lambda
    takes the place of the basis vector it overlaps most, with a positive
    overlap, so that eigenvalues that cross keep their place. Within a group
    of eigenvalues closer than `degenerate_tol`, where the eigenvectors are
    not well defined, the group is instead rotated as little as possible, and
    Lambda is diagonal only up to the splitting of the group.

    A symmetry of Lambda that permutes quasiparticles usually mixes its
    eigenvectors, so this basis breaks permutation and onsite symmetries
    stated for the bath orbitals, e.g., a layer swap given to
    :class:`.FockitImpuritySolver`.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix of a block, with a row for each quasiparticle.
    Lambda : numpy.ndarray
        Correlation potential of the quasiparticles of the block.
    degenerate_tol : float, optional
        Eigenvalues closer than this, relative to their size when it is
        above 1, are treated as degenerate. Default is ``1e-3``.

    Returns
    -------
    R : numpy.ndarray
        The rotated renormalization matrix.
    Lambda : numpy.ndarray
        The rotated correlation potential.

    """
    w, U = np.linalg.eigh(Lambda)
    rows, cols = linear_sum_assignment(-np.abs(U))
    order = cols[np.argsort(rows)]
    w, U = w[order], U[:, order]
    for group in _degenerate_groups(w, degenerate_tol):
        if len(group) > 1:
            A, _, Bh = np.linalg.svd(U[np.ix_(group, group)])
            U[:, group] = U[:, group] @ (A @ Bh).conj().T
        else:
            k = group[0]
            if U[k, k] != 0:
                U[:, k] = U[:, k] * (np.abs(U[k, k]) / U[k, k])
    Lambda_rotated: NDArray[np.inexact] = U.conj().T @ Lambda @ U
    R_rotated: NDArray[np.inexact] = U.conj().T @ R
    return R_rotated, Lambda_rotated


def _degenerate_groups(w: NDArray[np.floating], tol: float) -> list[NDArray[np.intp]]:
    # Indices of eigenvalues that are within tol of a neighbour in sorted order
    order = np.argsort(w)
    groups, start = [], 0
    for k in range(1, len(w) + 1):
        if k == len(w) or w[order[k]] - w[order[k - 1]] > tol * max(
            1.0, abs(w[order[k - 1]])
        ):
            groups.append(order[start:k])
            start = k
    return groups
