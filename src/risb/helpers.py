# Copyright (c) 2016-2023 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Helper functions to perform rotationally invariant slave-boson mean-field theory self-consistent loop."""

# from scipy.linalg import pinv
# from scipy.special import binom
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.linalg import inv
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from scipy.special import expit

from risb.types import BlockMatrix, BlockMatrixLike, BlockRealArray

# Eigenvalues of rho (1 - rho) below this are a completely full or empty orbital
EDGE_TOL = 1e-12

## Formula is (1-A)^{-1/2} = sum_r=0^{infty} (-1)^r * 1/2 choose r * A^r
# def one_sqrtm_inv(A, tol=np.finfo(float).eps, N=10000):
#    # Do r = 0 manually (it is just the identity)
#    A_r = np.eye(A.shape[0])
#    out = np.eye(A.shape[0])
#    for r in range(1,N+1):
#        old = out.copy()
#        A_r = A_r @ A
#        out += (-1)**r * binom(-1/2., r) * A_r
#        err = np.linalg.norm(out - old)
#        if err < tol:
#            break
#    print(r,err)
#    return out
#
# def get_K_sq_inv(pdensity, hdensity, tol=np.finfo(float).eps, N=10000):
#    return one_sqrtm_inv(A=pdensity, tol=tol, N=N) @ one_sqrtm_inv(A=hdensity, tol=tol, N=N)


def get_d(rho_qp: NDArray[np.inexact], ke: NDArray[np.inexact]) -> NDArray[np.inexact]:
    """
    Return hybridization matrix of impurity problem.

    Assumes the quasiparticle kinetic energy is lopsided as has not been multiplied
    by R on the left-hand side.

    Parameters
    ----------
    rho_qp : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.
    ke : numpy.ndarray
        Lopsided quasiparticle kinetic energy.

    Returns
    -------
    D : numpy.ndarray
        Hybridization coupling.

    Notes
    -----
    Eq. 35 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    D: NDArray[np.inexact] = _sqrt_k(rho_qp, inverse=True) @ ke.T
    return D


def get_d2(
    rho_qp: NDArray[np.inexact], ke: NDArray[np.inexact], R: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return hybridization matrix of impurity problem.

    This will invert R, so will not work in a Mott insulator when R is singular.

    Parameters
    ----------
    rho_qp : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.
    ke : numpy.ndarray
        Quasiparticle kinetic energy.
    R : numpy.ndarray
        Renormalization matrix from electrons to quasiparticles.

    Returns
    -------
    D : numpy.ndarray
        Hybridization coupling.

    Notes
    -----
    Eq. 35 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    The singular nature of R might be fixed by using the pseudo-inverse (untested).

    """
    K = rho_qp - rho_qp @ rho_qp
    D: NDArray[np.inexact] = inv(sqrtm(K)) @ (inv(R) @ ke).T
    return D


def get_lambda_c(
    rho_qp: NDArray[np.inexact],
    R: NDArray[np.inexact],
    Lambda: NDArray[np.inexact],
    D: NDArray[np.inexact],
) -> NDArray[np.inexact]:
    r"""
    Return bath matrix of impurity problem.

    Parameters
    ----------
    rho_qp : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.
    R : numpy.ndarray
        Renormalization matrix from electrons to quasiparticles.
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles.
    D : numpy.ndarray
        Hybridization coupling.

    Returns
    -------
    Lambda_c : numpy.ndarray
        Bath coupling.

    Notes
    -----
    Eq. 36 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__, with the matrix
    derivative of :math:`\sqrt{\rho (1 - \rho)}` evaluated with the
    Daleckii-Krein formula.
    ``Lambda_c`` is the transpose of the bath matrix there, because the bath
    term of the impurity problem is ``sum f_a Lambda_c[a, b] f^dag_b``.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    Lambda_c: NDArray[np.inexact] = -_sqrt_term_gradient(rho_qp, R, D).T - Lambda.T
    return Lambda_c


def get_lambda(
    R: NDArray[np.inexact],
    D: NDArray[np.inexact],
    Lambda_c: NDArray[np.inexact],
    rho_f: NDArray[np.inexact],
) -> NDArray[np.inexact]:
    """
    Return correlation potential matrix from impurity problem parameters and density matrices.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix from electrons to quasiparticles.
    D : numpy.ndarray
        Hybridization coupling.
    Lambda_c : numpy.ndarray
        Bath coupling.
    rho_f : numpy.ndarray
        f-electron density matrix from the impurity problem.

    Returns
    -------
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles.

    Notes
    -----
    Derived from Eq. 36 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__ by
    replacing the quasipartice density matrix with the f-electron density
    matrix using Eq. 39. See :func:`get_lambda_c` for the conventions.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    Lambda: NDArray[np.inexact] = -_sqrt_term_gradient(rho_f, R, D) - Lambda_c.T
    return Lambda


def get_r(
    rho_cf: NDArray[np.inexact], rho_f: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return renormalization matrix from impurity problem density matrices.

    Parameters
    ----------
    rho_cf : numpy.ndarray
        c,f-electron hybridization density matrix from the impurity problem.
    rho_f : numpy.ndarray
        f-electron density matrix from the impurity problem.

    Returns
    -------
    R : numpy.ndarray
        Renormalization matrix from electrons to quasiparticles.

    Notes
    -----
    Derived from Eq. 38 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__ by
    replacing the quasiparticle density matrix with the f-electron density
    matrix using Eq. 39.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    R: NDArray[np.inexact] = (rho_cf @ _sqrt_k(rho_f, inverse=True)).T
    return R


def get_f1(
    rho_cf: NDArray[np.inexact], rho_qp: NDArray[np.inexact], R: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return the first self-consistent equation of RISB.

    Parameters
    ----------
    rho_cf : numpy.ndarray
        c,f-electron hybridization density matrix from the impurity problem.
    rho_qp : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.
    R : numpy.ndarray
        Renormalization matrix from electrons to quasiparticles.

    Returns
    -------
    f1 : numpy.ndarray
        First self-consistency equation.

    Notes
    -----
    Eq. 38 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    f1: NDArray[np.inexact] = rho_cf - R.T @ _sqrt_k(rho_qp)
    return f1


def get_f2(
    rho_f: NDArray[np.inexact], rho_qp: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return the second self-consistent equation of RISB.

    Parameters
    ----------
    rho_f : numpy.ndarray
        f-electron density matrix from the impurity problem.
    rho_qp : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.

    Returns
    -------
    f2 : numpy.ndarray
        Second self-consistency equation.

    Notes
    -----
    Eq. 39 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    f2: NDArray[np.inexact] = rho_f - rho_qp
    return f2


def get_lagrangian(
    *,
    h0_kin_k: BlockMatrixLike,
    R: Sequence[BlockMatrixLike],
    Lambda: Sequence[BlockMatrixLike],
    D: Sequence[BlockMatrixLike],
    Lambda_c: Sequence[BlockMatrixLike],
    Delta_p: Sequence[BlockMatrixLike],
    rho_f: Sequence[BlockMatrixLike],
    rho_cf: Sequence[BlockMatrixLike],
    h_loc_energy: Sequence[float],
    beta: float,
    mu: float,
    projectors: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
    impurity_entropy: Sequence[float] | None = None,
) -> float:
    r"""
    Return the RISB Lagrangian.

    Its stationary points are the solutions of RISB, and its value at a
    solution is the grand potential of RISB, so minus its derivative with
    respect to the temperature at fixed chemical potential is the entropy of
    RISB. It is written with the expectation
    values of a fixed state (or thermal ensemble) of each impurity problem, so
    its derivatives with respect to ``D`` and ``Lambda_c`` are taken at fixed
    state. All arguments are keyword-only, and the matrices are lists with an
    entry for each cluster, as stored by :class:`risb.LatticeSolver`.

    Parameters
    ----------
    h0_kin_k : risb.types.BlockMatrixLike
        Single-particle dispersion without the local terms of the clusters,
        indexed as k, orb_i, orb_j.
    R : list of risb.types.BlockMatrixLike
        Renormalization matrix from electrons to quasiparticles.
    Lambda : list of risb.types.BlockMatrixLike
        Correlation potential of quasiparticles.
    D : list of risb.types.BlockMatrixLike
        Hybridization coupling of the impurity problem, indexed as
        ``D[i][bl][a, alpha]`` with ``a`` a bath orbital.
    Lambda_c : list of risb.types.BlockMatrixLike
        Bath coupling of the impurity problem.
    Delta_p : list of risb.types.BlockMatrixLike
        Quasiparticle density matrix, as a variable of the Lagrangian.
    rho_f : list of risb.types.BlockMatrixLike
        f-electron density matrix of the impurity problem.
    rho_cf : list of risb.types.BlockMatrixLike
        c,f-electron density matrix of the impurity problem.
    h_loc_energy : list of float
        Expectation value of the local Hamiltonian of the physical orbitals in
        the impurity problem.
    beta : float
        Inverse temperature of the Fermi-Dirac smearing of the quasiparticles.
    mu : float
        Chemical potential.
    projectors : list of risb.types.BlockMatrixLike, optional
        The projectors onto the subspace of each cluster, as in
        :class:`risb.LatticeSolver`.
    gf_struct_mapping : list of mapping, optional
        The mapping from the blocks of each cluster to the blocks of
        ``h0_kin_k``, as in :class:`risb.LatticeSolver`. Default assumes the
        blocks of every cluster are the blocks of ``h0_kin_k``.
    impurity_entropy : list of float, optional
        Entropy of the impurity problem of each cluster, in the order of the
        clusters, when its states are a thermal ensemble at the inverse
        temperature ``beta``, e.g.,
        :attr:`risb.impurity_solvers.FockitImpuritySolver.entropy`. The
        Lagrangian then has the free energy of each impurity problem instead of
        its energy. Default is zero, the ground state.

    Returns
    -------
    float
        The Lagrangian.

    Notes
    -----
    Eq. 40 in the supplemental of `10.1103/PhysRevLett.118.126401
    <PRL126401_>`__, with the zero-temperature limit replaced by Fermi-Dirac
    smearing, written in the conventions of risb:

    .. math::

        L = -\frac{1}{\beta N_k} \sum_k \mathrm{Tr} \log \left( 1 +
        e^{-\beta (\mathcal{R} \epsilon_k \mathcal{R}^{\dagger} + \lambda - \mu)}
        \right)
        + \sum_i \Big[ \langle \hat{H}^{\mathrm{loc}}_i \rangle
        + 2 \, \mathrm{Re} \, \mathrm{Tr} (\mathcal{D}_i \rho^{cf}_i)
        + \mathrm{Tr} (\lambda^c_i \rho^f_i)
        - \mathrm{Tr} (\lambda_i^T \Delta^p_i)
        - \mathrm{Tr} (\lambda^c_i \Delta^p_i)
        - 2 \, \mathrm{Re} \, \mathrm{Tr} \left( \mathcal{D}_i
        \mathcal{R}_i^T \sqrt{\Delta^p_i (1 - \Delta^p_i)} \right) \Big],

    summed over blocks, with :math:`\mathcal{R}` and :math:`\lambda` of the
    clusters in the blocks of ``h0_kin_k`` from :func:`upfold`, as in
    :class:`risb.LatticeSolver`. The terms with the
    normalization of the impurity ground states are zero and left out.

    With a thermal ensemble of each impurity problem (Gibbs weights at
    ``beta``) the expectation values are thermal averages and
    :math:`-S_i / \beta` is added for each cluster, with
    :math:`S_i = -\sum_n w_n \log w_n` over the Gibbs weights :math:`w_n` of
    the states of impurity problem :math:`i`, so the stationary points are
    the solutions with that ensemble.

    .. _PRL126401: https://doi.org/10.1103/PhysRevLett.118.126401

    """
    n_clusters = len(R)
    R_full = upfold(R, h0_kin_k, projectors, gf_struct_mapping, outside=1)
    Lambda_full = upfold(Lambda, h0_kin_k, projectors, gf_struct_mapping)

    lagrangian = 0.0
    for bl, h0 in h0_kin_k.items():
        energies = np.linalg.eigvalsh(get_h_qp(R_full[bl], Lambda_full[bl], h0, mu))
        n_k = energies.shape[0]
        lagrangian -= np.sum(np.logaddexp(0, -beta * energies)) / (beta * n_k)

    for i in range(n_clusters):
        lagrangian += h_loc_energy[i]
        if impurity_entropy is not None:
            lagrangian -= impurity_entropy[i] / beta
        for bl in R[i]:
            Dp = Delta_p[i][bl]
            sqrt_term = sqrtm(Dp - Dp @ Dp)
            lagrangian += 2 * np.real(np.trace(D[i][bl] @ rho_cf[i][bl]))
            lagrangian += np.real(np.trace(Lambda_c[i][bl] @ rho_f[i][bl]))
            lagrangian -= np.real(np.trace(Lambda[i][bl].T @ Dp))
            lagrangian -= np.real(np.trace(Lambda_c[i][bl] @ Dp))
            lagrangian -= 2 * np.real(np.trace(D[i][bl] @ R[i][bl].T @ sqrt_term))
    return float(lagrangian)


def get_entropy_qp(energies: BlockRealArray, beta: float, mu: float) -> float:
    """
    Return the entropy of the quasiparticles per unit cell.

    Parameters
    ----------
    energies : risb.types.BlockRealArray
        Energies of the quasiparticle bands in each block, indexed as k, band,
        e.g., :attr:`risb.LatticeSolver.energies_qp`.
    beta : float
        Inverse temperature of the Fermi-Dirac occupations.
    mu : float
        Chemical potential.

    Returns
    -------
    float
        ``-sum(f log f + (1 - f) log(1 - f)) / n_k`` over every band and
        k-point, with ``f`` the Fermi-Dirac occupations, as in the
        quasiparticle term of :func:`get_lagrangian`.

    """
    entropy = 0.0
    for e in energies.values():
        x = beta * (np.asarray(e) - mu)
        entropy += np.sum(np.logaddexp(0, -x) + x * expit(-x)) / x.shape[0]
    return float(entropy)


def get_entropy_impurity(weights: NDArray[np.float64]) -> float:
    """
    Return the entropy of the states of the impurity problem of a cluster.

    Parameters
    ----------
    weights : numpy.ndarray
        Weights of the states of the impurity problem, e.g., the Gibbs
        weights :attr:`risb.impurity_solvers.FockitImpuritySolver.weights`.

    Returns
    -------
    float
        ``-sum(weights * log(weights))``, zero for a single state.

    """
    w = np.asarray(weights)
    w = w[w > 0]
    return float(-np.sum(w * np.log(w)))


def get_entropy(
    *,
    energies: BlockRealArray,
    beta: float,
    mu: float,
    impurity_entropy: Sequence[float],
) -> float:
    """
    Return the entropy of RISB per unit cell.

    It is minus the derivative of :func:`get_lagrangian` at a solution with
    respect to the temperature at fixed chemical potential. All arguments are
    keyword-only.

    Parameters
    ----------
    energies : risb.types.BlockRealArray
        Energies of the quasiparticle bands in each block, indexed as k, band,
        e.g., :attr:`risb.LatticeSolver.energies_qp`.
    beta : float
        Inverse temperature.
    mu : float
        Chemical potential.
    impurity_entropy : list of float
        Entropy of the impurity problem of each cluster, in the order of the
        clusters, from :func:`get_entropy_impurity`, e.g.,
        :attr:`risb.impurity_solvers.FockitImpuritySolver.entropy`.

    Returns
    -------
    float
        :func:`get_entropy_qp` plus the sum of ``impurity_entropy``.

    """
    return get_entropy_qp(energies, beta, mu) + float(np.sum(impurity_entropy))


def upfold(
    A: Sequence[BlockMatrixLike],
    h0_k: BlockMatrixLike,
    projectors: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
    outside: float = 0.0,
) -> BlockMatrix:
    """
    Return the matrices of the clusters in the blocks of the lattice.

    Parameters
    ----------
    A : list of risb.types.BlockMatrixLike
        A matrix for each cluster, e.g., ``R`` or ``Lambda``.
    h0_k : risb.types.BlockMatrixLike
        Dispersion of the lattice, which sets its blocks and their size.
    projectors : list of risb.types.BlockMatrixLike, optional
        The projectors onto the subspace of each cluster, as in
        :class:`risb.LatticeSolver`. They can depend on k. Without them there
        is one cluster, the whole unit cell.
    gf_struct_mapping : list of mapping, optional
        The mapping from the blocks of each cluster to the blocks of ``h0_k``.
        Default assumes they are the same.
    outside : float, optional
        Multiple of the identity on the orbitals that are in no cluster, e.g.,
        1 for ``R`` and 0 for ``Lambda``.

    Returns
    -------
    risb.types.BlockMatrix
        ``sum_i P_i^dag A_i P_i + outside (1 - sum_i P_i^dag P_i)`` in each
        block of ``h0_k``, at each k-point if the projectors depend on k.

    """
    if gf_struct_mapping is None:
        gf_struct_mapping = [{bl: bl for bl in h0_k} for _ in range(len(A))]
    A_full: BlockMatrix = {}
    covered: BlockMatrix = {}
    for bl, h0 in h0_k.items():
        # Accumulated out of place, so it is only complex if needed
        A_full[bl] = np.zeros(h0.shape[1:])
        covered[bl] = np.zeros(h0.shape[1:])
    for i, A_i in enumerate(A):
        for bl, A_bl in A_i.items():
            bl_full = gf_struct_mapping[i][bl]
            if projectors is None:
                A_full[bl_full] = A_full[bl_full] + A_bl
                covered[bl_full] = covered[bl_full] + np.eye(A_bl.shape[-1])
            else:
                P = projectors[i][bl]
                A_full[bl_full] = A_full[bl_full] + _dagger(P) @ A_bl @ P
                covered[bl_full] = covered[bl_full] + _dagger(P) @ P
    for bl, h0 in h0_k.items():
        A_full[bl] = A_full[bl] + outside * (np.eye(h0.shape[-1]) - covered[bl])
    return A_full


def get_h_qp(
    R: NDArray[np.inexact],
    Lambda: NDArray[np.inexact],
    h0_kin_k: NDArray[np.inexact],
    mu: float = 0,
) -> NDArray[np.inexact]:
    r"""
    Construct the quasiparticle Hamiltonian :math:`\hat{H}^{\mathrm{qp}}`.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix) from electrons to quasiparticles. It can
        depend on k, indexed as k, orb_i, orb_j.
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles. It can depend on k.
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters. Indexed as
        k, orb_i, orb_j
    mu : float, optional
        Chemical potential

    Return
    ------
    h_qp : numpy.ndarray
        Indexed as k, orb_i, orb_j

    Notes
    -----
    Eq. A34 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    # h_qp = np.einsum('ac,cdk,db->kab', R, h0_kin_k, R.conj().T, optimize='optimal') + (Lambda - mu*np.eye(Lambda.shape[0]))
    h_qp: NDArray[np.inexact] = R @ h0_kin_k @ _dagger(R) + (
        Lambda - mu * np.eye(Lambda.shape[-1])
    )
    if not np.allclose(h_qp, np.swapaxes(h_qp, 1, 2).conj()):
        warnings.warn("H_qp is not Hermitian !", RuntimeWarning, stacklevel=2)
    # eig, vec = np.linalg.eigh(h_qp)
    # return (eig, vec)
    return h_qp


def get_h0_kin_k_R(
    R: NDArray[np.inexact], h0_kin_k: NDArray[np.inexact], vec: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return the matrix representation of the lopsided kinetic energy term in the quasiparticle Hamiltonian of RISB.

    Parameters
    ----------
    R : numpy.ndarray
        Rormalization matrix from electrons to quasiparticles.
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters. Indexed as
        k, orb_i, orb_j.
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian. Indexed as k, each column
        an eigenvector.

    Returns
    -------
    h0_kin_k_R : numpy.ndarray
        Matrix representation of lopsided quasiparticle Hamiltonian.

    Notes
    -----
    This is equivalent to the kinetic part of ``H^qp`` with the inverse of
    the renormalization matrix `R` multiplied on the left.

    """
    h0_kin_k_R: NDArray[np.inexact] = h0_kin_k @ _dagger(R) @ vec
    return h0_kin_k_R


def get_R_h0_kin_k_R(
    R: NDArray[np.inexact], h0_kin_k: NDArray[np.inexact], vec: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return the matrix representation of the kinetic energy term in the quasiparticle Hamiltonian of RISB.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix from quasiparticles to electrons.
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters. Indexed as
        k, orb_i, orb_j.
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian. Indexed as k, each column
        an eigenvector.

    Returns
    -------
    R_h0_kin_k_R : numpy.ndarray
        Matrix representation of kinetic part of quasiparticle
        Hamiltonian ``H^qp``.

    """
    R_h0_kin_k_R: NDArray[np.inexact] = np.einsum(
        "pa,kac,cd,kdb->kpb", R, h0_kin_k, R.conj().T, vec
    )
    return R_h0_kin_k_R


# \sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_rho_qp(
    vec: NDArray[np.inexact],
    kweights: NDArray[np.float64],
    P: NDArray[np.inexact] | None = None,
) -> NDArray[np.inexact]:
    r"""
    Return the single-particle density matrix of the quasiparticle Hamiltonian :math:`\hat{H}^{\mathrm{qp}}`.

    Parameters
    ----------
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian.
    kweights : numpy.ndarray
        Integration weights at each k-point for each band (eigenvector).
    P : numpy.ndarray, optional
        Projection matrix onto correlated subspace.

    Returns
    -------
    rho_qp : numpy.ndarray
        Quasiparticle density matrix from mean-field.

    Notes
    -----
    Eqs. 32 and 34 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__, but not in
    the natural-basis gauge. See Eq. 112 in the supplemental of
    `10.1103/PhysRevLett.118.126401 <PRL126401_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    .. _PRL126401: https://doi.org/10.1103/PhysRevLett.118.126401

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        rho_qp: NDArray[np.inexact] = np.einsum(
            "kan,kn,knb->ab", vec, kweights, vec_dag
        ).T
        return rho_qp
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", vec, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0).T


def get_ke(
    h0_kin_k_R: NDArray[np.inexact],
    vec: NDArray[np.inexact],
    kweights: NDArray[np.float64],
    P: NDArray[np.inexact] | None = None,
) -> NDArray[np.inexact]:
    """
    Return average lopsided kinetic energy matrix of the quasiparticle Hamiltonian in RISB.

    Parameters
    ----------
    h0_kin_k_R : numpy.ndarray
        Single-particle dispersion between local clusters with `R` matrix
        multiplied on the right.
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian.
    kweights : numpy.ndarray
        Integration weights at each k-point for each band (eigenvector).
    P : numpy.ndarray, optional
        Projection matrix onto correlated subspace.

    Returns
    -------
    ke : numpy.ndarray
        Lopsided quasiparticle kinetic energy from the mean-field.

    Notes
    -----
    Eq. 35 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        ke: NDArray[np.inexact] = np.einsum(
            "kan,kn,knb->ab", h0_kin_k_R, kweights, vec_dag
        )
        return ke
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", h0_kin_k_R, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0)


def get_ke2(
    R_h0_kin_k_R: NDArray[np.inexact],
    vec: NDArray[np.inexact],
    kweights: NDArray[np.float64],
    P: NDArray[np.inexact] | None = None,
) -> NDArray[np.inexact]:
    """
    Return average kinetic energy matrix of the quasiparticle Hamiltonian in RISB.

    Parameters
    ----------
    R_h0_kin_k_R : numpy.ndarray
        Kinetic part of quasiparticle Hamiltonain.
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian.
    kweights : numpy.ndarray
        Integration weights at each k-point for each band (eigenvector).
    P : numpy.ndarray, optional
        Projection matrix onto correlated subspace.

    Returns
    -------
    ke : numpy.ndarray
        Quasiparticle kinetic energy from the mean-field.

    Notes
    -----
    Eq. 35 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        ke: NDArray[np.inexact] = np.einsum(
            "kan,kn,knb->ab", R_h0_kin_k_R, kweights, vec_dag
        )
        return ke
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", R_h0_kin_k_R, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0)


def block_to_full(A: NDArray[np.inexact]) -> NDArray[np.inexact]:
    """
    Return a full block matrix from each block.

    Parameters
    ----------
    A : numpy.ndarray
        A block matrix indexed as ``A[...,block1,block2,orb1,orb2]``.

    Returns
    -------
    numpy.ndarray
        Matrix `A` of shape ``A[...,block * orb, block * orb]``.

    Notes
    -----
    FIXME this does not have to be a numpy array, it could be a ragged list of lists if each block has a different size.
    So we should not use shape and handle the ragged list differently.

    """
    if len(A.shape) < 4:
        msg = f"A.shape = {A.shape} must have at least ...,block,block,orb,orb structure !"
        raise ValueError(msg)
    na = A.shape[-4]
    nb = A.shape[-3]
    if na != nb:
        msg = f"Should be same number of blocks in i and j dimesnions, but got {na} and {nb} !"
        raise ValueError(msg)
    return np.block([[A[..., i, j, :, :] for j in range(na)] for i in range(na)])


def get_h0_loc_matrix(
    h0_k: NDArray[np.inexact], P: NDArray[np.inexact] | None = None
) -> NDArray[np.inexact]:
    """
    Return a matrix representation of the local terms in a non-interacting dispersion.

    If a projector ``P`` onto a local subspace is given the local terms are only in that subspace.

    Parameters
    ----------
    h0_k : numpy.ndarray
        Single-particle dispersion.
    P : numpy.ndarray, optional
        The projector onto a local cluster within the supercell.

    Returns
    -------
    numpy.ndarray
        The matrix of single-particle hopping/energies on a cluster defined by
        the projector.

    """
    n_k = h0_k.shape[0]
    if P is None:
        return np.sum(h0_k, axis=0) / float(n_k)
    return np.sum(P @ h0_k @ _dagger(P), axis=0) / float(n_k)


def get_h0_kin_k_mat(
    h0_k: NDArray[np.inexact], P: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return a matrix representation of the kinetic terms between clusters/local subspaces of a non-interacting dispersion.

    This function only subtracts off the local terms in a single subspace given by ``P``.

    Parameters
    ----------
    h0_k : numpy.ndarray
        Single-particle dispersion.
    P : numpy.ndarray
        The projector onto a local cluster within the supercell.

    Returns
    -------
    numpy.ndarray
        The single-particle hopping without the contribution from the cluster
        defined by the projector.

    """
    h0_loc_matrix = get_h0_loc_matrix(h0_k, P)
    return h0_k - P.conj().T @ h0_loc_matrix @ P


def get_h0_kin_k(
    h0_k: BlockMatrixLike,
    projectors: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
) -> BlockMatrix:
    """
    Return a non-interacting dispersion without the local terms of the clusters.

    Parameters
    ----------
    h0_k : risb.types.BlockMatrixLike
        Single-particle dispersion in each block.
    projectors : sequence of risb.types.BlockMatrixLike, optional
        The projectors onto each subspace of a local cluster within
        the supercell organized into single-particle symmetry blocks. They
        can depend on k.
    gf_struct_mapping : sequence of mapping, optional
        The mapping from the symmetry blocks in the subspace to the
        symmetry blocks of h0_k. Default assumes the keys in `projectors`
        are the same as the keys in `h0_k`.

    Returns
    -------
    risb.types.BlockMatrix
        The single-particle dispersion without the local terms of the
        clusters defined by the projectors, so with the local terms of the
        orbitals that are in no cluster.

    """
    h0_kin_k = dict(h0_k)

    if projectors is not None:
        n_clusters = len(projectors)
        if gf_struct_mapping is None:
            gf_struct_mapping = [{bl: bl for bl in h0_k} for i in range(n_clusters)]
        for i, P in enumerate(projectors):
            for bl in P:
                bl_full = gf_struct_mapping[i][bl]
                h0_loc_matrix = get_h0_loc_matrix(h0_k[bl_full], P[bl])
                h0_kin_k[bl_full] = (
                    h0_kin_k[bl_full] - _dagger(P[bl]) @ h0_loc_matrix @ P[bl]
                )
                # h0_kin_k[bl_full] = get_h0_kin_k_mat(h0_kin_k[bl_full], P[bl])
    else:
        for bl, mat in h0_k.items():
            h0_kin_k[bl] = h0_kin_k[bl] - get_h0_loc_matrix(mat)

    return h0_kin_k


def _sqrt_term_gradient(
    rho: NDArray[np.inexact], R: NDArray[np.inexact], D: NDArray[np.inexact]
) -> NDArray[np.inexact]:
    """
    Return the gradient of ``Tr[D R^T f(rho)] + c.c.`` with respect to ``rho``, with f(x) = sqrt(x (1 - x)).

    Element ``[a, b]`` is the derivative with respect to ``rho[a, b]``. The
    matrix derivative of f is evaluated with the Daleckii-Krein formula.
    """
    eigenvalues, V = np.linalg.eigh(rho)
    K = eigenvalues * (1 - eigenvalues)
    f = np.sqrt(np.clip(K, 0, None))
    # Between two completely full or empty orbitals the divided difference
    # diverges, but the entries of D R^T it multiplies vanish there
    edge = K < EDGE_TOL
    both_edge = edge[:, None] & edge[None, :]
    divided_differences = np.where(
        both_edge,
        0,
        (1 - eigenvalues[:, None] - eigenvalues[None, :])
        / np.where(both_edge, 1, f[:, None] + f[None, :]),
    )
    V_dag = V.conj().T
    W = V @ (divided_differences * (V_dag @ D @ R.T @ V)) @ V_dag
    gradient: NDArray[np.inexact] = W.T + W.conj()
    return gradient


def _sqrt_k(rho: NDArray[np.inexact], inverse: bool = False) -> NDArray[np.inexact]:
    """
    Return the square root of ``rho (1 - rho)``, or its pseudo-inverse.

    The pseudo-inverse is zero for completely full or empty orbitals.
    """
    eigenvalues, V = np.linalg.eigh(rho)
    K = eigenvalues * (1 - eigenvalues)
    f = np.sqrt(np.clip(K, 0, None))
    if inverse:
        f = np.where(K < EDGE_TOL, 0, 1 / np.where(K < EDGE_TOL, 1, f))
    sqrt_k: NDArray[np.inexact] = (V * f) @ V.conj().T
    return sqrt_k


def _dagger(A: NDArray[np.inexact]) -> NDArray[np.inexact]:
    """Return the conjugate transpose of the last two axes, e.g., at each k-point."""
    return np.swapaxes(A.conj(), -1, -2)
