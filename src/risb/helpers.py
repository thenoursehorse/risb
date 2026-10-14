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

    With ghosts, each block of ``R`` has a row for each of its :math:`M`
    quasiparticles and a column for each of its :math:`n` orbitals. The
    quasiparticle bands then hold :math:`n_g = \sum (M - n) / 2` more
    electrons than the physical ones, summed over the blocks of the clusters,
    so :math:`\mu n_g` is added for the value to be the grand potential.

    .. _PRL126401: https://doi.org/10.1103/PhysRevLett.118.126401

    """
    n_clusters = len(R)
    # With ghosts each block of R has more rows (quasiparticles) than columns
    bath_size = [{bl: R_bl.shape[0] for bl, R_bl in R_i.items()} for R_i in R]
    n_ghost_electrons = sum(
        (R_bl.shape[0] - R_bl.shape[1]) / 2 for R_i in R for R_bl in R_i.values()
    )
    projectors_qp = projectors
    if n_ghost_electrons > 0:
        projectors_qp = get_ghost_projectors(
            h0_kin_k, bath_size, projectors, gf_struct_mapping
        )
    R_full, Lambda_full = upfold_r_lambda(
        R, Lambda, h0_kin_k, projectors_qp, gf_struct_mapping
    )

    # The term of the quasiparticles has -mu n_g from the ghost electrons
    lagrangian = mu * n_ghost_electrons
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


def get_density_matrix_k(
    *,
    R_full: BlockMatrixLike,
    vectors: BlockMatrixLike,
    kweights: BlockRealArray,
    rho_c: Sequence[BlockMatrixLike],
    projectors: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
) -> BlockMatrix:
    r"""
    Return the density matrix of the electrons at each k-point.

    It is in the basis of the dispersion of the lattice, as the matrix of the
    single-particle density operator, ``rho[k, a, b] = <c^dag_b c_a>``. This is
    the transpose of the convention of :attr:`risb.LatticeSolver.rho_c`. All
    arguments are keyword-only.

    Parameters
    ----------
    R_full : risb.types.BlockMatrixLike
        Renormalization matrix of the lattice in each block, with a row for
        each quasiparticle and a column for each orbital, e.g.,
        :attr:`risb.LatticeSolver.R_full` or from :func:`upfold_r_lambda`. It
        can depend on k.
    vectors : risb.types.BlockMatrixLike
        Eigenvectors of the quasiparticle Hamiltonian at each k-point, e.g.,
        :attr:`risb.LatticeSolver.bloch_vector_qp`.
    kweights : risb.types.BlockRealArray
        Integration weights of each quasiparticle band at each k-point, on a
        uniform mesh, e.g., :attr:`risb.LatticeSolver.kweights`.
    rho_c : list of risb.types.BlockMatrixLike
        Density matrix of the physical orbitals in the impurity problem of
        each cluster, e.g., :attr:`risb.LatticeSolver.rho_c`, from the same
        cycle as the other arguments.
    projectors : list of risb.types.BlockMatrixLike, optional
        The projectors onto the subspace of each cluster, as in
        :class:`risb.LatticeSolver`.
    gf_struct_mapping : list of mapping, optional
        The mapping from the blocks of each cluster to the blocks of the
        lattice. Default assumes they are the same.

    Returns
    -------
    risb.types.BlockMatrix
        The density matrix at each k-point, indexed as k, orb_i, orb_j.

    Notes
    -----
    The quasiparticles give :math:`\mathcal{R}^{\dagger}(k) n^{\mathrm{qp}}(k)
    \mathcal{R}(k)`, with :math:`n^{\mathrm{qp}}(k)` their density matrix. On
    each cluster its local part is replaced by the density matrix of the
    impurity problem, which also has the weight that is not in the
    quasiparticles.

    """
    density: BlockMatrix = {}
    for bl, vec in vectors.items():
        n_k = vec.shape[0]
        n_qp = (vec * (kweights[bl] * n_k)[:, None, :]) @ _dagger(vec)
        density[bl] = _dagger(R_full[bl]) @ n_qp @ R_full[bl]
    if gf_struct_mapping is None:
        gf_struct_mapping = [{bl: bl for bl in vectors} for _ in range(len(rho_c))]
    for i, rho_c_i in enumerate(rho_c):
        for bl, rho in rho_c_i.items():
            bl_full = gf_struct_mapping[i][bl]
            P = np.eye(rho.shape[-1]) if projectors is None else projectors[i][bl]
            coherent = np.mean(P @ density[bl_full] @ _dagger(P), axis=0)
            density[bl_full] = (
                density[bl_full] + _dagger(P) @ (np.transpose(rho) - coherent) @ P
            )
    return density


def get_band_energy(h0_k: BlockMatrixLike, density_k: BlockMatrixLike) -> float:
    """
    Return the energy of a dispersion with a density matrix at each k-point.

    Parameters
    ----------
    h0_k : risb.types.BlockMatrixLike
        Single-particle dispersion, indexed as k, orb_i, orb_j.
    density_k : risb.types.BlockMatrixLike
        Density matrix at each k-point in the convention of
        :func:`get_density_matrix_k`, e.g., its result, or the occupations of
        the bands of a DFT calculation on the diagonal.

    Returns
    -------
    float
        ``sum_k Tr[h0_k rho(k)] / n_k`` summed over blocks, on a uniform mesh.

    """
    energy = 0.0
    for bl, h0 in h0_k.items():
        energy += np.real(np.einsum("kab,kba->", h0, density_k[bl])) / h0.shape[0]
    return float(energy)


def get_total_energy(
    *,
    dft_total_energy: float,
    dft_band_energy: float,
    band_energy: float,
    interaction_energy: float,
    double_counting_energy: float,
) -> float:
    """
    Return the total energy of DFT+RISB.

    All arguments are keyword-only.

    Parameters
    ----------
    dft_total_energy : float
        Total energy of the DFT calculation.
    dft_band_energy : float
        Energy of the DFT bands of the lattice dispersion with their DFT
        occupations, from :func:`get_band_energy`.
    band_energy : float
        Energy of the lattice dispersion with the density matrix of RISB,
        from :func:`get_band_energy` and :func:`get_density_matrix_k`.
    interaction_energy : float
        Expectation value of the interaction in the impurity problem of every
        cluster, summed.
    double_counting_energy : float
        Energy of the double counting, summed over clusters.

    Returns
    -------
    float
        ``dft_total_energy - dft_band_energy + band_energy +
        interaction_energy - double_counting_energy``.

    """
    return (
        dft_total_energy
        - dft_band_energy
        + band_energy
        + interaction_energy
        - double_counting_energy
    )


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
        block of ``h0_k``, at each k-point if the projectors depend on k. A
        block has as many rows and columns as the projectors into it have
        columns (or as ``A`` has, without projectors), and otherwise the size
        of the block of ``h0_k``. With the projectors from
        :func:`get_ghost_projectors` this is the number of quasiparticles,
        ghosts included.

    """
    if gf_struct_mapping is None:
        gf_struct_mapping = [{bl: bl for bl in h0_k} for _ in range(len(A))]
    size = {bl: h0.shape[-1] for bl, h0 in h0_k.items()}
    for i, A_i in enumerate(A):
        for bl, A_bl in A_i.items():
            P = A_bl if projectors is None else projectors[i][bl]
            size[gf_struct_mapping[i][bl]] = P.shape[-1]
    A_full: BlockMatrix = {}
    covered: BlockMatrix = {}
    for bl, n in size.items():
        # Accumulated out of place, so it is only complex if needed
        A_full[bl] = np.zeros((n, n))
        covered[bl] = np.zeros((n, n))
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
    for bl, n in size.items():
        A_full[bl] = A_full[bl] + outside * (np.eye(n) - covered[bl])
    return A_full


def upfold_r_lambda(
    R: Sequence[BlockMatrixLike],
    Lambda: Sequence[BlockMatrixLike],
    h0_kin_k: BlockMatrixLike,
    projectors_qp: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
) -> tuple[BlockMatrix, BlockMatrix]:
    """
    Return ``R`` and ``Lambda`` of the clusters in the blocks of the lattice, also with ghosts.

    Parameters
    ----------
    R : list of risb.types.BlockMatrixLike
        Renormalization matrix of each cluster, with a row for each
        quasiparticle and a column for each orbital.
    Lambda : list of risb.types.BlockMatrixLike
        Correlation potential of the quasiparticles of each cluster.
    h0_kin_k : risb.types.BlockMatrixLike
        Dispersion of the lattice, which sets its blocks and their size.
    projectors_qp : list of risb.types.BlockMatrixLike, optional
        The projectors onto the quasiparticles of each cluster, from
        :func:`get_ghost_projectors` with ghosts, and otherwise the projectors
        onto the subspace of each cluster, as in :class:`risb.LatticeSolver`.
    gf_struct_mapping : list of mapping, optional
        The mapping from the blocks of each cluster to the blocks of
        ``h0_kin_k``. Default assumes they are the same.

    Returns
    -------
    R_full : risb.types.BlockMatrix
        ``R`` in each block of ``h0_kin_k``, with a row for each quasiparticle
        and a column for each orbital, and ``R = 1`` on the orbitals in no
        cluster.
    Lambda_full : risb.types.BlockMatrix
        ``Lambda`` in each block of ``h0_kin_k``, a matrix on the
        quasiparticles, with ``Lambda = 0`` on the orbitals in no cluster.

    Notes
    -----
    With ghosts ``R`` has more rows than columns. It is padded with zero
    columns to be square and upfolded with :func:`upfold`, which leaves the
    columns after the orbitals of the block zero, so they are dropped.

    """
    R_padded = [
        {
            bl: np.hstack(
                [R_bl, np.zeros((R_bl.shape[0], R_bl.shape[0] - R_bl.shape[1]))]
            )
            for bl, R_bl in R_i.items()
        }
        for R_i in R
    ]
    R_full = upfold(R_padded, h0_kin_k, projectors_qp, gf_struct_mapping, outside=1)
    R_full = {bl: R_full[bl][..., : h0.shape[-1]] for bl, h0 in h0_kin_k.items()}
    Lambda_full = upfold(Lambda, h0_kin_k, projectors_qp, gf_struct_mapping)
    return R_full, Lambda_full


def get_ghost_projectors(
    h0_k: BlockMatrixLike,
    bath_size: Sequence[Mapping[str, int]],
    projectors: Sequence[BlockMatrixLike] | None = None,
    gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
) -> list[BlockMatrix]:
    """
    Return the projectors onto the quasiparticles of each cluster in ghost RISB.

    The quasiparticle space of a block of ``h0_k`` is its orbitals followed by
    the ghost orbitals of the clusters, in the order of the clusters and their
    blocks. A cluster block with ``n`` orbitals and ``bath_size`` ``M`` has
    ``M`` quasiparticles: the first ``n`` are on its orbitals, as in
    ``projectors``, and the other ``M - n`` are its ghost orbitals.

    Parameters
    ----------
    h0_k : risb.types.BlockMatrixLike
        Dispersion of the lattice, which sets its blocks and their size.
    bath_size : list of mapping
        Number of quasiparticles (bath orbitals of the impurity problem) of
        each block of each cluster.
    projectors : list of risb.types.BlockMatrixLike, optional
        The projectors onto the subspace of each cluster, as in
        :class:`risb.LatticeSolver`. They can depend on k. Without them there
        is one cluster, the whole unit cell.
    gf_struct_mapping : list of mapping, optional
        The mapping from the blocks of each cluster to the blocks of ``h0_k``.
        Default assumes they are the same.

    Returns
    -------
    list of risb.types.BlockMatrix
        Projector onto the quasiparticles of each cluster, with a row for each
        quasiparticle of the cluster block and a column for each quasiparticle
        of the block of ``h0_k``, at each k-point if ``projectors`` depend on
        k.

    Notes
    -----
    :func:`upfold_r_lambda` uses these projectors to put ``R`` and ``Lambda``
    of the clusters in the blocks of ``h0_k``.

    """
    if gf_struct_mapping is None:
        gf_struct_mapping = [{bl: bl for bl in h0_k} for _ in range(len(bath_size))]
    if projectors is None:
        projectors = [{bl: np.eye(h0.shape[-1]) for bl, h0 in h0_k.items()}]
    # Index of the first ghost orbital of each cluster block
    first_ghost: list[dict[str, int]] = []
    size = {bl: h0.shape[-1] for bl, h0 in h0_k.items()}
    for i, sizes in enumerate(bath_size):
        first_ghost.append({})
        for bl, M in sizes.items():
            bl_full = gf_struct_mapping[i][bl]
            first_ghost[i][bl] = size[bl_full]
            size[bl_full] += M - projectors[i][bl].shape[-2]
    projectors_qp: list[BlockMatrix] = []
    for i, sizes in enumerate(bath_size):
        projectors_qp.append({})
        for bl, M in sizes.items():
            P = np.asarray(projectors[i][bl])
            n_orb, n_band = P.shape[-2:]
            P_qp = np.zeros((*P.shape[:-2], M, size[gf_struct_mapping[i][bl]]), P.dtype)
            P_qp[..., :n_orb, :n_band] = P
            start = first_ghost[i][bl]
            P_qp[..., n_orb:, start : start + M - n_orb] = np.eye(M - n_orb)
            projectors_qp[i][bl] = P_qp
    return projectors_qp


def get_sigma_w(
    omega: complex | NDArray[np.inexact],
    R: NDArray[np.inexact],
    Lambda: NDArray[np.inexact],
    mu: float = 0.0,
    h0_loc: NDArray[np.inexact] | None = None,
) -> NDArray[np.inexact]:
    """
    Return the local self-energy of RISB, also with ghosts.

    Parameters
    ----------
    omega : complex or numpy.ndarray
        Frequencies, relative to the chemical potential, e.g., ``w + 1j * eta``
        or Matsubara frequencies.
    R : numpy.ndarray
        Renormalization matrix of a block of a cluster, with a row for each
        quasiparticle and a column for each orbital.
    Lambda : numpy.ndarray
        Correlation potential of the quasiparticles of the block.
    mu : float, optional
        Chemical potential.
    h0_loc : numpy.ndarray, optional
        Local non-interacting terms of the orbitals of the block.

    Returns
    -------
    numpy.ndarray
        ``Sigma(omega)``, with the frequencies along the first axes.

    Notes
    -----
    The self-energy of the Green's function ``R^dag (w + mu - H_qp)^-1 R`` at
    every k-point is ``Sigma(w) = w + mu - h0_loc - [R^dag (w + mu -
    Lambda)^-1 R]^-1``, Eq. 11 of `arXiv:2601.16401
    <https://arxiv.org/abs/2601.16401>`__. It is evaluated in the form of its
    poles, Eq. B7 there (equivalently Eq. B10), with ``z = w + mu``, ``S =
    R^dag R`` and ``Q`` the projector onto the quasiparticles that ``R`` does
    not reach (the ghosts),

    ``Sigma(z) = (1 - S^-1) z + S^-1 R^dag Lambda R S^-1
    + S^-1 R^dag Lambda Q (z - Q Lambda Q)^-1 Q Lambda R S^-1 - h0_loc``,

    which is only singular at its poles, the eigenvalues of ``Q Lambda Q`` on
    the ghosts. The sign of its second term is printed as minus in Eq. B7
    there. Without ghosts it is ``(1 - Z^-1) z + R^-1 Lambda R^-dag -
    h0_loc`` with ``Z = R^dag R``. The derivation is in
    :doc:`/explanations/self_energy`.

    """
    z = (np.asarray(omega) + mu)[..., None, None]
    S, poles, B = _self_energy_poles(R, Lambda)
    S_inv = np.linalg.inv(S)
    residues = (B / (z - poles)) @ _dagger(B)
    sigma: NDArray[np.inexact] = (
        (np.eye(len(S)) - S_inv) * z
        + S_inv @ _dagger(R) @ Lambda @ R @ S_inv
        + S_inv @ residues @ S_inv
    )
    if h0_loc is not None:
        sigma = sigma - h0_loc
    return sigma


def get_z(
    R: NDArray[np.inexact], Lambda: NDArray[np.inexact], mu: float = 0.0
) -> NDArray[np.inexact]:
    """
    Return the quasiparticle weight of RISB, also with ghosts.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix of a block of a cluster, with a row for each
        quasiparticle and a column for each orbital.
    Lambda : numpy.ndarray
        Correlation potential of the quasiparticles of the block.
    mu : float, optional
        Chemical potential.

    Returns
    -------
    numpy.ndarray
        ``Z = [1 - dSigma/dw (0)]^-1`` of :func:`get_sigma_w`. Without
        ghosts it is ``R^dag R``.

    Notes
    -----
    With the poles ``e_j`` of :func:`get_sigma_w`, the eigenvalues of ``Q
    Lambda Q`` on the ghosts with eigenvectors ``v_j``, and ``B`` the matrix
    with the columns ``R^dag Lambda v_j``, ``1 - dSigma/dw (0) = S^-1 (S
    + B diag(1 / (mu - e_j)^2) B^dag) S^-1``, so by the Woodbury identity
    ``Z = S - B [diag((mu - e_j)^2) + B^dag S^-1 B]^-1 B^dag``. It is finite
    when a pole is at ``mu``, e.g., ``Z = 0`` in a Mott insulator. The
    derivation is in :doc:`/explanations/self_energy`.

    """
    S, poles, B = _self_energy_poles(R, Lambda)
    middle = np.diag((mu - poles) ** 2) + _dagger(B) @ np.linalg.pinv(S) @ B
    Z: NDArray[np.inexact] = S - B @ np.linalg.pinv(middle) @ _dagger(B)
    return Z


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
    projector_qp: NDArray[np.inexact] | None = None,
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
    projector_qp : numpy.ndarray, optional
        Projection matrix onto the quasiparticles of the correlated subspace,
        from :func:`get_ghost_projectors`. Default is `P`.

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
    if P is None and projector_qp is None:
        ke: NDArray[np.inexact] = np.einsum(
            "kan,kn,knb->ab", h0_kin_k_R, kweights, vec_dag
        )
        return ke
    middle = np.einsum("kan,kn,knb->kab", h0_kin_k_R, kweights, vec_dag)
    if P is not None:
        middle = P @ middle
    projector_qp = P if projector_qp is None else projector_qp
    if projector_qp is not None:
        middle = middle @ _dagger(projector_qp)
    return np.sum(middle, axis=0)


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


def _self_energy_poles(
    R: NDArray[np.inexact], Lambda: NDArray[np.inexact]
) -> tuple[NDArray[np.inexact], NDArray[np.float64], NDArray[np.inexact]]:
    # Returns S = R^dag R, the poles of the self-energy, and the matrix with
    # the residue vectors R^dag Lambda v_j as columns. The poles are the
    # eigenvalues of Lambda on the ghosts, the quasiparticles that R does not
    # reach, with eigenvectors v_j. The ghosts are the left singular vectors
    # of R beyond its rank, with the rank tolerance of scipy.linalg.null_space
    U, singular_values, _ = np.linalg.svd(R)
    tol = np.max(singular_values, initial=0) * max(R.shape) * np.finfo(float).eps
    ghosts = U[:, np.sum(singular_values > tol) :]
    poles, vectors = np.linalg.eigh(_dagger(ghosts) @ Lambda @ ghosts)
    return _dagger(R) @ R, poles, _dagger(R) @ Lambda @ ghosts @ vectors


def _dagger(A: NDArray[np.inexact]) -> NDArray[np.inexact]:
    """Return the conjugate transpose of the last two axes, e.g., at each k-point."""
    return np.swapaxes(A.conj(), -1, -2)
