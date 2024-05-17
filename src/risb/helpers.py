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
from copy import deepcopy

import numpy as np
from scipy.linalg import inv, sqrtm

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


def get_d(rho_qp: np.ndarray, ke: np.ndarray) -> np.ndarray:
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
    K = rho_qp - rho_qp @ rho_qp
    return inv(sqrtm(K)) @ ke.T


def get_d2(rho_qp: np.ndarray, ke: np.ndarray, R: np.ndarray) -> np.ndarray:
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
    return inv(sqrtm(K)) @ (inv(R) @ ke).T


def get_lambda_c(
    rho_qp: np.ndarray, R: np.ndarray, Lambda: np.ndarray, D: np.ndarray
) -> np.ndarray:
    """
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
    Eq. 36 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    P = np.eye(rho_qp.shape[0]) - 2.0 * rho_qp
    K = rho_qp - rho_qp @ rho_qp
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return -np.real((R @ D).T @ K_sq_inv @ P).T - Lambda
    # lhs = ((R @ D).T @ K_sq_inv @ P ).T
    # return - Lambda - 0.5 * (lhs + lhs.conj())


def get_lambda(
    R: np.ndarray, D: np.ndarray, Lambda_c: np.ndarray, rho_f: np.ndarray
) -> np.ndarray:
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
        f-electron density matrix from impurity.

    Returns
    -------
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles.

    Notes
    -----
    Derived from Eq. 36 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__ by
    replacing the quasipartice density matrix with the f-electron density
    matrix using Eq. 39.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008

    """
    P = np.eye(rho_f.shape[0]) - 2.0 * rho_f
    K = rho_f - rho_f @ rho_f
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return -np.real((R @ D).T @ K_sq_inv @ P).T - Lambda_c
    # lhs = ( (R @ D).T @ K_sq_inv @ P ).T
    # return - Lambda_c - 0.5 * (lhs + lhs.conj())


def get_r(rho_cf: np.ndarray, rho_f: np.ndarray) -> np.ndarray:
    """
    Return renormalization matrix from impurity problem density matrices.

    Parameters
    ----------
    rho_cf : numpy.ndarray
        c,f-electron hybridization density matrix from impurity.
    rho_f : numpy.ndarray
        f-electron density matrix from impurity.

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
    K = rho_f - rho_f @ rho_f
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return (rho_cf @ K_sq_inv).T


def get_f1(rho_cf: np.ndarray, rho_qp: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Return the first self-consistent equation of RISB.

    Parameters
    ----------
    rho_cf : numpy.ndarray
        c,f-electron hybridization density matrix from impurity.
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
    K = rho_qp - rho_qp @ rho_qp
    K_sq = sqrtm(K)
    return rho_cf - R.T @ K_sq


def get_f2(rho_f: np.ndarray, rho_qp: np.ndarray) -> np.ndarray:
    """
    Return the second self-consistent equation of RISB.

    Parameters
    ----------
    rho_f : numpy.ndarray
        f-electron density matrix from impurity.
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
    return rho_f - rho_qp.T


def get_h_qp(
    R: np.ndarray, Lambda: np.ndarray, h0_kin_k: np.ndarray, mu: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Construct the quasiparticle Hamiltonian :math:`\hat{H}^{\mathrm{qp}}`.

    Parameters
    ----------
    R : numpy.ndarray
        Renormalization matrix) from electrons to quasiparticles
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles.
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
    h_qp = np.einsum("ac,kcd,db->kab", R, h0_kin_k, R.conj().T) + (
        Lambda - mu * np.eye(Lambda.shape[0])
    )
    if not np.allclose(h_qp, np.swapaxes(h_qp, 1, 2).conj()):
        warnings.warn("H_qp is not Hermitian !", RuntimeWarning, stacklevel=2)
    # eig, vec = np.linalg.eigh(h_qp)
    # return (eig, vec)
    return h_qp


def get_h0_kin_k_R(R: np.ndarray, h0_kin_k: np.ndarray, vec: np.ndarray) -> np.ndarray:
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
    return np.einsum("kac,cd,kdb->kab", h0_kin_k, R.conj().T, vec)


def get_R_h0_kin_k_R(
    R: np.ndarray, h0_kin_k: np.ndarray, vec: np.ndarray
) -> np.ndarray:
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
    return np.einsum("pa,kac,cd,kdb->kpb", R, h0_kin_k, R.conj().T, vec)


# \sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_rho_qp(
    vec: np.ndarray, kweights: np.ndarray, P: np.ndarray | None = None
) -> np.ndarray:
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
        return np.einsum("kan,kn,knb->ab", vec, kweights, vec_dag).T
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", vec, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0).T


def get_ke(
    h0_kin_k_R: np.ndarray,
    vec: np.ndarray,
    kweights: np.ndarray,
    P: np.ndarray | None = None,
) -> np.ndarray:
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
        return np.einsum("kan,kn,knb->ab", h0_kin_k_R, kweights, vec_dag)
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", h0_kin_k_R, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0)


def get_ke2(
    R_h0_kin_k_R: np.ndarray,
    vec: np.ndarray,
    kweights: np.ndarray,
    P: np.ndarray | None = None,
) -> np.ndarray:
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
        return np.einsum("kan,kn,knb->ab", R_h0_kin_k_R, kweights, vec_dag)
    P_dag = np.swapaxes(P.conj(), -2, -1)
    middle = np.einsum("kan,kn,knb->kab", R_h0_kin_k_R, kweights, vec_dag)
    return np.sum(P @ middle @ P_dag, axis=0)


def block_to_full(A: np.ndarray) -> np.ndarray:
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


def get_h0_loc_matrix(h0_k: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
    """
    Return a matrix representation of the local terms in a non-interacting dispersion.

    If a projector :attr:`P` onto a local subspace is given the local terms are only in that subspace.

    Parameters
    ----------
    h0_k : numpy.ndarray
        Single-particle dispersion.
    P : numpy.ndarray | None, optional
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
    return np.sum(P @ h0_k @ P.conj().T, axis=0) / float(n_k)


def get_h0_kin_k_mat(h0_k: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Return a matrix representation of the kinetic terms between clusters/local subspaces of a non-interacting dispersion.

    This function only subtracts off the local terms in a single subspace given by :attr:`P`.

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
    h0_k: dict[np.ndarray],
    projectors: list[dict[np.ndarray]] | None = None,
    gf_struct_mapping: list[dict[str, str]] | None = None,
) -> dict[np.ndarray]:
    """
    Return a matrix representation of only the kinetic terms between clusters/local subspaces of a non-interacting dispersion.

    Parameters
    ----------
    h0_k : dict[numpy.ndarray]
        Single-particle dispersion in each block.
    projectors : list[dict[numpy.ndarray]] | None, optional
        The projectors onto each subspace of a local cluster within
        the supercell organized into single-particle symmetry blocks.
    gf_struct_mapping : list[dict[str, str]] | None, optional
        The mapping from the symmetry blocks in the subspace to the
        symmetry blocks of h0_k. Default assumes the keys in `projectors`
        are the same as the keys in `h0_k`.

    Returns
    -------
    dict[numpy.ndarray]
        The single-particle hopping with only the kinetic contribution,
        without the single-particle terms from the clusters defined by
        the projectors.

    """
    h0_kin_k = deepcopy(h0_k)

    if projectors is not None:
        n_clusters = len(projectors)
        if gf_struct_mapping is None:
            gf_struct_mapping = [{bl: bl for bl in h0_k} for i in range(n_clusters)]
        for i, P in enumerate(projectors):
            for bl in P:
                bl_full = gf_struct_mapping[i][bl]
                h0_loc_matrix = get_h0_loc_matrix(h0_k[bl_full], P[bl])
                h0_kin_k[bl_full] -= P[bl].conj().T @ h0_loc_matrix @ P[bl]
                # h0_kin_k[bl_full] = get_h0_kin_k_mat(h0_kin_k[bl_full], P[bl])
    else:
        for bl in h0_k:
            h0_kin_k[bl] -= get_h0_loc_matrix(h0_k[bl])

    return h0_kin_k
