# Copyright (c) 2016 H. L. Nourse
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

import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import inv
#from scipy.linalg import pinv
#from scipy.special import binom

## Formula is (1-A)^{-1/2} = sum_r=0^{infty} (-1)^r * 1/2 choose r * A^r
#def one_sqrtm_inv(A, tol=np.finfo(float).eps, N=10000):
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
#def get_K_sq_inv(pdensity, hdensity, tol=np.finfo(float).eps, N=10000):
#    return one_sqrtm_inv(A=pdensity, tol=tol, N=N) @ one_sqrtm_inv(A=hdensity, tol=tol, N=N)

def get_d(pdensity : np.ndarray, 
          ke : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    pdensity : numpy.ndarray
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
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(pdensity, np.eye(pdensity.shape[0])-pdensity)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = inv(K_sq)
    return K_sq_inv @ ke.T

def get_lambda_c(pdensity : np.ndarray, 
                 R : np.ndarray, 
                 Lambda: np.ndarray, 
                 D: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    pdensity : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.

    R : numpy.ndarray
        Unitary transformation (renormalization matrix) from electrons to 
        quasiparticles.
    
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
    P = np.eye(pdensity.shape[0]) - 2.0*pdensity
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return -np.real( (R @ D).T @ K_sq_inv @ P ).T - Lambda

def get_lambda(R : np.ndarray, 
               D : np.ndarray, 
               Lambda_c : np.ndarray, 
               Nf : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    R : numpy.ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    D : numpy.ndarray
        Hybridization coupling.
    Lambda_c : numpy.ndarray
        Bath coupling.
    Nf : numpy.ndarray
        f-electron density matrix from impurity.

    Returns
    -------
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles. 
    
    Notes
    -----
    Derived from Eq. 36 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__ by replacing 
    the quasipartice density matrix with the f-electron density matrix using 
    Eq. 39.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    """
    P = np.eye(Nf.shape[0]) - 2.0*Nf
    K = Nf - Nf @ Nf
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    # FIXME check if .T or not. It won't ever matter because we always make
    # c and f particles the same basis, but in principle it should be correct
    return -np.real( (R @ D).T @ K_sq_inv @ P ).T - Lambda_c
    #return -np.real( (R @ D).T @ K_sq_inv @ P ) - Lambda_c 

def get_r(Mcf : np.ndarray, Nf : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    Mcf : numpy.ndarray
        c,f-electron hybridization density matrix from impurity.
    Nf : numpy.ndarray
        f-electron density matrix from impurity.

    Returns
    -------
    R : numpy.ndarray
        Renormalization (mean-field unitary transformation) matrix.
    
    Notes
    -----
    Derived from Eq. 38 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__ by 
    replacing the quasiparticle density matrix sith the f-electron density matrix 
    using Eq. 39.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    """
    K = Nf - Nf @ Nf
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return (Mcf @ K_sq_inv).T

def get_f1(Mcf : np.ndarray, pdensity : np.ndarray, R : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    Mcf : numpy.ndarray
        c,f-electron hybridization density matrix from impurity.
    pdensity : numpy.ndarray
        Quasiparticle density matrix obtained from the mean-field.
    R : numpy.ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.

    Returns
    -------
    f1 : numpy.ndarray
        First self-consistency equation.

    Notes
    -----
    Eq. 38 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    """
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    return Mcf - R.T @ K_sq

def get_f2(Nf : np.ndarray, pdensity : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    Nf : numpy.ndarray
        f-electron density matrix from impurity.
    pdensity : numpy.ndarray
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
    return Nf - pdensity.T

def get_h_qp(R : np.ndarray, 
             Lambda : np.ndarray, 
             h0_kin_k : np.ndarray, 
             mu : float = 0) -> tuple[ np.ndarray, np.ndarray ]:
    """
    Construct eigenvalues and eigenvectors of the quasiparticle Hamiltonian.
    
    Parameters
    ----------
    R : numpy.ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    Lambda : numpy.ndarray
        Correlation potential of quasiparticles.
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters. Indexed as k, orb_i, orb_j.
    mu : float, optional
        Chemical potential.

    Return
    ------
    eigenvalues : numpy.ndarray
        Indexed as k, band.
    eigenvectors : numpy.ndarray
        Indexed as k, each column an eigenvector.
    
    Notes
    -----
    Eq. A34 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    """
    #h_qp = np.einsum('ac,cdk,db->kab', R, h0_kin_k, R.conj().T, optimize='optimal') + (Lambda - mu*np.eye(Lambda.shape[0]))
    h_qp = np.einsum('ac,kcd,db->kab', R, h0_kin_k, R.conj().T) + \
        (Lambda - mu*np.eye(Lambda.shape[0]))
    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_h0_R(R : np.ndarray, h0_kin_k : np.ndarray, vec : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    R : numpy.ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters. Indexed as k, orb_i, orb_j.
    vec : numpy.ndarray
        Eigenvectors of quasiparticle Hamiltonian. Indexed as k, each column an eigenvector.

    Returns
    -------
    h0_kin_R : numpy.ndarray
        Matrix representation of lopsided quasiparticle Hamiltonian. 

    Notes
    -----
    This is ``H^qp`` with the inverse of the renormalization matrix R 
    multiplied on the left.
    """
    return np.einsum('kac,cd,kdb->kab', h0_kin_k, R.conj().T, vec)

#\sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_pdensity(vec : np.ndarray, kweights : np.ndarray, P : np.ndarray | None = None) -> np.ndarray:
    """
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
    pdensity : numpy.ndarray
        Quasiparticle density matrix from mean-field.
    
    Notes
    -----
    Eqs. 32 and 34 in `10.1103/PhysRevX.5.011008 <PRX011008_>`__, but not in the 
    natural-basis gauge. See Eq. 112 in the supplemental of 
    `10.1103/PhysRevLett.118.126401 <PRL126401_>`__.

    .. _PRX011008: https://doi.org/10.1103/PhysRevX.5.011008
    .. _PRL126401: https://doi.org/10.1103/PhysRevLett.118.126401

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        return np.real( np.einsum('kan,kn,knb->ab', vec, kweights, vec_dag).T )
    else:
        P_dag = np.swapaxes(P.conj(), -2, 1)
        middle = np.einsum('kan,kn,knb->kab', vec, kweights, vec_dag)
        return np.real( np.sum(P @ middle @ P_dag, axis=0).T )

def get_ke(h0_kin_R : np.ndarray, vec : np.ndarray, kweights : np.ndarray, P : np.ndarray | None = None) -> np.ndarray:
    """
    Parameters
    ----------
    h0_kin_k : numpy.ndarray
        Single-particle dispersion between local clusters.
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
        return np.einsum('kan,kn,knb->ab', h0_kin_R, kweights, vec_dag)
    else:
        P_dag = np.swapaxes(P.conj(), -2, 1)
        middle = np.einsum('kan,kn,knb->kab', h0_kin_R, kweights, vec_dag)
        return np.sum(P @ middle @ P_dag, axis=0)