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
from scipy.special import binom

# Formula is (1-A)^{-1/2} = sum_r=0^{infty} (-1)^r * 1/2 choose r * A^r
def one_sqrtm_inv(A, tol=np.finfo(float).eps, N=10000):
    # Do r = 0 manually (it is just the identity)
    A_r = np.eye(A.shape[0])
    out = np.eye(A.shape[0])
    for r in range(1,N+1):
        old = out.copy()
        A_r = A_r @ A
        out += (-1)**r * binom(-1/2., r) * A_r
        err = np.linalg.norm(out - old)
        if err < tol:
            break
    print(r,err)
    return out

def get_K_sq_inv(pdensity, hdensity, tol=np.finfo(float).eps, N=10000):
    return one_sqrtm_inv(A=pdensity, tol=tol, N=N) @ one_sqrtm_inv(A=hdensity, tol=tol, N=N)

def get_d(pdensity, ke):
    """
    Return the hybridization coupling for rotationally invariant slave-bosons.
    
    This is Eq. 35 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------

    pdensity : ndarray
        Quasiparticle density matrix obtained from the mean-field.

    ke : ndarray
        Lopsided quasiparticle kinetic energy.

    """
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    #K_sq_inv = get_K_sq_inv(pdensity, np.eye(pdensity.shape[0])-pdensity)
    #K_sq_inv = pinv(K_sq)
    K_sq_inv = inv(K_sq)
    return K_sq_inv @ ke.T

def get_lambda_c(pdensity, R, Lambda, D):
    """
    Return the bath coupling for rotationally invariant slave-bosons.
    
    This is Eq. 36 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------

    pdensity : ndarray
        Quasiparticle density matrix obtained from the mean-field.

    R : ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    
    Lambda : ndarray
        Correlation potential of quasiparticles.

    D : ndarray
        Hybridization coupling.

    """
    P = np.eye(pdensity.shape[0]) - 2.0*pdensity
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return -np.real( (R @ D).T @ K_sq_inv @ P ).T - Lambda

def get_lambda(R, D, Lambda_c, Nf):
    """
    Return the correlation potential of the quasiparticles for rotationally 
    invariant slave-bosons.
    
    This is derived from Eq. 36 in 10.1103/PhysRevX.5.011008 by replacing 
    the quasipartice density matrix with the f-electron density matrix using 
    Eq. 39.

    Parameters
    ----------
    
    R : ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    
    D : ndarray
        Hybridization coupling.

    Lambda_c : ndarray
        Bath coupling.

    Nf : ndarray
        f-electron density matrix from impurity.
    
    """
    P = np.eye(Nf.shape[0]) - 2.0*Nf
    K = Nf - Nf @ Nf
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    # FIXME check if .T or not. It won't ever matter because we always make
    # c and f particles the same basis, but in principle it should be correct
    return -np.real( (R @ D).T @ K_sq_inv @ P ).T - Lambda_c
    #return -np.real( (R @ D).T @ K_sq_inv @ P ) - Lambda_c 

def get_r(Mcf, Nf):
    """
    Return the renormalization matrix for rotationally invariant slave-bosons.

    Parameters
    ----------
    
    Mcf : ndarray
        c,f-electron hybridization density matrix from impurity.
    
    Nf : ndarray
        f-electron density matrix from impurity.
    
    This is derived from Eq. 38 in 10.1103/PhysRevX.5.011008 by replacing 
    the quasipartice density matrix with the f-electron density matrix using 
    Eq. 39.

    """
    K = Nf - Nf @ Nf
    K_sq = sqrtm(K)
    K_sq_inv = inv(K_sq)
    return (Mcf @ K_sq_inv).T

def get_f1(Mcf, pdensity, R):
    """
    Return the first self-consistency equation for rotationally invariant 
    slave-bosons.

    This is Eq. 38 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------
    
    Mcf : ndarray
        c,f-electron hybridization density matrix from impurity.
    
    pdensity : ndarray
        Quasiparticle density matrix obtained from the mean-field.

    R : ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.

    """
    K = pdensity - pdensity @ pdensity
    K_sq = sqrtm(K)
    return Mcf - R.T @ K_sq

def get_f2(Nf, pdensity):
    """
    Return the second self-consistency equation for rotationally invariant 
    slave-bosons.

    This is Eq. 39 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------
    
    Nf : ndarray
        f-electron density matrix from impurity.
    
    pdensity : ndarray
        Quasiparticle density matrix obtained from the mean-field.

    """
    return Nf - pdensity.T

def get_h_qp(R, Lambda, h0_kin_k, mu=0):
    """
    Return the eigenvalues and eigenvectors of the quasiparticle Hamiltonian 
    for rotationally invariant slave-bosons.
    
    This is Eq. A34 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------
    
    R : ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    
    Lambda : ndarray
        Correlation potential of quasiparticles.

    h0_kin_k : ndarray
        Single-particle dispersion between local clusters.

    mu : optional, float
        Chemical potential.

    """
    #h_qp = np.einsum('ac,cdk,db->kab', R, h0_kin_k, R.conj().T, optimize='optimal') + (Lambda - mu*np.eye(Lambda.shape[0]))
    h_qp = np.einsum('ac,kcd,db->kab', R, h0_kin_k, R.conj().T) + \
        (Lambda - mu*np.eye(Lambda.shape[0]))
    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_h0_R(R, h0_kin_k, vec):
    """
    Return the matrix representation of the lopsided quasiparticle Hamiltonian 
    for rotationally invariant slave-bosons.

    This is ``H^qp`` with the inverse of the renormalization matrix R 
    multiplied on the left.

    Parameters
    ----------
    
    R : ndarray
        Unitary transformation (renormalization matrix) from quasiparticles to 
        electrons.
    
    h0_kin_k : ndarray
        Single-particle dispersion between local clusters.
    
    vec : ndarray
        Eigenvectors of quasiparticle Hamiltonian.

    """
    return np.einsum('kac,cd,kdb->kab', h0_kin_k, R.conj().T, vec)

#\sum_n \sum_k [A_k P_k]_{an} [D_k]_n  [P_k^+ B_k]_{nb}
def get_pdensity(vec, wks, P=None):
    """
    Return the quasiparticle density matrix for rotationally invariant 
    slave-bosons.
    
    This is Eqs. 32 and 34 in 10.1103/PhysRevX.5.011008, but not in the 
    natural-basis gauge. See Eq. 112 in the supplemental of 
    10.1103/PhysRevLett.118.126401.

    Parameters
    ----------
    
    vec : ndarray
        Eigenvectors of quasiparticle Hamiltonian.

    wks : ndarray
        Integration weights at each k-point.

    P : optional, ndarray
        Projection matrix onto correlated subspace.

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        return np.real( np.einsum('kan,kn,knb->ab', vec, wks, vec_dag).T )
    else:
        P_dag = np.swapaxes(P.conj(), -2, 1)
        middle = np.einsum('kan,kn,knb->kab', vec, wks, vec_dag)
        return np.real( np.sum(P @ middle @ P_dag, axis=0).T )

def get_ke(h0_kin_R, vec, wks, P=None):
    """
    Return the lopsided quasiparticle kinetic energy for rotationally invariant 
    slave-bosons.
    
    This is Eq. 35 in 10.1103/PhysRevX.5.011008.

    Parameters
    ----------
    
    h0_kin_k : ndarray
        Single-particle dispersion between local clusters.
    
    vec : ndarray
        Eigenvectors of quasiparticle Hamiltonian.

    wks : ndarray
        Integration weights at each k-point.

    P : optional, ndarray
        Projection matrix onto correlated subspace.

    """
    vec_dag = np.swapaxes(vec.conj(), -1, -2)
    if P is None:
        return np.einsum('kan,kn,knb->ab', h0_kin_R, wks, vec_dag)
    else:
        P_dag = np.swapaxes(P.conj(), -2, 1)
        middle = np.einsum('kan,kn,knb->kab', h0_kin_R, wks, vec_dag)
        return np.sum(P @ middle @ P_dag, axis=0)