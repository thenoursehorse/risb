
# Copyright (c) 2023 H. L. Nourse
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
from triqs.operators import Operator, c, c_dag
from triqs.gf import BlockGf, inverse
from risb.helpers import get_h0_loc_matrix, get_h_qp

def get_C_Op(gf_struct : list[tuple[str,int]], dagger : bool = False) -> dict[list[Operator]]:
    """
    Parameters
    ----------
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    dagger : bool
        Whether to return the creation operator or not.

    Returns
    -------
    dict[list[triqs.operators.Operator]]
        For each block in `gf_struct`, a vector of all creation/annihilation 
        operators in its subspace.
    """
    C_Op = dict()
    for bl, bl_size in gf_struct:
        if dagger:
            C_Op[bl] = [c_dag(bl, o) for o in range(bl_size)]
        else:
            C_Op[bl] = [c(bl, o) for o in range(bl_size)]
    return C_Op

def matrix_to_Op(A : dict[np.ndarray], gf_struct : list[tuple[str,int]]) -> dict[Operator]:
    """
    Parameters
    ----------
    A : dict[numpy.ndarray]
        Single-particle matrix, where each key is a different block.
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    Returns
    -------
    dict[triqs.operators.Operator]
        The single-particle matrix as a quadratic TRIQS operator.
    """
    C_dag_Op = get_C_Op(gf_struct=gf_struct, dagger=True)
    C_Op = get_C_Op(gf_struct=gf_struct, dagger=False)
    Op = dict()
    for bl in A:
        Op[bl] = C_dag_Op[bl] @ A[bl] @ C_Op[bl]
    return Op

def get_h0_loc_blocks(h0_k : dict[np.ndarray], 
                      P : dict[np.ndarray], 
                      gf_struct : list[tuple[str,int]] | None = None, 
                      gf_struct_mapping : dict[str,str] | None = None, 
                      force_real: bool = True) -> dict[Operator]:
    """
    Parameters
    ----------
    h0_k : dict[numpy.ndarray]
        Single-particle dispersion in each block.
    P : dict[numpy.ndarray]
        The projector onto a local cluster within the supercell.
    gf_struct : list of pairs [ (str,int), ...] | None, optional
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``. Default is 
        structure worked out from the projector P.
    gf_struct_mapping : dict[str, str] | None, optional
        The mapping from the symmetry blocks in the subspace of P to the 
        symmetry blocks of h0_k. Default assumes the keys in `P`
        are the same as the keys in `h0_k`.
    force_real : bool
        Whether to make the resulting matrix real or not.

    Returns
    -------
    dict[triqs.operators.Operator]
        For each single-particle symmetry block the non-interacting 
        terms in the cluster defined by the projector `P`.
    """
    if gf_struct is None:
        gf_struct = [(k, v.shape[-2]) for k, v in P.items()]
    if gf_struct_mapping is None:
        gf_struct_mapping = {bl:bl for bl in h0_k.keys()}
    
    h0_loc_matrix = dict()
    for bl_sub in P.keys(): # sub = subspace of full space defined by h0_k
        bl = gf_struct_mapping[bl_sub]
        if force_real:
            h0_loc_matrix[bl_sub] = get_h0_loc_matrix(h0_k[bl], P[bl_sub]).real
        else:
            h0_loc_matrix[bl_sub] = get_h0_loc_matrix(h0_k[bl], P[bl_sub])

    return matrix_to_Op(A=h0_loc_matrix, gf_struct=gf_struct)

def get_h0_loc(h0_k : dict[np.ndarray], 
               P : dict[np.ndarray], 
               gf_struct : list[tuple[str,int]] | None = None, 
               gf_struct_mapping : dict[str,str] | None = None, 
               force_real : bool = True) -> Operator:
    """
    Parameters
    ----------
    h0_k : dict[numpy.ndarray]
        Single-particle dispersion in each block.
    P : dict[numpy.ndarray]
        The projector onto a local cluster within the supercell.
    gf_struct : list of pairs [ (str,int), ...] | None, optional
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``. Default is 
        structure worked out from the projector P.
    gf_struct_mapping : dict[str, str] | None, optional
        The mapping from the symmetry blocks in the subspace of P to the 
        symmetry blocks of h0_k. Default assumes the keys in `P`
        are the same as the keys in `h0_k`.
    force_real : bool
        Whether to make the resulting matrix real or not.

    Returns
    -------
    triqs.operators.Operator
        Non-interacting terms in the cluster defined by the projector `P`.
    """
    h0_loc_blocks = get_h0_loc_blocks(h0_k=h0_k, P=P, gf_struct=gf_struct, gf_struct_mapping=gf_struct_mapping, force_real=force_real)
    h0_loc = Operator()
    for Op in h0_loc_blocks.values():
        h0_loc += Op
    return h0_loc

def get_gf_struct_from_g(block_gf):
    gf_struct = []
    for bl, gf in block_gf:
        gf_struct.append( [bl, gf.data.shape[-1]] )
    return gf_struct

def get_sigma_w(gf_struct, mesh, Lambda, R, mu=0, h_loc=None):
    sigma_w = BlockGf(mesh=mesh, gf_struct=gf_struct)
    for bl, gf in sigma_w:
        id = np.eye(*gf.data.shape[-2::]) # Last two indices are the orbital structure
        Z = R[bl] @ R[bl].conj().T
        id_Z = (id - np.linalg.inv(Z))
        id_Z_mu = id_Z * mu
        hf = np.linalg.inv(R[bl]) @ Lambda[bl] @ np.linalg.inv(R[bl].conj().T)
        if h_loc is not None:
            for w in gf.mesh:
                gf[w] = id_Z * w + hf + id_Z_mu - h_loc[bl]
        else:
            for w in gf.mesh:
                gf[w] = id_Z * w + hf + id_Z_mu
    return sigma_w

# FIXME have to check h0_kin_k shares the same mesh 
def get_g_qp_k_w(gf_struct, mesh, h0_kin_k, Lambda, R, mu=0):
    g_qp_k_w = BlockGf(mesh=mesh, gf_struct=gf_struct)
    for bl, gf in g_qp_k_w:
        h_qp = get_h_qp(R=R[bl], Lambda=Lambda[bl], h0_kin_k=h0_kin_k[bl], mu=mu)
        for k, w in g_qp_k_w.mesh:
            gf[k,w] = inverse(w - h_qp[k.data_index])
    return g_qp_k_w

def get_g_k_w(**kwargs):
    if ('g0_k_w' in kwargs) and ('sigma_w' in kwargs):
        g0_k_w = kwargs['g0_k_w']
        sigma_w = kwargs['sigma_w']
        g_k_w = g0_k_w.copy()
        g_k_w.zero()
        for bl, gf in g_k_w:
            for k, w in gf.mesh:            
                gf[k,w] = inverse(inverse(g0_k_w[bl][k,w]) - sigma_w[bl][w])
    elif ('g_qp_k_w' in kwargs) and ('R' in kwargs):
        g_qp_k_w = kwargs['g_qp_k_w']
        R = kwargs['R']
        g_k_w = g_qp_k_w.copy()
        g_k_w.zero()
        for bl, gf in g_qp_k_w:
            g_k_w[bl] = R[bl].conj().T @ gf @ R[bl]
    else:
        msg = 'Required kwargs are g0_k_w and sigma_w, or g_qp_k_w and R !'
        raise ValueError(msg)
    return g_k_w

def get_g_w_loc(g_k_w):
    k_mesh = g_k_w.mesh[0]
    w_mesh = g_k_w.mesh[1]
    gf_struct = get_gf_struct_from_g(g_k_w)
    g_w_loc = BlockGf(mesh=w_mesh, gf_struct=gf_struct)
    for bl, gf in g_w_loc:
        for k in k_mesh:
            gf += g_k_w[bl][k,:]
        gf /= np.prod(k_mesh.dims)
    return g_w_loc