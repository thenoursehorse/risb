
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
from triqs.operators import Operator, c, c_dag, dagger
from risb.helpers import get_h0_loc_mat

def get_C_Op(gf_struct, dagger=False):
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

def mat_to_Op(mat, C_Op):
    C_dag_Op = dict()
    for bl in C_Op:
        C_dag_Op[bl] = [dagger(Op) for Op in C_Op[bl]]
    Op = dict()
    for bl in mat:
        Op[bl] = C_dag_Op[bl] @ mat[bl] @ C_Op[bl]
    return Op

def get_h0_loc_blocks(h0_k, P, gf_struct=None, force_real=True):
    """
    """
    if gf_struct is None:
        gf_struct = [(k, v.shape[-2]) for k, v in P.items()]

    C_Op = get_C_Op(gf_struct=gf_struct)
    C_dag_Op = get_C_Op(gf_struct=gf_struct, dagger=True)
    
    # FIXME if h0_k and P do not have the same blocks,
    # need mapping from P blocks to h0_k blocks
    h0_loc_mat = dict()
    for bl_sub in P.keys(): # sub = subspace of full space defined by h0_k
        bl = bl_sub # FIXME correct mapping
        if force_real:
            h0_loc_mat[bl_sub] = get_h0_loc_mat(h0_k[bl], P[bl_sub]).real
        else:
            h0_loc_mat[bl_sub] = get_h0_loc_mat(h0_k[bl], P[bl_sub])

    h0_loc = dict()
    for bl_sub in h0_loc_mat.keys():    
        h0_loc[bl_sub] = C_dag_Op[bl_sub] @ h0_loc_mat[bl_sub] @ C_Op[bl_sub]

    return h0_loc

def get_h0_loc(h0_k, P, gf_struct=None, force_real=True):
    h0_loc_blocks = get_h0_loc_blocks(h0_k=h0_k, P=P, gf_struct=gf_struct, force_real=force_real)
    h0_loc = Operator()
    for Op in h0_loc_blocks.values():
        h0_loc += Op
    return h0_loc