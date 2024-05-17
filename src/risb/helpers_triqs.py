# Copyright (c) 2016-2024 H. L. Nourse
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

"""Functions used in RISB based on the TRIQS library."""

import numpy as np
from triqs.gf import BlockGf, MeshImFreq, MeshProduct, MeshReFreq, inverse
from triqs.operators import Operator, c, c_dag

from risb.helpers import get_h0_loc_matrix, get_h_qp


def get_C_Op(
    gf_struct: list[tuple[str, int]], dagger: bool = False
) -> dict[list[Operator]]:
    """
    Return all creation operators in Hilbert space.

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
    C_Op = {}
    for bl, bl_size in gf_struct:
        if dagger:
            C_Op[bl] = [c_dag(bl, o) for o in range(bl_size)]
        else:
            C_Op[bl] = [c(bl, o) for o in range(bl_size)]
    return C_Op


def matrix_to_Op(
    A: dict[np.ndarray], gf_struct: list[tuple[str, int]]
) -> dict[Operator]:
    """
    Return a TRIQS operator from a matrix representation of quadratic operators.

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
    Op = {}
    for bl in A:
        Op[bl] = C_dag_Op[bl] @ A[bl] @ C_Op[bl]
    return Op


def get_h0_loc_blocks(
    h0_k: dict[np.ndarray],
    P: dict[np.ndarray],
    gf_struct: list[tuple[str, int]] | None = None,
    gf_struct_mapping: dict[str, str] | None = None,
    force_real: bool = True,
) -> dict[Operator]:
    """
    Return a TRIQS operator of the non-interacting terms in the subspace given by :attr:`P`.

    This function splits the terms into the symmetry blocks given by :attr:`gf_struct`.

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
        gf_struct_mapping = {bl: bl for bl in h0_k}

    h0_loc_matrix = {}
    for bl_sub in P:  # sub = subspace of full space defined by h0_k
        bl = gf_struct_mapping[bl_sub]
        if force_real:
            h0_loc_matrix[bl_sub] = get_h0_loc_matrix(h0_k[bl], P[bl_sub]).real
        else:
            h0_loc_matrix[bl_sub] = get_h0_loc_matrix(h0_k[bl], P[bl_sub])

    return matrix_to_Op(A=h0_loc_matrix, gf_struct=gf_struct)


def get_h0_loc(
    h0_k: dict[np.ndarray],
    P: dict[np.ndarray],
    gf_struct: list[tuple[str, int]] | None = None,
    gf_struct_mapping: dict[str, str] | None = None,
    force_real: bool = True,
) -> Operator:
    """
    Return a TRIQS operator of the non-interacting terms in the subspace given by :attr:`P`.

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
    h0_loc_blocks = get_h0_loc_blocks(
        h0_k=h0_k,
        P=P,
        gf_struct=gf_struct,
        gf_struct_mapping=gf_struct_mapping,
        force_real=force_real,
    )
    h0_loc = Operator()
    for Op in h0_loc_blocks.values():
        h0_loc += Op
    return h0_loc


def get_gf_struct_from_g(block_gf: BlockGf) -> list[tuple[str, int]]:
    """
    Return the block structure of a TRIQS Green's function.

    Parameters
    ----------
    block_gf : triqs.gf.BlockGf
        Block Green's function.

    Returns
    -------
    list[tuple[str,int]]
        Green's function's structure.

    """
    gf_struct = []
    for bl, gf in block_gf:
        gf_struct.append([bl, gf.data.shape[-1]])
    return gf_struct


# FIXME have to check h0_k shares the same mesh as Gf
def get_g0_k_w(
    gf_struct: list[tuple[str, int]],
    mesh: MeshProduct,
    h0_k: dict[np.ndarray] | None = None,
    h0_k_gf=None,
    mu: float = 0,
    use_broadcasting: bool = True,
) -> BlockGf:
    """
    Return a TRIQS non-interacting lattice Green's function.

    Parameters
    ----------
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    mesh : triqs.gf.MeshProduct
        A meshproduct where first index is a triqs.gf.MeshBrZone mesh and
        the second index is a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq
        mesh. MeshProduct is a fancy list.
    h0_k : dict[numpy.ndarray], optional
        Non-interacting dispersion indexed as k, orb_i, orb_j.
        Each key in dictionary must follow :attr:`gf_struct`.
    h0_k : triqs.gf.BlockGf
        Non-interacting dispersion as a triqs.gf.BlockGf.
        Must follow the structure given by :attr:`gf_struct`, on
        the mesh given by :attr:`mesh`.
    mu : float, optional
        Chemical potential.
    use_broadcasting : bool, optional
        Whether to treat triqs.gf.Gf with its underlying numpy.ndarray
        data structure, or to use iterators over for loops and lazy
        expressions from TRIQS.

    Returns
    -------
    triqs.gf.BlockGf
        Non-interacting Green's function from a non-interacting dispersion
        relation :attr:`h0_k`.

    """
    g0_k_w = BlockGf(mesh=mesh, gf_struct=gf_struct)
    for bl, gf in g0_k_w:
        identity = np.eye(
            *gf.data.shape[-2::]
        )  # Last two indices are the orbital structure
        if use_broadcasting:
            w = np.fromiter(gf.mesh[1].values(), dtype=complex)
            if h0_k is not None:
                # gf.data[...] = np.linalg.inv( ((w + mu)[:,None,None] * identity[None,:])[None,:,...] - h0_k[bl][:,None,...] )
                gf.data[...] = ((w + mu)[:, None, None] * identity[None, :])[
                    None, :, ...
                ] - h0_k[bl][:, None, ...]
                gf.invert()
            elif h0_k_gf is not None:
                gf.data[...] = ((w + mu)[:, None, None] * identity[None, :])[
                    None, :, ...
                ] - h0_k_gf[bl].data[:, None, ...]
                gf.invert()
            else:
                msg = "Require a kwarg of h0_k or h0_k_gf !"
                raise ValueError(msg)
        else:
            if h0_k is not None:
                for k, w in gf.mesh:
                    gf[k, w] = inverse(w + mu - h0_k[bl][k.data_index])
            elif h0_k_gf is not None:
                for k, w in gf.mesh:
                    gf[k, w] = inverse(w + mu - h0_k_gf[bl][k])
            else:
                msg = "Require a kwarg of h0_k or h0_k_gf !"
                raise ValueError(msg)
    return g0_k_w


def get_sigma_w(
    gf_struct: list[tuple[str, int]],
    mesh: MeshReFreq | MeshImFreq,
    Lambda: dict[np.ndarray],
    R: dict[np.ndarray],
    mu: float = 0,
    h0_loc: dict[np.ndarray] | None = None,
    use_broadcasting: bool = True,
) -> BlockGf:
    r"""
    Return a TRIQS local self-energy from RISB.

    Parameters
    ----------
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    mesh : triqs.gf.meshes.MeshReFreq | triqs.gf.meshes.MeshImFreq
        Frequency mesh of the returned self-energy.
    Lambda : dict[numpy.ndarray]
        Correlation potential of quasiparticles.
        Each key in dictionary must follow :attr:`gf_struct`.
    R : dict[numpy.ndarray]
        Rormalization matrix from electrons to quasiparticles.
        Each key in dictionary must follow :attr:`gf_struct`.
    mu : float, optional
        Chemical potential.
    h0_loc : dict[numpy.ndarray], optional
        Matrix of non-interacting hopping terms in each local subspace.
        Each key in dictionary must follow :attr:`gf_struct`.
    use_broadcasting : bool, optional
        Whether to treat triqs.gf.Gf with its underlying numpy.ndarray
        data structure, or to use iterators over for loops and lazy
        expressions from TRIQS.

    Returns
    -------
    triqs.gf.BlockGf
        RISB local self-energy :math:`\Sigma(\omega)`.

    """
    sigma_w = BlockGf(mesh=mesh, gf_struct=gf_struct)
    for bl, gf in sigma_w:
        identity = np.eye(
            *gf.data.shape[-2::]
        )  # Last two indices are the orbital structure
        Z_inv = np.linalg.inv(R[bl] @ R[bl].conj().T)
        hf = np.linalg.inv(R[bl]) @ Lambda[bl] @ np.linalg.inv(R[bl].conj().T)
        if use_broadcasting:
            w = np.fromiter(gf.mesh.values(), dtype=complex)
            if h0_loc is not None:
                gf.data[...] = (
                    (identity - Z_inv) * w[:, None, None]
                    + hf
                    + (identity - Z_inv) * mu
                    - h0_loc[bl]
                )
            else:
                gf.data[...] = (
                    (identity - Z_inv) * w[:, None, None] + hf + (identity - Z_inv) * mu
                )
        else:
            if h0_loc is not None:
                for w in gf.mesh:
                    gf[w] = (
                        (identity - Z_inv) * w
                        + hf
                        + (identity - Z_inv) * mu
                        - h0_loc[bl]
                    )
            else:
                for w in gf.mesh:
                    gf[w] = (identity - Z_inv) * w + hf + (identity - Z_inv) * mu
    return sigma_w


# FIXME have to check h0_kin_k shares the same mesh
# FIXME allow h0_kin_k_gf structure as well?
def get_g_qp_k_w(
    gf_struct: list[tuple[str, int]],
    mesh: MeshProduct,
    h0_kin_k: dict[np.ndarray],
    Lambda: dict[np.ndarray],
    R: dict[np.ndarray],
    mu: float = 0,
    use_broadcasting: bool = True,
) -> BlockGf:
    r"""
    Return a TRIQS lattice RISB quasiparticle Green's function.

    Parameters
    ----------
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
    mesh : triqs.gf.MeshProduct
        A meshproduct where first index is a triqs.gf.MeshBrZone mesh and
        the second index is a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq
        mesh. MeshProduct is a fancy list.
    h0_kin_k : dict[numpy.ndarray]
        Single-particle dispersion between local clusters. Indexed as
        k, orb_i, orb_j. Each key in dictionary must follow :attr:`gf_struct`.
    Lambda : dict[numpy.ndarray]
        Correlation potential of quasiparticles.
        Each key in dictionary must follow :attr:`gf_struct`.
    R : dict[numpy.ndarray]
        Rormalization matrix from electrons to quasiparticles.
        Each key in dictionary must follow :attr:`gf_struct`.
    mu : float, optional
        Chemical potential.
    use_broadcasting : bool, optional
        Whether to treat triqs.gf.Gf with its underlying numpy.ndarray
        data structure, or to use iterators over for loops and lazy
        expressions from TRIQS.

    Returns
    -------
    triqs.gf.BlockGf
        Quasiparticle Green's function :math:`G^{\mathrm{qp}}(k,\omega)`.

    """
    g_qp_k_w = BlockGf(mesh=mesh, gf_struct=gf_struct)
    for bl, gf in g_qp_k_w:
        identity = np.eye(
            *gf.data.shape[-2::]
        )  # Last two indices are the orbital structure
        h_qp = get_h_qp(R=R[bl], Lambda=Lambda[bl], h0_kin_k=h0_kin_k[bl], mu=mu)
        if use_broadcasting:
            w = np.fromiter(gf.mesh[1].values(), dtype=complex)
            # gf.data[...] = inverse( (w[:,None,None] * identity[None,:])[None,:,...] - h_qp[:,None,...] )
            gf.data[...] = (w[:, None, None] * identity[None, :])[None, :, ...] - h_qp[
                :, None, ...
            ]
            gf.invert()
        else:
            for k, w in g_qp_k_w.mesh:
                gf[k, w] = inverse(w - h_qp[k.data_index])
    return g_qp_k_w


def get_g_k_w(
    g0_k_w: BlockGf | None = None,
    sigma_w: BlockGf | None = None,
    g_qp_k_w: BlockGf | None = None,
    R: dict[np.ndarray] | None = None,
    use_broadcasting: bool = True,
) -> BlockGf:
    r"""
    Return a TRIQS lattice interacting Green's function, with a local self-energy.

    Must pass g0_k_w and sigma_w, or g_qp_k_w and R. Passing g_qp_k_w and R is specific
    to RISB. Passing g0_k_w and sigma_w is valid for any interacting theory with a local
    self-energy (it is just Dyson's equation at each k-point).

    Parameters
    ----------
    g0_k_w : triqs.gf.BlockGf, optional
        Non-interacting Green's function on a meshproduct where
        the first index is a triqs.gf.MeshBrZone mesh and
        the second index is a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq mesh.
    sigma_w : triqs.gf.BlockGf, optional
        Local self-energy on a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq mesh.
    gp_k_w : triqs.gf.BlockGf, optional
        Quasiparticle Green's function on a meshproduct where
        the first index is a triqs.gf.MeshBrZone mesh and
        the second index is a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq mesh.
    R : dict[numpy.ndarray], optional
        Rormalization matrix from electrons to quasiparticles.
        Each key in dictionary must follow the gf_struct in
        :attr:`g0_k_w` and :attr:`sigma_w`.
    use_broadcasting : bool, optional
        Whether to treat triqs.gf.Gf with its underlying numpy.ndarray
        data structure, or to use iterators over for loops and lazy
        expressions from TRIQS.

    Returns
    -------
    triqs.gf.BlockGf
        Physical c-electrons Green's function :math:`G(k,\omega)`.

    """
    if (g0_k_w is not None) and (sigma_w is not None):
        g_k_w = g0_k_w.copy()
        g_k_w.zero()
        for bl, gf in g_k_w:
            if use_broadcasting:
                gf.data[...] = inverse(g0_k_w[bl]).data - sigma_w[bl].data[None, ...]
                gf.invert()
            else:
                for k, w in gf.mesh:
                    gf[k, w] = inverse(inverse(g0_k_w[bl][k, w]) - sigma_w[bl][w])
    elif (g_qp_k_w is not None) and (R is not None):
        g_k_w = g_qp_k_w.copy()
        g_k_w.zero()
        for bl, gf in g_qp_k_w:
            g_k_w[bl] = R[bl].conj().T @ gf @ R[bl]
    else:
        msg = "Required kwargs are one of the pairs g0_k_w and sigma_w, or g_qp_k_w and R !"
        raise ValueError(msg)
    return g_k_w


def get_g_w_loc(g_k_w: BlockGf, use_broadcasting: bool = True) -> BlockGf:
    """
    Return a TRIQS local Green's function from a lattice Green's function.

    Parameters
    ----------
    g_k_w : triqs.gf.BlockGf
        A Green's function on a meshproduct where
        the first index is a triqs.gf.MeshBrZone mesh and
        the second index is a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq mesh.
    use_broadcasting : bool, optional
        Whether to treat triqs.gf.Gf with its underlying numpy.ndarray
        data structure, or to use iterators over for loops and lazy
        expressions from TRIQS.

    Returns
    -------
    triqs.gf.BlockGf
        k-integrated Green's function on a triqs.gf.MeshReFreq or triqs.gf.MeshImFreq mesh.

    """
    k_mesh = g_k_w.mesh[0]
    w_mesh = g_k_w.mesh[1]
    gf_struct = get_gf_struct_from_g(g_k_w)
    g_w_loc = BlockGf(mesh=w_mesh, gf_struct=gf_struct)
    for bl, gf in g_k_w:
        if use_broadcasting:
            g_w_loc[bl].data[...] = np.sum(gf.data, axis=0)
        else:
            for k in k_mesh:
                g_w_loc[bl] += gf[k, :]
        g_w_loc[bl] /= np.prod(k_mesh.dims)
    return g_w_loc
