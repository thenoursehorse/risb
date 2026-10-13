# Copyright (c) 2026 H. L. Nourse
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

"""Functions used in RISB based on the fockit library."""

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
from fockit.algebra import OpSum, c, cdag, hc, nc, zero
from numpy.typing import NDArray

from risb.helpers import get_h0_loc_matrix
from risb.types import BlockMatrixLike


def N_op(spin_names: Sequence[str], n_orb: int) -> OpSum:
    r"""
    Return the total particle-number operator.

    .. math:: \hat{N} = \sum_{i\sigma} \hat{c}_{i\sigma}^\dagger \hat{c}_{i\sigma}.

    Adapted from TRIQS's :func:`triqs.operators.util.observables.N_op` with
    ``off_diag=True``.

    Parameters
    ----------
    spin_names : sequence of str
        Names of the spins, e.g. ``['up', 'dn']``.
    n_orb : int
        Number of orbitals.

    Returns
    -------
    fockit.algebra.OpSum
        The operator with modes ``(spin, orbital)``.

    """
    N = zero()
    for s, o in product(spin_names, range(n_orb)):
        N += nc(s, o)
    return N


def S2_op(spin_names: Sequence[str], n_orb: int) -> OpSum:
    r"""
    Return the square of the total spin operator.

    .. math:: \hat{S}^2 = \hat{S}_z^2 + \frac{1}{2} (\hat{S}_+ \hat{S}_- + \hat{S}_- \hat{S}_+).

    Adapted from TRIQS's :func:`triqs.operators.util.observables.S2_op` with
    ``off_diag=True``.

    Parameters
    ----------
    spin_names : sequence of str
        Names of the spin up and spin down blocks, in that order, e.g.
        ``['up', 'dn']``.
    n_orb : int
        Number of orbitals.

    Returns
    -------
    fockit.algebra.OpSum
        The operator with modes ``(spin, orbital)``.

    """
    up, dn = spin_names
    Sz = zero()
    Sp = zero()
    for o in range(n_orb):
        Sz += 0.5 * (nc(up, o) - nc(dn, o))
        Sp += cdag(up, o) * c(dn, o)
    return Sz * Sz + 0.5 * (Sp * hc(Sp) + hc(Sp) * Sp)


def h_int_kanamori(
    spin_names: Sequence[str],
    n_orb: int,
    U: NDArray[np.float64],
    Uprime: NDArray[np.float64],
    J_hund: float,
) -> OpSum:
    r"""
    Return the Kanamori interaction: density-density, spin-flip and pair-hopping terms.

    .. math::
        \hat{H} = \frac{1}{2} \sum_{(i \sigma) \neq (j \sigma')} U_{i j}^{\sigma \sigma'} \hat{n}_{i \sigma} \hat{n}_{j \sigma'}
            - J \sum_{i \neq j} \hat{c}^\dagger_{i \uparrow} \hat{c}_{i \downarrow} \hat{c}^\dagger_{j \downarrow} \hat{c}_{j \uparrow}
            + J \sum_{i \neq j} \hat{c}^\dagger_{i \uparrow} \hat{c}^\dagger_{i \downarrow} \hat{c}_{j \downarrow} \hat{c}_{j \uparrow},

    where :math:`U^{\sigma \sigma} = U` and :math:`U^{\sigma \bar{\sigma}} = U'`.

    Adapted from TRIQS's :func:`triqs.operators.util.hamiltonians.h_int_kanamori` with
    ``off_diag=True``.

    Parameters
    ----------
    spin_names : sequence of str
        Names of the spins, e.g. ``['up', 'dn']``.
    n_orb : int
        Number of orbitals.
    U : numpy.ndarray
        Orbital matrix :math:`U_{ij}` of density-density interactions between
        same spins.
    Uprime : numpy.ndarray
        Orbital matrix :math:`U'_{ij}` of density-density interactions between
        opposite spins.
    J_hund : float
        Hund's coupling :math:`J` of the spin-flip and pair-hopping terms.

    Returns
    -------
    fockit.algebra.OpSum
        The operator with modes ``(spin, orbital)``.

    """
    h_int = zero()
    for s1, s2, a1, a2 in product(spin_names, spin_names, range(n_orb), range(n_orb)):
        U_val = U[a1, a2] if s1 == s2 else Uprime[a1, a2]
        h_int += 0.5 * float(U_val) * nc(s1, a1) * nc(s2, a2)
        if s1 != s2 and a1 != a2:
            # spin flip and pair hopping
            h_int += -0.5 * J_hund * cdag(s1, a1) * c(s2, a1) * cdag(s2, a2) * c(s1, a2)
            h_int += 0.5 * J_hund * cdag(s1, a1) * cdag(s2, a1) * c(s2, a2) * c(s1, a2)
    return h_int


def matrix_to_Op(A: BlockMatrixLike) -> dict[str, OpSum]:
    """
    Return fockit operators from a matrix representation of quadratic operators.

    Parameters
    ----------
    A : risb.types.BlockMatrixLike
        Single-particle matrix, where each key is a different block.

    Returns
    -------
    dict of fockit.algebra.OpSum
        For each block ``bl``, the operator
        ``sum_ij A[bl][i, j] * cdag(bl, i) * c(bl, j)``.

    """
    Op: dict[str, OpSum] = {}
    for bl, mat in A.items():
        Op[bl] = zero()
        for i, j in np.ndindex(mat.shape):
            Op[bl] += mat[i, j].item() * cdag(bl, i) * c(bl, j)
    return Op


def get_h0_loc(
    h0_k: BlockMatrixLike,
    P: BlockMatrixLike,
    gf_struct_mapping: Mapping[str, str] | None = None,
    force_real: bool = True,
) -> OpSum:
    """
    Return a fockit operator of the non-interacting terms in the subspace given by ``P``.

    Parameters
    ----------
    h0_k : risb.types.BlockMatrixLike
        Single-particle dispersion in each block.
    P : risb.types.BlockMatrixLike
        The projector onto a local cluster within the supercell.
    gf_struct_mapping : mapping, optional
        The mapping from the symmetry blocks in the subspace of P to the
        symmetry blocks of h0_k. Default assumes the keys in `P`
        are the same as the keys in `h0_k`.
    force_real : bool
        Whether to make the resulting matrix real or not.

    Returns
    -------
    fockit.algebra.OpSum
        Non-interacting terms in the cluster defined by the projector `P`,
        with modes ``(block, index)`` for the blocks of `P`.

    """
    if gf_struct_mapping is None:
        gf_struct_mapping = {bl: bl for bl in h0_k}
    h0_loc_matrix = {}
    for bl_sub, P_sub in P.items():
        h0_loc_matrix[bl_sub] = get_h0_loc_matrix(
            h0_k[gf_struct_mapping[bl_sub]], P_sub
        )
        if force_real:
            h0_loc_matrix[bl_sub] = h0_loc_matrix[bl_sub].real
    h0_loc = zero()
    for Op in matrix_to_Op(h0_loc_matrix).values():
        h0_loc += Op
    return h0_loc
