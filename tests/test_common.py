# ruff: noqa: D100, D103
from itertools import product

import numpy as np
from numpy.typing import NDArray

from risb.types import BlockMatrix, GfStruct


def symmetrize_blocks(A: list[BlockMatrix]) -> list[BlockMatrix]:
    for A_i in A:
        A_sym = np.mean(list(A_i.values()), axis=0)
        for bl in A_i:
            A_i[bl] = A_sym
    return A


def build_cubic_h0_k(
    gf_struct: GfStruct | None = None,
    nkx: int = 6,
    spatial_dim: int = 2,
    t: float = 1,
    a: float = 1,
) -> dict[str, NDArray[np.float64]]:
    if gf_struct is None:
        gf_struct = [("up", 1), ("dn", 1)]
    n_orb = gf_struct[0][1]
    for _, bsize in gf_struct:
        if bsize != n_orb:
            msg = "Each block must have the same number of orbitals !"
            raise ValueError(msg)

    t_scaled = -t / float(spatial_dim)
    n_k = nkx**spatial_dim

    # Make mesh
    mesh = np.empty(shape=(n_k, spatial_dim))
    coords = [range(nkx) for _ in range(spatial_dim)]
    for idx, coord in enumerate(product(*coords)):
        for i in range(len(coord)):
            mesh[idx, i] = coord[i] / float(nkx)

    # Make hopping matrix
    h0_k: dict[str, NDArray[np.float64]] = {}
    for bl, n_orb in gf_struct:
        di = np.diag_indices(n_orb)
        h0_k[bl] = np.zeros([n_k, n_orb, n_orb])
        h0_k[bl][:, di[0], di[1]] = (
            -2.0 * t_scaled * np.sum(np.cos(2.0 * a * np.pi * mesh), axis=1)[:, None]
        )

    return h0_k


def build_block_mf_matrices(
    gf_struct: GfStruct | None = None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    if gf_struct is None:
        gf_struct = [("up", 1), ("dn", 1)]
    R: dict[str, NDArray[np.float64]] = {}
    Lambda: dict[str, NDArray[np.float64]] = {}
    for bl, bsize in gf_struct:
        R[bl] = np.zeros((bsize, bsize))
        Lambda[bl] = np.zeros((bsize, bsize))
        np.fill_diagonal(R[bl], 1)
    return (R, Lambda)
