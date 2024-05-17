# ruff: noqa: T201, D100, D103
from itertools import product

import numpy as np


def symmetrize_blocks(A: list[dict[np.ndarray]]):
    n_clusters = len(A)
    A_sym = [0 for i in range(n_clusters)]
    for i in range(n_clusters):
        for bl in A[i]:
            A_sym[i] += A[i][bl] / len(A[i])
        for bl in A[i]:
            A[i][bl] = A_sym[i]
    return A


def build_cubic_h0_k(gf_struct=None, nkx=6, spatial_dim=2, t=1, a=1):
    if gf_struct is None:
        gf_struct = [("up", 1), ("dn", 1)]
    for _, bsize in gf_struct:
        n_orb = bsize
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
    h0_k = {}
    for bl, n_orb in gf_struct:
        di = np.diag_indices(n_orb)
        h0_k[bl] = np.zeros([n_k, n_orb, n_orb])
        h0_k[bl][:, di[0], di[1]] = (
            -2.0 * t_scaled * np.sum(np.cos(2.0 * a * np.pi * mesh), axis=1)[:, None]
        )

    return h0_k


def build_block_mf_matrices(gf_struct=None):
    if gf_struct is None:
        gf_struct = [("up", 1), ("dn", 1)]
    R = {}
    Lambda = {}
    for bl, bsize in gf_struct:
        R[bl] = np.zeros((bsize, bsize))
        Lambda[bl] = np.zeros((bsize, bsize))
        np.fill_diagonal(R[bl], 1)
    return (R, Lambda)
