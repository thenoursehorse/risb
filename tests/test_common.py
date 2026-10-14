# ruff: noqa: D100, D103
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fockit.algebra import OpSum, to_triqs
from numpy.typing import NDArray

from risb.impurity_solvers import FockitImpuritySolver
from risb.types import BlockMatrix, GfStruct

if TYPE_CHECKING:
    from risb.impurity_solvers import AtomDiagImpuritySolver

IMPURITY_SOLVERS = ["fockit", "atom_diag"]


def make_impurity_solver(
    name: str, h_int: OpSum, gf_struct: GfStruct
) -> FockitImpuritySolver | AtomDiagImpuritySolver:
    if name == "fockit":
        return FockitImpuritySolver(h_int, gf_struct)
    pytest.importorskip("triqs")
    from risb.impurity_solvers import AtomDiagImpuritySolver  # noqa: PLC0415

    return AtomDiagImpuritySolver(to_triqs(h_int), gf_struct)


def overlap(
    impurity_solver: FockitImpuritySolver | AtomDiagImpuritySolver, Op: OpSum
) -> float | complex:
    if isinstance(impurity_solver, FockitImpuritySolver):
        return impurity_solver.overlap(Op)
    return impurity_solver.overlap(to_triqs(Op))


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


def build_two_orbital_complex_h0_k(nkx: int = 6) -> dict[str, NDArray[np.complex128]]:
    """Cubic dispersion of two orbitals with a crystal field and complex inter-orbital hopping."""
    k_mesh = 2 * np.pi * np.array(list(product(range(nkx), repeat=3))) / nkx
    h0_k = np.zeros([len(k_mesh), 2, 2], dtype=complex)
    h0_k[:] = -(2 / 3) * np.sum(np.cos(k_mesh), axis=1)[:, None, None] * np.eye(2)
    h0_k += np.diag([0.2, -0.2])
    hopping = 0.3 * (np.cos(k_mesh[:, 0]) + 1j * np.sin(k_mesh[:, 1]))
    h0_k[:, 0, 1] += hopping
    h0_k[:, 1, 0] += hopping.conj()
    return {bl: h0_k.copy() for bl in ["up", "dn"]}


def random_unitary(n: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    return Q
