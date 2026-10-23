# ruff: noqa: D100, D103
from importlib.util import find_spec
from itertools import product
from typing import Any

import numpy as np
import pytest
from fockit.algebra import OpSum
from numpy.typing import NDArray

from risb.impurity_solvers import FockitImpuritySolver
from risb.types import BlockMatrix, GfStruct

# The ground-state impurity solvers, block2 with each kind of symmetry
_needs_block2 = pytest.mark.skipif(find_spec("pyblock2") is None, reason="needs block2")
SOLVERS = [
    "fockit",
    pytest.param("block2", marks=_needs_block2),
    pytest.param("block2 sz", marks=_needs_block2),
    pytest.param("block2 s", marks=_needs_block2),
]

# How close each solver gets to the exact ground state
TOLERANCE = {"fockit": 1e-12, "block2": 1e-7}


def make_impurity_solver(
    name: str, h_int: OpSum, gf_struct: GfStruct, **kwargs: Any
) -> Any:
    if name == "fockit":
        return FockitImpuritySolver(h_int, gf_struct, **kwargs)
    from risb.impurity_solvers import Block2ImpuritySolver  # noqa: PLC0415

    all_symmetries: dict[str, dict[str, float | None]] = {
        "block2": {},
        "block2 sz": {"sz": 0},
        "block2 s": {"s": None},
    }
    symmetries = all_symmetries[name]
    return Block2ImpuritySolver(
        h_int, gf_struct, bond_dim=200, symmetries=symmetries, **kwargs
    )


def tolerance(name: str) -> float:
    return TOLERANCE[name.split(maxsplit=1)[0]]


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


def one_band_cubic_h0_k(nkx: int, supercell: bool) -> NDArray[np.complex128]:
    """One band on the cubic lattice, or the same in a cell of two sites along x."""
    t = -1.0 / 3.0
    mesh = 2 * np.pi * np.array(list(product(range(nkx), repeat=3))) / nkx
    if not supercell:
        return (2 * t * np.sum(np.cos(mesh), axis=1))[:, None, None].astype(complex)
    # The two k of the one-site cell folded onto each k of the two-site cell
    mesh = mesh[mesh[:, 0] < np.pi]
    h0_k = np.zeros((len(mesh), 2, 2), dtype=complex)
    for n, k in enumerate(mesh):
        folded = [k, k + np.array([np.pi, 0, 0])]
        energies = [2 * t * np.sum(np.cos(q)) for q in folded]
        bloch = np.array(
            [[np.exp(1j * q[0] * site) for q in folded] for site in range(2)]
        ) / np.sqrt(2)
        h0_k[n] = bloch @ np.diag(energies) @ bloch.conj().T
    return h0_k


def bethe_h0_k(n_k: int) -> NDArray[np.float64]:
    """One band with the semicircular density of states of half bandwidth 1.

    The energies are at equal steps of its cumulative distribution, so equal
    weights at each k give the density of states.
    """
    energies = np.linspace(-1, 1, 100001)
    cdf = 0.5 + (energies * np.sqrt(1 - energies**2) + np.arcsin(energies)) / np.pi
    steps = (np.arange(n_k) + 0.5) / n_k
    return np.interp(steps, cdf, energies)[:, None, None]
