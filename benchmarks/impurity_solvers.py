# ruff: noqa: D103
"""
Time FockitImpuritySolver.

Run with ``uv run python benchmarks/impurity_solvers.py``.
"""

import logging
import time
from collections.abc import Callable
from itertools import product

import numpy as np
from fockit.algebra import OpSum, nc, zero
from fockit.hilbert import to_matrix
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from risb import LatticeSolver
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix

logger = logging.getLogger(__name__)

SPINS = ["up", "dn"]
# Metallic at half filling on the cubic lattice, as in the lattice tests
U = 3.0
J = 0.2 * U
U_PRIME = U - 2 * J


def hubbard(n_orb: int) -> OpSum:
    h_int = zero()
    for o in range(n_orb):
        h_int += U * nc("up", o) * nc("dn", o)
    return h_int


def kanamori(n_orb: int) -> OpSum:
    off_diag = 1 - np.eye(n_orb)
    return h_int_kanamori(
        SPINS,
        n_orb,
        U=(U_PRIME - J) * off_diag,
        Uprime=U * np.eye(n_orb) + U_PRIME * off_diag,
        J_hund=J,
    )


def impurity_couplings(
    n_orb: int, is_complex: bool
) -> tuple[BlockMatrix, BlockMatrix, BlockMatrix]:
    rng = np.random.default_rng(0)

    def random(scale: float) -> NDArray[np.inexact]:
        A = rng.normal(size=(n_orb, n_orb))
        if is_complex:
            return scale * (A + 1j * rng.normal(size=(n_orb, n_orb)))
        return scale * A

    A = random(0.2)
    h0 = A + A.conj().T
    B = random(0.1).real
    Lambda_c = -0.5 * U * np.eye(n_orb) + B + B.T
    D = -0.3 * np.eye(n_orb) + random(0.05)
    return (
        dict.fromkeys(SPINS, Lambda_c),
        dict.fromkeys(SPINS, D),
        dict.fromkeys(SPINS, h0),
    )


def best_time(f: Callable[[], object]) -> float:
    times: list[float] = []
    while len(times) < 3 and sum(times) < 1:
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return min(times)


def time_impurity_solve(name: str, h_int: OpSum, n_orb: int, is_complex: bool) -> None:
    gf_struct = [(bl, n_orb) for bl in SPINS]
    Lambda_c, D, h0 = impurity_couplings(n_orb, is_complex)

    start = time.perf_counter()
    fockit = FockitImpuritySolver(h_int, gf_struct)
    t_basis = time.perf_counter() - start
    t_h_imp = best_time(lambda: fockit.set_h_imp(Lambda_c, D, h0))
    H = to_matrix(fockit.h_imp, fockit.basis)
    t_matrix = best_time(lambda: to_matrix(fockit.h_imp, fockit.basis))
    t_eigsh = best_time(lambda: eigsh(H, k=1, which="SA"))
    t_solve = best_time(fockit.solve)
    row = (
        f"{name:9s} {n_orb} {'complex' if is_complex else 'real':7s} "
        f"{fockit.basis.dim:7d} | {t_h_imp + t_solve:9.4f} {t_basis:8.4f} "
        f"{t_h_imp:8.4f} {t_matrix:8.4f} {t_eigsh:8.4f} "
        f"{t_solve - t_matrix - t_eigsh:8.4f}"
    )
    logger.info(row)


def cubic_h0_k(n_orb: int, V: float = 0, nkx: int = 10) -> BlockMatrix:
    mesh = np.array(list(product(range(nkx), repeat=3))) / nkx
    eps = (2 / 3) * np.sum(np.cos(2 * np.pi * mesh), axis=1)
    h0_k = eps[:, None, None] * np.eye(n_orb) + V * (1 - np.eye(n_orb))
    return dict.fromkeys(SPINS, h0_k)


def paramagnetic(A: list[BlockMatrix]) -> list[BlockMatrix]:
    for A_i in A:
        A_i["up"] = A_i["dn"] = 0.5 * (A_i["up"] + A_i["dn"])
    return A


def time_lattice_solve(name: str, h_int: OpSum, n_orb: int, V: float = 0) -> None:
    gf_struct = [(bl, n_orb) for bl in SPINS]
    S = LatticeSolver(
        h0_k=cubic_h0_k(n_orb, V),
        gf_struct=gf_struct,
        impurity_solvers=[FockitImpuritySolver(h_int, gf_struct)],
        update_weights=SmearingKWeight(beta=40, n_target=n_orb).update_weights,
        symmetrizers=[paramagnetic],
    )
    start = time.perf_counter()
    S.solve(tol=1e-8)
    elapsed = time.perf_counter() - start
    logger.info(
        f"{name:12s} {elapsed:8.2f} s, "
        f"{S.optimize.n:4d} iterations, success {S.optimize.success}, "
        f"Z {S.Z[0]['up'][0, 0].real:.8f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("risb").setLevel(logging.WARNING)
    logger.info(
        "model     n type        dim |  fockit:  total    basis    h_imp   matrix"
        "    eigsh      rho"
    )
    for n_orb in range(1, 6):
        time_impurity_solve("hubbard", hubbard(n_orb), n_orb, is_complex=False)
        time_impurity_solve("kanamori", kanamori(n_orb), n_orb, is_complex=False)
        time_impurity_solve("kanamori", kanamori(n_orb), n_orb, is_complex=True)

    logger.info("")
    time_lattice_solve("hubbard 1", hubbard(1), 1)
    time_lattice_solve("bilayer 2", hubbard(2), 2, V=0.25)
    time_lattice_solve("kanamori 2", kanamori(2), 2)
    time_lattice_solve("kanamori 3", kanamori(3), 3)
