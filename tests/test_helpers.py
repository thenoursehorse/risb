# ruff: noqa: D100, D103

from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from test_common import build_two_orbital_complex_h0_k, random_unitary

from risb import LatticeSolver, helpers
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight

# FIXME finish helpers: lambda, r, f1, f2

abs = 1e-12
helpers_filename = Path(__file__).parent / Path("data_helpers.h5")


def test_get_h_qp(subtests: pytest.Subtests) -> None:
    with h5py.File(helpers_filename, "r") as f:
        h0_k = f["h0_k"][:]
        eig_expected = f["eig"][:]
        vec_expected = f["vec"][:]
    R = np.zeros(shape=(2, 2))
    np.fill_diagonal(R, 1)
    Lambda = np.zeros(shape=(2, 2))
    np.fill_diagonal(Lambda, 0.5)
    h_qp = helpers.get_h_qp(R, Lambda, h0_k)
    eig, vec = np.linalg.eigh(h_qp)
    with subtests.test(msg="eigenvalues"):
        assert eig == pytest.approx(eig_expected, abs=abs)
    with subtests.test(msg="eigenvectors"):
        assert vec == pytest.approx(vec_expected, abs=abs)


def test_get_h0_kin_k_R() -> None:
    R = np.zeros(shape=(2, 2))
    np.fill_diagonal(R, 1)
    with h5py.File(helpers_filename, "r") as f:
        h0_k = f["h0_k"][:]
        vec = f["vec"][:]
        h0_k_R_expected = f["h0_R"][:]
    h0_k_R = helpers.get_h0_kin_k_R(R, h0_k, vec)
    assert h0_k_R == pytest.approx(h0_k_R_expected, abs=abs)


def test_get_ke() -> None:
    with h5py.File(helpers_filename, "r") as f:
        h0_k_R = f["h0_R"][:]
        vec = f["vec"][:]
        wks = f["wks"][:]
    ke = helpers.get_ke(h0_k_R, vec, wks)
    ke_expected = np.array([[-0.36035732126514364, 0], [0, -0.36035732126514364]])
    assert ke == pytest.approx(ke_expected, abs=abs)


def test_get_rho_qp() -> None:
    with h5py.File(helpers_filename, "r") as f:
        vec = f["vec"][:]
        wks = f["wks"][:]
    rho_qp = helpers.get_rho_qp(vec, wks)
    rho_qp_expected = np.array([[0.30667105643085796, 0], [0, 0.30667105643085796]])
    assert rho_qp == pytest.approx(rho_qp_expected, abs=abs)


def test_get_d() -> None:
    rho_qp = np.array([[0.19618454, 0.0], [0.0, 0.19618454]])
    ke = np.array([[-0.13447044, 0.0], [0.0, -0.13447044]])
    D = helpers.get_d(rho_qp, ke)
    D_expected = np.array([[-0.33862284815908383, 0.0], [0.0, -0.33862284815908383]])
    assert pytest.approx(D_expected, abs=abs) == D


def test_get_lambda_c() -> None:
    Lambda = np.array([[0.5, 0.0], [0.0, 0.5]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_qp = np.array([[0.19618454, 0.0], [0.0, 0.19618454]])
    D = np.array([[-0.33862285, 0.0], [0.0, -0.33862285]])
    Lambda_c = helpers.get_lambda_c(rho_qp, R, Lambda, D)
    Lambda_c_expected = np.array(
        [[0.018138135818154377, 0.0], [0.0, 0.018138135818154377]]
    )
    assert Lambda_c == pytest.approx(Lambda_c_expected, abs=abs)


def test_get_lambda() -> None:
    # R = np.zeros(shape=(2,2))
    # np.fill_diagonal(R, 1.0)
    # Lambda = helpers.get_lambda(R, D, Lambda_c, rho_f)
    pass


def test_get_R() -> None:
    # R = helpers.get_r(rho_cf, rho_f)
    pass


def test_get_f1() -> None:
    # R = np.zeros(shape=(2,2))
    # np.fill_diagonal(R, 1.0)
    # f1 = helpers.get_f1(rho_cf, rho_qp, R)
    pass


def test_get_f2() -> None:
    # f2 = helpers.get_f2(rho_f, rho_qp)
    pass


def test_get_h0_kin_k_removes_local_terms_with_complex_projector() -> None:
    rng = np.random.default_rng(0)
    h0_k = {"up": rng.random((4, 2, 2))}
    P = np.array([[1j, 0]])
    h0_kin_k = helpers.get_h0_kin_k(h0_k, [{"up": P}])
    h0_loc = (P @ h0_k["up"] @ P.conj().T).mean(axis=0)
    assert h0_kin_k["up"] == pytest.approx(h0_k["up"] - P.conj().T @ h0_loc @ P)


def test_converged_solution_is_a_stationary_point_of_the_lagrangian() -> None:
    U, J = 3.0, 0.6
    Up = U - 2 * J
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * (1 - np.eye(2)),
        Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
        J_hund=J,
    )
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    kweight = SmearingKWeight(beta=40, n_target=2)
    impurity_solver = FockitImpuritySolver(h_int, gf_struct)
    S = LatticeSolver(
        h0_k=build_two_orbital_complex_h0_k(),
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        update_weights=kweight.update_weights,
    )
    # A quasiparticle basis rotated away from R = 1
    u = random_unitary(2, seed=1)
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = 1.5 * np.eye(2)
        S.R[0][bl] = u.conj().T
    S.solve(tol=1e-8)
    assert S.optimize.success
    assert kweight.mu is not None

    variables = {
        "R": S.R[0],
        "Lambda": S.Lambda[0],
        "D": S.D[0],
        "Lambda_c": S.Lambda_c[0],
        "Delta_p": S.rho_qp[0],
    }
    h_loc_energy = float(
        np.real(impurity_solver.overlap(impurity_solver.h_int + impurity_solver.h0_loc))
    )
    mu = kweight.mu

    def lagrangian(name: str, bl: str, E: NDArray[np.inexact]) -> float:
        v = {**variables, name: {**variables[name], bl: variables[name][bl] + E}}
        return helpers.get_lagrangian(
            S.h0_kin_k,
            v["R"],
            v["Lambda"],
            v["D"],
            v["Lambda_c"],
            v["Delta_p"],
            S.rho_f[0],
            S.rho_cf[0],
            h_loc_energy,
            beta=40,
            mu=mu,
        )

    # R and D are general matrices, Lambda, Lambda_c and Delta_p Hermitian
    step = 1e-6
    for name in variables:
        for bl, _ in gf_struct:
            for i, j in product(range(2), repeat=2):
                for phase in [1, 1j]:
                    E = np.zeros((2, 2), dtype=complex)
                    E[i, j] = phase
                    if name not in ("R", "D"):
                        E = E + E.conj().T
                    derivative = (
                        lagrangian(name, bl, step * E) - lagrangian(name, bl, -step * E)
                    ) / (2 * step)
                    assert derivative == pytest.approx(0, abs=1e-7), (name, bl, i, j)
