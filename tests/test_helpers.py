# ruff: noqa: D100, D103

from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pytest
from fockit.algebra import nc
from numpy.typing import NDArray
from test_common import (
    build_two_orbital_complex_h0_k,
    one_band_cubic_h0_k,
    random_unitary,
)

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


def assert_lagrangian_is_stationary(
    S: LatticeSolver,
    impurity_solvers: list[FockitImpuritySolver],
    kweight: SmearingKWeight,
) -> None:
    assert kweight.mu is not None
    mu = kweight.mu
    variables = {
        "R": S.R,
        "Lambda": S.Lambda,
        "D": S.D,
        "Lambda_c": S.Lambda_c,
        "Delta_p": S.rho_qp,
    }
    h_loc_energy = [
        float(np.real(s.overlap(s.h_int + s.h0_loc))) for s in impurity_solvers
    ]

    def lagrangian(name: str, i: int, bl: str, E: NDArray[np.inexact]) -> float:
        shifted = [dict(cluster) for cluster in variables[name]]
        shifted[i][bl] = shifted[i][bl] + E
        v = {**variables, name: shifted}
        return helpers.get_lagrangian(
            h0_kin_k=S.h0_kin_k,
            R=v["R"],
            Lambda=v["Lambda"],
            D=v["D"],
            Lambda_c=v["Lambda_c"],
            Delta_p=v["Delta_p"],
            rho_f=S.rho_f,
            rho_cf=S.rho_cf,
            h_loc_energy=h_loc_energy,
            beta=kweight.beta,
            mu=mu,
            projectors=S.projectors,
            gf_struct_mapping=S.gf_struct_mapping,
            impurity_entropy=[s.entropy for s in impurity_solvers],
        )

    # R and D are general matrices, Lambda, Lambda_c and Delta_p Hermitian
    step = 1e-6
    for name in variables:
        for i in range(S.n_clusters):
            for bl, n in S.gf_struct[i]:
                for a, b in product(range(n), repeat=2):
                    for phase in [1, 1j]:
                        E = np.zeros((n, n), dtype=complex)
                        E[a, b] = phase
                        if name not in ("R", "D"):
                            E = E + E.conj().T
                        derivative = (
                            lagrangian(name, i, bl, step * E)
                            - lagrangian(name, i, bl, -step * E)
                        ) / (2 * step)
                        assert derivative == pytest.approx(0, abs=1e-7), (
                            name,
                            i,
                            bl,
                            a,
                            b,
                        )


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
    assert_lagrangian_is_stationary(S, [impurity_solver], kweight)


def test_converged_solution_with_a_thermal_impurity_is_a_stationary_point_of_the_lagrangian() -> (
    None
):
    # At this temperature the excited states of the impurity problem change
    # the solution
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
    beta = 20
    kweight = SmearingKWeight(beta=beta, n_target=2)
    impurity_solver = FockitImpuritySolver(h_int, gf_struct, beta=beta)
    S = LatticeSolver(
        h0_k=build_two_orbital_complex_h0_k(),
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        update_weights=kweight.update_weights,
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = 1.5 * np.eye(2)
    S.solve(tol=1e-8)
    assert S.optimize.success
    assert impurity_solver.entropy > 0.01
    assert_lagrangian_is_stationary(S, [impurity_solver], kweight)


def test_converged_solution_with_several_clusters_is_a_stationary_point_of_the_lagrangian() -> (
    None
):
    # One band on a cell of two sites, with a cluster on each site. The
    # hopping along y and z is larger on site 0, so the sites are inequivalent
    h0_k = one_band_cubic_h0_k(6, supercell=True)
    h0_k[:, 0, 0] *= 1.3
    spin_names = ["up", "dn"]
    gf_struct = [[(bl, 1) for bl in spin_names]] * 2
    impurity_solvers = [
        FockitImpuritySolver(4 * nc("up", 0) * nc("dn", 0), gf) for gf in gf_struct
    ]
    kweight = SmearingKWeight(beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=dict.fromkeys(spin_names, h0_k),
        gf_struct=gf_struct,
        impurity_solvers=impurity_solvers,
        update_weights=kweight.update_weights,
        projectors=[
            dict.fromkeys(spin_names, np.array([[1.0, 0.0]])),
            dict.fromkeys(spin_names, np.array([[0.0, 1.0]])),
        ],
    )
    for i in range(2):
        for bl in spin_names:
            S.Lambda[i][bl] = np.array([[2.0]])
    S.solve(tol=1e-8)
    assert S.optimize.success
    assert_lagrangian_is_stationary(S, impurity_solvers, kweight)


def test_completely_full_orbital_is_decoupled_and_the_rest_is_unchanged() -> None:
    rho = np.diag([1.0, 0.3])
    # No correlation between the physical orbitals and the full orbital a = 0
    rho_cf = np.array([[0.0, 0.2], [0.0, 0.1]])
    ke = np.array([[0.4, 0.3], [0.1, 0.2]])
    R = helpers.get_r(rho_cf, rho)
    D = helpers.get_d(rho, ke)
    Lambda_c = helpers.get_lambda_c(rho, R, np.diag([0.5, -0.2]), D)
    Lambda = helpers.get_lambda(R, D, Lambda_c, rho)
    for A in [R, D, Lambda_c, Lambda]:
        assert np.all(np.isfinite(A))
    assert R[0] == pytest.approx(np.zeros(2), abs=1e-12)
    assert D[0] == pytest.approx(np.zeros(2), abs=1e-12)
    assert R[1] == pytest.approx(rho_cf[:, 1] / np.sqrt(0.3 * 0.7), abs=1e-12)
    assert D[1] == pytest.approx(ke[:, 1] / np.sqrt(0.3 * 0.7), abs=1e-12)


def test_converged_solution_with_an_orbital_in_no_cluster_is_a_stationary_point_of_the_lagrangian() -> (
    None
):
    # One band on a cell of two sites, with a cluster on site 0 only
    spin_names = ["up", "dn"]
    gf_struct = [[(bl, 1) for bl in spin_names]]
    impurity_solvers = [
        FockitImpuritySolver(4 * nc("up", 0) * nc("dn", 0), gf_struct[0])
    ]
    kweight = SmearingKWeight(beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=dict.fromkeys(spin_names, one_band_cubic_h0_k(6, supercell=True)),
        gf_struct=gf_struct,
        impurity_solvers=impurity_solvers,
        update_weights=kweight.update_weights,
        projectors=[dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))],
    )
    for bl in spin_names:
        S.Lambda[0][bl] = np.array([[2.0]])
    S.solve(tol=1e-8)
    assert S.optimize.success
    assert_lagrangian_is_stationary(S, impurity_solvers, kweight)


@pytest.mark.parametrize(
    "n_clusters", [2, 1], ids=["every site", "a site in no cluster"]
)
def test_entropy_is_minus_the_temperature_derivative_of_the_lagrangian(
    n_clusters: int,
) -> None:
    # One band on a cell of two sites, a cluster on each site or on site 0
    # only, with thermal impurity problems. At fixed chemical potential the
    # Lagrangian at a solution is the grand potential
    spin_names = ["up", "dn"]
    U, mu = 2.0, 1.0
    h0_k = dict.fromkeys(spin_names, one_band_cubic_h0_k(4, supercell=True))
    sites = [
        dict.fromkeys(spin_names, np.array([[1.0, 0.0]])),
        dict.fromkeys(spin_names, np.array([[0.0, 1.0]])),
    ]

    def solve(beta: float) -> tuple[float, float]:
        gf_struct = [[(bl, 1) for bl in spin_names]] * n_clusters
        impurity_solvers = [
            FockitImpuritySolver(U * nc("up", 0) * nc("dn", 0), gf, beta=beta)
            for gf in gf_struct
        ]
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=impurity_solvers,
            update_weights=SmearingKWeight(beta=beta, mu=mu).update_weights,
            projectors=sites[:n_clusters],
        )
        for i in range(n_clusters):
            for bl in spin_names:
                S.Lambda[i][bl] = np.array([[mu]])
        S.solve(tol=1e-12)
        assert S.optimize.success
        impurity_entropy = [s.entropy for s in impurity_solvers]
        lagrangian = helpers.get_lagrangian(
            h0_kin_k=S.h0_kin_k,
            R=S.R,
            Lambda=S.Lambda,
            D=S.D,
            Lambda_c=S.Lambda_c,
            Delta_p=S.rho_qp,
            rho_f=S.rho_f,
            rho_cf=S.rho_cf,
            h_loc_energy=[
                float(np.real(s.overlap(s.h_int + s.h0_loc))) for s in impurity_solvers
            ],
            beta=beta,
            mu=mu,
            projectors=S.projectors,
            gf_struct_mapping=S.gf_struct_mapping,
            impurity_entropy=impurity_entropy,
        )
        entropy = helpers.get_entropy(
            energies=S.energies_qp, beta=beta, mu=mu, impurity_entropy=impurity_entropy
        )
        return lagrangian, entropy

    T = 0.1
    dT = 1e-3 * T
    lagrangian_minus, _ = solve(1 / (T - dT))
    _, entropy = solve(1 / T)
    lagrangian_plus, _ = solve(1 / (T + dT))
    assert entropy == pytest.approx(
        -(lagrangian_plus - lagrangian_minus) / (2 * dT), abs=1e-5
    )
