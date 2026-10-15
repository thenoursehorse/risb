# ruff: noqa: D100, D103

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
from risb.types import BlockMatrix

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

    # R and D are general matrices, Lambda, Lambda_c and Delta_p Hermitian.
    # Ghost orbitals can be nearly full or empty, where sqrt(Delta_p (1 -
    # Delta_p)) is steep, so the step is small
    step = 1e-7
    for name in variables:
        for i in range(S.n_clusters):
            for bl, _ in S.gf_struct[i]:
                shape = variables[name][i][bl].shape
                for a, b in np.ndindex(shape):
                    for phase in [1, 1j]:
                        E = np.zeros(shape, dtype=complex)
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


def test_converged_solution_with_ghosts_is_a_stationary_point_of_the_lagrangian() -> (
    None
):
    # One band on a cell of two sites, with three quasiparticles per spin on
    # site 0 and site 1 in no cluster
    spin_names = ["up", "dn"]
    gf_struct = [[(bl, 1) for bl in spin_names]]
    bath_size = [dict.fromkeys(spin_names, 3)]
    impurity_solvers = [
        FockitImpuritySolver(
            4 * nc("up", 0) * nc("dn", 0), gf_struct[0], bath_size=bath_size[0]
        )
    ]
    kweight = SmearingKWeight(beta=40, n_target=2 + 2)
    S = LatticeSolver(
        h0_k=dict.fromkeys(spin_names, one_band_cubic_h0_k(6, supercell=True)),
        gf_struct=gf_struct,
        impurity_solvers=impurity_solvers,
        update_weights=kweight.update_weights,
        projectors=[dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))],
        bath_size=bath_size,
    )
    S.solve(tol=1e-8)
    assert S.optimize.success
    assert S.self_consistency_error < 1e-8
    assert_lagrangian_is_stationary(S, impurity_solvers, kweight)


@pytest.mark.parametrize(
    ("n_clusters", "n_qp", "T"),
    # With ghosts the cycle does not converge in this model above T ~ 0.03
    [(2, 1, 0.1), (1, 1, 0.1), (1, 3, 0.025)],
    ids=["every site", "a site in no cluster", "ghosts on a site"],
)
def test_entropy_is_minus_the_temperature_derivative_of_the_lagrangian(
    n_clusters: int, n_qp: int, T: float
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
        bath_size = [dict.fromkeys(spin_names, n_qp)] * n_clusters
        impurity_solvers = [
            FockitImpuritySolver(
                U * nc("up", 0) * nc("dn", 0), gf, beta=beta, bath_size=bath_size[0]
            )
            for gf in gf_struct
        ]
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=impurity_solvers,
            update_weights=SmearingKWeight(beta=beta, mu=mu).update_weights,
            projectors=sites[:n_clusters],
            bath_size=bath_size,
        )
        for i in range(n_clusters):
            for bl in spin_names:
                S.Lambda[i][bl] = S.Lambda[i][bl] + mu * np.eye(n_qp)
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

    dT = 1e-3 * T
    lagrangian_minus, _ = solve(1 / (T - dT))
    _, entropy = solve(1 / T)
    lagrangian_plus, _ = solve(1 / (T + dT))
    assert entropy == pytest.approx(
        -(lagrangian_plus - lagrangian_minus) / (2 * dT), abs=1e-5
    )


def solved_lattice(
    model: str,
) -> tuple[LatticeSolver, list[FockitImpuritySolver], SmearingKWeight, int]:
    spin_names = ["up", "dn"]
    if model == "complex":
        U, J = 3.0, 0.6
        Up = U - 2 * J
        h_int = h_int_kanamori(
            spin_names,
            2,
            U=(Up - J) * (1 - np.eye(2)),
            Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
            J_hund=J,
        )
        gf_struct = [[(bl, 2) for bl in spin_names]]
        impurity_solvers = [FockitImpuritySolver(h_int, gf_struct[0])]
        h0_k = build_two_orbital_complex_h0_k()
        projectors = None
        n_target, start = 2, 1.5 * np.eye(2)
        bath_size = None
        n_ghost_electrons = 0
    else:
        # One band on a cell of two sites, with a cluster on site 0 only. The
        # cluster has three quasiparticles per spin in the model "ghosts"
        n_qp = 3 if model == "ghosts" else 1
        gf_struct = [[(bl, 1) for bl in spin_names]]
        bath_size = [dict.fromkeys(spin_names, n_qp)]
        impurity_solvers = [
            FockitImpuritySolver(
                2 * nc("up", 0) * nc("dn", 0), gf_struct[0], bath_size=bath_size[0]
            )
        ]
        h0_k = dict.fromkeys(spin_names, one_band_cubic_h0_k(4, supercell=True))
        projectors = [dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))]
        n_target, start = 2, np.eye(n_qp)
        n_ghost_electrons = n_qp - 1
    kweight = SmearingKWeight(beta=40, n_target=n_target + n_ghost_electrons)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solvers,
        update_weights=kweight.update_weights,
        projectors=projectors,
        bath_size=bath_size,
    )
    for bl in spin_names:
        S.Lambda[0][bl] = S.Lambda[0][bl] + start
    S.solve(tol=1e-12)
    assert S.optimize.success
    return S, impurity_solvers, kweight, n_target


@pytest.mark.parametrize("model", ["complex", "a site in no cluster", "ghosts"])
def test_density_matrix_k_has_the_local_density_of_each_impurity(model: str) -> None:
    S, _, _, n_target = solved_lattice(model)
    density = helpers.get_density_matrix_k(
        R_full=S.R_full,
        vectors=S.bloch_vector_qp,
        kweights=S.kweights,
        rho_c=S.rho_c,
        projectors=S.projectors,
    )
    assert sum(np.trace(rho.mean(axis=0)).real for rho in density.values()) == (
        pytest.approx(n_target, abs=1e-10)
    )
    for bl, rho in density.items():
        assert rho == pytest.approx(np.swapaxes(rho.conj(), -1, -2), abs=1e-12)
        P = np.eye(rho.shape[-1]) if S.projectors is None else S.projectors[0][bl]
        local = np.mean(P @ rho @ np.swapaxes(P.conj(), -1, -2), axis=0)
        # <c^dag_b c_a>, the transpose of rho_c
        assert local == pytest.approx(S.rho_c[0][bl].T, abs=1e-10)


@pytest.mark.parametrize("model", ["complex", "a site in no cluster", "ghosts"])
def test_band_and_interaction_energy_is_the_energy_of_risb(model: str) -> None:
    S, impurity_solvers, kweight, n_target = solved_lattice(model)
    assert kweight.mu is not None
    density = helpers.get_density_matrix_k(
        R_full=S.R_full,
        vectors=S.bloch_vector_qp,
        kweights=S.kweights,
        rho_c=S.rho_c,
        projectors=S.projectors,
    )
    energy = helpers.get_band_energy(S.h0_k, density) + sum(
        float(np.real(s.overlap(s.h_int))) for s in impurity_solvers
    )
    # The grand potential plus mu N plus T S
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
        beta=kweight.beta,
        mu=kweight.mu,
        projectors=S.projectors,
    )
    entropy = helpers.get_entropy(
        energies=S.energies_qp,
        beta=kweight.beta,
        mu=kweight.mu,
        impurity_entropy=[s.entropy for s in impurity_solvers],
    )
    assert energy == pytest.approx(
        lagrangian + kweight.mu * n_target + entropy / kweight.beta, abs=1e-8
    )


def test_ghost_projectors_without_ghosts_are_the_projectors() -> None:
    h0_k = {"up": np.zeros((4, 3, 3))}
    U_k = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 3, 3)) + 0j)[0]
    for projectors in [
        [{"up": np.array([[1.0, 0, 0]])}, {"up": np.array([[0, 1.0, 0]])}],
        [{"up": U_k[:, :2, :]}],
    ]:
        bath_size = [{"up": P["up"].shape[-2]} for P in projectors]
        projectors_qp = helpers.get_ghost_projectors(h0_k, bath_size, projectors)
        for P, P_qp in zip(projectors, projectors_qp, strict=True):
            assert P_qp["up"] == pytest.approx(P["up"], abs=abs)


@pytest.mark.parametrize("k_dependent", [False, True])
def test_upfolded_ghost_quasiparticle_hamiltonian_has_the_ghost_risb_form(
    k_dependent: bool,
) -> None:
    # A two-site cell with site 0 in a cluster of 3 quasiparticles and site 1 in
    # no cluster. In the quasiparticle space (site 0, site 1, ghost 1, ghost 2)
    # H_qp is T eps T^dag + Lambda on (site 0, ghost 1, ghost 2), with T the
    # rows of R on those orbitals and the identity on site 1
    rng = np.random.default_rng(2)
    n_k = 5
    eps = rng.normal(size=(n_k, 2, 2)) + 1j * rng.normal(size=(n_k, 2, 2))
    eps = eps + np.swapaxes(eps.conj(), -1, -2)
    R = rng.normal(size=(3, 1)) + 1j * rng.normal(size=(3, 1))
    Lambda = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Lambda = Lambda + Lambda.conj().T
    cluster = [0, 2, 3]
    T = np.zeros((4, 2), complex)
    T[cluster, 0] = R[:, 0]
    T[1, 1] = 1
    expected = T @ eps @ T.conj().T
    expected[:, *np.ix_(cluster, cluster)] += Lambda
    # The same Hamiltonian in a basis of the cell that rotates with k
    U = np.eye(2)
    if k_dependent:
        U = np.linalg.qr(
            rng.normal(size=(n_k, 2, 2)) + 1j * rng.normal(size=(n_k, 2, 2))
        )[0]
    U_dag = np.swapaxes(U.conj(), -1, -2)
    h0_k = {"up": U @ eps @ U_dag}
    projectors = [{"up": np.array([[1.0, 0.0]]) @ U_dag}]
    projectors_qp = helpers.get_ghost_projectors(h0_k, [{"up": 3}], projectors)
    R_padded = np.hstack([R, np.zeros((3, 2))])
    R_full = helpers.upfold([{"up": R_padded}], h0_k, projectors_qp, outside=1)["up"]
    Lambda_full = helpers.upfold([{"up": Lambda}], h0_k, projectors_qp)["up"]
    assert R_full[..., 2:] == pytest.approx(np.zeros_like(R_full[..., 2:]), abs=abs)
    h_qp = helpers.get_h_qp(R_full[..., :2], Lambda_full, h0_k["up"])
    assert np.linalg.eigvalsh(h_qp) == pytest.approx(
        np.linalg.eigvalsh(expected), abs=abs
    )


def test_self_energy_with_ghosts_gives_the_green_function_at_every_k() -> None:
    rng = np.random.default_rng(3)
    n_orb, n_qp = 2, 6
    R = rng.normal(size=(n_qp, n_orb)) + 1j * rng.normal(size=(n_qp, n_orb))
    Lambda = rng.normal(size=(n_qp, n_qp)) + 1j * rng.normal(size=(n_qp, n_qp))
    Lambda = Lambda + Lambda.conj().T
    h0_kin_k = rng.normal(size=(4, n_orb, n_orb)) + 1j * rng.normal(
        size=(4, n_orb, n_orb)
    )
    h0_kin_k = h0_kin_k + np.swapaxes(h0_kin_k.conj(), -1, -2)
    h0_loc = np.diag([0.2, -0.1])
    mu = 0.3
    omega = np.array([0.4 + 0.1j, -1.3 + 0.05j, 2.5j])
    sigma = helpers.get_sigma_w(omega, R, Lambda, mu, h0_loc)
    z = (omega + mu)[:, None, None, None]
    G = np.linalg.inv(z * np.eye(n_orb) - h0_kin_k - h0_loc - sigma[:, None])
    G_qp = np.linalg.inv(z * np.eye(n_qp) - R @ h0_kin_k @ R.conj().T - Lambda)
    assert pytest.approx(R.conj().T @ G_qp @ R, abs=abs) == G


def test_self_energy_without_ghosts_is_linear_in_frequency() -> None:
    rng = np.random.default_rng(4)
    R = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Lambda = np.diag([0.3, -0.2])
    h0_loc = np.diag([0.1, 0.0])
    mu = 0.2
    # Includes w + mu at the eigenvalues of Lambda
    omega = np.array([0.1, -0.4, 1.7, 0.5j])
    Z_inv = np.linalg.inv(R.conj().T @ R)
    R_inv = np.linalg.inv(R)
    expected = (
        (np.eye(2) - Z_inv) * (omega + mu)[:, None, None]
        + R_inv @ Lambda @ R_inv.conj().T
        - h0_loc
    )
    assert helpers.get_sigma_w(omega, R, Lambda, mu, h0_loc) == pytest.approx(
        expected, abs=abs
    )


@pytest.mark.parametrize(("n_orb", "n_qp"), [(2, 2), (2, 6)])
def test_quasiparticle_weight_is_set_by_the_slope_of_the_self_energy(
    n_orb: int, n_qp: int
) -> None:
    rng = np.random.default_rng(5)
    R = rng.normal(size=(n_qp, n_orb)) + 1j * rng.normal(size=(n_qp, n_orb))
    Lambda = rng.normal(size=(n_qp, n_qp)) + 1j * rng.normal(size=(n_qp, n_qp))
    Lambda = Lambda + Lambda.conj().T
    mu, h = 0.3, 1e-5
    slope = (
        helpers.get_sigma_w(h, R, Lambda, mu) - helpers.get_sigma_w(-h, R, Lambda, mu)
    ) / (2 * h)
    Z = helpers.get_z(R, Lambda, mu)
    assert pytest.approx(np.linalg.inv(np.eye(n_orb) - slope), abs=1e-8) == Z


def test_green_functions_do_not_depend_on_the_quasiparticle_basis() -> None:
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
    h0_k = build_two_orbital_complex_h0_k(nkx=6)
    kweight = SmearingKWeight(beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        update_weights=kweight.update_weights,
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = 1.5 * np.eye(2)
    S.solve(tol=1e-10)
    assert kweight.mu is not None
    mu = kweight.mu
    # Matsubara frequencies
    iw = 1j * (2 * np.arange(32) + 1) * np.pi / 40

    def green_functions(
        R: BlockMatrix, Lambda: BlockMatrix
    ) -> tuple[BlockMatrix, BlockMatrix, BlockMatrix]:
        sigma = {
            bl: helpers.get_sigma_w(iw, R[bl], Lambda[bl], mu, S.h0_loc_matrix[0][bl])
            for bl, _ in gf_struct
        }
        g0 = helpers.get_g0_k_w(iw, h0_k, mu)
        g_dyson = {
            bl: np.linalg.inv(np.linalg.inv(g0[bl]) - sigma[bl][None])
            for bl, _ in gf_struct
        }
        g_qp = helpers.get_g_k_w(iw, R, Lambda, S.h0_kin_k, mu)
        return (
            sigma,
            helpers.get_g_w_loc(g_dyson)[0],
            helpers.get_g_w_loc(g_qp)[0],
        )

    u = random_unitary(2, seed=4)
    sigma, g_dyson, g_qp = green_functions(S.R[0], S.Lambda[0])
    sigma_rotated, g_dyson_rotated, g_qp_rotated = green_functions(
        {bl: u.conj().T @ S.R[0][bl] for bl, _ in gf_struct},
        {bl: u.conj().T @ S.Lambda[0][bl] @ u for bl, _ in gf_struct},
    )
    for bl, _ in gf_struct:
        assert sigma_rotated[bl] == pytest.approx(sigma[bl], abs=1e-10)
        assert g_dyson[bl] == pytest.approx(g_qp[bl], abs=1e-10)
        assert g_dyson_rotated[bl] == pytest.approx(g_qp_rotated[bl], abs=1e-10)


def test_green_function_with_ghosts_and_projectors_is_dyson_with_the_upfolded_self_energy() -> (
    None
):
    # A two-orbital and a one-orbital cluster, both with three quasiparticles
    # per orbital, in a basis that rotates with k, and an orbital in no cluster
    rng = np.random.default_rng(7)
    n_k, n_band = 5, 4
    h0 = rng.normal(size=(n_k, n_band, n_band)) + 1j * rng.normal(
        size=(n_k, n_band, n_band)
    )
    h0_k = {"up": h0 + np.swapaxes(h0.conj(), -1, -2)}
    U = np.linalg.qr(
        rng.normal(size=(n_k, n_band, n_band))
        + 1j * rng.normal(size=(n_k, n_band, n_band))
    )[0]
    projectors = [{"up": U[:, 0:2, :]}, {"up": U[:, 2:3, :]}]
    R, Lambda, h0_loc = [], [], []
    for P in projectors:
        n_orb = P["up"].shape[-2]
        n_qp = 3 * n_orb
        R.append({"up": rng.normal(size=(n_qp, n_orb)) + 0.5j})
        L = rng.normal(size=(n_qp, n_qp)) + 1j * rng.normal(size=(n_qp, n_qp))
        Lambda.append({"up": L + L.conj().T})
        h0_loc.append(helpers.get_h0_loc_matrix(h0_k["up"], P["up"]))
    h0_kin_k = helpers.get_h0_kin_k(h0_k, projectors)
    projectors_qp = helpers.get_ghost_projectors(
        h0_kin_k, [{"up": R_i["up"].shape[0]} for R_i in R], projectors
    )
    R_full, Lambda_full = helpers.upfold_r_lambda(R, Lambda, h0_kin_k, projectors_qp)
    mu = 0.2
    omega = np.array([0.3 + 0.15j, -1.0 + 0.05j, 2.0j])
    g = helpers.get_g_k_w(omega, R_full, Lambda_full, h0_kin_k, mu)["up"]
    g0 = helpers.get_g0_k_w(omega, h0_k, mu)["up"]
    for w, z in enumerate(omega):
        sigma = [
            {"up": helpers.get_sigma_w(z, R[i]["up"], Lambda[i]["up"], mu, h0_loc[i])}
            for i in range(2)
        ]
        sigma_k = helpers.upfold(sigma, h0_k, projectors)["up"]
        g_dyson = np.linalg.inv(np.linalg.inv(g0[:, w]) - sigma_k)
        assert g[:, w] == pytest.approx(g_dyson, abs=abs)


def test_local_green_function_does_not_depend_on_a_basis_that_rotates_with_k() -> None:
    # A two-site cell with a cluster on site 0, three quasiparticles per
    # orbital, and site 1 in no cluster
    rng = np.random.default_rng(8)
    n_k = 5
    eps = rng.normal(size=(n_k, 2, 2)) + 1j * rng.normal(size=(n_k, 2, 2))
    eps = eps + np.swapaxes(eps.conj(), -1, -2)
    R = [{"up": rng.normal(size=(3, 1)) + 0.3j}]
    L = rng.normal(size=(3, 3))
    Lambda = [{"up": L + L.T}]
    omega = np.array([0.4 + 0.1j, 1.5j])
    U = np.linalg.qr(rng.normal(size=(n_k, 2, 2)) + 1j * rng.normal(size=(n_k, 2, 2)))[
        0
    ]
    g_w_loc = []
    for basis in [np.eye(2), U]:
        basis_dag = np.swapaxes(basis.conj(), -1, -2)
        h0_k = {"up": basis @ eps @ basis_dag}
        projectors = [{"up": np.array([[1.0, 0.0]]) @ basis_dag}]
        h0_kin_k = helpers.get_h0_kin_k(h0_k, projectors)
        projectors_qp = helpers.get_ghost_projectors(h0_kin_k, [{"up": 3}], projectors)
        R_full, Lambda_full = helpers.upfold_r_lambda(
            R, Lambda, h0_kin_k, projectors_qp
        )
        g = helpers.get_g_k_w(omega, R_full, Lambda_full, h0_kin_k, mu=0.1)
        g_w_loc.append(helpers.get_g_w_loc(g, projectors)[0]["up"])
    assert g_w_loc[1] == pytest.approx(g_w_loc[0], abs=abs)
