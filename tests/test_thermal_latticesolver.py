# ruff: noqa: D100, D103

import numpy as np
import pytest
from fockit.algebra import OpSum, nc
from test_common import (
    EXACT_THERMAL,
    bethe_h0_k,
    build_two_orbital_complex_h0_k,
    one_band_cubic_h0_k,
    random_unitary,
    symmetrize_blocks,
)

from risb import LatticeSolver, ThermalLatticeSolver, helpers
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver, FockitThermalImpuritySolver
from risb.kweight import SmearingKWeight
from risb.optimize import LinearMixing, ScipyRoot
from risb.types import BlockMatrixLike, GfStruct

ONE_BAND: GfStruct = [("up", 1), ("dn", 1)]


def hubbard(U: float) -> OpSum:
    return U * nc("up", 0) * nc("dn", 0) - U / 2 * (nc("up", 0) + nc("dn", 0))


def bethe_lattice(
    h_int: OpSum,
    B: int,
    T: float,
    *,
    mu: float | None = 0.0,
    n_target: float | None = None,
) -> tuple[ThermalLatticeSolver, FockitThermalImpuritySolver]:
    bath_size = {bl: B for bl, _ in ONE_BAND}
    impurity_solver = FockitThermalImpuritySolver(
        h_int, ONE_BAND, bath_size=bath_size, beta=1 / T, **EXACT_THERMAL
    )
    S = ThermalLatticeSolver(
        h0_k={bl: bethe_h0_k(400) for bl, _ in ONE_BAND},
        gf_struct=ONE_BAND,
        impurity_solvers=impurity_solver,
        beta=1 / T,
        mu=mu,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=[bath_size],
        n_target=n_target,
    )
    return S, impurity_solver


def two_site_cell(
    T: float, mu: float = 1.0
) -> tuple[ThermalLatticeSolver, FockitThermalImpuritySolver]:
    # One band on a cell of two sites, with two ghosts per orbital on site 0
    # and site 1 in no cluster
    spin_names = ["up", "dn"]
    bath_size = dict.fromkeys(spin_names, 3)
    impurity_solver = FockitThermalImpuritySolver(
        2.0 * nc("up", 0) * nc("dn", 0),
        ONE_BAND,
        bath_size=bath_size,
        beta=1 / T,
        **EXACT_THERMAL,
    )
    S = ThermalLatticeSolver(
        h0_k=dict.fromkeys(spin_names, one_band_cubic_h0_k(4, supercell=True)),
        gf_struct=ONE_BAND,
        impurity_solvers=impurity_solver,
        beta=1 / T,
        mu=mu,
        projectors=[dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))],
        bath_size=[bath_size],
    )
    return S, impurity_solver


def grand_potential(
    S: ThermalLatticeSolver, impurity_solver: FockitThermalImpuritySolver
) -> float:
    return helpers.get_thermal_grand_potential(
        energies=S.energies_qp,
        R=S.R,
        Lambda=S.Lambda,
        D=S.D,
        Lambda_c=S.Lambda_c,
        beta=S.beta,
        mu=S.mu,
        impurity_grand_potential=[impurity_solver.grand_potential],
    )


def entropy(
    S: ThermalLatticeSolver, impurity_solver: FockitThermalImpuritySolver
) -> float:
    return helpers.get_thermal_entropy(
        energies=S.energies_qp,
        R=S.R,
        Lambda=S.Lambda,
        D=S.D,
        Lambda_c=S.Lambda_c,
        beta=S.beta,
        mu=S.mu,
        impurity_entropy=[impurity_solver.entropy],
    )


@pytest.mark.parametrize("model", ["complex_two_orbitals", "one_band_ghosts"])
@pytest.mark.parametrize("fixed", ["mu", "filling"])
def test_low_temperature_solution_is_the_zero_temperature_solution(
    model: str, fixed: str
) -> None:
    gf_struct: GfStruct
    h0_k: BlockMatrixLike
    if model == "complex_two_orbitals":
        U, J = 3.0, 0.6
        Up = U - 2 * J
        h = h_int_kanamori(
            ["up", "dn"],
            2,
            U=(Up - J) * (1 - np.eye(2)),
            Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
            J_hund=J,
        )
        gf_struct, h0_k, B = [("up", 2), ("dn", 2)], build_two_orbital_complex_h0_k(), 1
    else:
        h = hubbard(2.0)
        gf_struct, h0_k, B = ONE_BAND, {bl: bethe_h0_k(400) for bl, _ in ONE_BAND}, 3
    beta = 400
    bath_size = {bl: B * size for bl, size in gf_struct}
    # Half filled. The quasiparticle bands of LatticeSolver also hold the
    # ghost electrons
    n_target = sum(size for _, size in gf_struct) / 2
    kweight = SmearingKWeight(beta=beta, n_target=n_target * B)
    S0 = LatticeSolver(
        h0_k,
        gf_struct,
        FockitImpuritySolver(h, gf_struct, bath_size=bath_size),
        kweight.update_weights,
        symmetrizers=[symmetrize_blocks],
        bath_size=[bath_size],
    )
    S0.solve(tol=1e-11)
    assert kweight.mu is not None
    S = ThermalLatticeSolver(
        h0_k,
        gf_struct,
        FockitThermalImpuritySolver(
            h, gf_struct, bath_size=bath_size, beta=beta, **EXACT_THERMAL
        ),
        beta=beta,
        mu=kweight.mu if fixed == "mu" else None,
        symmetrizers=[symmetrize_blocks],
        bath_size=[bath_size],
        n_target=n_target if fixed == "filling" else None,
    )
    if model == "complex_two_orbitals":
        # A complex R, in another quasiparticle basis
        for bl, _ in gf_struct:
            S.R[0][bl] = random_unitary(2, seed=1).conj().T
    S.solve(tol=1e-10)
    assert S.self_consistency_error < 1e-8
    assert S.mu == pytest.approx(kweight.mu, abs=1e-8)
    for bl, _ in gf_struct:
        assert S.Z[0][bl] == pytest.approx(S0.Z[0][bl], abs=1e-8)
        assert S.rho_c[0][bl] == pytest.approx(S0.rho_c[0][bl], abs=1e-8)


def test_thermodynamics_follow_the_finite_temperature_ghost_paper() -> None:
    # Fig. 7 of arXiv:2603.20559 at U = 2: d falls from T = 0 to a minimum
    # near T = 0.2 (about 0.052), then rises to about 0.19 at T = 2. The
    # energy (E - mu N) / N is about -1.08 at low T and -0.68 at T = 2, and
    # the entropy approaches ln 4 at high T
    docc, energy, S_T = {}, {}, {}
    for T in [0.005, 0.2, 2.0, 10.0]:
        S, impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
        S.solve(tol=1e-10)
        assert S.self_consistency_error < 1e-6
        docc[T] = float(np.real(impurity_solver.overlap(nc("up", 0) * nc("dn", 0))))
        S_T[T] = entropy(S, impurity_solver)
        energy[T] = grand_potential(S, impurity_solver) + T * S_T[T]
    assert docc[0.2] < docc[0.005]
    assert docc[0.2] == pytest.approx(0.052, abs=3e-3)
    assert docc[2.0] == pytest.approx(0.19, abs=3e-3)
    assert energy[0.005] == pytest.approx(-1.08, abs=0.01)
    assert energy[2.0] == pytest.approx(-0.68, abs=0.01)
    assert S_T[10.0] == pytest.approx(np.log(4), abs=0.01)


@pytest.mark.parametrize("model", ["one_band_ghosts", "site_in_no_cluster"])
def test_entropy_is_minus_the_temperature_derivative_of_the_grand_potential(
    model: str,
) -> None:
    T, h = 0.1, 1e-4

    def solve(
        T: float, start: ThermalLatticeSolver | None = None
    ) -> tuple[ThermalLatticeSolver, FockitThermalImpuritySolver]:
        if model == "one_band_ghosts":
            S, impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
        else:
            S, impurity_solver = two_site_cell(T)
        if start is not None:
            S.R = [{bl: A.copy() for bl, A in R_i.items()} for R_i in start.R]
            S.Lambda = [{bl: A.copy() for bl, A in L_i.items()} for L_i in start.Lambda]
        S.solve(tol=1e-13)
        assert S.self_consistency_error < 1e-11
        return S, impurity_solver

    S, impurity_solver = solve(T)
    derivative = (
        grand_potential(*solve(T + h, S)) - grand_potential(*solve(T - h, S))
    ) / (2 * h)
    assert entropy(S, impurity_solver) == pytest.approx(-derivative, abs=1e-6)


def test_grand_potential_at_low_temperature_is_the_zero_temperature_lagrangian() -> (
    None
):
    # Half filled at mu = 1, so the quasiparticle bands of the ghosts are
    # below mu
    beta, mu = 400, 1.0
    h_int = hubbard(2.0) + mu * (nc("up", 0) + nc("dn", 0))
    S, impurity_solver = bethe_lattice(h_int, 3, 1 / beta, mu=mu)
    S.solve(tol=1e-12)
    bath_size = {bl: 3 for bl, _ in ONE_BAND}
    impurity_solver_0 = FockitImpuritySolver(
        impurity_solver.h_int, ONE_BAND, bath_size=bath_size
    )
    S0 = LatticeSolver(
        h0_k={bl: bethe_h0_k(400) for bl, _ in ONE_BAND},
        gf_struct=ONE_BAND,
        impurity_solvers=impurity_solver_0,
        update_weights=SmearingKWeight(beta=beta, mu=mu).update_weights,
        symmetrizers=[symmetrize_blocks],
        bath_size=[bath_size],
    )
    for bl, _ in ONE_BAND:
        S0.Lambda[0][bl] = S0.Lambda[0][bl] + mu * np.eye(3)
    S0.solve(tol=1e-12)
    lagrangian = helpers.get_lagrangian(
        h0_kin_k=S0.h0_kin_k,
        R=S0.R,
        Lambda=S0.Lambda,
        D=S0.D,
        Lambda_c=S0.Lambda_c,
        Delta_p=S0.rho_qp,
        rho_f=S0.rho_f,
        rho_cf=S0.rho_cf,
        h_loc_energy=[
            float(np.real(impurity_solver_0.overlap(impurity_solver_0.h_int)))
        ],
        beta=beta,
        mu=mu,
    )
    assert grand_potential(S, impurity_solver) == pytest.approx(lagrangian, abs=1e-9)


def test_four_ghosts_per_orbital_converge_to_the_two_ghost_solution() -> None:
    U, T = 2.4, 0.005
    S3, impurity_solver3 = bethe_lattice(hubbard(U), 3, T)
    S3.solve(tol=1e-10)
    S5, impurity_solver5 = bethe_lattice(hubbard(U), 5, T)
    # Two more ghosts with a little of the weight, at levels outside those
    # of the other ghosts
    for bl, _ in ONE_BAND:
        S5.R[0][bl] = np.zeros((5, 1))
        S5.R[0][bl][:3] = np.sqrt(0.98) * S3.R[0][bl]
        S5.R[0][bl][3:] = np.sqrt(0.01)
        S5.Lambda[0][bl] = np.diag([0, 0, 0, -1.5, 1.5])
        S5.Lambda[0][bl][:3, :3] = S3.Lambda[0][bl]
    S5.solve(tol=1e-10)
    assert S5.self_consistency_error < 1e-8
    docc = nc("up", 0) * nc("dn", 0)
    assert impurity_solver5.overlap(docc) == pytest.approx(
        impurity_solver3.overlap(docc), abs=1e-3
    )


def test_scipy_root_converges_near_a_solution() -> None:
    T = 0.1
    S, impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
    S.solve(tol=1e-10)
    S_root, impurity_solver_root = bethe_lattice(hubbard(2.0), 3, T)
    for bl, _ in ONE_BAND:
        S_root.R[0][bl] = 0.99 * S.R[0][bl]
        S_root.Lambda[0][bl] = 1.01 * S.Lambda[0][bl]
    optimizer = ScipyRoot(error_fun="root", method="krylov", options={"fatol": 1e-10})
    result = S_root.solve(optimizer=optimizer)
    assert result.converged
    assert S_root.get_x() == pytest.approx(result.x, abs=1e-12)
    docc = nc("up", 0) * nc("dn", 0)
    assert impurity_solver_root.overlap(docc) == pytest.approx(
        impurity_solver.overlap(docc), abs=1e-8
    )
    assert S_root.Z[0]["up"] == pytest.approx(S.Z[0]["up"], abs=1e-8)


@pytest.mark.parametrize("optimizer", ["DIIS", "LinearMixing"])
def test_fixed_filling_is_the_fixed_mu_solution_at_the_mu_it_finds(
    optimizer: str,
) -> None:
    T, n_target = 0.1, 0.9
    S, _ = bethe_lattice(hubbard(2.0), 3, T, mu=None, n_target=n_target)
    if optimizer == "DIIS":
        S.solve(tol=1e-10)
    else:
        S.solve(optimizer=LinearMixing(error_fun="root"), tol=1e-10)
    assert S.n_electrons == pytest.approx(n_target, abs=1e-8)
    assert S.mu < 0
    S_mu, _ = bethe_lattice(hubbard(2.0), 3, T, mu=S.mu)
    S_mu.solve(tol=1e-11)
    assert S_mu.n_electrons == pytest.approx(n_target, abs=1e-8)
    assert S_mu.Z[0]["up"] == pytest.approx(S.Z[0]["up"], abs=1e-7)
    assert S_mu.rho_c[0]["up"] == pytest.approx(S.rho_c[0]["up"], abs=1e-7)


def test_electrons_are_minus_the_mu_derivative_of_the_grand_potential() -> None:
    # With site 1 in no cluster, its quasiparticle band holds electrons too
    T, mu, h = 0.1, 1.0, 1e-4

    def omega(mu: float) -> float:
        S, impurity_solver = two_site_cell(T, mu)
        S.solve(tol=1e-13)
        return grand_potential(S, impurity_solver)

    S, _ = two_site_cell(T, mu)
    S.solve(tol=1e-13)
    derivative = (omega(mu + h) - omega(mu - h)) / (2 * h)
    assert S.n_electrons == pytest.approx(-derivative, abs=1e-6)
    assert S.n_electrons > np.trace(S.rho_c[0]["up"] + S.rho_c[0]["dn"]).real + 1


@pytest.mark.parametrize(
    ("model", "T"),
    [("one_band_ghosts", 1.0), ("site_in_no_cluster", 1.0), ("one_band_ghosts", 10.0)],
)
def test_default_solve_converges_at_high_temperature(model: str, T: float) -> None:
    # At high T the equations barely fix some combinations of the ghost
    # parameters, and plain DIIS takes hundreds of cycles or does not converge
    if model == "one_band_ghosts":
        S, impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
    else:
        S, impurity_solver = two_site_cell(T)
    result = S.solve(tol=1e-10)
    assert result.converged
    assert result.cycles < 100
    assert S.self_consistency_error < 1e-7
    if (model, T) == ("one_band_ghosts", 1.0):
        docc = impurity_solver.overlap(nc("up", 0) * nc("dn", 0))
        assert docc == pytest.approx(0.141837, abs=1e-6)


def test_a_cycle_leaves_the_solver_at_its_input() -> None:
    S, _ = bethe_lattice(hubbard(2.0), 3, 1.0)
    x = S.get_x()
    x_new, _ = S.cycle(x)
    assert np.linalg.norm(x_new - x) > 1e-3
    assert S.get_x() == pytest.approx(x, abs=1e-12)


def test_an_anchor_does_not_carry_over_to_the_next_solve() -> None:
    T = 0.1
    S, impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
    S.anchor(alpha=1e6)
    S.solve(tol=1e-10, callbacks=[])
    fresh, fresh_impurity_solver = bethe_lattice(hubbard(2.0), 3, T)
    fresh.solve(tol=1e-10, callbacks=[])
    docc = nc("up", 0) * nc("dn", 0)
    assert impurity_solver.overlap(docc) == pytest.approx(
        fresh_impurity_solver.overlap(docc), abs=1e-8
    )
    assert S.Z[0]["up"] == pytest.approx(fresh.Z[0]["up"], abs=1e-8)
