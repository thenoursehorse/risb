# ruff: noqa: D100, D103
from copy import deepcopy

import numpy as np
import pytest
from fockit.algebra import nc, zero
from test_common import bethe_h0_k, build_cubic_h0_k, symmetrize_blocks

from risb import LatticeSolver, PhysicalMuLatticeSolver
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator

SPINS = ["up", "dn"]


def hubbard(
    U: float, B: int, n_target: float | None = None, mu: float | None = None
) -> PhysicalMuLatticeSolver:
    gf_struct = [(bl, 1) for bl in SPINS]
    bath_size = dict.fromkeys(SPINS, B)
    return PhysicalMuLatticeSolver(
        h0_k=dict.fromkeys(SPINS, bethe_h0_k(400)),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(
            U * nc("up", 0) * nc("dn", 0), gf_struct, bath_size=bath_size
        ),
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=100),
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=[bath_size],
        mu=mu,
        n_target=n_target,
    )


def band_mu_hubbard(U: float, B: int, n_target: float) -> LatticeSolver:
    gf_struct = [(bl, 1) for bl in SPINS]
    bath_size = dict.fromkeys(SPINS, B)
    band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=100)
    S = LatticeSolver(
        h0_k=dict.fromkeys(SPINS, bethe_h0_k(400)),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(
            U * nc("up", 0) * nc("dn", 0), gf_struct, bath_size=bath_size
        ),
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=[bath_size],
    )
    band_integrator.n_target = n_target + S.n_ghost_electrons
    return S


@pytest.mark.parametrize("B", [1, 3])
def test_half_filled_hubbard_model_has_mu_at_half_the_interaction(B: int) -> None:
    # The model is particle-hole symmetric, and is a metal at this U
    U = 2.0
    S = hubbard(U, B, n_target=1.0)
    assert S.solve(tol=1e-10).converged
    assert S.n_electrons == pytest.approx(1.0, abs=1e-8)
    assert S.mu == pytest.approx(U / 2, abs=1e-6)

    expected = band_mu_hubbard(U, B, n_target=1.0)
    expected.solve(tol=1e-10)
    for bl in SPINS:
        assert np.linalg.eigvalsh(S.Z[0][bl]) == pytest.approx(
            np.linalg.eigvalsh(expected.Z[0][bl]), abs=1e-6
        )


@pytest.mark.parametrize("B", [1, 3])
def test_a_mu_far_above_the_bands_returns_to_the_half_filled_solution(B: int) -> None:
    U = 2.0
    S = hubbard(U, B, n_target=1.0)
    x0 = S.get_x()
    x0[-1] += 5.0
    assert S.solve(tol=1e-9, x0=x0).converged
    assert S.mu == pytest.approx(U / 2, abs=1e-6)
    assert S.n_electrons == pytest.approx(1.0, abs=1e-8)


def test_fixed_mu_gives_the_filling_of_the_solution_at_that_mu() -> None:
    U = 2.0
    S = hubbard(U, 1, n_target=0.8)
    assert S.solve(tol=1e-10).converged

    fixed = hubbard(U, 1, mu=S.mu)
    assert fixed.solve(tol=1e-10).converged
    assert fixed.n_electrons == pytest.approx(0.8, abs=1e-6)
    for bl in SPINS:
        assert fixed.Z[0][bl] == pytest.approx(S.Z[0][bl], abs=1e-6)


def test_uncorrelated_ligand_shares_the_mu_of_the_metal() -> None:
    # One metal orbital with three quasiparticles, hybridized with an
    # uncorrelated ligand orbital. The solution is that of LatticeSolver at the
    # same filling, with the chemical potential of its quasiparticle bands
    U, n_target = 6.0, 3.0
    e = bethe_h0_k(2000)
    h0_k = np.block([[0.5 * e, 0.5 + 0 * e], [0.5 + 0 * e, (U / 2 - 0.5) + e]])
    gf_struct = [(bl, 1) for bl in SPINS]
    bath_size = dict.fromkeys(SPINS, 3)
    sz = nc("up", 0) - nc("dn", 0)
    sz += sum((nc("bath_up", a) - nc("bath_dn", a) for a in range(3)), zero())

    def common() -> dict[str, object]:
        return {
            "h0_k": dict.fromkeys(SPINS, h0_k),
            "gf_struct": gf_struct,
            "impurity_solvers": FockitImpuritySolver(
                U * nc("up", 0) * nc("dn", 0),
                gf_struct,
                diagonal_symmetries={sz: 0},
                bath_size=bath_size,
            ),
            "projectors": [dict.fromkeys(SPINS, np.eye(2)[:1])],
            "symmetrizers": [symmetrize_blocks],
            "force_real": True,
            "bath_size": [bath_size],
        }

    S = PhysicalMuLatticeSolver(
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=100),
        n_target=n_target,
        **common(),  # type: ignore[arg-type]
    )
    assert S.solve(tol=1e-9, maxiter=300).converged
    assert S.n_electrons == pytest.approx(n_target, abs=1e-8)

    band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=100)
    expected = LatticeSolver(band_integrator=band_integrator, **common())  # type: ignore[arg-type]
    band_integrator.n_target = n_target + expected.n_ghost_electrons
    expected.R = deepcopy(S.R)
    expected.Lambda = deepcopy(S.Lambda)
    result = expected.solve(tol=1e-9)
    assert result.converged
    assert result.cycles <= 2
    assert band_integrator.mu == pytest.approx(S.mu, abs=1e-8)
    for bl in SPINS:
        assert np.trace(S.rho_c[0][bl]).real == pytest.approx(
            np.trace(expected.rho_c[0][bl]).real, abs=1e-6
        )
        assert np.linalg.eigvalsh(S.Z[0][bl]) == pytest.approx(
            np.linalg.eigvalsh(expected.Z[0][bl]), abs=1e-6
        )


def test_physical_mu_finds_the_neel_antiferromagnet() -> None:
    # Half filling of a bipartite lattice with a cluster on each sublattice,
    # from a start with opposite spin splittings on the two sublattices
    U, B = 4.0, 3
    e = bethe_h0_k(400)
    h0_k = np.block([[0 * e, e], [e, 0 * e]])
    gf_struct = [(bl, 1) for bl in SPINS]
    bath_size = dict.fromkeys(SPINS, B)
    S = PhysicalMuLatticeSolver(
        h0_k=dict.fromkeys(SPINS, h0_k),
        gf_struct=[gf_struct] * 2,
        impurity_solvers=[
            FockitImpuritySolver(
                U * nc("up", 0) * nc("dn", 0), gf_struct, bath_size=bath_size
            )
            for _ in range(2)
        ],
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=100),
        projectors=[
            dict.fromkeys(SPINS, np.eye(2)[:1]),
            dict.fromkeys(SPINS, np.eye(2)[1:]),
        ],
        force_real=True,
        bath_size=[bath_size] * 2,
        n_target=2.0,
    )
    for i, sign in [(0, 1), (1, -1)]:
        for bl, split in [("up", -0.1), ("dn", 0.1)]:
            S.Lambda[i][bl] = S.Lambda[i][bl] + sign * split * np.eye(B)
    assert S.solve(tol=1e-9).converged
    assert S.n_electrons == pytest.approx(2.0, abs=1e-8)
    m = [np.trace(S.rho_c[i]["up"] - S.rho_c[i]["dn"]).real for i in range(2)]
    assert m[0] > 0.9
    assert m[1] == pytest.approx(-m[0], abs=1e-6)


def test_three_orbital_kanamori_is_a_paramagnetic_mott_insulator() -> None:
    n_orb = 3
    U = 3
    J = 0.2 * U
    Up = U - 2 * J
    gf_struct = [(bl, n_orb) for bl in SPINS]
    off_diag = 1 - np.eye(n_orb)
    h_int = h_int_kanamori(
        SPINS,
        n_orb,
        U=(Up - J) * off_diag,
        Uprime=U * np.eye(n_orb) + Up * off_diag,
        J_hund=J,
    )
    S = PhysicalMuLatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct, nkx=10, spatial_dim=3),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=40),
        symmetrizers=[symmetrize_blocks],
        n_target=n_orb,
    )
    assert S.solve(tol=1e-8).converged
    for bl in S.Z[0]:
        assert S.Z[0][bl] == pytest.approx(np.zeros((n_orb, n_orb)), abs=1e-6)
    assert S.rho_c[0]["up"] == pytest.approx(S.rho_c[0]["dn"], abs=1e-10)


def test_decoupled_ghosts_of_a_metal_on_ligands_converge() -> None:
    # Two metal orbitals with three quasiparticles each, hybridized with two
    # uncorrelated ligand orbitals. Two ghosts per spin decouple with
    # occupation 1/2
    U, J = 3.0, 0.5
    e = bethe_h0_k(2000)
    eye = np.eye(2)
    metal = np.diag([0.5, -0.5]) + 0.5 * e * eye
    ligand = ((3 * U - 5 * J) / 2 - 1.5) * eye + e * eye
    hybridization = np.broadcast_to(0.5 * eye, metal.shape)
    h0_k = np.block([[metal, hybridization], [hybridization, ligand]])
    gf_struct = [(bl, 2) for bl in SPINS]
    bath_size = dict.fromkeys(SPINS, 6)
    h_int = h_int_kanamori(
        SPINS,
        2,
        U=(U - 3 * J) * (1 - eye),
        Uprime=U * eye + (U - 2 * J) * (1 - eye),
        J_hund=0,
    )
    sz = sum((nc("up", a) - nc("dn", a) for a in range(2)), zero())
    sz += sum((nc("bath_up", a) - nc("bath_dn", a) for a in range(6)), zero())
    S = PhysicalMuLatticeSolver(
        h0_k=dict.fromkeys(SPINS, h0_k),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(
            h_int, gf_struct, diagonal_symmetries={sz: 0}, bath_size=bath_size
        ),
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=100),
        projectors=[dict.fromkeys(SPINS, np.eye(4)[:2])],
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=[bath_size],
        n_target=6.0,
    )
    assert S.solve(tol=1e-9, maxiter=100).converged
    assert S.self_consistency_error < 1e-8
    assert S.n_electrons == pytest.approx(6.0, abs=1e-8)
    for bl in SPINS:
        rho_f = np.linalg.eigvalsh(S.rho_f[0][bl])
        assert rho_f[2:4] == pytest.approx([0.5, 0.5], abs=1e-8)
