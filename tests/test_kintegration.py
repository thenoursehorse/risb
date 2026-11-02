# ruff: noqa: D100, D103
from io import StringIO

import numpy as np
import pytest
from fockit.algebra import zero

from risb import LatticeSolver, helpers
from risb.diagnostics import KIntegrationDiagnostics
from risb.impurity_solvers import FockitImpuritySolver
from risb.kintegration import (
    BandKIntegrator,
    IntegralValues,
    KPoints,
    TemperatureExtrapolatedKIntegrator,
    WeightedKPointIntegrator,
    fermi_function,
    gaussian_function,
    methfessel_paxton_function,
)
from risb.types import BlockMatrix, BlockMatrixLike, SmearFunction

energies = {
    block: np.linspace(-1, 1, 100).reshape(100, 1, 1) for block in ["first", "second"]
}


@pytest.mark.parametrize(
    "smearing", [fermi_function, gaussian_function, methfessel_paxton_function]
)
def test_band_filling_matches_the_target_with_nonuniform_k_weights(
    smearing: SmearFunction,
) -> None:
    kintegrator = BandKIntegrator(
        WeightedKPointIntegrator(weights=np.linspace(1, 2, 100)),
        smearing=smearing,
    )
    result = kintegrator.integrate(energies, beta=10, n_target=1.2)
    assert result.n_electrons == pytest.approx(1.2)
    assert result.quadrature_error is None


def test_band_integration_without_mu_or_n_target_raises() -> None:
    kintegrator = BandKIntegrator(WeightedKPointIntegrator())
    with pytest.raises(ValueError):  # noqa: PT011
        kintegrator.integrate(energies, beta=10)


@pytest.mark.parametrize(
    "smearing", [fermi_function, gaussian_function, methfessel_paxton_function]
)
def test_mu_of_a_flat_band_more_than_half_filled(smearing: SmearFunction) -> None:
    flat = {block: np.zeros((100, 1, 1)) for block in ["first", "second"]}
    result = BandKIntegrator(WeightedKPointIntegrator(), smearing=smearing).integrate(
        flat, beta=10, n_target=1.5
    )
    assert result.n_electrons == pytest.approx(1.5)
    assert result.mu > 0
    if smearing is fermi_function:
        assert result.mu == pytest.approx(np.log(3) / 10)


def test_extrapolation_recovers_zero_temperature_integrals_of_a_complex_multiblock_metal() -> (
    None
):
    kpoints = np.arange(8192)[:, None] / 8192
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)

    def hamiltonian(k: KPoints) -> BlockMatrix:
        diagonal = np.zeros((len(k), 2, 2))
        diagonal[:, 0, 0] = -2 * np.cos(2 * np.pi * k[:, 0])
        diagonal[:, 1, 1] = 4
        return dict.fromkeys(["alpha", "beta"], u @ diagonal @ u.conj().T)

    def get_k_observables(
        k: KPoints | None, density_k: BlockMatrixLike
    ) -> IntegralValues:
        assert k is not None
        h = hamiltonian(k)
        return {
            **{("density", block): density for block, density in density_k.items()},
            **{
                ("kinetic", block): h[block] @ density
                for block, density in density_k.items()
            },
        }

    bands = BandKIntegrator(WeightedKPointIntegrator(kpoints))
    extrapolation = TemperatureExtrapolatedKIntegrator(
        bands,
        betas=[20, 35, 65, 120, 250, 500],
        atol=1e-8,
        rtol=0,
    )
    result = extrapolation.integrate(
        hamiltonian, observables=get_k_observables, beta=np.inf, n_target=1.4
    )
    expected_density = u @ np.diag([0.7, 0]) @ u.conj().T
    expected_ke = u @ np.diag([-2 / np.pi * np.sin(0.7 * np.pi), 0]) @ u.conj().T
    assert result.converged
    assert result.n_electrons == pytest.approx(1.4, abs=1e-10)
    assert result.mu == pytest.approx(-2 * np.cos(0.7 * np.pi), abs=1e-7)
    for block in ["alpha", "beta"]:
        assert result.values["density", block] == pytest.approx(
            expected_density, abs=1e-8
        )
        assert result.values["kinetic", block] == pytest.approx(expected_ke, abs=1e-8)
        for observable in ("density", "kinetic"):
            error = result.temperature_errors[observable, block]
            assert 0 <= error.max_abs <= error.norm <= 1e-8
    assert result.quadrature_error is None

    too_short = TemperatureExtrapolatedKIntegrator(bands, betas=[20], atol=1e-8, rtol=0)
    assert not too_short.integrate(hamiltonian, n_target=1.4).converged


def test_lattice_cycle_accepts_temperature_extrapolation() -> None:
    kpoints = np.arange(8192)[:, None] / 8192
    gf_struct = [("left", 1), ("right", 1)]

    def hamiltonian(k: KPoints) -> BlockMatrix:
        return {
            block: (-2 * np.cos(2 * np.pi * k[:, 0]))[:, None, None]
            for block, _ in gf_struct
        }

    output = StringIO()
    with KIntegrationDiagnostics(level="debug", output=output) as diagnostics:
        kintegrator = TemperatureExtrapolatedKIntegrator(
            BandKIntegrator(
                WeightedKPointIntegrator(kpoints, diagnostics=diagnostics),
                n_target=1,
                diagnostics=diagnostics,
            ),
            betas=[10, 20, 40, 80, 160, 320],
            atol=1e-8,
            rtol=0,
            diagnostics=diagnostics,
        )
        solver = LatticeSolver(
            hamiltonian(kpoints),
            gf_struct,
            FockitImpuritySolver(zero(), gf_struct),
            kintegrator,
            h0_k_evaluator=hamiltonian,
        )
        assert solver.solve(tol=1e-9).converged
    assert not output.closed
    report = output.getvalue()
    for field in (
        "ksum.result",
        "kintegration.mu",
        "kintegration.result",
        "kextrapolation.sample",
        "kextrapolation.result",
        "temperature_error=",
        "kextrapolation.matrix_error",
        "error_norm=",
        "error_max_abs=",
        "quadrature_error=None",
        "betas=",
    ):
        assert field in report
    for block, _ in gf_struct:
        assert solver.Z[0][block] == pytest.approx(np.eye(1), abs=1e-8)
        assert solver.rho_qp[0][block] == pytest.approx(np.array([[0.5]]), abs=1e-8)
        assert solver.lopsided_ke_qp[0][block] == pytest.approx(
            np.array([[-2 / np.pi]]), abs=1e-8
        )
    assert solver.kintegration_result is not None
    density = helpers.get_density_matrix_k(
        R_full=solver.R_full,
        qp_density_k=solver.qp_density_k,
        rho_c=solver.rho_c,
        weights=solver.kintegration_result.weights,
    )
    assert sum(
        np.trace(rho, axis1=1, axis2=2).real.mean() for rho in density.values()
    ) == pytest.approx(1)
