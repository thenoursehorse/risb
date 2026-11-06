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
    TemperatureExtrapolatedBandKIntegrator,
    WeightedKPointIntegrator,
    fermi_function,
    gaussian_function,
    max_occupation_slope,
    methfessel_paxton_function,
)
from risb.types import BlockMatrix, BlockMatrixLike, SmearFunction

eigenvalues = {
    block: np.linspace(-1, 1, 100).reshape(100, 1, 1) for block in ["first", "second"]
}


@pytest.mark.parametrize(
    "smearing", [fermi_function, gaussian_function, methfessel_paxton_function]
)
def test_band_filling_matches_the_target_with_nonuniform_k_weights(
    smearing: SmearFunction,
) -> None:
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(weights=np.linspace(1, 2, 100)),
        smearing=smearing,
    )
    result = band_integrator.integrate(eigenvalues, beta=10, n_target=1.2)
    assert result.converged
    assert result.n_electrons == pytest.approx(1.2)
    assert result.quadrature_error is None


@pytest.mark.parametrize(
    ("smearing", "slope_per_beta"),
    [(fermi_function, 1 / 4), (gaussian_function, 1 / np.sqrt(np.pi))],
)
def test_max_occupation_slope_of_fermi_and_gaussian_smearing(
    smearing: SmearFunction, slope_per_beta: float
) -> None:
    assert max_occupation_slope(smearing, 800.0) == pytest.approx(
        800 * slope_per_beta, rel=1e-6
    )
    assert max_occupation_slope(smearing, np.inf) == np.inf


def test_band_integration_without_mu_or_n_target_raises() -> None:
    band_integrator = BandKIntegrator(WeightedKPointIntegrator())
    with pytest.raises(ValueError):  # noqa: PT011
        band_integrator.integrate(eigenvalues, beta=10)


@pytest.mark.parametrize(
    "smearing", [fermi_function, gaussian_function, methfessel_paxton_function]
)
def test_band_filling_handles_partial_flat_levels_and_gaps(
    smearing: SmearFunction,
) -> None:
    flat = {block: np.zeros((100, 1, 1)) for block in ["first", "second"]}
    band_integrator = BandKIntegrator(WeightedKPointIntegrator(), smearing=smearing)
    for beta in (10, 1e12):
        result = band_integrator.integrate(flat, beta=beta, n_target=1.5)
        assert result.converged
        assert result.n_electrons == pytest.approx(
            1.5, abs=band_integrator.electron_count_atol
        )
        assert result.mu > 0
        if smearing is fermi_function:
            assert beta * result.mu == pytest.approx(np.log(3), abs=1e-10)

    if smearing is fermi_function:
        shifted = {block: h + 1 for block, h in flat.items()}
        cold = band_integrator.integrate(shifted, beta=1e16, n_target=1.5)
        assert cold.converged
        assert cold.n_electrons == pytest.approx(
            1.5, abs=band_integrator.electron_count_atol
        )
        assert cold.mu_reference is not None
        assert cold.density_k is not None
        for block, band_eigenvalues in cold.eigenvalues.items():
            weights = fermi_function(
                band_eigenvalues - cold.mu_reference, cold.beta, cold.mu_correction
            )
            assert weights == pytest.approx(np.full((100, 1), 0.75), abs=1e-10)
            assert cold.density_k[block] == pytest.approx(
                np.full((100, 1, 1), 0.75), abs=1e-10
            )

        extrapolation = TemperatureExtrapolatedBandKIntegrator(
            BandKIntegrator(WeightedKPointIntegrator(), n_target=1.5),
            betas=[1e15, 2e15, 4e15, 8e15],
            atol=1e-8,
            rtol=0,
        )
        zero = extrapolation.integrate(
            shifted,
            observables=lambda _k, density: dict(density.items()),
        )
        assert zero.converged
        assert zero.mu == pytest.approx(1, abs=1e-10)
        assert zero.mu_extrapolation_error is not None
        assert zero.mu_extrapolation_error < 1e-27
        assert zero.n_electrons == pytest.approx(1.5, abs=1e-8)
        assert zero.density_k is not None
        for density in zero.density_k.values():
            assert density == pytest.approx(np.full((100, 1, 1), 0.75), abs=1e-8)
        finite = extrapolation.integrate(shifted, beta=1.6e16)
        assert finite.converged
        assert finite.mu_reference is not None
        assert (finite.mu_reference - 1) + finite.mu_correction == pytest.approx(
            np.log(3) / finite.beta, abs=1e-27
        )

    gapped = {"auxiliary": np.broadcast_to(np.diag([-2.0, 2.0, -0.2, 0.1]), (3, 4, 4))}
    mesh = WeightedKPointIntegrator(weights=[0.2, 0.3, 0.5])
    midpoint = BandKIntegrator(mesh, smearing=smearing, use_midgap=True).integrate(
        gapped, beta=800, n_target=2
    )
    direct = BandKIntegrator(mesh, smearing=smearing, use_midgap=False).integrate(
        gapped, beta=800, n_target=2
    )
    assert midpoint.mu == pytest.approx(-0.05)
    assert midpoint.n_electrons == pytest.approx(2, abs=1e-10)
    assert direct.n_electrons == pytest.approx(2, abs=1e-10)
    assert midpoint.density_k is not None
    assert direct.density_k is not None
    expected = np.broadcast_to(np.diag([1.0, 0.0, 1.0, 0.0]), (3, 4, 4))
    assert midpoint.density_k["auxiliary"] == pytest.approx(expected, abs=1e-10)
    assert direct.density_k["auxiliary"] == pytest.approx(expected, abs=1e-10)


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
    extrapolation = TemperatureExtrapolatedBandKIntegrator(
        bands,
        betas=[20, 35, 65, 120, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
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
    levels = np.sort(-2 * np.cos(2 * np.pi * kpoints[:, 0]))
    expected_mu = levels[int(0.7 * len(levels))]
    assert result.mu_extrapolation_error is not None
    assert np.isfinite(result.mu_extrapolation_error)
    assert abs(result.mu - expected_mu) <= result.mu_extrapolation_error + 1e-10
    for block in ["alpha", "beta"]:
        assert result.values["density", block] == pytest.approx(
            expected_density, abs=1e-8
        )
        assert result.values["kinetic", block] == pytest.approx(expected_ke, abs=1e-8)
        for observable in ("density", "kinetic"):
            error = result.temperature_extrapolation_errors[observable, block]
            assert 0 <= error.max_abs <= error.norm <= 1e-8
    assert result.quadrature_error is None

    too_short = TemperatureExtrapolatedBandKIntegrator(
        bands, betas=extrapolation.betas[-3:], atol=1e-8, rtol=0
    )
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
        band_integrator = TemperatureExtrapolatedBandKIntegrator(
            BandKIntegrator(
                WeightedKPointIntegrator(kpoints, diagnostics=diagnostics),
                n_target=1,
                use_midgap=True,
                diagnostics=diagnostics,
            ),
            betas=[10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480],
            atol=1e-8,
            rtol=0,
            diagnostics=diagnostics,
        )
        solver = LatticeSolver(
            hamiltonian(kpoints),
            gf_struct,
            FockitImpuritySolver(zero(), gf_struct),
            band_integrator,
            h0_k_evaluator=hamiltonian,
        )
        assert solver.solve(tol=1e-9).converged
    assert not output.closed
    report = output.getvalue()
    for field in (
        "ksum.result",
        "kintegration.mu",
        "kintegration.result",
        "use_midgap=True",
        "midgap_available=True",
        "midgap_accepted=True",
        "confirmed=True",
        "kextrapolation.sample",
        "kextrapolation.result",
        "temperature_extrapolation_error=",
        "mu_extrapolation_error=",
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
        levels = -2 * np.cos(2 * np.pi * kpoints[:, 0])
        expected_ke = np.minimum(levels, 0).mean()
        assert solver.lopsided_ke_qp[0][block] == pytest.approx(
            np.array([[expected_ke]]), abs=1e-8
        )
    assert solver.band_integration_result is not None
    density = helpers.get_density_matrix_k(
        R_full=solver.R_full,
        density_qp_k=solver.density_qp_k,
        rho_c=solver.rho_c,
        weights=solver.band_integration_result.weights,
    )
    assert sum(
        np.trace(rho, axis1=1, axis2=2).real.mean() for rho in density.values()
    ) == pytest.approx(1)
