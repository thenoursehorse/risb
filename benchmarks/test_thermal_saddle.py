"""Physical checks for finite-temperature nested and coupled saddle solves."""

import numpy as np
import pytest
from thermal_saddle import ThermalSaddle, run_method


@pytest.mark.parametrize("temperature", [0.2, 1.0])
def test_thermal_potential_gradient_and_gibbs_response(temperature: float) -> None:
    """Analytic derivatives match displaced interacting thermal functionals."""
    problem = ThermalSaddle(temperature, nk=80)
    generator = np.random.default_rng(2)
    point = np.r_[problem.reference_x, problem.reference_y]
    point += 0.01 * np.maximum(np.abs(point), 0.2) * generator.normal(size=12)
    sample = problem.joint(point[:6], point[6:])
    hessian = problem.hessian(sample)
    direction = generator.normal(size=12)
    direction /= np.linalg.norm(direction)
    step = 2e-5
    plus = problem.joint(
        point[:6] + step * direction[:6], point[6:] + step * direction[6:]
    )
    minus = problem.joint(
        point[:6] - step * direction[:6], point[6:] - step * direction[6:]
    )
    assert (plus.value - minus.value) / (2 * step) == pytest.approx(
        sample.gradient @ direction, abs=1e-8
    )
    finite_response = (plus.gradient - minus.gradient) / (2 * step)
    assert finite_response == pytest.approx(hessian @ direction, abs=1e-7)


def test_nested_thermal_maximum_leaves_a_local_energy_minimum() -> None:
    """A nonsymmetric restart returns the matched PH thermal minimum branch."""
    problem = ThermalSaddle(0.2, nk=80)
    reference = problem.joint(problem.reference_x, problem.reference_y)
    generator = np.random.default_rng(1)
    x = reference.x + 0.01 * np.maximum(np.abs(reference.x), 0.2) * generator.normal(
        size=6
    )
    y = reference.y + 0.01 * np.maximum(np.abs(reference.y), 0.2) * generator.normal(
        size=6
    )
    result = run_method(problem, x, y, "nested-PH")
    assert result["minimum_converged"]
    assert result["strict_minimum"]
    assert result["auxiliary_largest_curvature"] < -1e-6
    assert result["outer_smallest_curvature"] > 1e-6
    assert result["residual"] < 1e-7
    assert result["gradient_norm"] < 1e-8
    assert result["value"] == pytest.approx(reference.value, abs=1e-9)
    assert result["double_occupancy"] == pytest.approx(
        reference.double_occupancy, abs=1e-7
    )


@pytest.mark.parametrize("method", ["coupled-Newton", "coupled-extragradient"])
def test_coupled_thermal_solve_matches_the_reference(method: str) -> None:
    """Simultaneous field updates reproduce the Gibbs state and constraints."""
    problem = ThermalSaddle(0.2, nk=80)
    reference = problem.joint(problem.reference_x, problem.reference_y)
    generator = np.random.default_rng(1)
    x = reference.x + 0.01 * np.maximum(np.abs(reference.x), 0.2) * generator.normal(
        size=6
    )
    y = reference.y + 0.01 * np.maximum(np.abs(reference.y), 0.2) * generator.normal(
        size=6
    )
    result = run_method(problem, x, y, method)
    assert result["stationary_converged"]
    assert result["residual"] < 1e-7
    assert result["gradient_norm"] < 1e-8
    assert result["value"] == pytest.approx(reference.value, abs=1e-8)
    assert result["double_occupancy"] == pytest.approx(
        reference.double_occupancy, abs=1e-7
    )


def test_common_bath_shift_does_not_remove_every_saddle_direction() -> None:
    """Stationary common charge still leaves negative reduced curvature."""
    problem = ThermalSaddle(0.2, nk=80)
    result = run_method(
        problem, problem.reference_x, problem.reference_y, "nested-common"
    )
    assert result["stationary_converged"]
    assert not result["minimum_converged"]
    assert result["auxiliary_largest_curvature"] < 0
    assert result["outer_smallest_curvature"] < -1e-3
