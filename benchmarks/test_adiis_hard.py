"""Physical checks for the small thermal ghost benchmark."""

import numpy as np
import pytest
from adiis import run_adiis
from adiis_hard import ThermalObjective, thermal_observation


def test_full_thermal_reduction_preserves_reference_thermodynamics() -> None:
    """All six bath-field coordinates reproduce the unprojected reference."""
    problem = ThermalObjective(1.0, symmetric=False)
    reference = thermal_observation(problem)
    sample = problem.evaluate(problem.reference_y)
    observed = problem.observe(sample)
    assert len(sample.y) == 6
    assert sample.residual < 1e-7
    for name in ("grand_potential", "energy", "entropy", "double_occupancy", "filling"):
        assert observed[name] == pytest.approx(reference[name], abs=1e-7)


def test_thermal_envelope_gradient_differentiates_the_grand_potential() -> None:
    """A bath-charge perturbation includes the complete interacting trace."""
    problem = ThermalObjective(1.0, symmetric=False)
    y = problem.reference_y
    sample = problem.evaluate(y)
    direction = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    step = 1e-4
    plus = problem.evaluate(y + step * direction).energy
    minus = problem.evaluate(y - step * direction).energy
    assert sample.gradient @ direction == pytest.approx(
        (plus - minus) / (2 * step), abs=1e-6
    )


def test_symmetric_thermal_adiis_recovers_the_physical_solution() -> None:
    """The symmetry-restricted control matches the original thermal equations."""
    problem = ThermalObjective(1.0)
    reference = problem.evaluate(problem.reference_y)
    problem.map_outputs = True
    result, _ = run_adiis(problem, reference.y * np.array([1.3, 0.7, 1.3]), maxiter=30)
    assert result.residual < 1e-7
    assert result.energy == pytest.approx(reference.energy, abs=1e-7)
    assert result.double_occupancy == pytest.approx(
        reference.double_occupancy, abs=1e-6
    )
    assert problem.observe(result)["filling"] == pytest.approx(1.0, abs=1e-7)
