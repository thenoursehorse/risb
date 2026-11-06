"""Physical checks for the standalone reduced-energy ADIIS prototype."""

import numpy as np
import pytest
from adiis import BetheObjective, run_adiis


@pytest.mark.parametrize("bath_size", [1, 3])
@pytest.mark.parametrize("perturbation", [0.03, 0.3])
def test_adiis_reproduces_the_constrained_risb_solution(
    bath_size: int, perturbation: float
) -> None:
    """ADIIS recovers energy and observables from perturbed embedding fields."""
    objective = BetheObjective(bath_size)
    reference = objective.evaluate(objective.reference_y)
    initial = reference.y * (
        1 + perturbation * np.array([1.0, -1.0, 1.0])[: len(reference.y)]
    )
    result, _ = run_adiis(objective, initial, maxiter=100)
    assert result.residual < 1e-7
    assert result.energy == pytest.approx(reference.energy, abs=1e-9)
    assert result.double_occupancy == pytest.approx(
        reference.double_occupancy, abs=1e-6
    )
    assert result.z == pytest.approx(reference.z, abs=1e-6)
    if bath_size == 1:
        kinetic_free = 2 * np.mean(np.minimum(objective.eps, 0))
        interaction_critical = -8 * kinetic_free
        assert result.double_occupancy == pytest.approx(
            (1 - 1 / interaction_critical) / 4, abs=1e-7
        )
        assert result.z == pytest.approx(1 - 1 / interaction_critical**2, abs=1e-7)


def test_adjoint_is_the_gradient_of_the_constrained_energy() -> None:
    """The interacting response differentiates the complete reduced energy."""
    objective = BetheObjective(3)
    initial = objective.reference_y * np.array([1.1, 0.9, 1.1])
    sample = objective.evaluate(initial)
    finite = []
    for direction in np.eye(len(initial)):
        plus = objective.evaluate(initial + 1e-5 * direction).energy
        minus = objective.evaluate(initial - 1e-5 * direction).energy
        finite.append((plus - minus) / 2e-5)
    assert sample.gradient == pytest.approx(finite, abs=1e-7)


def test_adiis_recovers_from_a_distorted_ghost_embedding() -> None:
    """The nonmonotone proposal recovers the physical ghost solution."""
    objective = BetheObjective(3)
    reference = objective.evaluate(objective.reference_y)
    result, _ = run_adiis(
        objective, reference.y * np.array([0.5, 0.5, 1.5]), maxiter=100
    )
    assert result.residual < 1e-7
    assert result.energy == pytest.approx(reference.energy, abs=1e-9)
    assert result.double_occupancy == pytest.approx(
        reference.double_occupancy, abs=1e-6
    )
    assert result.z == pytest.approx(reference.z, abs=1e-6)
