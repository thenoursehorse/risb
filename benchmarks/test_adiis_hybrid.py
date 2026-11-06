"""Physical checks for ADIIS-to-DIIS on a fixed reduced energy objective."""

import numpy as np
import pytest
from adiis import BetheObjective
from adiis_hard import ThermalObjective
from adiis_hybrid import run_energy_hybrid, run_root_diis


@pytest.mark.parametrize("threshold", [1e-3, 1e-4])
def test_hybrid_finishes_the_same_near_mott_energy_minimization(
    threshold: float,
) -> None:
    """Both coefficient rules converge in the original embedding coordinates."""
    problem = BetheObjective(3, interaction=2.8)
    reference = problem.evaluate(problem.reference_y)
    initial = reference.y + 0.15 * np.maximum(np.abs(reference.y), 0.2) * (
        np.random.default_rng(1).normal(size=3)
    )
    start = problem.evaluate(initial)
    problem.band_initial = np.einsum("aij,ij->a", problem.basis, start.Lambda)
    result = run_energy_hybrid(problem, initial, switch_tolerance=threshold)
    assert result["physical_converged"]
    assert result["attempted_evaluations"] <= 150
    final = problem.evaluate(np.array(result["y"]))
    assert final.energy == pytest.approx(reference.energy, abs=1e-10)
    assert np.max(np.abs(final.gradient)) < 1e-8
    assert final.residual < 1e-7


def test_adiis_cycle_cap_preserves_the_reduced_energy_problem() -> None:
    """A cycle-limited switch retains an admissible converged variational state."""
    problem = BetheObjective(3, interaction=2.4)
    reference = problem.evaluate(problem.reference_y)
    initial = reference.y * np.array([1.1, 0.9, 1.1])
    result = run_energy_hybrid(
        problem, initial, switch_tolerance=1e-30, adiis_limit=3, budget=100
    )
    assert result["switch_reason"] == "cycle_limit"
    assert result["switch_evaluation"] == 3
    assert result["physical_converged"]
    assert result["attempted_evaluations"] <= 100
    final = problem.evaluate(np.array(result["y"]))
    assert final.energy == pytest.approx(reference.energy, abs=1e-10)
    assert np.max(np.abs(final.gradient)) < 1e-8
    assert final.residual < 1e-7


def test_thermal_hybrid_keeps_the_reduced_grand_potential() -> None:
    """The symmetry-restricted hybrid finishes on the same thermal functional."""
    problem = ThermalObjective(1.0)
    reference = problem.evaluate(problem.reference_y)
    initial = reference.y + 0.15 * np.maximum(np.abs(reference.y), 0.2) * (
        np.random.default_rng(1).normal(size=3)
    )
    start = problem.evaluate(initial)
    problem.inner_initial = problem.pack(start.R, start.Lambda)
    result = run_energy_hybrid(problem, initial, switch_tolerance=1e-3, budget=50)
    assert result["physical_converged"]
    assert result["attempted_evaluations"] <= 50
    final = problem.evaluate(np.array(result["y"]))
    assert final.energy == pytest.approx(reference.energy, abs=1e-7)
    assert final.double_occupancy == pytest.approx(reference.double_occupancy, abs=1e-6)
    assert np.max(np.abs(final.gradient)) < 1e-8
    assert final.residual < 1e-7


def test_root_diis_reaches_the_same_near_mott_state() -> None:
    """The root baseline reaches the energy and full matching reference."""
    problem = BetheObjective(3, interaction=2.4)
    reference = problem.evaluate(problem.reference_y)
    initial = reference.y * np.array([1.1, 0.9, 1.1])
    start = problem.evaluate(initial)
    problem.band_initial = np.einsum("aij,ij->a", problem.basis, start.Lambda)
    result = run_root_diis(problem, initial, start)
    assert result["physical_converged"]
    assert result["value"] == pytest.approx(reference.energy, abs=1e-10)
    assert result["double_occupancy"] == pytest.approx(
        reference.double_occupancy, abs=1e-7
    )
    assert result["gradient_norm"] < 1e-8
    assert result["residual"] < 1e-7
