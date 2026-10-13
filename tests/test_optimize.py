# ruff: noqa: D100, D103
import numpy as np
import pytest
from numpy.typing import NDArray

from risb.optimize import DIIS, LinearMixing, NewtonSolver


def fixed_point(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    g_x = 0.5 * x + 1.0
    return g_x, g_x - x


@pytest.mark.parametrize(
    "solver", [DIIS(), DIIS(alpha=0.5), LinearMixing(), LinearMixing(alpha=0.5)]
)
def test_solver_finds_fixed_point(solver: NewtonSolver) -> None:
    x = solver.solve(fixed_point, np.zeros(3), tol=1e-10)
    assert solver.success
    assert x == pytest.approx(2.0)


def test_diis_keeps_history_size_vectors() -> None:
    solver = DIIS(history_size=4)
    solver.solve(fixed_point, np.zeros(3), tol=0, maxiter=10)
    assert len(solver.error) == 4


@pytest.mark.parametrize(("n_restart", "n_vectors"), [(None, 3), (2, 1)])
def test_diis_restarts_history_after_n_restart_iterations(
    n_restart: int | None, n_vectors: int
) -> None:
    solver = DIIS(n_restart=n_restart)
    solver.solve(fixed_point, np.zeros(3), tol=0, maxiter=3)
    assert len(solver.error) == n_vectors
