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


def test_diis_is_anderson_acceleration_on_a_linear_problem() -> None:
    rng = np.random.default_rng(0)
    n = 8
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    A = Q @ np.diag(rng.uniform(-0.9, 0.9, n)) @ Q.T
    b = rng.normal(size=n)
    x_star = np.linalg.solve(np.eye(n) - A, b)

    def linear(
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        g_x = A @ x + b
        return g_x, g_x - x

    # Without the periodic linear mixing steps. With every vector in the
    # history it converges in at most n + 1 iterations, like GMRES
    solver = DIIS(history_size=n + 2, n_period=10**9)
    x = solver.solve(linear, np.zeros(n), tol=1e-12)
    assert solver.success
    assert solver.n <= n + 1
    assert x == pytest.approx(x_star, abs=1e-10)

    # With a short history it keeps its rate down to the tolerance (textbook
    # Anderson acceleration with 3 vectors needs 84 iterations here)
    solver = DIIS(history_size=3, n_period=10**9)
    solver.solve(linear, np.zeros(n), tol=1e-12, maxiter=100)
    assert solver.success
