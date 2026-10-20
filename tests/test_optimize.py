# ruff: noqa: D100, D102, D103
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray

from risb.optimize import DIIS, Driver, LinearMixing, NewtonSolver, ScipyRoot, drive
from risb.types import FixedPointProblem


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

    # With a short history it still converges to the tolerance
    solver = DIIS(history_size=3, n_period=10**9)
    solver.solve(linear, np.zeros(n), tol=1e-12, maxiter=100)
    assert solver.success


@pytest.mark.parametrize("solver_class", [DIIS, LinearMixing])
def test_solving_another_problem_does_not_depend_on_the_one_before(
    solver_class: type[NewtonSolver],
) -> None:
    def other_fixed_point(
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        g_x = -0.3 * x + np.array([1.0, -2.0, 0.5])
        return g_x, g_x - x

    fresh = solver_class()
    x_fresh = fresh.solve(other_fixed_point, np.zeros(3), tol=1e-10)
    reused = solver_class()
    reused.solve(fixed_point, np.zeros(3), tol=1e-10)
    x_reused = reused.solve(other_fixed_point, np.zeros(3), tol=1e-10)
    assert reused.n == fresh.n
    assert x_reused == pytest.approx(x_fresh, abs=1e-14)


class Linear:
    """The fixed-point problem x = A x + b, with the residual (A - 1) x + b."""

    def __init__(self, n: int = 6, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        self.A = Q @ np.diag(rng.uniform(-0.9, 0.9, n)) @ Q.T
        self.b = rng.normal(size=n)
        self.x = np.zeros(n)
        self.solution = np.linalg.solve(np.eye(n) - self.A, self.b)

    def get_x(self) -> NDArray[np.float64]:
        return self.x

    def cycle(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.x = self.A @ x + self.b
        return self.x, self.x - x


def test_drive_solves_a_problem_like_the_optimizer_on_its_own() -> None:
    problem: FixedPointProblem = Linear()
    result = drive(problem, DIIS(), tol=1e-12)
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)
    alone = DIIS()
    alone.solve(Linear().cycle, np.zeros(6), tol=1e-12)
    assert result.cycles == alone.n + 1
    assert len(result.history) == result.cycles


def test_drive_stops_on_the_error_the_optimizer_chooses() -> None:
    class Scaled(Linear):
        # The residual is much smaller than the change of x
        @override
        def cycle(
            self, x: NDArray[np.float64]
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            x_new, change = super().cycle(x)
            return x_new, 1e-6 * change

    root = drive(Scaled(), LinearMixing(error_fun="root"), tol=1e-10)
    recursion = drive(Scaled(), LinearMixing(error_fun="recursion"), tol=1e-10)
    assert root.converged
    assert recursion.converged
    assert root.cycles < recursion.cycles
    assert recursion.x == pytest.approx(Linear().solution, abs=1e-8)


def test_callbacks_are_called_after_every_cycle_and_can_stop_the_solve() -> None:
    seen: list[int] = []

    def record(driver: Driver, _problem: FixedPointProblem) -> None:
        seen.append(driver.cycle)
        if driver.cycle == 3:
            driver.stop()

    result = drive(Linear(), DIIS(), callbacks=[record], tol=1e-14)
    assert seen == [1, 2, 3]
    assert not result.converged
    assert result.cycles == 3


def test_a_callback_can_replace_the_optimizer_mid_solve() -> None:
    def switch(driver: Driver, _problem: FixedPointProblem) -> None:
        if driver.cycle == 2:
            driver.optimizer = DIIS(history_size=10, n_period=10**9)

    result = drive(Linear(), LinearMixing(alpha=0.1), callbacks=[switch], tol=1e-12)
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)


def test_scipy_root_solves_a_problem_in_one_shot() -> None:
    def never(_driver: Driver, _problem: FixedPointProblem) -> None:
        raise AssertionError

    result = drive(Linear(), ScipyRoot(method="hybr", tol=1e-12), callbacks=[never])
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)
    assert result.cycles == len(result.history)
