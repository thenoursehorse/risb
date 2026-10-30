# ruff: noqa: D100, D102, D103
from io import StringIO
from itertools import pairwise
from typing import override

import numpy as np
import pytest
from numpy.typing import NDArray

from risb.diagnostics import LatticeDiagnostics
from risb.optimize import (
    DIIS,
    FixedPointLoop,
    LinearMixing,
    NewtonSolver,
    OnPlateau,
    ScipyRoot,
    solve_fixed_point,
)
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


def test_diis_with_a_history_of_one_is_linear_mixing_with_the_same_step() -> None:
    def nonlinear(
        x: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        g_x = np.cos(x)
        return g_x, g_x - x

    diis = DIIS(history_size=1, n_period=None, alpha=0.3)
    mixing = LinearMixing(alpha=0.3)
    x_diis = diis.solve(nonlinear, np.zeros(2), tol=0, maxiter=5)
    x_mixing = mixing.solve(nonlinear, np.zeros(2), tol=0, maxiter=5)
    assert x_diis == pytest.approx(x_mixing, abs=1e-14)


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
    solver = DIIS(history_size=n + 2, n_period=None)
    x = solver.solve(linear, np.zeros(n), tol=1e-12)
    assert solver.success
    assert solver.n <= n + 1
    assert x == pytest.approx(x_star, abs=1e-10)

    # With a short history it still converges to the tolerance
    solver = DIIS(history_size=3, n_period=None)
    solver.solve(linear, np.zeros(n), tol=1e-12, maxiter=100)
    assert solver.success


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


def test_solve_fixed_point_solves_a_problem_like_the_optimizer_on_its_own() -> None:
    problem: FixedPointProblem = Linear()
    result = solve_fixed_point(problem, DIIS(), tol=1e-12)
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)
    alone = DIIS()
    alone.solve(Linear().cycle, np.zeros(6), tol=1e-12)
    assert result.cycles == alone.n + 1
    assert len(result.history) == result.cycles


def test_solve_fixed_point_stops_on_the_error_the_optimizer_chooses() -> None:
    class Scaled(Linear):
        # The residual is much smaller than the change of x
        @override
        def cycle(
            self, x: NDArray[np.float64]
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            x_new, change = super().cycle(x)
            return x_new, 1e-6 * change

    root = solve_fixed_point(Scaled(), LinearMixing(error_fun="root"), tol=1e-10)
    recursion = solve_fixed_point(
        Scaled(), LinearMixing(error_fun="recursion"), tol=1e-10
    )
    assert root.converged
    assert recursion.converged
    assert root.cycles < recursion.cycles
    assert recursion.x == pytest.approx(Linear().solution, abs=1e-8)


def test_callbacks_are_called_after_every_cycle_and_can_stop_the_solve() -> None:
    seen: list[int] = []

    def record(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        seen.append(loop.cycle)
        if loop.cycle == 3:
            loop.stop()

    result = solve_fixed_point(Linear(), DIIS(), callbacks=[record], tol=1e-14)
    assert seen == [1, 2, 3]
    assert not result.converged
    assert result.cycles == 3


def test_callbacks_see_each_input_and_next_guess_up_to_the_converged_cycle() -> None:
    seen: list[tuple[NDArray[np.float64], NDArray[np.float64], bool]] = []

    def record(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        seen.append((loop.x, loop.x_next, loop.converged))

    result = solve_fixed_point(Linear(), DIIS(), callbacks=[record], tol=1e-12)
    assert result.converged
    assert len(seen) == result.cycles
    assert [converged for _, _, converged in seen] == [False] * (len(seen) - 1) + [True]
    for (_, x_next, _), (x, _, _) in pairwise(seen):
        assert x == pytest.approx(x_next)
    assert seen[-1][0] == pytest.approx(result.x, rel=0, abs=0)
    assert seen[-1][1] == pytest.approx(result.x_next, rel=0, abs=0)


def test_a_plateau_on_the_converged_cycle_leaves_the_solve_to_continue() -> None:
    plain = DIIS()
    result = solve_fixed_point(Linear(), plain, tol=1e-12)
    # A plateau from the second cycle on would start on the converged cycle
    on_plateau = OnPlateau(patience=result.cycles - 2, factor=1e100)
    optimizer = DIIS()
    with_plateau = solve_fixed_point(
        Linear(), optimizer, callbacks=[on_plateau], tol=1e-12
    )
    assert on_plateau.restarts == 0
    # The optimizer keeps its history for the next solve. Exact, since near the
    # solution a reset changes the next guess only at the rounding level
    after_plain = solve_fixed_point(Linear(), plain, maxiter=1, x0=result.x_next)
    after = solve_fixed_point(Linear(), optimizer, maxiter=1, x0=with_plateau.x_next)
    assert after.x_next == pytest.approx(after_plain.x_next, rel=0, abs=0)


def test_a_callback_can_replace_the_optimizer_mid_solve() -> None:
    def switch(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        if loop.cycle == 2:
            loop.optimizer = DIIS(history_size=10, n_period=None)

    result = solve_fixed_point(
        Linear(), LinearMixing(alpha=0.1), callbacks=[switch], tol=1e-12
    )
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)


def test_scipy_root_solves_a_problem_in_one_shot() -> None:
    def never(_loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        raise AssertionError

    output = StringIO()
    with LatticeDiagnostics(output=output) as diagnostics:
        result = solve_fixed_point(
            Linear(),
            ScipyRoot(method="hybr", tol=1e-12, options={"maxfev": 100}),
            callbacks=[never],
            diagnostics=diagnostics,
        )
    assert result.converged
    assert result.x == pytest.approx(Linear().solution, abs=1e-10)
    assert result.cycles == len(result.history)
    assert result.x_next is None
    assert "optimizer.evaluation" in output.getvalue()
    assert "optimizer.cycle" not in output.getvalue()
    assert "optimizer.finished" in output.getvalue()


def test_solving_one_cycle_at_a_time_from_x_next_is_one_solve() -> None:
    problem = Linear()
    optimizer = DIIS()
    result = solve_fixed_point(problem, optimizer, maxiter=1)
    for _ in range(4):
        result = solve_fixed_point(problem, optimizer, maxiter=1, x0=result.x_next)
    whole = solve_fixed_point(Linear(), DIIS(), maxiter=5)
    assert result.x == pytest.approx(whole.x)
    assert result.x_next == pytest.approx(whole.x_next)


class Stuck:
    """A problem whose residual grows from `level` until it is anchored, then is solved at once."""

    def __init__(self, level: float, can_anchor: bool) -> None:
        self.level = level
        self.anchors = 0
        self.x = np.ones(3)
        self.cycles = 0
        self.anchored_at: NDArray[np.float64] | None = None
        if not can_anchor:
            self.anchor = None  # type: ignore[assignment]

    def get_x(self) -> NDArray[np.float64]:
        return self.x

    def anchor(self) -> None:
        self.anchors += 1

    def cycle(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.anchors:
            if self.anchored_at is None:
                self.anchored_at = x
            return x, np.zeros(3)
        self.x = x + 0.1
        self.cycles += 1
        return self.x, self.level * self.cycles * np.ones(3)


def test_a_plateau_near_a_solution_anchors_the_problem_once() -> None:
    problem = Stuck(1e-8, can_anchor=True)
    on_plateau = OnPlateau(patience=5)
    result = solve_fixed_point(
        problem, DIIS(error_fun="root"), callbacks=[on_plateau], tol=1e-12
    )
    assert result.converged
    assert problem.anchors == 1
    # The first guess has the smallest error
    assert problem.anchored_at == pytest.approx(np.ones(3))
    assert on_plateau.anchored
    assert on_plateau.restarts == 0


@pytest.mark.parametrize(("level", "can_anchor"), [(1.0, True), (1e-8, False)])
def test_a_plateau_far_from_a_solution_or_without_an_anchor_restarts(
    level: float, can_anchor: bool
) -> None:
    problem = Stuck(level, can_anchor)
    on_plateau = OnPlateau(patience=5)
    result = solve_fixed_point(
        problem, DIIS(error_fun="root"), callbacks=[on_plateau], maxiter=20
    )
    assert not result.converged
    assert problem.anchors == 0
    # A plateau is 6 cycles without progress, counted from the first cycle
    # after a restart: restarts at cycles 7 and 14
    assert on_plateau.restarts == 2
