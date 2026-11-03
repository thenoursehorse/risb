"""Solve a fixed-point problem with an optimizer and callbacks."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from risb.diagnostics import DiagnosticObserver
from risb.types import Checkpointable, FixedPointProblem

from .solver_newton import NewtonSolver

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """The outcome of :func:`solve_fixed_point`."""

    #: numpy.ndarray : Last accepted input of :meth:`risb.types.FixedPointProblem.cycle`.
    x: NDArray[np.float64]

    #: bool : Whether the error fell below the tolerance.
    converged: bool

    #: int : Total number of cycle evaluations, including rejected steps
    #: and any extra cycles needed to restore a checkpoint.
    cycles: int

    #: float : Error norm at the last accepted guess.
    error: float

    #: list of float : Error norms at accepted points.
    history: list[float] = field(default_factory=list)

    #: numpy.ndarray or None : Next guess for continuing the solve.
    #: None when using :class:`ScipyRoot`.
    x_next: NDArray[np.float64] | None = None

    #: list of str : Residual definition used for each history value.
    history_error_funs: list[Literal["root", "recursion"]] = field(
        default_factory=list, kw_only=True
    )


class ScipyRoot:
    """
    Use :func:`scipy.optimize.root` with :func:`solve_fixed_point`.

    SciPy controls the iterations. The callbacks of :func:`solve_fixed_point`
    are not called.

    Parameters
    ----------
    error_fun : {'recursion', 'root'}, optional
        The error to make zero, see :class:`.NewtonSolver`.
    **kwargs
        Keyword arguments of :func:`scipy.optimize.root`, e.g., ``method`` and
        ``options``.

    """

    def __init__(
        self, error_fun: Literal["recursion", "root"] = "recursion", **kwargs: object
    ) -> None:
        self.error_fun = error_fun
        self.kwargs = kwargs


type Callback = Callable[["FixedPointLoop", FixedPointProblem], None]
"""Called after each accepted step as ``callback(loop, problem)``."""


class FixedPointLoop:
    """
    Run a fixed-point iteration with an optimizer and callbacks.

    Callbacks run after each accepted step, including the last. They can
    inspect the loop, replace :attr:`optimizer` or the next guess
    :attr:`x_next`, or call :meth:`stop` to end the solve.

    Parameters
    ----------
    problem : risb.types.FixedPointProblem
        The problem to solve. If a cycle depends on state other than x,
        implement ``checkpoint()`` so the optimizer can restore that state
        after rejecting a step.
    optimizer : risb.optimize.NewtonSolver
        Gives the next guess after each cycle. It keeps its history from
        before; :meth:`~risb.optimize.NewtonSolver.reset` clears it.
    callbacks : sequence of callable, optional
        Called after each accepted step as ``callback(loop, problem)``.
    tol : float, optional
        Stop when the 2-norm of the error is below this.
    maxiter : int, optional
        Maximum number of cycle evaluations. This includes rejected steps
        and any extra cycles needed to restore a checkpoint.
        A budget too small to complete an optimizer step raises ValueError.
        The loop stops before a step if too few evaluations remain to finish
        it. Continue from ``x_next`` with the same optimizer.
    x0 : numpy.ndarray, optional
        Where to start. Default is the current guess of `problem`.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Receive progress events independently of callbacks.

    """

    def __init__(
        self,
        problem: FixedPointProblem,
        optimizer: NewtonSolver,
        callbacks: Sequence[Callback] = (),
        tol: float = 1e-12,
        maxiter: int = 1000,
        x0: NDArray[np.float64] | None = None,
        *,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.problem = problem
        self.diagnostics = diagnostics

        #: risb.optimize.NewtonSolver : Gives the next guess after each cycle.
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.tol = tol
        self.maxiter = maxiter

        #: numpy.ndarray : The last accepted guess.
        self.x = problem.get_x() if x0 is None else x0

        #: numpy.ndarray : The input of the next cycle, from the optimizer.
        self.x_next = self.x

        #: int : Number of cycles so far.
        self.cycle = 0

        #: float : Error norm at the last accepted guess.
        self.error = np.inf

        #: list of float : Error norms at accepted points.
        self.history: list[float] = []
        self.history_error_funs: list[Literal["root", "recursion"]] = []

        #: bool : Whether the error of the last cycle is below the tolerance.
        self.converged = False

        self._errors: dict[Literal["root", "recursion"], float] = {
            "root": np.inf,
            "recursion": np.inf,
        }
        self._best_errors = self._errors.copy()
        self._best_x = {"root": self.x.copy(), "recursion": self.x.copy()}
        self._stopped = False

    def get_error(self, error_fun: Literal["root", "recursion"] | None = None) -> float:
        """Return the current residual norm; None follows the active optimizer."""
        return self._errors[
            self.optimizer.error_fun if error_fun is None else error_fun
        ]

    def get_best_error(
        self, error_fun: Literal["root", "recursion"] | None = None
    ) -> float:
        """Return the smallest selected residual norm seen so far."""
        return self._best_errors[
            self.optimizer.error_fun if error_fun is None else error_fun
        ]

    def get_best_x(
        self, error_fun: Literal["root", "recursion"] | None = None
    ) -> NDArray[np.float64]:
        """Return the iterate with the smallest selected residual norm."""
        return self._best_x[
            self.optimizer.error_fun if error_fun is None else error_fun
        ]

    @property
    def best_x(self) -> NDArray[np.float64]:
        """Best iterate for the active optimizer's residual."""
        return self.get_best_x()

    def clear_best_points(self) -> None:
        """Discard saved errors and best iterates after changing the fixed-point map."""
        self._errors = {"root": np.inf, "recursion": np.inf}
        self._best_errors = self._errors.copy()
        self._best_x = {"root": self.x.copy(), "recursion": self.x.copy()}
        self.error = np.inf
        self.converged = False

    def stop(self) -> None:
        """End the loop after the current cycle."""
        self._stopped = True

    def emit(self, event: str, **fields: object) -> None:
        """Report an event independently of the control callbacks."""
        if self.diagnostics is not None:
            self.diagnostics.emit(event, cycle=self.cycle, **fields)

    def run(self) -> SolveResult:
        """
        Run cycles until the error is below the tolerance.

        Returns
        -------
        risb.optimize.SolveResult

        """
        self.optimizer.step_fits_budget(self.maxiter, self.cycle)
        while self.cycle < self.maxiter:
            saved = (
                self.problem.checkpoint()
                if self.optimizer.can_reject
                and self.optimizer.x
                and isinstance(self.problem, Checkpointable)
                else None
            )
            if not self.optimizer.step_fits_budget(
                self.maxiter, self.cycle, checkpoint=saved
            ):
                if self.optimizer.x:
                    self.x = self.optimizer.x[0].copy()
                    self._errors["root"] = float(
                        np.linalg.norm(self.optimizer.residual[0])
                    )
                    self._errors["recursion"] = float(
                        np.linalg.norm(self.optimizer.g_x[0] - self.x)
                    )
                    self.error = self.get_error()
                    self.converged = self.error < self.tol
                break
            x_new, residual = self.problem.cycle(self.x_next)
            self.x, x_new, residual, self.x_next, extra = self.optimizer.step(
                self.x_next, x_new, residual, self.problem.cycle, checkpoint=saved
            )
            self.cycle += 1 + extra
            self._errors["root"] = float(np.linalg.norm(residual))
            self._errors["recursion"] = float(np.linalg.norm(x_new - self.x))
            for error_fun, norm in self._errors.items():
                if norm < self._best_errors[error_fun]:
                    self._best_errors[error_fun] = norm
                    self._best_x[error_fun] = self.x.copy()
            self.error = self.get_error()
            self.history.append(self.error)
            self.history_error_funs.append(self.optimizer.error_fun)
            logger.info(f"cycle: {self.cycle}, error: {self.error}")
            self.converged = self.error < self.tol
            self.optimizer.report_step(self, error=self.error, converged=self.converged)
            for callback in self.callbacks:
                callback(self, self.problem)
                self.error = self.get_error()
                self.converged = self.error < self.tol
            if self.converged or self._stopped:
                break
        self.emit(
            "optimizer.finished",
            reason="tolerance"
            if self.converged
            else "callback"
            if self._stopped
            else "maxiter",
        )
        return SolveResult(
            self.x,
            self.converged,
            self.cycle,
            self.error,
            self.history,
            self.x_next,
            history_error_funs=self.history_error_funs,
        )


def solve_fixed_point(
    problem: FixedPointProblem,
    optimizer: NewtonSolver | ScipyRoot,
    callbacks: Sequence[Callback] = (),
    tol: float = 1e-12,
    maxiter: int = 1000,
    x0: NDArray[np.float64] | None = None,
    *,
    diagnostics: DiagnosticObserver | None = None,
) -> SolveResult:
    """
    Solve a fixed-point problem, starting from its current guess or `x0`.

    Parameters
    ----------
    problem : risb.types.FixedPointProblem
        The problem to solve, e.g., a :class:`risb.LatticeSolver`. If a cycle
        depends on state other than x, implement ``checkpoint()`` so the
        optimizer can restore that state after rejecting a step.
    optimizer : risb.optimize.NewtonSolver or risb.optimize.ScipyRoot
        Gives the next guess after each cycle, e.g., :class:`.DIIS`. It keeps
        its history from before; :meth:`~risb.optimize.NewtonSolver.reset`
        clears it.
    callbacks : sequence of callable, optional
        Called after each accepted step as ``callback(loop, problem)``, with the
        :class:`FixedPointLoop`. Not called with a :class:`ScipyRoot`.
    tol : float, optional
        Stop when the error norm is below this. With :class:`ScipyRoot`,
        set the tolerance in its constructor instead.
    maxiter : int, optional
        Maximum number of cycle evaluations. This includes rejected steps
        and any extra cycles needed to restore a checkpoint.
        A budget too small to complete an optimizer step raises ValueError.
        The loop stops before a step if too few evaluations remain to finish
        it. Continue from ``x_next`` with the same optimizer.
        Not used with a :class:`ScipyRoot`.
    x0 : numpy.ndarray, optional
        Where to start. Default is the current guess of `problem`.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Receive progress events independently of callbacks.

    Returns
    -------
    risb.optimize.SolveResult

    """
    if isinstance(optimizer, ScipyRoot):
        return _solve_fixed_point_scipy(problem, optimizer, x0, diagnostics)
    return FixedPointLoop(
        problem, optimizer, callbacks, tol, maxiter, x0, diagnostics=diagnostics
    ).run()


def _solve_fixed_point_scipy(
    problem: FixedPointProblem,
    optimizer: ScipyRoot,
    x0: NDArray[np.float64] | None,
    diagnostics: DiagnosticObserver | None,
) -> SolveResult:
    history: list[float] = []
    if x0 is None:
        x0 = problem.get_x()
    last = x0

    def error(x: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal last
        last = x.copy()
        x_new, residual = problem.cycle(x)
        e: NDArray[np.float64] = (
            residual if optimizer.error_fun == "root" else x_new - x
        )
        history.append(float(np.linalg.norm(e)))
        if diagnostics is not None:
            diagnostics.emit(
                "optimizer.evaluation",
                evaluation=len(history),
                error=history[-1],
                error_fun=optimizer.error_fun,
            )
        return e

    result = root(error, x0, **optimizer.kwargs)  # type: ignore[call-overload]
    # The problem is left at the solution, which need not be the last guess
    if not np.array_equal(last, result.x):
        problem.cycle(result.x)
    if diagnostics is not None:
        diagnostics.emit(
            "optimizer.finished",
            success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            evaluations=int(result.nfev),
        )
    return SolveResult(
        np.asarray(result.x),
        bool(result.success),
        len(history),
        history[-1],
        history,
        history_error_funs=[optimizer.error_fun] * len(history),
    )
