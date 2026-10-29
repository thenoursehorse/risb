"""Solve a fixed-point problem with an optimizer and callbacks."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from risb.diagnostics import DiagnosticObserver
from risb.types import FixedPointProblem

from .solver_newton import NewtonSolver

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """The outcome of :func:`solve_fixed_point`."""

    #: numpy.ndarray : The last input of :meth:`risb.types.FixedPointProblem.cycle`.
    x: NDArray[np.float64]

    #: bool : Whether the error fell below the tolerance.
    converged: bool

    #: int : Number of cycles.
    cycles: int

    #: float : 2-norm of the error of the last cycle.
    error: float

    #: list of float : 2-norm of the error of every cycle.
    history: list[float] = field(default_factory=list)

    #: numpy.ndarray or None : The next guess, where a following solve can
    #: start. None with a :class:`ScipyRoot`.
    x_next: NDArray[np.float64] | None = None


class ScipyRoot:
    """
    :func:`scipy.optimize.root` as a one-shot optimizer for :func:`solve_fixed_point`.

    It runs to the end in one call, so the callbacks are not called.

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
"""Called after every cycle of :func:`solve_fixed_point` as ``callback(loop, problem)``."""


class FixedPointLoop:
    """
    The loop of :func:`solve_fixed_point`.

    After every cycle, including the last, each callback gets the loop. It
    can read :attr:`cycle`, :attr:`error`, :attr:`history`, :attr:`converged`,
    the input :attr:`x` of the cycle, and :attr:`best_x`, replace
    :attr:`optimizer` or the next input :attr:`x_next`, or end the loop with
    :meth:`stop`.

    Parameters
    ----------
    problem : risb.types.FixedPointProblem
        The problem to solve.
    optimizer : risb.optimize.NewtonSolver
        Gives the next guess after each cycle. It keeps its history from
        before; :meth:`~risb.optimize.NewtonSolver.reset` clears it.
    callbacks : sequence of callable, optional
        Called after every cycle as ``callback(loop, problem)``.
    tol : float, optional
        Stop when the 2-norm of the error is below this.
    maxiter : int, optional
        Maximum number of cycles.
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

        #: numpy.ndarray : The input of the last cycle.
        self.x = problem.get_x() if x0 is None else x0

        #: numpy.ndarray : The input of the next cycle, from the optimizer.
        self.x_next = self.x

        #: int : Number of cycles so far.
        self.cycle = 0

        #: float : 2-norm of the error of the last cycle.
        self.error = np.inf

        #: list of float : 2-norm of the error of every cycle.
        self.history: list[float] = []

        #: bool : Whether the error of the last cycle is below the tolerance.
        self.converged = False

        #: numpy.ndarray : The input of the cycle with the smallest error so
        #: far.
        self.best_x = self.x

        self._stopped = False

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
        while self.cycle < self.maxiter:
            self.x = self.x_next
            x_new, residual = self.problem.cycle(self.x)
            self.cycle += 1
            self.error = float(
                np.linalg.norm(self.optimizer.error_of(self.x, x_new, residual))
            )
            if self.error < min(self.history, default=np.inf):
                self.best_x = self.x
            self.history.append(self.error)
            error_fun = self.optimizer.error_fun
            logger.info(f"cycle: {self.cycle}, error: {self.error}")
            self.x_next = self.optimizer.step(self.x, x_new, residual)
            self.converged = self.error < self.tol
            for callback in self.callbacks:
                callback(self, self.problem)
            self.emit(
                "optimizer.cycle",
                error=self.error,
                error_fun=error_fun,
                converged=self.converged,
            )
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
        The problem to solve, e.g., a :class:`risb.LatticeSolver`.
    optimizer : risb.optimize.NewtonSolver or risb.optimize.ScipyRoot
        Gives the next guess after each cycle, e.g., :class:`.DIIS`. It keeps
        its history from before; :meth:`~risb.optimize.NewtonSolver.reset`
        clears it.
    callbacks : sequence of callable, optional
        Called after every cycle as ``callback(loop, problem)``, with the
        :class:`FixedPointLoop`. Not called with a :class:`ScipyRoot`.
    tol : float, optional
        Stop when the 2-norm of the error is below this. A :class:`ScipyRoot`
        stops by its own options.
    maxiter : int, optional
        Maximum number of cycles. Not used with a :class:`ScipyRoot`.
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
        np.asarray(result.x), bool(result.success), len(history), history[-1], history
    )
