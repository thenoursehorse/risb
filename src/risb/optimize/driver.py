"""Drive a fixed-point problem to its solution with an optimizer and callbacks."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from risb.types import FixedPointProblem

from .solver_newton import NewtonSolver

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """The outcome of :func:`drive`."""

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


class ScipyRoot:
    """
    :func:`scipy.optimize.root` as a one-shot optimizer for :func:`drive`.

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


type Callback = Callable[["Driver", FixedPointProblem], None]
"""Called after every cycle of :func:`drive` as ``callback(driver, problem)``."""


class Driver:
    """
    The loop of :func:`drive`.

    After every cycle, each callback gets the driver. It can read
    :attr:`cycle`, :attr:`error`, :attr:`history`, and :attr:`best_x`, replace
    :attr:`optimizer` or the next input :attr:`x`, or end the loop with
    :meth:`stop`.

    Parameters
    ----------
    problem : risb.types.FixedPointProblem
        The problem to solve.
    optimizer : risb.optimize.NewtonSolver
        Gives the next guess after each cycle.
    callbacks : sequence of callable, optional
        Called after every cycle as ``callback(driver, problem)``.
    tol : float, optional
        Stop when the 2-norm of the error is below this.
    maxiter : int, optional
        Maximum number of cycles.

    """

    def __init__(
        self,
        problem: FixedPointProblem,
        optimizer: NewtonSolver,
        callbacks: Sequence[Callback] = (),
        tol: float = 1e-12,
        maxiter: int = 1000,
    ) -> None:
        self.problem = problem

        #: risb.optimize.NewtonSolver : Gives the next guess after each cycle.
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.tol = tol
        self.maxiter = maxiter

        #: numpy.ndarray : The input of the next cycle.
        self.x = problem.get_x()

        #: int : Number of cycles so far.
        self.cycle = 0

        #: float : 2-norm of the error of the last cycle.
        self.error = np.inf

        #: list of float : 2-norm of the error of every cycle.
        self.history: list[float] = []

        #: numpy.ndarray : The input of the cycle with the smallest error so
        #: far.
        self.best_x = self.x

        self._stopped = False

    def stop(self) -> None:
        """End the loop after the current cycle."""
        self._stopped = True

    def run(self) -> SolveResult:
        """
        Run cycles until the error is below the tolerance.

        Returns
        -------
        risb.optimize.SolveResult

        """
        self.optimizer.reset()
        converged = False
        x = self.x
        while self.cycle < self.maxiter:
            x = self.x
            x_new, residual = self.problem.cycle(x)
            self.cycle += 1
            self.error = float(
                np.linalg.norm(self.optimizer.error_of(x, x_new, residual))
            )
            if self.error < min(self.history, default=np.inf):
                self.best_x = x
            self.history.append(self.error)
            logger.info(f"cycle: {self.cycle}, error: {self.error}")
            if self.error < self.tol:
                converged = True
                break
            self.x = self.optimizer.step(x, x_new, residual)
            for callback in self.callbacks:
                callback(self, self.problem)
            if self._stopped:
                break
        return SolveResult(x, converged, self.cycle, self.error, self.history)


def drive(
    problem: FixedPointProblem,
    optimizer: NewtonSolver | ScipyRoot,
    callbacks: Sequence[Callback] = (),
    tol: float = 1e-12,
    maxiter: int = 1000,
) -> SolveResult:
    """
    Solve a fixed-point problem, starting from its current guess.

    Parameters
    ----------
    problem : risb.types.FixedPointProblem
        The problem to solve, e.g., a :class:`risb.LatticeSolver`.
    optimizer : risb.optimize.NewtonSolver or risb.optimize.ScipyRoot
        Gives the next guess after each cycle, e.g., :class:`.DIIS`.
    callbacks : sequence of callable, optional
        Called after every cycle as ``callback(driver, problem)``, with the
        :class:`Driver`. Not called with a :class:`ScipyRoot`.
    tol : float, optional
        Stop when the 2-norm of the error is below this. A :class:`ScipyRoot`
        stops by its own options.
    maxiter : int, optional
        Maximum number of cycles. Not used with a :class:`ScipyRoot`.

    Returns
    -------
    risb.optimize.SolveResult

    """
    if isinstance(optimizer, ScipyRoot):
        return _drive_scipy(problem, optimizer)
    return Driver(problem, optimizer, callbacks, tol, maxiter).run()


def _drive_scipy(problem: FixedPointProblem, optimizer: ScipyRoot) -> SolveResult:
    history: list[float] = []
    last = problem.get_x()

    def error(x: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal last
        last = x.copy()
        x_new, residual = problem.cycle(x)
        e: NDArray[np.float64] = (
            residual if optimizer.error_fun == "root" else x_new - x
        )
        history.append(float(np.linalg.norm(e)))
        return e

    result = root(error, problem.get_x(), **optimizer.kwargs)  # type: ignore[call-overload]
    # The problem is left at the solution, which need not be the last guess
    if not np.array_equal(last, result.x):
        problem.cycle(result.x)
    return SolveResult(
        np.asarray(result.x), bool(result.success), len(history), history[-1], history
    )
