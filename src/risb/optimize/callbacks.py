"""Callbacks for :func:`risb.optimize.solve_fixed_point`."""

import logging

import numpy as np

from risb.types import FixedPointProblem

from .fixed_point import FixedPointLoop

logger = logging.getLogger(__name__)


class OnPlateau:
    """
    Restart or anchor a solve whose error has stopped falling.

    The error is on a plateau when it has not fallen by `factor` for
    `patience` cycles. If the smallest error so far is below `near`, and the
    problem has an ``anchor`` method that has not been called yet, the solve
    goes back to the guess with the smallest error so far, the problem is
    anchored there, and the optimizer switches to ``error_fun='recursion'``.
    Otherwise the optimizer restarts its history from the current guess.

    Parameters
    ----------
    patience : int, optional
        Cycles without the error falling by `factor` that make a plateau.
        Default is ``20``.
    near : float, optional
        A plateau is near a solution when the smallest error so far is below
        this. Default is ``1e-6``.
    factor : float, optional
        How much the error must fall to count as progress. Default is ``10``.

    """

    def __init__(
        self, patience: int = 20, near: float = 1e-6, factor: float = 10.0
    ) -> None:
        self.patience = patience
        self.near = near
        self.factor = factor

        #: bool : Whether the problem has been anchored.
        self.anchored = False

        #: int : Number of restarts of the optimizer.
        self.restarts = 0

        self._smallest = np.inf
        self._mark = np.inf
        self._since = 0

    def __call__(self, loop: FixedPointLoop, problem: FixedPointProblem) -> None:
        """Check for a plateau after a cycle of `loop`, and act on it."""
        if loop.converged:
            return
        self._smallest = min(self._smallest, loop.error)
        if loop.error < self._mark / self.factor:
            self._mark = loop.error
            self._since = loop.cycle
        if loop.cycle - self._since <= self.patience:
            return
        anchor = getattr(problem, "anchor", None)
        if self._smallest < self.near and anchor is not None and not self.anchored:
            # At the anchored solution the root functions need not be zero,
            # but the guess stops changing
            loop.x_next = loop.best_x
            anchor()
            loop.optimizer.error_fun = "recursion"
            self.anchored = True
            logger.info(f"cycle: {loop.cycle}, anchored")
        else:
            self.restarts += 1
            logger.info(f"cycle: {loop.cycle}, restarted")
        loop.optimizer.reset()
        self._smallest = np.inf
        self._mark = np.inf
        self._since = loop.cycle
