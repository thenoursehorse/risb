"""Callbacks for :func:`risb.optimize.solve_fixed_point`."""

import logging
from typing import Literal

import numpy as np

from risb.types import FixedPointProblem

from .fixed_point import FixedPointLoop
from .schedule import OptimizerSchedule

logger = logging.getLogger(__name__)


class OnPlateau:
    """
    Recover a self-consistency iteration whose error has stopped decreasing.

    Reset the active optimizer when progress stalls. Near a solution, also
    anchor the problem at its best iterate and switch to the recursion
    residual, if the problem supports anchoring.

    Parameters
    ----------
    patience : int, optional
        Cycle evaluations without a factor-fold error reduction. Default is 20.
    near : float, optional
        Attempt anchoring when the best error is below this. Default is 1e-6.
    factor : float, optional
        Error reduction needed to count as progress. Default is 10.
    error_fun : {'root', 'recursion'} or None, optional
        Residual to monitor. None follows the active optimizer and starts a
        new comparison when its residual definition changes.
    reset_on_scheduler_phase_entry : bool, optional
        Start a new plateau comparison on each stage entry, including
        re-entry. Default is False.

    """

    def __init__(
        self,
        patience: int = 20,
        near: float = 1e-6,
        factor: float = 10.0,
        *,
        error_fun: Literal["root", "recursion"] | None = None,
        reset_on_scheduler_phase_entry: bool = False,
    ) -> None:
        self.patience = patience
        self.near = near
        self.factor = factor
        self.error_fun = error_fun
        self.reset_on_scheduler_phase_entry = reset_on_scheduler_phase_entry

        #: bool : Whether the problem has been anchored.
        self.anchored = False

        #: int : Number of restarts of the optimizer.
        self.restarts = 0

        self._smallest = np.inf
        self._mark = np.inf
        self._since = 0
        self._loop: FixedPointLoop | None = None
        self._error_fun: Literal["root", "recursion"] | None = None
        self._phase_entry: tuple[OptimizerSchedule, int] | None = None

    def __call__(self, loop: FixedPointLoop, problem: FixedPointProblem) -> None:
        """Check for a plateau after a cycle of `loop`, and act on it."""
        if loop.converged:
            return
        error_fun = (
            loop.optimizer.error_fun if self.error_fun is None else self.error_fun
        )
        phase_entry = None
        if self.reset_on_scheduler_phase_entry and isinstance(
            loop.optimizer, OptimizerSchedule
        ):
            state = loop.optimizer.get_state()
            phase_entry = loop.optimizer, state.n_steps - state.n_phase_steps
        if (
            self._loop is not loop
            or self._error_fun != error_fun
            or self._phase_entry != phase_entry
        ):
            self._smallest = np.inf
            self._mark = np.inf
            self._since = loop.cycle
        self._loop = loop
        self._error_fun = error_fun
        self._phase_entry = phase_entry
        error = loop.get_error(error_fun)
        self._smallest = min(self._smallest, error)
        if error < self._mark / self.factor:
            self._mark = error
            self._since = loop.cycle
        if loop.cycle - self._since <= self.patience:
            return
        anchor = getattr(problem, "anchor", None)
        if self._smallest < self.near and anchor is not None and not self.anchored:
            # At the anchored solution the root functions need not be zero,
            # but the guess stops changing
            loop.x_next = loop.get_best_x(error_fun).copy()
            anchor()
            loop.optimizer.error_fun = "recursion"
            loop.clear_best_points()
            self.anchored = True
            logger.info(f"cycle: {loop.cycle}, anchored")
            event = "optimizer.anchored"
        else:
            self.restarts += 1
            logger.info(f"cycle: {loop.cycle}, restarted")
            event = "optimizer.restarted"
        loop.optimizer.reset_active_optimizer(
            clear_norm_history=event == "optimizer.anchored"
        )
        self._smallest = np.inf
        self._mark = np.inf
        self._since = loop.cycle
        loop.emit(event, reason="plateau")
