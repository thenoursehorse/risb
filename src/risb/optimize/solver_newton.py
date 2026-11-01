# Copyright (c) 2021-2023 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Abstract base class for quasi-Newton methods."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Concatenate, Literal

import numpy as np
from numpy.typing import NDArray

from risb.types import Checkpoint, FixedPointFunction

logger = logging.getLogger(__name__)


class NewtonSolver(ABC):
    """
    Base class for quasi-Newton methods to find the root of a function.

    Use :meth:`step` to take one iteration at a time, as
    :func:`risb.optimize.solve_fixed_point` does, or :meth:`solve` to run until it
    converges.

    Parameters
    ----------
    history_size : int, optional
        Maximum size of subspace.
    error_fun : {'recursion', 'root'}, optional
        The error of an iteration: the change ``g_x - x`` of the fixed-point
        function (``'recursion'``), or the residual of the root functions
        (``'root'``). Default is ``'recursion'``.

    Notes
    -----
    :meth:`update_x` must be defined in the inherited class.

    """

    def __init__(
        self,
        history_size: int = 6,
        error_fun: Literal["recursion", "root"] = "recursion",
    ) -> None:
        if history_size < 1:
            msg = "history_size must be positive."
            raise ValueError(msg)
        self.history_size = history_size
        if error_fun not in {"recursion", "root"}:
            msg = f"Unrecognized error_fun {error_fun} !"
            raise ValueError(msg)

        #: str : The error to make zero, ``'recursion'`` (``g_x - x``) or
        #: ``'root'`` (the residual of the root functions).
        self.error_fun = error_fun

        #: list of numpy.ndarray : History of guesses to the root problem.
        self.x: list[NDArray[np.float64]] = []

        #: list of numpy.ndarray : History of fixed point function with ``x`` as the input.
        self.g_x: list[NDArray[np.float64]] = []

        #: list of numpy.ndarray : History of error vector of ``x``.
        self.error: list[NDArray[np.float64]] = []

        #: int : Iteration counter for solver.
        self.n: int = 0

        #: bool : Whether the solver converged to within tolerance.
        self.success: bool = False

        # float : 2-norm of :attr:`error`.
        self.norm: float = np.inf

    @property
    def error_fun(self) -> Literal["recursion", "root"]:
        """Residual used for extrapolation and stopping."""
        return self._error_fun

    @error_fun.setter
    def error_fun(self, value: Literal["recursion", "root"]) -> None:
        self._error_fun = value

    @property
    def can_reject(self) -> bool:
        """Whether the method can reject a guess after evaluating it."""
        return False

    @property
    def max_candidate_evaluations(self) -> int:
        """Maximum evaluations per step, excluding checkpoint restoration."""
        return 1

    @staticmethod
    def _insert_vector(
        vec: list[NDArray[np.float64]],
        vec_new: NDArray[np.float64],
        max_size: int | None = None,
    ) -> None:
        vec.insert(0, vec_new)
        if max_size is not None and len(vec) > max_size:
            vec.pop()

    @abstractmethod
    def update_x(self) -> NDArray[np.float64]:
        """
        Calculate the next guess from the stored history.

        Parameters of the method, such as a mixing step, are set when the
        solver is constructed.

        Returns
        -------
        numpy.ndarray
            Next guess, not yet evaluated.

        """

    def reset(self) -> None:
        """Clear the history, so that the next :meth:`step` starts afresh."""
        self.x = []
        self.g_x = []
        self.error = []
        self.n = 0
        self.success = False
        self.norm = np.inf

    def error_of(
        self,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Return the error vector of an iteration, as chosen by :attr:`error_fun`.

        Parameters
        ----------
        x : numpy.ndarray
            The input of the iteration.
        g_x : numpy.ndarray
            The fixed-point function of ``x``.
        residual : numpy.ndarray
            The residual of the root functions at ``x``.

        Returns
        -------
        numpy.ndarray
            ``g_x - x`` or ``residual``.

        """
        if self.error_fun == "root":
            return residual
        error: NDArray[np.float64] = g_x - x
        return error

    def step(
        self,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
        cycle: FixedPointFunction,  # noqa: ARG002
        *,
        checkpoint: Checkpoint | None = None,  # noqa: ARG002
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """
        Accept an evaluated point and calculate the next guess.

        Parameters
        ----------
        x, g_x, residual : numpy.ndarray
            Input x, its fixed-point output g_x, and the root residual at x.
        cycle : callable
            Evaluate a fallback guess if this guess is rejected.
        checkpoint : risb.types.Checkpoint, optional
            State to restore before evaluating a fallback guess.

        Returns
        -------
        x, g_x, residual : numpy.ndarray
            Accepted guess, its fixed-point output, and its root residual.
        x_next : numpy.ndarray
            Next guess, not yet evaluated.
        extra_cycles : int
            Extra evaluations needed to restore a checkpoint and evaluate a fallback.

        """
        error = self.error_of(x, g_x, residual)
        self._insert_vector(self.x, x.copy(), self.history_size)
        self._insert_vector(self.g_x, g_x.copy(), self.history_size)
        self._insert_vector(self.error, error.copy(), self.history_size)
        x_next = self.update_x()
        self.n += 1
        return x, g_x, residual, x_next, 0

    def solve(
        self,
        fun: Callable[
            Concatenate[NDArray[np.float64], ...],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ],
        x0: NDArray[np.float64],
        args: tuple[object, ...] = (),
        tol: float = 1e-12,
        maxiter: int = 1000,
        *,
        checkpoint: Callable[[], Checkpoint] | None = None,
    ) -> NDArray[np.float64]:
        """
        Find the root of a function.

        Reuse the stored history. Call :meth:`reset` to clear it.

        Parameters
        ----------
        fun : callable
            Called as ``fun(x, *args)``, it returns ``(g_x, residual)``: the
            fixed-point function and the residual of the root functions at
            ``x``.
        x0 : numpy.ndarray
            Initial guess of the parameters.
        args : tuple, optional
            Additional arguments to pass to ``fun``.
        tol : float, optional
            Stop when the 2-norm of the error (see :attr:`error_fun`) is below
            this.
        maxiter : int, optional
            Maximum number of function evaluations. This includes evaluations
            of rejected guesses and any extra calls needed to restore a checkpoint.
        checkpoint : callable, optional
            Save the state of the last accepted calculation. Supply this if
            the function uses state other than x and the optimizer can reject steps.

        Returns
        -------
        numpy.ndarray
            Last accepted guess, even if the evaluation limit was reached.

        """
        self.success = False
        x = x0.copy()
        accepted_x = self.x[0].copy() if self.x else x.copy()
        cycles = 0

        def cycle(
            value: NDArray[np.float64],
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return fun(value, *args)

        while cycles < maxiter:
            saved = (
                checkpoint()
                if self.can_reject and self.x and checkpoint is not None
                else None
            )
            needed = self.max_candidate_evaluations if self.x else 1
            if saved is not None:
                needed += (needed - 1) * saved.evaluations
            if cycles + needed > maxiter:
                break
            g_x, residual = cycle(x)
            accepted_x, g_x, residual, x, extra = self.step(
                x, g_x, residual, cycle, checkpoint=saved
            )
            cycles += 1 + extra
            self.norm = float(np.linalg.norm(self.error_of(accepted_x, g_x, residual)))
            logger.info(f"n: {self.n}, error: {self.norm}")
            if self.norm < tol:
                self.success = True
                break
        return accepted_x
