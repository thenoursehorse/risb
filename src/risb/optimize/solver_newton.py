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
from copy import deepcopy
from typing import Concatenate, Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class NewtonSolver(ABC):
    """
    Base class for quasi-Newton methods to find the root of a function.

    Use :meth:`step` to take one iteration at a time, as
    :func:`risb.optimize.drive` does, or :meth:`solve` to run until it
    converges.

    Parameters
    ----------
    history_size : int, optional
        Maximum size of subspace.
    n_restart : int, optional
        Fully reset subspace after this many iterations. Defaults to never.
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
        n_restart: int | None = None,
        error_fun: Literal["recursion", "root"] = "recursion",
    ) -> None:
        self.history_size = history_size
        self.n_restart = n_restart
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

    @staticmethod
    def _insert_vector(
        vec: list[NDArray[np.float64]],
        vec_new: NDArray[np.float64],
        max_size: int | None = None,
    ) -> None:
        # Note these operations are mutable on input list
        vec.insert(0, vec_new)
        if max_size is not None and len(vec) > max_size:
            vec.pop()

    @abstractmethod
    def update_x(self) -> NDArray[np.float64]:
        """
        Return a single iteration for the new guess for :attr:`x`.

        Parameters of the method, such as a mixing step, are set when the
        solver is constructed.

        Returns
        -------
        numpy.ndarray
            New guess for x to add to history.

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
    ) -> NDArray[np.float64]:
        """
        Add an iteration to the history and return the next guess.

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
            The next guess for x.

        """
        error = self.error_of(x, g_x, residual)
        if self.history_size > 0:
            self._insert_vector(self.x, x, self.history_size)
            self._insert_vector(self.g_x, g_x, self.history_size)
            self._insert_vector(self.error, error, self.history_size)
        x_next = self.update_x()
        if self.n_restart is not None and (self.n + 1) % self.n_restart == 0:
            self.x = []
            self.g_x = []
            self.error = []
        self.n += 1
        return x_next

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
    ) -> NDArray[np.float64]:
        """
        Find the root of a function.

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
            Maximum number of iterations.

        Returns
        -------
        numpy.ndarray
            Root of ``fun``.

        """
        self.reset()
        x = deepcopy(x0)
        for _ in range(maxiter):
            g_x, residual = fun(x, *args)
            self.norm = float(np.linalg.norm(self.error_of(x, g_x, residual)))
            logger.info(f"n: {self.n}, error: {self.norm}")
            if self.norm < tol:
                self.success = True
                break
            x = self.step(x, g_x, residual)
        return x
