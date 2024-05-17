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
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


# TODO fix up verbose messages
class NewtonSolver(ABC):
    """
    Base class for quasi-Newton methods to find the root of a function.

    Parameters
    ----------
    history_size : int, optional
        Maximum size of subspace.
    n_restart : int, optional
        Fully reset subspace after this many iterations.

    Notes
    -----
    :meth:`update_x` must be defined in the inherited class.

    """

    def __init__(self, history_size: int = 6, n_restart: float = np.inf) -> None:
        self.history_size = history_size
        self.n_restart = n_restart
        self.initialized = False

        #: list[numpy.ndarray] : History of guesses to the root problem.
        self.x: list[ArrayLike] = []

        #: list[numpy.ndarray] : History of fixed point function with ``x`` as the input.
        self.g_x: list[ArrayLike] = []

        #: list[numpy.ndarray] : History of error vector of ``x``.
        self.error: list[ArrayLike] = []

        #: int : Iteration counter for solver.
        self.n: int = 0

        #: bool : Whether the solver converged to within tolerance.
        self.success: bool = False

        # float : 2-norm of :attr:`error`.
        self.norm: float = np.inf

    @staticmethod
    def _load_history(
        x: list[ArrayLike], error: list[ArrayLike], max_size: int
    ) -> tuple[list[ArrayLike], list[ArrayLike]]:
        if (len(x) != len(error)) and (len(x) != (len(error) + 1)):
            msg = "x and error are the wrong lengths !"
            raise ValueError(msg)

        x_out = deepcopy(x)
        error_out = deepcopy(error)

        while len(x_out) >= max_size:
            x_out.pop()
        while len(error_out) >= max_size:
            error_out.pop()

        return x_out, error_out

    @staticmethod
    def _insert_vector(
        vec: list[ArrayLike], vec_new: ArrayLike, max_size: int | None = None
    ) -> None:
        # Note these operations are mutable on input list
        vec.insert(0, vec_new)
        if max_size is not None and len(vec) >= max_size:
            vec.pop()

    @abstractmethod
    def update_x(self, **kwargs) -> np.ndarray:
        """
        Return a single iteration for the new guess for :attr:`x`.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specific to the solver implementation.

        Returns
        -------
        numpy.ndarray
            New guess for x to add to history.

        """

    def solve(
        self,
        fun: Callable[..., ArrayLike],
        x0: ArrayLike,
        args: tuple[Any, ...] = (),
        tol: float = 1e-12,
        maxiter: int = 1000,
        options: dict | None = None,
    ) -> ArrayLike:
        """
        Find the root of a function. It is called similarly to :func:scipy.optimize.root.

        Parameters
        ----------
        fun : callable
            The function to find the root of. It must be callable as
            ``fun(x, *args)``.
        x0 : numpy.ndarray
            Initial guess of the parameters. This does not neccessarily have to
            be flattened, but it usually is.
        args : tuple, optional
            Additional arguments to pass to ``fun``.
        tol : float, optional
            The tolerance. When the 2-norm difference of the return of ``fun``
            is less than this, the solver stops.
        maxiter : int, optional
            Maximum number of iterations.
        options : dict, optional
            keyword arguments to pass to :meth:`update_x`.

        Returns
        -------
        numpy.ndarray
            Root of ``fun``.

        """
        if options is None:
            options = {}
        self.success = False
        x = deepcopy(x0)
        if self.history_size > 0:
            self._insert_vector(self.x, x, self.history_size)

        for self.n in range(maxiter):
            g_x, error = fun(x, *args)

            if self.history_size > 0:
                self._insert_vector(self.g_x, g_x, self.history_size)
                self._insert_vector(self.error, error, self.history_size)

            self.norm = np.linalg.norm(error)
            logger.info(f"n: {self.n}, rms(risb): {self.norm}")
            if self.norm < tol:
                self.success = True
                break

            x = self.update_x(**options)

            if (self.n % self.n_restart) == 0:
                self.x = []
                self.g_x = []
                self.error = []

            if self.history_size > 0:
                self._insert_vector(self.x, x, self.history_size)

        if self.success:
            logger.info(f"The solution converged. nit: {self.n}, tol: {self.norm}")
        else:
            logger.info(
                f"The solution did NOT converge. nit: {self.n} tol: {self.norm}"
            )

        return x
