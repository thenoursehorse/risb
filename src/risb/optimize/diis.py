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

"""DIIS optimizers."""

import numpy as np
import scipy
from numpy.typing import ArrayLike

from .solver_newton import NewtonSolver


class DIIS(NewtonSolver):
    """
    Direct inversion in the iterative subspace to minimize a function.

    Parameters
    ----------
    n_period : int, optional
        Take a single linear mixing step afer this many iterations. Default
        round(history_size / 2).

    Notes
    -----
    Algorithm 2 (Anderson type) in `10.1051/m2an/2021069 <hal-02492983v5_>`__.

    .. _hal-02492983v5: https://doi.org/10.1051/m2an/2021069

    """

    def __init__(self, /, n_period: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)

        if n_period == 0:
            self.n_period = int(np.round(self.history_size / 2.0))
        else:
            self.n_period = n_period

        self.solve = super().solve

    @staticmethod
    def extrapolate(
        x: list[ArrayLike], g_x: list[ArrayLike], error: list[ArrayLike]
    ) -> np.ndarray:
        """DIIS extrapolation algorithm for the new guess for :attr:`x`."""
        # Construct the B matrix
        m = len(error)
        B = np.empty(shape=(m, m))
        for i in range(m):
            for j in range(m):
                B[i, j] = np.dot(error[i], error[j])

        # Add the constraint lambda
        B = np.column_stack((B, -np.ones(B.shape[0])))
        B = np.vstack((B, -np.ones(B.shape[1])))
        B[m, m] = 0.0

        # Solve for the c coefficients (last element in c gives lambda constraint)
        rhs = np.zeros(B.shape[0])
        rhs[-1] = -1.0

        c = np.dot(scipy.linalg.pinv(B), rhs)

        # Calculate optimal x(n)
        x_opt = np.zeros(x[0].shape)
        for i in range(m):
            x_opt += c[i] * g_x[i]
        return x_opt

    # x_i, error(x_i), g(x_i) where g(x_i) is the fixed-point function
    # that gives a new x_i
    def update_x(self, alpha: float = 1.0) -> np.ndarray:
        """Return a new guess for the vector x."""
        if (self.n + 1) % self.n_period == 0:
            # Do linear mixing
            x_opt = self.x[0] + alpha * (self.g_x[0] - self.x[0])
        else:
            # Do DIIS
            x_opt = self.extrapolate(self.x, self.g_x, self.error)

        return x_opt
