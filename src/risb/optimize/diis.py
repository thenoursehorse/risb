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

"""Pulay mixing and safeguarded Anderson acceleration."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, override

import numpy as np
import scipy
from numpy.typing import NDArray

from risb.types import Checkpoint, FixedPointFunction

from .solver_newton import NewtonSolver


@dataclass
class Pulay:
    """
    Periodic Pulay mixing.

    Parameters
    ----------
    alpha : float, optional
        Fraction of the extrapolated fixed-point correction to take.
    mixing_period : int or None, optional
        Use linear mixing every this many steps, or never if None.
    restart_period : int or None, optional
        Clear the subspace every this many steps, or never if None.
    error_fun : {'root', 'recursion'}, optional
        Minimize the supplied root residual or the change ``g(x) - x``.

    """

    alpha: float = 1.0
    mixing_period: int | None = 3
    restart_period: int | None = None
    error_fun: Literal["root", "recursion"] = "root"

    can_reject = False
    max_candidate_evaluations = 1

    def update_x(
        self,
        x: Sequence[NDArray[np.float64]],
        g_x: Sequence[NDArray[np.float64]],
        error: Sequence[NDArray[np.float64]],
        iteration: int,
    ) -> NDArray[np.float64]:
        """Mix the latest point or extrapolate from the subspace."""
        if self.mixing_period is not None and (iteration + 1) % self.mixing_period == 0:
            return x[0] + self.alpha * (g_x[0] - x[0])
        return DIIS.extrapolate(x, g_x, error, self.alpha)


@dataclass
class BianLi:
    """
    Bian-Li Algorithm 1, using the recursion residual ``g(x) - x``.

    Accept an Anderson step when its residual norm is at most
    ``gamma * max(history norms)``. Otherwise take a damped fixed-point
    step from the last accepted point and restart the subspace.

    Parameters
    ----------
    schedule : {'iteration', 'residual', 'constant', 'combined'}, optional
        With step index k, largest history norm W and initial norm W0:
        ``iteration`` uses ``1 - 1/(k+2)``;
        ``residual`` uses ``1 - W/(2*(W0+W))``;
        ``constant`` uses `gamma`; and
        ``combined`` uses ``1 - max(1/(k+2), W/(2*W0))``.
    fallback_mixing : float, optional
        Mixing fraction lambda for the fallback step. Use less than one
        for the paper's nonexpansive case.
    gamma : float, optional
        Acceptance factor for the constant schedule.

    Notes
    -----
    See Bian and Li, https://arxiv.org/abs/2609.28950.
    The convergence results require assumptions on the fixed-point map;
    these have not been established for RISB. The combined rule raises
    ValueError if its required residual bound fails.

    """

    schedule: Literal["iteration", "residual", "constant", "combined"] = "iteration"
    fallback_mixing: float = 0.5
    gamma: float = 0.5

    can_reject = True
    max_candidate_evaluations = 2

    def __post_init__(self) -> None:
        """Check the schedule and mixing fraction."""
        if self.schedule not in {"iteration", "residual", "constant", "combined"}:
            msg = f"Unknown Bian-Li schedule: {self.schedule}"
            raise ValueError(msg)
        if not 0 < self.fallback_mixing <= 1:
            msg = "fallback_mixing must be in (0, 1]."
            raise ValueError(msg)
        if not 0 < self.gamma < 1:
            msg = "gamma must be in (0, 1)."
            raise ValueError(msg)

    def acceptance_factor(
        self, iteration: int, window_norm: float, initial_norm: float
    ) -> float:
        """Return the paper's gamma for this step."""
        if self.schedule == "iteration":
            gamma = 1 - 1 / (iteration + 2)
        elif self.schedule == "residual":
            gamma = 1 - window_norm / (2 * (initial_norm + window_norm))
        elif self.schedule == "constant":
            gamma = self.gamma
        else:
            gamma = 1 - max(1 / (iteration + 2), window_norm / (2 * initial_norm))
        if not 0 < gamma < 1:
            msg = "Bian-Li gamma is outside (0, 1); the schedule's assumptions failed."
            raise ValueError(msg)
        return gamma

    def update_x(
        self,
        x: Sequence[NDArray[np.float64]],
        g_x: Sequence[NDArray[np.float64]],
        error: Sequence[NDArray[np.float64]],
        _iteration: int,
    ) -> NDArray[np.float64]:
        """Return the unregularized Anderson step."""
        return DIIS.extrapolate(x, g_x, error)


class DIIS(NewtonSolver):
    """
    Direct inversion in the iterative subspace.

    Parameters
    ----------
    history_size : int, optional
        Maximum number of accepted points in the subspace. This is the
        paper's memory parameter m plus one.
    policy : Pulay or BianLi, optional
        Mixing, acceptance and restart rules. Defaults to Pulay.

    Notes
    -----
    With recursion residuals this is Anderson acceleration: algorithm 2 in
    https://doi.org/10.1051/m2an/2021069. A history of one gives linear mixing.

    """

    def __init__(
        self, history_size: int = 6, *, policy: Pulay | BianLi | None = None
    ) -> None:
        self.policy = Pulay() if policy is None else policy
        super().__init__(history_size=history_size, error_fun=self.error_fun)
        self._initial_norm = 0.0

    @property  # type: ignore[explicit-override]
    @override
    def error_fun(self) -> Literal["root", "recursion"]:
        """Residual chosen by the policy."""
        return self.policy.error_fun if isinstance(self.policy, Pulay) else "recursion"

    @error_fun.setter
    @override
    def error_fun(self, value: Literal["root", "recursion"]) -> None:
        if value not in {"root", "recursion"}:
            msg = f"Unknown error_fun: {value}"
            raise ValueError(msg)
        if isinstance(self.policy, Pulay):
            self.policy.error_fun = value
        elif value != "recursion":
            msg = "BianLi uses the recursion residual."
            raise ValueError(msg)

    @override
    @property
    def can_reject(self) -> bool:
        """Whether a step can reject an evaluated point."""
        return self.policy.can_reject

    @override
    @property
    def max_candidate_evaluations(self) -> int:
        """Maximum candidate evaluations needed to complete a step."""
        return self.policy.max_candidate_evaluations

    @staticmethod
    def extrapolate(
        x: Sequence[NDArray[np.float64]],
        g_x: Sequence[NDArray[np.float64]],
        error: Sequence[NDArray[np.float64]],
        alpha: float = 1.0,
    ) -> NDArray[np.float64]:
        """Minimize the combined residual with coefficients that sum to one."""
        m = len(error)
        B = np.array([[np.dot(a, b) for b in error] for a in error])
        if np.max(np.abs(B)) > 0:
            B = B / np.max(np.abs(B))
        B = np.column_stack((B, -np.ones(m)))
        B = np.vstack((B, -np.ones(m + 1)))
        B[m, m] = 0.0
        rhs = np.zeros(m + 1)
        rhs[-1] = -1.0
        c = scipy.linalg.pinv(B) @ rhs
        x_bar = np.zeros(x[0].shape)
        g_bar = np.zeros(x[0].shape)
        for i in range(m):
            x_bar += c[i] * x[i]
            g_bar += c[i] * g_x[i]
        return x_bar + alpha * (g_bar - x_bar)

    @override
    def update_x(self) -> NDArray[np.float64]:
        """Return the next guess using the policy."""
        return self.policy.update_x(self.x, self.g_x, self.error, self.n)

    @override
    def step(
        self,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
        cycle: FixedPointFunction,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """Accept the evaluated point, or restore and evaluate the Bian-Li fallback."""
        extra_cycles = 0
        if isinstance(self.policy, BianLi):
            if self.n == 0:
                self._initial_norm = float(np.linalg.norm(g_x - x))
            if self.x:
                window = max(float(np.linalg.norm(e)) for e in self.error)
                gamma = (
                    self.policy.acceptance_factor(
                        self.n - 1, window, self._initial_norm
                    )
                    if window > 0
                    else 0.5
                )
                if not np.linalg.norm(g_x - x) <= gamma * window:
                    x = self.x[0] + self.policy.fallback_mixing * (
                        self.g_x[0] - self.x[0]
                    )
                    if checkpoint is not None:
                        checkpoint.restore()
                        extra_cycles += checkpoint.evaluations
                    g_x, residual = cycle(x)
                    extra_cycles += 1
                    self.x = []
                    self.g_x = []
                    self.error = []
        x, g_x, residual, x_next, _ = super().step(x, g_x, residual, cycle)
        if (
            isinstance(self.policy, Pulay)
            and self.policy.restart_period is not None
            and self.n % self.policy.restart_period == 0
        ):
            self.x = []
            self.g_x = []
            self.error = []
        return x, g_x, residual, x_next, extra_cycles
