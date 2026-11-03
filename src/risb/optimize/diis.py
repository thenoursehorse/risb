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
from typing import Literal, Protocol, override

import numpy as np
import scipy
from numpy.typing import NDArray

from risb.types import Checkpoint, FixedPointFunction

from .solver_newton import NewtonSolver, OptimizerHistory


class DIISPolicy(Protocol):
    """DIIS mixing, acceptance and history rules."""

    error_fun: Literal["root", "recursion"]
    can_reject: bool
    max_candidate_evaluations: int

    def update_x(
        self,
        x: Sequence[NDArray[np.float64]],
        g_x: Sequence[NDArray[np.float64]],
        error: Sequence[NDArray[np.float64]],
        iteration: int,
        /,
    ) -> NDArray[np.float64]:
        """Calculate the next iterate from accepted history."""
        ...

    def step(
        self,
        optimizer: NewtonSolver,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
        cycle: FixedPointFunction,
        *,
        initial_norm: float,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """Accept a trial or evaluate a replacement, then calculate the next iterate."""
        ...


@dataclass
class Pulay:
    """
    Periodic Pulay mixing.

    Parameters
    ----------
    alpha : float, optional
        Fraction of the extrapolated fixed-point correction to take.
    mixing_period : int or None, optional
        Take a linear mixing step every ``mixing_period`` steps.
        None disables periodic linear mixing.
    restart_period : int or None, optional
        Clear the history every ``restart_period`` steps.
        None disables scheduled restarts.
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
        return extrapolate(x, g_x, error, self.alpha)

    def step(
        self,
        optimizer: NewtonSolver,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
        cycle: FixedPointFunction,
        *,
        initial_norm: float,  # noqa: ARG002
        checkpoint: Checkpoint | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """Accept the trial and apply the periodic history reset."""
        result = NewtonSolver.step(
            optimizer, x, g_x, residual, cycle, checkpoint=checkpoint
        )
        if self.restart_period is not None and optimizer.n % self.restart_period == 0:
            optimizer.clear_history()
        return result


@dataclass
class BianLi:
    """
    Bian-Li Algorithm 1, using the recursion residual ``g(x) - x``.

    Accept an Anderson guess if the norm of ``g(x) - x`` is at most gamma
    times the largest residual norm in the current history. Otherwise,
    restore the last accepted state, take a damped fixed-point step, and
    clear the history.

    Parameters
    ----------
    schedule : {'iteration', 'residual', 'constant', 'combined'}, optional
        For step index k, let W be the largest residual norm in the current
        history and W0 the initial residual norm:
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

    @property
    def error_fun(self) -> Literal["root", "recursion"]:
        """Residual required by Bian-Li acceptance."""
        return "recursion"

    @error_fun.setter
    def error_fun(self, value: Literal["root", "recursion"]) -> None:
        if value != "recursion":
            msg = "BianLi uses the recursion residual."
            raise ValueError(msg)

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
        """Return the factor used to accept or reject the Anderson guess."""
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
        """Calculate the next guess with Anderson acceleration."""
        return extrapolate(x, g_x, error)

    def step(
        self,
        optimizer: NewtonSolver,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
        cycle: FixedPointFunction,
        *,
        initial_norm: float,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
    ]:
        """Accept the trial or restore the last accepted state and take a damped step."""
        extra_cycles = 0
        if optimizer.x:
            window = max(float(np.linalg.norm(e)) for e in optimizer.error)
            gamma = (
                self.acceptance_factor(optimizer.n - 1, window, initial_norm)
                if window > 0
                else 0.5
            )
            if not np.linalg.norm(g_x - x) <= gamma * window:
                x = optimizer.x[0] + self.fallback_mixing * (
                    optimizer.g_x[0] - optimizer.x[0]
                )
                if checkpoint is not None:
                    checkpoint.restore()
                    extra_cycles += checkpoint.evaluations
                g_x, residual = cycle(x)
                extra_cycles += 1
                optimizer.clear_history()
        x, g_x, residual, x_next, _ = NewtonSolver.step(
            optimizer, x, g_x, residual, cycle, checkpoint=checkpoint
        )
        return x, g_x, residual, x_next, extra_cycles


class DIIS(NewtonSolver):
    """
    Direct inversion in the iterative subspace.

    Parameters
    ----------
    history_size : int, optional
        Maximum number of accepted guesses kept in the history.
        For BianLi, this equals the paper's memory parameter m plus one.
    policy : DIISPolicy, optional
        Rules for mixing guesses, rejecting steps and clearing history.
        None selects Pulay.

    Notes
    -----
    With recursion residuals this is Anderson acceleration: algorithm 2 in
    https://doi.org/10.1051/m2an/2021069. A history of one gives linear mixing.

    """

    def __init__(
        self, history_size: int = 6, *, policy: DIISPolicy | None = None
    ) -> None:
        self.policy: DIISPolicy = Pulay() if policy is None else policy
        super().__init__(history_size=history_size, error_fun=self.error_fun)
        self._initial_norm = 0.0

    @property  # type: ignore[explicit-override]
    @override
    def error_fun(self) -> Literal["root", "recursion"]:
        """Residual chosen by the policy."""
        return self.policy.error_fun

    @error_fun.setter
    @override
    def error_fun(self, value: Literal["root", "recursion"]) -> None:
        if value not in {"root", "recursion"}:
            msg = f"Unknown error_fun: {value}"
            raise ValueError(msg)
        self.policy.error_fun = value

    @override
    @property
    def can_reject(self) -> bool:
        """Whether the policy can reject a guess after evaluating it."""
        return self.policy.can_reject

    @override
    @property
    def max_candidate_evaluations(self) -> int:
        """Maximum evaluations per step, excluding checkpoint restoration."""
        return self.policy.max_candidate_evaluations

    @staticmethod
    def extrapolate(
        x: Sequence[NDArray[np.float64]],
        g_x: Sequence[NDArray[np.float64]],
        error: Sequence[NDArray[np.float64]],
        alpha: float = 1.0,
    ) -> NDArray[np.float64]:
        """Return the DIIS guess using coefficients that minimize the residual."""
        return extrapolate(x, g_x, error, alpha)

    @override
    def set_history(self, history: OptimizerHistory) -> None:
        """Load history, using the newest iterate as the initial reference."""
        super().set_history(history)
        self._initial_norm = self.norm if self.x else 0.0

    @override
    def update_x(self) -> NDArray[np.float64]:
        """Return the next guess using the policy."""
        return self.policy.update_x(self.x, self.g_x, self.error, self.n - 1)

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
        """Apply the policy's acceptance and history rules."""
        if self.n == 0:
            self._initial_norm = float(np.linalg.norm(self.error_of(x, g_x, residual)))
        return self.policy.step(
            self,
            x,
            g_x,
            residual,
            cycle,
            initial_norm=self._initial_norm,
            checkpoint=checkpoint,
        )


def extrapolate(
    x: Sequence[NDArray[np.float64]],
    g_x: Sequence[NDArray[np.float64]],
    error: Sequence[NDArray[np.float64]],
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Extrapolate an iterate by minimizing the norm of the combined residual."""
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
