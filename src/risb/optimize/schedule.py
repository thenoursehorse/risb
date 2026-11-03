"""Schedules that switch between fixed-point optimizers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
from numpy.typing import NDArray

from risb.diagnostics import DiagnosticObserver
from risb.types import Checkpoint, FixedPointFunction

from .solver_newton import NewtonSolver, OptimizerHistory

type ScheduleCondition = Callable[[SchedulerState], bool]


@dataclass(frozen=True)
class SchedulerState:
    """Iteration counts and histories of residual norms (oldest first)."""

    scheduler_phase_index: int
    n_steps: int
    n_evaluations: int
    n_phase_steps: int
    n_phase_evaluations: int
    error_fun: Literal["root", "recursion"]
    root_norms: tuple[float, ...]
    recursion_norms: tuple[float, ...]
    phase_root_norms: tuple[float, ...]
    phase_recursion_norms: tuple[float, ...]


@dataclass
class SchedulerPhase:
    """One optimizer stage, with a switching condition and history choice."""

    optimizer: NewtonSolver
    until: ScheduleCondition | None = field(default=None, kw_only=True)
    history: Literal["carry", "clear"] = field(default="carry", kw_only=True)
    next_phase: int | Callable[[SchedulerState], int] | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        """Check history selection and terminal phases."""
        if self.history not in {"carry", "clear"}:
            msg = f"Unknown history selection: {self.history}"
            raise ValueError(msg)
        if self.until is None and self.next_phase is not None:
            msg = "next_phase requires an exit condition."
            raise ValueError(msg)


class OptimizerSchedule(NewtonSolver):
    """
    Switch optimizers during a fixed-point iteration.

    Parameters
    ----------
    phases : sequence of SchedulerPhase
        Stages of the iteration. An exit condition selects the next stage,
        unless next_phase chooses another. Without an exit condition, the
        optimizer runs until the solve stops.

    """

    def __init__(self, phases: Sequence[SchedulerPhase]) -> None:
        if not phases:
            msg = "An optimizer schedule needs at least one phase."
            raise ValueError(msg)
        self._phases = tuple(phases)
        for index, phase in enumerate(self._phases):
            target = phase.next_phase
            if isinstance(target, int) and not 0 <= target < len(self._phases):
                msg = f"Scheduler phase index out of range: {target}"
                raise ValueError(msg)
            if phase.until is not None and target is None and index == len(phases) - 1:
                msg = (
                    "The last phase needs an explicit next_phase or no exit condition."
                )
                raise ValueError(msg)
        self._scheduler_phase_index = 0
        self._transition: tuple[int, int] | None = None
        self._n_evaluations = 0
        self._n_phase_steps = 0
        self._n_phase_evaluations = 0
        self._root_norms: list[float] = []
        self._recursion_norms: list[float] = []
        self._phase_start_index = 0
        active = self.active_optimizer
        super().__init__(history_size=active.history_size, error_fun=active.error_fun)
        active.set_history(active.get_history())
        self._sync_history()
        self.norm = active.norm
        if self.x:
            self._record_norms(self.x[0], self.g_x[0], self.residual[0])

    @property
    def active_optimizer(self) -> NewtonSolver:
        """Optimizer that generates the next guess."""
        return self._phases[self._scheduler_phase_index].optimizer

    @property  # type: ignore[explicit-override]
    @override
    def error_fun(self) -> Literal["root", "recursion"]:
        """Residual selected by the active optimizer."""
        return self.active_optimizer.error_fun

    @error_fun.setter
    @override
    def error_fun(self, value: Literal["root", "recursion"]) -> None:
        self.active_optimizer.error_fun = value

    @override
    @property
    def can_reject(self) -> bool:
        """Whether the active optimizer can reject a trial iterate."""
        return self.active_optimizer.can_reject

    @override
    @property
    def max_candidate_evaluations(self) -> int:
        """Maximum trial evaluations per step, excluding state restoration."""
        return self.active_optimizer.max_candidate_evaluations

    @override
    def error_of(
        self,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the error using the active optimizer's residual."""
        return self.active_optimizer.error_of(x, g_x, residual)

    @override
    def step_fits_budget(
        self,
        maxiter: int,
        cycles_used: int,
        *,
        checkpoint: Checkpoint | None = None,
    ) -> bool:
        """Check whether the next optimizer step fits within maxiter."""
        return self.active_optimizer.step_fits_budget(
            maxiter, cycles_used, checkpoint=checkpoint
        )

    def get_state(self) -> SchedulerState:
        """Return counters and norm histories for the schedule and current stage."""
        root_norms = tuple(self._root_norms)
        recursion_norms = tuple(self._recursion_norms)
        return SchedulerState(
            scheduler_phase_index=self._scheduler_phase_index,
            n_steps=self.n,
            n_evaluations=self._n_evaluations,
            n_phase_steps=self._n_phase_steps,
            n_phase_evaluations=self._n_phase_evaluations,
            error_fun=self.error_fun,
            root_norms=root_norms,
            recursion_norms=recursion_norms,
            phase_root_norms=root_norms[self._phase_start_index :],
            phase_recursion_norms=recursion_norms[self._phase_start_index :],
        )

    @override
    def get_history(self) -> OptimizerHistory:
        """Copy the active optimizer's accepted iterates."""
        return self.active_optimizer.get_history()

    @override
    def set_history(self, history: OptimizerHistory) -> None:
        """Reset the schedule and load history into its first optimizer."""
        self.reset()
        active = self.active_optimizer
        active.set_history(history)
        self._sync_history()
        self.norm = active.norm
        if self.x:
            self._record_norms(self.x[0], self.g_x[0], self.residual[0])

    @override
    def clear_history(self) -> None:
        """Clear the active optimizer's history, keeping scheduler progress."""
        self.active_optimizer.clear_history()
        self._sync_history()

    @override
    def reset_active_optimizer(self, *, clear_norm_history: bool = False) -> None:
        """
        Reset the active optimizer, keeping the current stage and counters.

        Set clear_norm_history to discard residual norms after changing the
        fixed-point map.
        """
        self.active_optimizer.reset_active_optimizer(
            clear_norm_history=clear_norm_history
        )
        if clear_norm_history:
            self._root_norms = []
            self._recursion_norms = []
            self._phase_start_index = 0
        self._sync_history()
        self.success = False
        self.norm = np.inf

    @override
    def report_step(
        self,
        observer: DiagnosticObserver,
        *,
        error: float,
        converged: bool,
    ) -> None:
        """Report the active optimizer and any switch."""
        if self._transition is not None:
            source, target = self._transition
            observer.emit(
                "optimizer.switched",
                from_scheduler_phase=source,
                to_scheduler_phase=target,
                history=self._phases[target].history,
                optimizer=type(self.active_optimizer).__name__,
                error_fun=self.error_fun,
            )
            self._transition = None
        observer.emit(
            "optimizer.cycle",
            scheduler_phase_index=self._scheduler_phase_index,
            optimizer=type(self.active_optimizer).__name__,
            error=error,
            error_fun=self.error_fun,
            converged=converged,
        )

    @override
    def reset(self) -> None:
        """Reset every optimizer and start the schedule again."""
        for phase in self._phases:
            phase.optimizer.reset()
        self._scheduler_phase_index = 0
        super().reset()
        self._transition = None
        self._n_evaluations = 0
        self._n_phase_steps = 0
        self._n_phase_evaluations = 0
        self._root_norms = []
        self._recursion_norms = []
        self._phase_start_index = 0

    @override
    def update_x(self) -> NDArray[np.float64]:
        """Calculate the next iterate with the active optimizer."""
        x_next = self.active_optimizer.update_x()
        self._sync_history()
        return x_next

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
        """Accept or reject a trial, then choose the optimizer for the next iterate."""
        self._transition = None
        outgoing = self.active_optimizer
        x, g_x, residual, x_next, extra = outgoing.step(
            x, g_x, residual, cycle, checkpoint=checkpoint
        )
        self.n += 1
        self._n_evaluations += 1 + extra
        self._n_phase_steps += 1
        self._n_phase_evaluations += 1 + extra
        self._record_norms(x, g_x, residual)
        self._sync_history()
        phase = self._phases[self._scheduler_phase_index]
        state = self.get_state()
        if phase.until is not None and phase.until(state):
            target = phase.next_phase
            if target is None:
                target = self._scheduler_phase_index + 1
            elif callable(target):
                target = target(state)
            if not 0 <= target < len(self._phases):
                msg = f"Scheduler phase index out of range: {target}"
                raise ValueError(msg)
            history = (
                outgoing.get_history()
                if self._phases[target].history == "carry"
                else OptimizerHistory()
            )
            if not history.x:
                history = OptimizerHistory([x], [g_x], [residual])
            self._transition = self._scheduler_phase_index, target
            self._scheduler_phase_index = target
            self._n_phase_steps = 0
            self._n_phase_evaluations = 0
            self._phase_start_index = len(self._root_norms) - 1
            incoming = self.active_optimizer
            incoming.set_history(history)
            self._sync_history()
            x_next = self.update_x()
        return x, g_x, residual, x_next, extra

    def _sync_history(self) -> None:
        active = self.active_optimizer
        self.history_size = active.history_size
        self.x = active.x
        self.g_x = active.g_x
        self.error = active.error
        self.residual = active.residual

    def _record_norms(
        self,
        x: NDArray[np.float64],
        g_x: NDArray[np.float64],
        residual: NDArray[np.float64],
    ) -> None:
        self._root_norms.append(float(np.linalg.norm(residual)))
        self._recursion_norms.append(float(np.linalg.norm(g_x - x)))


def after_steps(
    n_steps: int, *, scope: Literal["phase", "solve"] = "phase"
) -> ScheduleCondition:
    """Switch after n_steps accepted iterations."""
    if n_steps < 1:
        msg = "n_steps must be positive."
        raise ValueError(msg)
    if scope not in {"phase", "solve"}:
        msg = f"Unknown progress scope: {scope}"
        raise ValueError(msg)

    def condition(state: SchedulerState) -> bool:
        count = state.n_phase_steps if scope == "phase" else state.n_steps
        return count >= n_steps

    return condition


def residual_below(
    threshold: float,
    *,
    error_fun: Literal["root", "recursion"] | None = None,
) -> ScheduleCondition:
    """Switch when the selected residual's 2-norm is below threshold."""

    def condition(state: SchedulerState) -> bool:
        norms = _norms(state, error_fun, "solve")
        return bool(norms and norms[-1] < threshold)

    return condition


def reduced_by(
    factor: float,
    *,
    error_fun: Literal["root", "recursion"] | None = None,
    scope: Literal["phase", "solve"] = "phase",
) -> ScheduleCondition:
    """Switch after the residual norm has decreased by factor."""
    if factor <= 1:
        msg = "factor must be greater than one."
        raise ValueError(msg)

    def condition(state: SchedulerState) -> bool:
        norms = _norms(state, error_fun, scope)
        return bool(norms and norms[-1] <= norms[0] / factor)

    return condition


def plateau(
    patience: int,
    *,
    factor: float = 10.0,
    error_fun: Literal["root", "recursion"] | None = None,
    scope: Literal["phase", "solve"] = "phase",
) -> ScheduleCondition:
    """Switch after patience iterations without a factor-fold error reduction."""
    if patience < 1:
        msg = "patience must be positive."
        raise ValueError(msg)
    if factor <= 1:
        msg = "factor must be greater than one."
        raise ValueError(msg)

    def condition(state: SchedulerState) -> bool:
        norms = _norms(state, error_fun, scope)
        mark = np.inf
        last_progress = 0
        for index, norm in enumerate(norms):
            if norm < mark / factor:
                mark = norm
                last_progress = index
        return len(norms) - 1 - last_progress >= patience

    return condition


def any_of(*conditions: ScheduleCondition) -> ScheduleCondition:
    """Switch when any supplied condition is met."""

    def condition(state: SchedulerState) -> bool:
        return any(check(state) for check in conditions)

    return condition


def all_of(*conditions: ScheduleCondition) -> ScheduleCondition:
    """Switch when every supplied condition is met."""

    def condition(state: SchedulerState) -> bool:
        return all(check(state) for check in conditions)

    return condition


def _norms(
    state: SchedulerState,
    error_fun: Literal["root", "recursion"] | None,
    scope: Literal["phase", "solve"],
) -> tuple[float, ...]:
    if scope not in {"phase", "solve"}:
        msg = f"Unknown progress scope: {scope}"
        raise ValueError(msg)
    if error_fun is None:
        error_fun = state.error_fun
    if error_fun not in {"root", "recursion"}:
        msg = f"Unknown error_fun: {error_fun}"
        raise ValueError(msg)
    if scope == "phase":
        return (
            state.phase_root_norms
            if error_fun == "root"
            else state.phase_recursion_norms
        )
    return state.root_norms if error_fun == "root" else state.recursion_norms
