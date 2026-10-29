"""Diagnostics for the fockit impurity solvers."""

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from .output import DiagnosticOutput, Output


class FockitObserver(Protocol):
    """Reporting accepted by the fockit solvers."""

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Receive an event and its measurements."""
        ...


@dataclass(frozen=True)
class FockitContext:
    """Supply rank information to the timing helper."""

    observer: FockitObserver
    rank: int
    size: int

    def emit(self, event: str, **fields: object) -> None:
        """Forward an event with rank information."""
        self.observer.emit(event, rank=self.rank, size=self.size, **fields)


class FockitDiagnostics:
    """
    Report fockit impurity progress.

    Parameters
    ----------
    level : str or None, optional
        "info" reports results; "debug" adds construction and solve details.
        None disables reporting.
    output : str, pathlib.Path or typing.TextIO, optional
        Append to this file or stream. None uses standard error.

    """

    def __init__(self, level: str | None = "info", *, output: Output = None) -> None:
        self._output = DiagnosticOutput(level, output)
        self._solve = 0

    def __enter__(self) -> Self:
        """Return this reporter."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close owned files."""
        self.close()

    def close(self) -> None:
        """Close owned files; leave supplied streams open."""
        self._output.close()

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Report one impurity event."""
        if event == "solve.started":
            self._solve += 1
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event in {"configuration", "ground_state.summary"}
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        if rank == 0 or level != logging.INFO:
            self._output.write(
                event, level, rank=rank, size=size, solve=self._solve, **fields
            )


class FockitThermalDiagnostics:
    """
    Report thermal fockit impurity progress.

    Parameters
    ----------
    level : str or None, optional
        "info" reports results; "debug" adds construction and solve details.
        None disables reporting.
    output : str, pathlib.Path or typing.TextIO, optional
        Append to this file or stream. None uses standard error.

    """

    def __init__(self, level: str | None = "info", *, output: Output = None) -> None:
        self._output = DiagnosticOutput(level, output)
        self._solve = 0

    def __enter__(self) -> Self:
        """Return this reporter."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close owned files."""
        self.close()

    def close(self) -> None:
        """Close owned files; leave supplied streams open."""
        self._output.close()

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Report one impurity event."""
        if event == "solve.started":
            self._solve += 1
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event in {"configuration", "thermal.summary"}
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        if rank == 0 or level != logging.INFO:
            self._output.write(
                event, level, rank=rank, size=size, solve=self._solve, **fields
            )
