"""Diagnostics for the lattice solvers."""

import logging
from types import TracebackType
from typing import Self

from risb.helpers_mpi import Comm, SerialComm

from .output import DiagnosticOutput, Output


class LatticeDiagnostics:
    """
    Report zero-temperature lattice progress.

    Parameters
    ----------
    level : str or None, optional
        "info" reports progress and results; "debug" adds stages and details.
        None disables reporting.
    output : str, pathlib.Path or typing.TextIO, optional
        Append to this file or stream. None uses standard error.
    comm : risb.helpers_mpi.Comm, optional
        MPI communicator. File paths gain .rankN; summaries use rank 0.

    """

    def __init__(
        self,
        level: str | None = "info",
        *,
        output: Output = None,
        comm: Comm | None = None,
    ) -> None:
        self._output = DiagnosticOutput(level, output)
        self._comm = SerialComm() if comm is None else comm
        self._solve = 0
        self._cycle = 0

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

    def emit(self, event: str, **fields: object) -> None:
        """Report one solver event."""
        if event == "solve.started":
            self._solve += 1
            self._cycle = 0
        if event == "cycle.started":
            self._cycle += 1
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event
            in {
                "configuration",
                "cycle.summary",
                "optimizer.cycle",
                "optimizer.evaluation",
                "optimizer.finished",
                "optimizer.restarted",
            }
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        if self._comm.rank == 0 or level != logging.INFO:
            self._output.write(
                event,
                level,
                rank=self._comm.rank,
                size=self._comm.size,
                solve=self._solve,
                **({"cycle": self._cycle} | fields),
            )


class ThermalLatticeDiagnostics:
    """
    Report thermal lattice progress and fitting results.

    Parameters
    ----------
    level : str or None, optional
        "info" reports progress and results; "debug" adds stages and details.
        None disables reporting.
    output : str, pathlib.Path or typing.TextIO, optional
        Append to this file or stream. None uses standard error.
    comm : risb.helpers_mpi.Comm, optional
        MPI communicator. File paths gain .rankN; summaries use rank 0.

    """

    def __init__(
        self,
        level: str | None = "info",
        *,
        output: Output = None,
        comm: Comm | None = None,
    ) -> None:
        self._output = DiagnosticOutput(level, output)
        self._comm = SerialComm() if comm is None else comm
        self._solve = 0
        self._cycle = 0

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

    def emit(self, event: str, **fields: object) -> None:
        """Report one solver event."""
        if event == "solve.started":
            self._solve += 1
            self._cycle = 0
        if event == "cycle.started":
            self._cycle += 1
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event
            in {
                "configuration",
                "cycle.summary",
                "optimizer.cycle",
                "optimizer.evaluation",
                "optimizer.finished",
                "optimizer.restarted",
                "thermal.filling",
                "optimizer.anchored",
                "anchor.set",
                "anchor.released",
            }
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        if self._comm.rank == 0 or level != logging.INFO:
            self._output.write(
                event,
                level,
                rank=self._comm.rank,
                size=self._comm.size,
                solve=self._solve,
                **({"cycle": self._cycle} | fields),
            )
