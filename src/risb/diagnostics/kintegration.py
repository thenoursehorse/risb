"""Diagnostics for k-space integration and temperature extrapolation."""

import logging
from types import TracebackType
from typing import Self

from risb.helpers_mpi import Comm, SerialComm

from .output import DiagnosticOutput, Output


class KIntegrationDiagnostics:
    """Report k-integrals, error estimates and temperature extrapolation.

    Parameters
    ----------
    level : str or None, optional
        "info" reports results; "debug" also reports timings, beta samples
        and chemical-potential searches. None disables reporting.
    output : str, pathlib.Path or typing.TextIO, optional
        Append to this file or stream. None uses standard error.
    comm : risb.helpers_mpi.Comm, optional
        MPI communicator. File paths gain .rankN; summaries use rank 0.

    Notes
    -----
    Attach this reporter to the k-integrator's diagnostics argument.
    Reporters can be shared, but are not passed to wrapped k-integrators
    automatically.

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
        self._calls: dict[str, int] = {}

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
        """Report one k-integration event."""
        operation = event.rsplit(".", 1)[0]
        if event.endswith(".started"):
            self._calls[operation] = self._calls.get(operation, 0) + 1
        level = logging.INFO if event.endswith(".result") else logging.DEBUG
        if event.endswith(".failed"):
            level = logging.ERROR
        elif fields.get("converged") is False:
            level = logging.WARNING
        if self._comm.rank == 0 or level != logging.INFO:
            self._output.write(
                event,
                level,
                rank=self._comm.rank,
                size=self._comm.size,
                call=self._calls.get(operation, 0),
                **fields,
            )
