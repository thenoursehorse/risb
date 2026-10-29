"""Optional reporting and output capture for the block2 impurity solvers."""

import ctypes
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Concatenate, Protocol, Self, TextIO

from risb.helpers_mpi import Comm

from .output import DiagnosticOutput, Output


class Block2Observer(Protocol):
    """Reporting and backend capture accepted by the block2 solvers."""

    @property
    def iprint(self) -> int:
        """Native block2 verbosity."""
        ...

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Receive an event and its measurements."""
        ...

    def capture(
        self, stage: str, *, rank: int, size: int
    ) -> AbstractContextManager[None]:
        """Route backend output for the duration of a stage."""
        ...


class _Observed(Protocol):
    diagnostics: Block2Observer | None
    comm: Comm


class Block2Diagnostics:
    """
    Report solver events and route native block2 output independently.

    Parameters
    ----------
    level : str or None, optional
        Python logging level, e.g. ``"info"`` or ``"debug"``. None disables
        event reporting. Default is ``"info"``.
    risb_output : str, pathlib.Path or typing.TextIO, optional
        Destination for solver events. None uses standard error.
    iprint : int, optional
        Native block2 verbosity: 0 is quiet, 1 reports sweeps, 2 reports
        iterations within sweeps. Default is 0.
    block2_output : str, pathlib.Path or typing.TextIO, optional
        Destination for native output. None keeps the standard streams.

    Notes
    -----
    Use a context manager or :meth:`close` for file cleanup. Files append;
    supplied streams stay open. MPI files gain ``.rankN`` before the suffix.
    Capture changes process-wide streams; calls in one process must be serial.
    Native output to streams without a file descriptor is copied after the stage.

    """

    def __init__(
        self,
        level: str | None = "info",
        *,
        risb_output: Output = None,
        iprint: int = 0,
        block2_output: Output = None,
    ) -> None:
        self.iprint = iprint
        self._report = DiagnosticOutput(level, risb_output)
        self._outputs = {"risb": risb_output, "block2": block2_output}
        self._stack = ExitStack()
        self._streams: dict[tuple[str, int, int], TextIO] = {}
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
        self._report.close()
        self._stack.close()
        self._streams.clear()

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Report an event to this instance's destination."""
        if event == "solve.started":
            self._solve += 1
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event
            in {
                "configuration",
                "sector.selected",
                "ground_state.summary",
                "thermal.summary",
            }
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        self._report.write(
            event, level, rank=rank, size=size, solve=self._solve, **fields
        )

    @contextmanager
    def capture(self, stage: str, *, rank: int, size: int) -> Iterator[None]:
        """Route Python and native backend output."""
        if not self.iprint or self._outputs["block2"] is None:
            yield
            return
        stream = self._stream("block2", rank, size)
        stream.write(f"\nsolve={self._solve} rank={rank}/{size} stage={stage}\n")
        stream.flush()
        with ExitStack() as buffers:
            target = stream
            try:
                target.fileno()
            except AttributeError, OSError:
                target = buffers.enter_context(
                    tempfile.TemporaryFile(mode="w+", encoding="utf-8")
                )
            sys.stdout.flush()
            sys.stderr.flush()
            libc = ctypes.CDLL(None)
            libc.fflush(None)
            try:
                with ExitStack() as redirects:
                    for descriptor in (1, 2):
                        saved = os.dup(descriptor)
                        redirects.callback(os.close, saved)
                        os.dup2(target.fileno(), descriptor)
                        redirects.callback(os.dup2, saved, descriptor)
                    with redirect_stdout(target), redirect_stderr(target):
                        try:
                            yield
                        finally:
                            target.flush()
                            libc.fflush(None)
            finally:
                if target is not stream:
                    target.seek(0)
                    stream.write(target.read())
                stream.flush()

    def _stream(self, channel: str, rank: int, size: int) -> TextIO:
        key = channel, rank, size
        if key not in self._streams:
            output = self._outputs[channel]
            stream: TextIO
            if isinstance(output, (str, Path)):
                path = Path(output)
                if size > 1:
                    path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
                stream = self._stack.enter_context(path.open("a", encoding="utf-8"))
            else:
                stream = sys.stderr if output is None else output
            self._streams[key] = stream
        return self._streams[key]


def diagnostic_stage[S: _Observed, **P, R](
    method: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """Report stage start, duration and failure."""

    @wraps(method)
    def wrapped(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
        diagnostics = self.diagnostics
        if diagnostics is None:
            return method(self, *args, **kwargs)
        stage = method.__name__.lstrip("_")
        context = {"rank": self.comm.rank, "size": self.comm.size}
        diagnostics.emit(f"{stage}.started", **context, solver=type(self).__name__)
        start = perf_counter()
        try:
            result = method(self, *args, **kwargs)
        except Exception as error:
            diagnostics.emit(
                f"{stage}.failed",
                **context,
                error=type(error).__name__,
                elapsed=perf_counter() - start,
            )
            raise
        diagnostics.emit(f"{stage}.finished", **context, elapsed=perf_counter() - start)
        return result

    return wrapped


def capture_block2[S: _Observed, **P, R](
    method: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """Route output around a backend operation."""

    @wraps(method)
    def wrapped(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.diagnostics is None:
            return method(self, *args, **kwargs)
        with self.diagnostics.capture(
            method.__name__.lstrip("_"), rank=self.comm.rank, size=self.comm.size
        ):
            return method(self, *args, **kwargs)

    return wrapped
