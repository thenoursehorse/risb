"""Output and timing for passive diagnostic events."""

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import perf_counter
from typing import Protocol, TextIO

type Output = str | Path | TextIO | None


class DiagnosticObserver(Protocol):
    """Receive event names and measurements."""

    def emit(self, event: str, **fields: object) -> None:
        """Record an event."""
        ...


class DiagnosticOutput:
    """Write events to an owned file or a supplied stream."""

    def __init__(self, level: str | None = "info", output: Output = None) -> None:
        self._level = None if level is None else level.upper()
        self._output = output
        self._stack = ExitStack()
        self._loggers: dict[tuple[int, int], logging.Logger] = {}

    def close(self) -> None:
        """Close owned files; leave supplied streams open."""
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.flush()
                handler.close()
        self._loggers.clear()
        self._stack.close()

    def write(
        self, event: str, level: int, *, rank: int = 0, size: int = 1, **fields: object
    ) -> None:
        """Write one event at its reporting level."""
        if self._level is None:
            return
        key = rank, size
        if key not in self._loggers:
            output = self._output
            stream: TextIO
            if isinstance(output, (str, Path)):
                path = Path(output)
                if size > 1:
                    path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
                stream = self._stack.enter_context(path.open("a", encoding="utf-8"))
            else:
                stream = sys.stderr if output is None else output
                if stream is sys.stdout or stream is sys.stderr:
                    # Preserve the destination during native stream capture.
                    try:
                        descriptor = stream.fileno()
                    except AttributeError, OSError:
                        pass
                    else:
                        stream = self._stack.enter_context(
                            os.fdopen(os.dup(descriptor), "w", encoding="utf-8")
                        )
            # Each instance owns its handlers.
            logger = logging.Logger(__name__, self._level)  # noqa: LOG001
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(handler)
            self._loggers[key] = logger
        logger = self._loggers[key]
        if logger.isEnabledFor(level):
            logger.log(
                level,
                "rank=%d/%d %s %s",
                rank,
                size,
                event,
                " ".join(f"{key}={value}" for key, value in fields.items()),
            )


@contextmanager
def stage(
    observer: DiagnosticObserver | None, name: str, **fields: object
) -> Iterator[None]:
    """Report the start, duration and failure of a stage."""
    if observer is None:
        yield
        return
    observer.emit(f"{name}.started", **fields)
    start = perf_counter()
    try:
        yield
    except Exception as error:
        observer.emit(
            f"{name}.failed",
            **fields,
            error=type(error).__name__,
            seconds=perf_counter() - start,
        )
        raise
    observer.emit(f"{name}.finished", **fields, seconds=perf_counter() - start)
