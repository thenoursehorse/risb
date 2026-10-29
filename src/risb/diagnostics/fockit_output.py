"""Event interface and rank context for fockit diagnostics."""

from dataclasses import dataclass
from typing import Protocol


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
