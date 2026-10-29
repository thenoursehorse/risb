"""Diagnostics for the thermal block2 impurity solver."""

import logging

from .block2_output import Block2Output


class Block2ThermalDiagnostics(Block2Output):
    """Report thermal block2 events."""

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Report a thermal impurity event."""
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event in {"configuration", "thermal.summary"}
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        self.write(event, level, rank=rank, size=size, **fields)
