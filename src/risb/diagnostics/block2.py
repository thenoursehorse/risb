"""Diagnostics for the ground-state block2 impurity solver."""

import logging

from .block2_output import Block2Output


class Block2Diagnostics(Block2Output):
    """Report ground-state block2 events."""

    def emit(self, event: str, *, rank: int, size: int, **fields: object) -> None:
        """Report a ground-state impurity event."""
        level = (
            logging.INFO
            if event.startswith("solve.")
            or event in {"configuration", "sector.selected", "ground_state.summary"}
            else logging.DEBUG
        )
        if event.endswith(".failed"):
            level = logging.ERROR
        self.write(event, level, rank=rank, size=size, **fields)
