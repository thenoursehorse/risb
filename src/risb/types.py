"""Type aliases used across risb."""

import numpy as np
from numpy.typing import NDArray

type GfStruct = list[tuple[str, int]]
"""Block structure of one cluster: (block name, block size) pairs."""

type BlockMatrix = dict[str, NDArray[np.inexact]]
"""Real or complex array for each block, e.g. R, Lambda, densities, h0_k."""

type BlockRealArray = dict[str, NDArray[np.float64]]
"""Real array for each block, e.g. band energies, k-weights."""
