# Copyright (c) 2023 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Embedding solver that acts as a copy of another solver without calculating anything."""

from collections.abc import Callable, Sequence
from typing import override

import numpy as np
from numpy.typing import NDArray

from risb.types import BlockMatrix, BlockMatrixLike, Embedding


class EmbeddingDummy(Embedding):
    """
    Dummy impurity solver.

    Does not solve anything and instead references
    variables of another embedding solver. This is useful when
    some inequivalent clusters are the same up to rotations, so only
    a single impurity problem has to be solved and then the single-particle
    matrices are rotated into the relevant basis.

    Parameters
    ----------
    embedding : risb.types.Embedding
        The embedding solver to reference. It must be solved before this one.
    rotations : sequence of callable, optional
        Functions applied in order to the density matrices of ``embedding``.
        Each takes and returns a dictionary of block matrices with the
        blocks of ``embedding.gf_struct``.

    """

    def __init__(
        self,
        embedding: Embedding,
        rotations: Sequence[Callable[[BlockMatrix], BlockMatrix]] = (),
    ) -> None:
        self.embedding = embedding
        self.rotations = rotations
        self.gf_struct = embedding.gf_struct
        self.rho_f: BlockMatrix = {}
        self.rho_cf: BlockMatrix = {}
        self.rho_c: BlockMatrix = {}

    @override
    def set_h_emb(
        self,
        Lambda_c: BlockMatrixLike,
        D: BlockMatrixLike,
        h0_loc_matrix: BlockMatrixLike | None = None,
    ) -> None:
        """Do nothing, because the referenced embedding solver is the one solved."""

    @override
    def solve(self) -> None:
        """Copy the density matrices of the referenced embedding solver and apply :attr:`rotations`."""
        rho_f = dict(self.embedding.rho_f)
        rho_cf = dict(self.embedding.rho_cf)
        rho_c = dict(self.embedding.rho_c)
        for rotate in self.rotations:
            rho_f = rotate(rho_f)
            rho_cf = rotate(rho_cf)
            rho_c = rotate(rho_c)
        self.rho_f = rho_f
        self.rho_cf = rho_cf
        self.rho_c = rho_c

    @override
    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_f[bl]

    @override
    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_cf[bl]

    @override
    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        return self.rho_c[bl]
