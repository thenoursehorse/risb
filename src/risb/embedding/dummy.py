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

import numpy as np


class EmbeddingDummy:
    """
    Dummy impurity solver.

    Does not solve anything and instead references
    variables of another embedding solver. This is useful when
    some inequivalent clusters are the same up to rotations, so only
    a single impurity problem has to be solved and then the single-particle
    matrices are rotated into the relevant basis.

    Parameters
    ----------
    embedding : class
        The embedding solver class to reference.
    rotations : list[callable], optional
        A list of rotation functions to apply to matrices.

    """

    def __init__(self, embedding, rotations=None):
        if rotations is None:
            rotations = []
        self.embedding = embedding
        self.rotations = rotations

    def set_h_emb(self, *args, **kwargs):  # noqa: D102
        pass

    def solve(self, *args, **kwargs):  # noqa: D102
        pass

    def get_rho_f(self, bl: str) -> np.ndarray:  # noqa: D102
        if bl not in self.embedding.rho_f:
            rho_f = self.embedding.get_rho_f(bl)
        else:
            rho_f = self.embedding.rho_f[bl]
        for func in self.rotations:
            rho_f = func(rho_f)
        return rho_f

    def get_rho_c(self, bl: str) -> np.ndarray:  # noqa: D102
        if bl not in self.embedding.rho_c:
            rho_c = self.embedding.get_rho_c(bl)
        else:
            rho_c = self.embedding.rho_c[bl]
        for func in self.rotations:
            rho_c = func(rho_c)
        return rho_c

    def get_rho_cf(self, bl: str) -> np.ndarray:  # noqa: D102
        if bl not in self.embedding.rho_cf:
            rho_cf = self.embedding.get_rho_cf(bl)
        else:
            rho_cf = self.embedding.rho_cf[bl]
        for func in self.rotations:
            rho_cf = func(rho_cf)
        return rho_cf

    # FIXME rotate Op correctly, is this even correctly possible?
    def overlap(self, Op, force_real: bool = True) -> float | complex:  # noqa: D102
        return self.embedding.overlap(Op, force_real)
