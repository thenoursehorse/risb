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

import numpy as np

class EmbeddingDummy:
    """
    Dummy impurity solver. Does not solve anything and instead references 
    variables of another embedding solver. This is useful when 
    some inequivalent clusters are the same up to rotations, so only 
    a single impurity problem has to be solved and then the single-particle 
    matrices are rotated into the relevant basis.

    Parameters
    ----------
    embedding : class
        The embedding solving class to reference.
    rotations : list[callable], optional
        A list of rotation functions to apply to matrices.
    """
    def __init__(self, embedding, rotations = []):
        self.embedding = embedding
        self.rotations = rotations
    
    def set_h_emb(self, *args, **kwargs):
        pass
    
    def solve(self, *args, **kwargs):
        pass

    def get_nf(self, bl : str) -> np.ndarray:
        nf = self.embedding.Nf[bl]
        for func in self.rotations:
            nf = func(nf)
        return nf
    
    def get_nc(self, bl : str) -> np.ndarray:
        nc = self.embedding.Nc[bl]
        for func in self.rotations:
            nc = func(nc)
        return nc
    
    def get_mcf(self, bl : str) -> np.ndarray:
        mcf = self.embedding.Mcf[bl]
        for func in self.rotations:
            mcf = func(mcf)
        return mcf
    
    def overlap(self, Op, force_real : bool = True) -> float | complex:
        return self.embedding.overlap(Op, force_real)