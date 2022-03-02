################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2016-2018, N. Wentzell
# Copyright (C) 2018-2019, The Simons Foundation
#   author: N. Wentzell
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

r"""
DOC

"""
from triqs.operators import Operator
from .embedding_atom_diag_module import EmbeddingAtomDiagReal, EmbeddingAtomDiagComplex

# Construct real/complex (literally copied from triqs.atom_diag)
def EmbeddingAtomDiag(*args, **kwargs):
    """Wrapper to triqs.atom_diag to solve H_emb"""
    if 'is_complex' in kwargs: is_complex = True
    else: is_complex = False

    if is_complex :
        return EmbeddingAtomDiagComplex(*args, **kwargs)
    else:
        return EmbeddingAtomDiagReal(*args, **kwargs)


__all__ = ['EmbeddingAtomDiag']
