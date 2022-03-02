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
from .functions_module import EigSystemReal, EigSystemComplex
from .functions_module import get_embedding_space, get_h_emb
from .functions_module import get_h_qp, get_disp_R
from .functions_module import get_ke, get_pdensity
from .functions_module import get_d, get_lambda_c, get_r, get_lambda
from .functions_module import get_pdensity_gf, get_ke_gf
from .functions_module import get_sigma_z, get_delta_z

def EigSystem(*args, **kwargs):
    """"""
    vec = args[-1]
    if any(abs(vec.imag) > 0):
        return EigSystemComplex(*args, **kwargs)
    else:
        return EigSystemReal(*args, **kwargs)

__all__ = ['EigSystem','get_embedding_space','get_h_qp','get_disp_R','get_ke','get_pdensity','get_d','get_lambda_c','get_r','get_lambda','get_h_emb','get_pdensity_gf','get_ke_gf','get_sigma_z','get_delta_z']
