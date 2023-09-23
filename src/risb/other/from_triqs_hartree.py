# Copyright (c) 2022 Simons Foundation
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
# Authors: Jonathan Karp, Alexander Hampel, Nils Wentzell, Hugo U. R. Strand, Olivier Parcollet

import numpy as np
from scipy.optimize import brentq

# Cdoe below copy/derived from github.com/TRIQS/hartree_fock

def fermi(e, beta):
    """
    Numerically stable version of the Fermi function

    Parameters
    ----------
    e : float or ndarray
        Energy minus chemical potential

    beta: float
        Inverse temperature

    """
    return np.exp(-beta * e * (e > 0))/(1 + np.exp(-beta*np.abs(e)))

def update_mu(n_target, energies, beta, n_k, smear_function):
    e_min = np.inf
    e_max = -np.inf
    for en in energies.values():
        bl_min = en.min()
        bl_max = en.max()
        if bl_min < e_min:
            e_min = bl_min
        if bl_max > e_max:
            e_max = bl_max
            
    def target_function(mu):
        n = 0
        for en in energies.values():
            n += np.sum(smear_function(en, beta, mu)) / n_k
        return n - n_target
    return brentq(target_function, e_min, e_max)

def flatten(Lambda, R, is_real):
    if len(R) != len(Lambda):
        raise ValueError(f'len(R) = {len(R)} and len(Lambda) = {len(Lambda)} must have the same number of clusters !')
    n_clusters = len(Lambda)
    x = []
    for i in range(n_clusters):
        x = np.append(x, [mat.flatten().real for mat in Lambda[i].values()])
        if is_real:
            x = np.append(x, [mat.flatten().real for mat in R[i].values()])
        else:
            x = np.append(x, [mat.flatten().view(float) for mat in R[i].values()])
    return x
    
def unflatten(x, gf_struct, is_real):
    n_clusters = len(gf_struct)
    Lambda = [dict() for i in range(n_clusters)]
    R = [dict() for i in range(n_clusters)]
    offset = 0
    for i in range(n_clusters):
        for bl, bl_size in gf_struct[i]:
            Lambda[i][bl] = x[list(range(offset, offset + bl_size**2))].reshape(bl_size, bl_size)
            offset += bl_size**2
        for bl, bl_size in gf_struct[i]:
            if is_real:
                R[i][bl] = x[list(range(offset, offset + bl_size**2))].reshape(bl_size, bl_size)
                offset += bl_size**2
            else:
                R[i][bl] = x[list(range(offset, offset + 2*bl_size**2))].view(complex).reshape(bl_size, bl_size)
                offset += 2*bl_size**2
    # FIXME check offest = len(x)
    return Lambda, R