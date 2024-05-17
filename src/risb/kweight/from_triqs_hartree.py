# noqa: D100
#
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

# Code below copied from github.com/TRIQS/hartree_fock


def fermi(e, beta):
    """
    Numerically stable version of the Fermi function.

    Parameters
    ----------
    e : float or ndarray
        Energy minus chemical potential
    beta: float
        Inverse temperature

    """
    return np.exp(-beta * e * (e > 0)) / (1 + np.exp(-beta * np.abs(e)))


def update_mu(n_target, energies, beta, n_k, smear_function):
    """
    Update the chemical potential using :func:`scipy.optimize.brentq` for smearing k-space integrations.

    Parameters
    ----------
    n_target : float
        Filling target
    energies : numpy.ndarray
        Array of eigenenergies
    beta : float
        Inverse temperature
    n_k : int
        Number of unit cells on lattice
    smear_function:
        The function that smears the energy at each k-point.

    """
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
