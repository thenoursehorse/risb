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

"""k-space integrator on a grid using smearing functions."""

import numpy as np
from numpy.typing import ArrayLike

from .from_triqs_hartree import fermi, update_mu


class SmearingKWeight:
    """
    Obtain weights for k-space integrals using smearing functions.

    Parameters
    ----------
    beta : float
        Inverse temperature
    mu : float or None, optional
        Chemical potential. One of :attr:`mu` or :attr:`n_target` needs to be provided.
    n_target : float or None, optional
        Target lattice filling per unit cell. One of :attr:`mu` or :attr:`n_target`
        needs to be provided.
    method : str, 'fermi' | 'gaussian' | 'methfessel-paxton'
        Smearing method.

    """

    def __init__(
        self,
        beta: float,
        mu: float | None = None,
        n_target: float | None = None,
        method: str = "fermi",
    ) -> None:
        #: float : Inverse temperature
        self.beta = beta

        #: float : Chemical potential.
        self.mu = mu

        #: float : Target lattice filling per unit cell.
        self.n_target = n_target

        #: dict[numpy.ndarray] : Energies in each band at each k-point.
        self.energies: dict[ArrayLike]

        #: dict[numpy.ndarray] : Integration weights at each k-point in the same shape as energies.
        self.weight: dict[ArrayLike]

        #: int : Number of k-points.
        self.n_k: int

        if method == "fermi":
            self.smear_function = self._fermi
        elif method == "gaussian":
            self.smear_function = self._gaussian
        elif method == "methfessel-paxton":
            self.smear_function = self._methfessel_paxton
        else:
            msg = "Unrecognized smearing function !"
            raise ValueError(msg)

    @staticmethod
    def _fermi(energies: ArrayLike, beta: float, mu: float) -> ArrayLike:
        e = energies - mu
        return fermi(e, beta)

    @staticmethod
    def _gaussian(energies: ArrayLike, beta: float, mu: float) -> ArrayLike:
        from scipy.special import erfc

        return 0.5 * erfc(beta * (energies - mu))

    @staticmethod
    def _methfessel_paxton(
        energies: ArrayLike, beta: float, mu: float, N: int = 1
    ) -> ArrayLike:
        from scipy.special import erfc, factorial, hermite

        def A_n(n):
            return (-1) ** n / (factorial(n) * 4**n * np.sqrt(np.pi))

        x = beta * (energies - mu)

        S = 0.5 * erfc(x)  # S_0
        for n in range(1, N + 1):
            H_n = hermite(2 * n - 1)
            S += A_n(n) * H_n(x) * np.exp(-(x**2))
        return S

    def _update_n_k(self) -> int:
        if isinstance(self.energies, dict):
            first_key = next(iter(self.energies))
            self.n_k = self.energies[first_key].shape[0]
            for en in self.energies.values():
                if self.n_k != en.shape[0]:
                    # FIXME Must they? I don't see why, but its weird to not
                    msg = "Blocks must be on the same sized grid !"
                    raise ValueError(msg)
        else:
            self.n_k = self.energies.shape[0]
        return self.n_k

    def update_weights(self, energies: dict[ArrayLike]) -> dict[ArrayLike]:
        """
        Update the integral weighting factors at each k-point.

        Parameters
        ----------
        energies : dict[numpy.ndarray]
            Energies at each k-point. Each key in dictionary is a different
            symmetry block, and its associated value is a list of energies.

        Returns
        -------
        dict[numpy.ndarray]
            Integration weights at each k-point in each band.

        """
        self.energies = energies
        self._update_n_k()
        if self.n_target is not None:
            self.mu = update_mu(
                self.n_target, self.energies, self.beta, self.n_k, self.smear_function
            )

        self.weights = {}
        for bl in self.energies:
            self.weights[bl] = (
                self.smear_function(self.energies[bl], self.beta, self.mu) / self.n_k
            )
        return self.weights
