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

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, factorial, hermite

from risb.types import BlockRealArray, SmearFunction

from ._from_triqs_hartree import fermi, update_mu


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
    method : {'fermi', 'gaussian', 'methfessel-paxton'}, optional
        Smearing method.

    """

    def __init__(
        self,
        beta: float,
        mu: float | None = None,
        n_target: float | None = None,
        method: Literal["fermi", "gaussian", "methfessel-paxton"] = "fermi",
    ) -> None:
        #: float : Inverse temperature
        self.beta = beta

        #: float : Chemical potential.
        self.mu = mu

        #: float : Target lattice filling per unit cell.
        self.n_target = n_target

        #: risb.types.BlockRealArray : Energies in each band at each k-point.
        self.energies: BlockRealArray

        #: risb.types.BlockRealArray : Integration weights at each k-point in the same shape as energies.
        self.weights: BlockRealArray

        #: int : Number of k-points.
        self.n_k: int

        #: risb.types.SmearFunction : Smearing function called as
        #: ``smear_function(energies, beta, mu)``.
        self.smear_function: SmearFunction
        smear_functions: dict[str, SmearFunction] = {
            "fermi": self._fermi,
            "gaussian": self._gaussian,
            "methfessel-paxton": self._methfessel_paxton,
        }
        if method not in smear_functions:
            msg = "Unrecognized smearing function !"
            raise ValueError(msg)
        self.smear_function = smear_functions[method]

    @staticmethod
    def _fermi(
        energies: NDArray[np.float64], beta: float, mu: float
    ) -> NDArray[np.float64]:
        e = energies - mu
        return fermi(e, beta)

    @staticmethod
    def _gaussian(
        energies: NDArray[np.float64], beta: float, mu: float
    ) -> NDArray[np.float64]:
        occupation: NDArray[np.float64] = 0.5 * erfc(beta * (energies - mu))
        return occupation

    @staticmethod
    def _methfessel_paxton(
        energies: NDArray[np.float64], beta: float, mu: float, N: int = 1
    ) -> NDArray[np.float64]:
        def A_n(n: int) -> float:
            return float((-1) ** n / (factorial(n) * 4**n * np.sqrt(np.pi)))

        x = beta * (energies - mu)

        S: NDArray[np.float64] = 0.5 * erfc(x)  # S_0
        for n in range(1, N + 1):
            H_n = hermite(2 * n - 1)
            S += A_n(n) * H_n(x) * np.exp(-(x**2))
        return S

    def _update_n_k(self) -> int:
        first_key = next(iter(self.energies))
        self.n_k = self.energies[first_key].shape[0]
        for en in self.energies.values():
            if self.n_k != en.shape[0]:
                # FIXME Must they? I don't see why, but its weird to not
                msg = "Blocks must be on the same sized grid !"
                raise ValueError(msg)
        return self.n_k

    def update_weights(self, energies: BlockRealArray) -> BlockRealArray:
        """
        Update the integral weighting factors at each k-point.

        Parameters
        ----------
        energies : risb.types.BlockRealArray
            Energies at each k-point. Each key in dictionary is a different
            symmetry block, and its associated value is a list of energies.

        Returns
        -------
        risb.types.BlockRealArray
            Integration weights at each k-point in each band.

        """
        self.energies = energies
        self._update_n_k()
        if self.n_target is not None:
            self.mu = update_mu(
                self.n_target, self.energies, self.beta, self.n_k, self.smear_function
            )
        elif self.mu is None:
            msg = "One of mu or n_target must be provided !"
            raise ValueError(msg)

        self.weights = {}
        for bl in self.energies:
            self.weights[bl] = (
                self.smear_function(self.energies[bl], self.beta, self.mu) / self.n_k
            )
        return self.weights
