"""Smearing functions for band integrals."""

from functools import cache

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, eval_hermite, expit, factorial

from risb.types import SmearFunction


def fermi_function(
    eigenvalues: NDArray[np.float64], beta: float, mu: float
) -> NDArray[np.float64]:
    """Return the Fermi function at inverse temperature beta and chemical potential mu."""
    if np.isinf(beta):
        return np.heaviside(mu - eigenvalues, 0.5)
    return expit(-beta * (eigenvalues - mu))


def gaussian_function(
    eigenvalues: NDArray[np.float64], beta: float, mu: float
) -> NDArray[np.float64]:
    """Return the integrated Gaussian with width 1/beta, centred at mu."""
    if np.isinf(beta):
        return np.heaviside(mu - eigenvalues, 0.5)
    return 0.5 * erfc(beta * (eigenvalues - mu))


def methfessel_paxton_function(
    eigenvalues: NDArray[np.float64], beta: float, mu: float, order: int = 1
) -> NDArray[np.float64]:
    """Return Methfessel-Paxton weights of the given order (Phys. Rev. B 40, 3616)."""
    if np.isinf(beta):
        return np.heaviside(mu - eigenvalues, 0.5)
    x = beta * (eigenvalues - mu)
    result = 0.5 * erfc(x)
    for n in range(1, order + 1):
        coefficient = (-1) ** n / (factorial(n) * 4**n * np.sqrt(np.pi))
        result += coefficient * eval_hermite(2 * n - 1, x) * np.exp(-(x**2))
    return result


@cache
def max_occupation_slope(smearing: SmearFunction, beta: float) -> float:
    """
    Return the largest ``|df/de|`` of the occupation function ``f(e) = smearing(e, beta, 0)``.

    The derivative is taken numerically for ``|e| < 10 / beta``.

    Parameters
    ----------
    smearing : callable
        Occupation function f(eigenvalues, beta, mu).
    beta : float
        Inverse temperature, or inverse smearing width.

    Returns
    -------
    float
        Infinite for ``beta = inf``.

    """
    if np.isinf(beta):
        return np.inf
    energies = np.linspace(-10 / beta, 10 / beta, 20001)
    slopes = np.gradient(smearing(energies, beta, 0.0), energies)
    return float(np.max(np.abs(slopes)))
