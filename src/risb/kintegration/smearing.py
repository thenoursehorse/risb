"""Smearing functions for band integrals."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, eval_hermite, expit, factorial


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
