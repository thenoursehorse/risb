"""Weighted k-point sums and integrals of band observables."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from risb.diagnostics.output import DiagnosticObserver, stage
from risb.types import BlockMatrix, BlockMatrixLike, BlockRealArray, SmearFunction

from .smearing import fermi_function

type KPoints = NDArray[np.float64]
type IntegralValues = dict[Hashable, NDArray[np.inexact]]
type KIntegrand = Callable[
    [NDArray[np.float64]], Mapping[Hashable, NDArray[np.inexact]]
]
type Hamiltonian = BlockMatrixLike | Callable[[NDArray[np.float64]], BlockMatrixLike]
type BandObservables = Callable[
    [NDArray[np.float64] | None, BlockMatrixLike],
    Mapping[Hashable, NDArray[np.inexact]],
]

_ELECTRON_COUNT = object()


@dataclass
class KIntegrationResult:
    """Results of a k-space sum or Brillouin-zone integral.

    The error estimate is None when the method does not provide one.
    For a supplied mesh, weights contains its normalized k-point weights.
    A successful sum does not establish convergence with respect to k-points.
    """

    values: IntegralValues
    error: float | None = None
    converged: bool = True
    n_evaluations: int = 0
    weights: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class KIntegralError:
    """Frobenius norm and largest absolute entry of a k-integral error estimate."""

    norm: float
    max_abs: float


@dataclass
class BandKIntegrationResult:
    """Brillouin-zone averages at the requested beta and mu.

    With a supplied mesh, density_k contains the density matrix at each
    k-point, and weights contains the normalized mesh weights.
    quadrature_error estimates the error of the k-integration method;
    temperature_error estimates the error of temperature extrapolation.
    temperature_errors gives the norm and maximum absolute entry of the
    extrapolation error for each requested matrix. An empty mapping means
    that per-matrix estimates are unavailable; None marks an unavailable
    scalar estimate.
    """

    values: IntegralValues
    mu: float
    beta: float
    n_electrons: float
    converged: bool = True
    quadrature_error: float | None = None
    temperature_error: float | None = None
    temperature_errors: dict[Hashable, KIntegralError] = field(default_factory=dict)
    betas: tuple[float, ...] = ()
    density_k: BlockMatrix | None = None
    energies: BlockRealArray = field(default_factory=dict)
    vectors: BlockMatrix = field(default_factory=dict)
    band_weights: BlockRealArray = field(default_factory=dict)
    weights: NDArray[np.float64] | None = None


class KIntegrator(Protocol):
    """Compute Brillouin-zone averages of scalars and matrices."""

    def integrate(
        self,
        integrand: KIntegrand | Mapping[Hashable, NDArray[np.inexact]],
    ) -> KIntegrationResult:
        """Sum or integrate each supplied quantity over k."""


class BandKIntegration(Protocol):
    """K-space integration of band observables at fixed mu or filling."""

    kpoint_integrator: KIntegrator

    def integrate(
        self,
        hamiltonian: Hamiltonian,
        *,
        observables: BandObservables | None = None,
        beta: float | None = None,
        mu: float | None = None,
        n_target: float | None = None,
    ) -> BandKIntegrationResult:
        """Return Brillouin-zone averages for the supplied Bloch Hamiltonian."""


class WeightedKPointIntegrator:
    """Evaluate k-space sums on a supplied mesh.

    Parameters
    ----------
    kpoints : numpy.ndarray, optional
        K-point coordinates with shape (n_k, dimension). Supply these when
        the integrand is a function rather than an array on the mesh.
    weights : sequence of float, optional
        Nonnegative k-point weights. They are normalized to sum to one;
        omitting them gives equal weights.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics.

    """

    def __init__(
        self,
        kpoints: KPoints | None = None,
        weights: Sequence[float] | NDArray[np.float64] | None = None,
        *,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        self.kpoints = None if kpoints is None else np.asarray(kpoints, dtype=float)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and (
            self.weights.ndim != 1
            or not np.all(np.isfinite(self.weights))
            or np.any(self.weights < 0)
            or self.weights.sum() <= 0
        ):
            msg = "k-point weights must be finite, nonnegative and have a positive sum."
            raise ValueError(msg)

    def integrate(
        self,
        integrand: KIntegrand | Mapping[Hashable, NDArray[np.inexact]],
    ) -> KIntegrationResult:
        """Return the weighted k-point sums without estimating the mesh error."""
        with stage(self.diagnostics, "ksum"):
            if callable(integrand):
                if self.kpoints is None:
                    msg = "Supply k-points to integrate a callable."
                    raise ValueError(msg)
                samples = integrand(self.kpoints)
            else:
                samples = integrand
            n_k = next(iter(samples.values())).shape[0]
            if self.kpoints is not None and len(self.kpoints) != n_k:
                msg = "The k-points and sampled arrays have different lengths."
                raise ValueError(msg)
            weights = (
                np.full(n_k, 1 / n_k)
                if self.weights is None
                else self.weights / self.weights.sum()
            )
            if len(weights) != n_k:
                msg = "The k-point weights and sampled arrays have different lengths."
                raise ValueError(msg)
            values: IntegralValues = {}
            for name, array in samples.items():
                if array.shape[0] != n_k:
                    msg = "All integrands must use the same k-points."
                    raise ValueError(msg)
                values[name] = np.asarray(np.tensordot(weights, array, axes=(0, 0)))
            result = KIntegrationResult(values, n_evaluations=n_k, weights=weights)
            if self.diagnostics is not None:
                self.diagnostics.emit(
                    "ksum.result",
                    n_k=n_k,
                    n_observables=len(values),
                    quadrature_error=result.error,
                    converged=result.converged,
                )
            return result


class BandKIntegrator:
    """Calculate Brillouin-zone averages from a Bloch Hamiltonian.

    Parameters
    ----------
    kpoint_integrator : KIntegrator
        Method for summing over supplied k-points or integrating over the BZ.
    smearing : callable, optional
        Function f(energies, beta, mu) used to weight each band.
    beta : float, optional
        Inverse temperature in inverse Hamiltonian-energy units. For Gaussian
        and Methfessel-Paxton smearing, 1/beta is the broadening width.
    mu : float, optional
        Chemical potential. With n_target, it is updated after solving for
        the required filling.
    n_target : float, optional
        Number of electrons represented by these bands per unit cell.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics.

    Notes
    -----
    Arguments to integrate override these settings. Supplying mu alone fixes
    mu for that call; supplying n_target instead solves for it.

    """

    def __init__(
        self,
        kpoint_integrator: KIntegrator,
        *,
        smearing: SmearFunction = fermi_function,
        beta: float = np.inf,
        mu: float | None = None,
        n_target: float | None = None,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        self.kpoint_integrator = kpoint_integrator
        self.smearing = smearing
        self.beta = beta
        self.mu = mu
        self.n_target = n_target

    def integrate(
        self,
        hamiltonian: Hamiltonian,
        *,
        observables: BandObservables | None = None,
        beta: float | None = None,
        mu: float | None = None,
        n_target: float | None = None,
    ) -> BandKIntegrationResult:
        """Calculate the requested k-integrals at fixed mu or fixed filling.

        Hamiltonian blocks have shape (n_k, n_orbitals, n_orbitals). A
        Hamiltonian function receives k-points of shape (n_k, dimension)
        and returns those blocks.

        The observables function receives the k-points and density matrices
        P(k) = U(k) f(k) U(k)^dagger. It returns a named array for each
        quantity to integrate, with k as the first index. When the Hamiltonian
        is supplied as arrays, its k-points argument is None and the density
        matrices follow the input mesh.
        """
        with stage(self.diagnostics, "kintegration"):
            beta = self.beta if beta is None else beta
            if np.isnan(beta) or beta <= 0:
                msg = "Supply a positive inverse temperature beta."
                raise ValueError(msg)
            if mu is None and n_target is None:
                mu, n_target = self.mu, self.n_target
            if mu is None and n_target is None:
                msg = "Supply mu or n_target."
                raise ValueError(msg)
            eigenstates: tuple[BlockRealArray, BlockMatrix] | None = None
            if not callable(hamiltonian):
                eigenstates = _diagonalize(hamiltonian)
            density_k: BlockMatrix = {}
            band_weights: BlockRealArray = {}
            energies: BlockRealArray = {}
            vectors: BlockMatrix = {}

            def integrate_at_mu(
                chemical_potential: float, include_observables: bool
            ) -> KIntegrationResult:
                def get_values(kpoints: KPoints | None) -> IntegralValues:
                    nonlocal density_k, band_weights, energies, vectors
                    if eigenstates is None:
                        assert callable(hamiltonian)
                        assert kpoints is not None
                        energies, vectors = _diagonalize(hamiltonian(kpoints))
                    else:
                        energies, vectors = eigenstates
                    band_weights = {
                        block: self.smearing(e, beta, chemical_potential)
                        for block, e in energies.items()
                    }
                    count = sum(np.sum(f, axis=1) for f in band_weights.values())
                    values: IntegralValues = {_ELECTRON_COUNT: np.asarray(count)}
                    if include_observables:
                        density_k = {
                            block: (v * band_weights[block][:, None, :])
                            @ v.conj().swapaxes(-1, -2)
                            for block, v in vectors.items()
                        }
                        if observables is not None:
                            values.update(observables(kpoints, density_k))
                    return values

                if eigenstates is not None:
                    return self.kpoint_integrator.integrate(get_values(None))
                return self.kpoint_integrator.integrate(get_values)

            if n_target is not None:
                if not np.isfinite(beta):
                    msg = "Solving mu at beta=infinity requires temperature extrapolation."
                    raise ValueError(msg)

                def excess(chemical_potential: float) -> float:
                    result = integrate_at_mu(chemical_potential, False)
                    if not result.converged:
                        msg = (
                            "k-integration failed during the chemical-potential solve."
                        )
                        raise RuntimeError(msg)
                    count = float(result.values[_ELECTRON_COUNT].real)
                    if self.diagnostics is not None:
                        self.diagnostics.emit(
                            "kintegration.mu",
                            mu=chemical_potential,
                            n_electrons=count,
                            n_target=n_target,
                            filling_residual=count - n_target,
                            quadrature_error=result.error,
                        )
                    return count - n_target

                if eigenstates is not None:
                    low = (
                        min(float(e.min()) for e in eigenstates[0].values()) - 50 / beta
                    )
                    high = (
                        max(float(e.max()) for e in eigenstates[0].values()) + 50 / beta
                    )
                else:
                    centre, margin = 0.0 if mu is None else mu, max(1.0, 50 / beta)
                    low, high = centre - margin, centre + margin
                    for _ in range(60):
                        if excess(low) <= 0 <= excess(high):
                            break
                        margin *= 2
                        low, high = centre - margin, centre + margin
                    else:
                        msg = "Could not bracket the target band filling."
                        raise ValueError(msg)
                mu = float(brentq(excess, low, high))
            assert mu is not None
            result = integrate_at_mu(mu, True)
            count = float(result.values.pop(_ELECTRON_COUNT).real)
            self.mu = mu
            sampled = result.weights is not None
            band_result = BandKIntegrationResult(
                values=result.values,
                mu=mu,
                beta=beta,
                n_electrons=count,
                converged=result.converged,
                quadrature_error=result.error,
                density_k=density_k if sampled else None,
                energies=energies if sampled else {},
                vectors=vectors if sampled else {},
                band_weights=band_weights if sampled else {},
                weights=result.weights,
            )

            if self.diagnostics is not None:
                self.diagnostics.emit(
                    "kintegration.result",
                    beta=beta,
                    mu=mu,
                    n_electrons=count,
                    quadrature_error=band_result.quadrature_error,
                    converged=band_result.converged,
                )
            return band_result


def _diagonalize(hamiltonian: BlockMatrixLike) -> tuple[BlockRealArray, BlockMatrix]:
    energies: BlockRealArray = {}
    vectors: BlockMatrix = {}
    for block, matrix in hamiltonian.items():
        e, vectors[block] = np.linalg.eigh(matrix)
        energies[block] = np.asarray(e, dtype=float)
    return energies, vectors
