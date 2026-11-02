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
    temperature_error estimates the combined temperature-extrapolation error
    of the requested integrals and electron count. mu_temperature_error
    separately estimates the temperature-extrapolation error of mu; it does
    not include errors from the filling search or k integration.
    When solving for mu, converged also requires the final electron count
    to meet the BandKIntegrator electron_count_atol.
    During temperature extrapolation, converged checks the integrals and
    electron count; mu's extrapolation error does not set this flag.
    temperature_errors gives the norm and maximum absolute entry of the
    extrapolation error for each requested matrix. An empty mapping means
    that per-matrix estimates are unavailable; None marks an unavailable
    scalar estimate.

    mu is the rounded sum of mu_reference and mu_correction. Reproduce
    finite-beta band weights with
    smearing(energies - mu_reference, beta, mu_correction). Adding the
    reference and correction before calling smearing can lose precision.
    A None reference means only mu is available. After temperature
    extrapolation, the pair describes the extrapolated mu; the densities
    combine finite-beta results.
    """

    values: IntegralValues
    mu: float
    beta: float
    n_electrons: float
    converged: bool = True
    quadrature_error: float | None = None
    temperature_error: float | None = None
    mu_temperature_error: float | None = None
    temperature_errors: dict[Hashable, KIntegralError] = field(default_factory=dict)
    betas: tuple[float, ...] = ()
    density_k: BlockMatrix | None = None
    energies: BlockRealArray = field(default_factory=dict)
    vectors: BlockMatrix = field(default_factory=dict)
    band_weights: BlockRealArray = field(default_factory=dict)
    weights: NDArray[np.float64] | None = None
    mu_reference: float | None = None
    mu_correction: float = 0.0


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
        Function f(energies, beta, mu) used to weight each band. It must
        depend on energies and mu through their difference; the filling
        search may shift both by a common reference energy.
    beta : float, optional
        Inverse temperature in inverse Hamiltonian-energy units. For Gaussian
        and Methfessel-Paxton smearing, 1/beta is the broadening width.
    mu : float, optional
        Chemical potential. With n_target, it is updated after solving for
        the required filling.
    n_target : float, optional
        Number of electrons represented by these bands per unit cell.
    use_midgap : bool, optional
        When solving for mu, try the midpoint of the occupied and empty
        sampled levels first. If its electron count misses the target,
        use Brent's method. Requires band energies and k-point weights;
        otherwise use Brent's method directly. All bands enter the count,
        including auxiliary bands with no physical spectral weight.
    electron_count_atol : float, optional
        Maximum absolute error in the electron count per cell when solving
        for mu. Applies to midpoint selection and the final integrals.
        If the count misses this tolerance, converged is False.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics.

    Notes
    -----
    beta, mu and n_target passed to integrate override these settings.
    Supplying mu alone fixes mu for that call; supplying n_target solves
    for it. A gap between sampled levels need not be a bulk band gap.

    If the initial mu search misses electron_count_atol, Brent's method
    solves for a small correction relative to that mu. Band weights retain
    the correction separately to avoid rounding it away. The final count
    must still meet tolerance.

    References
    ----------
    The midpoint-first search follows VASP's
    `EFERMI=MIDGAP policy
    <https://vasp.at/wiki/index.php/Smearing_technique#Calculation_of_the_Fermi_energy>`__.
    Brent's method is used for the fallback search.

    """

    def __init__(
        self,
        kpoint_integrator: KIntegrator,
        *,
        smearing: SmearFunction = fermi_function,
        beta: float = np.inf,
        mu: float | None = None,
        n_target: float | None = None,
        use_midgap: bool = True,
        electron_count_atol: float = 1e-10,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        if not np.isfinite(electron_count_atol) or electron_count_atol < 0:
            msg = "electron_count_atol must be finite and nonnegative."
            raise ValueError(msg)
        self.use_midgap = use_midgap
        self.electron_count_atol = electron_count_atol
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

            mu_reference = 0.0
            mu_correction = mu

            def integrate_at_mu(
                mu_correction: float, include_observables: bool
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
                        block: self.smearing(e - mu_reference, beta, mu_correction)
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

            midgap_available: bool | None = None
            midgap_accepted = False
            sampled_gap: float | None = None
            if n_target is not None:
                if not np.isfinite(beta):
                    msg = "Solving mu at beta=infinity requires temperature extrapolation."
                    raise ValueError(msg)

                count_result: KIntegrationResult | None = None
                filling_residuals: dict[tuple[float, float], float] = {}

                def electron_count_residual(mu_correction: float) -> float:
                    nonlocal count_result
                    key = (mu_reference, mu_correction)
                    if key in filling_residuals:
                        return filling_residuals[key]
                    result = integrate_at_mu(mu_correction, False)
                    if not result.converged:
                        msg = (
                            "k-integration failed during the chemical-potential solve."
                        )
                        raise RuntimeError(msg)
                    count_result = result
                    count = float(result.values[_ELECTRON_COUNT].real)
                    filling_residuals[key] = count - n_target
                    if self.diagnostics is not None:
                        self.diagnostics.emit(
                            "kintegration.mu",
                            mu=mu_reference + mu_correction,
                            mu_reference=mu_reference,
                            mu_correction=mu_correction,
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
                        if (
                            electron_count_residual(low)
                            <= 0
                            <= electron_count_residual(high)
                        ):
                            break
                        margin *= 2
                        low, high = centre - margin, centre + margin
                    else:
                        msg = "Could not bracket the target band filling."
                        raise ValueError(msg)
                if self.use_midgap:
                    if count_result is None:
                        electron_count_residual(low)
                    assert count_result is not None
                    midgap_available = count_result.weights is not None
                    if count_result.weights is not None:
                        occupied_edge, empty_edge = get_filling_edges(
                            energies,
                            count_result.weights,
                            n_target,
                            electron_count_atol=self.electron_count_atol,
                        )
                        sampled_gap = empty_edge - occupied_edge
                        midpoint = occupied_edge + (empty_edge - occupied_edge) / 2
                        midgap_accepted = (
                            abs(electron_count_residual(midpoint))
                            <= self.electron_count_atol
                        )
                        if midgap_accepted:
                            mu = midpoint
                if not midgap_accepted:
                    mu = float(
                        brentq(
                            electron_count_residual,
                            low,
                            high,
                            xtol=np.nextafter(0.0, 1.0),
                        )
                    )
                assert mu is not None
                count_error = electron_count_residual(mu)
                mu_reference, mu_correction = mu, 0.0
                if abs(count_error) > self.electron_count_atol:
                    margin = max(50 / beta, 16 * abs(np.spacing(mu_reference)))
                    mu_correction = float(
                        brentq(
                            electron_count_residual,
                            -margin,
                            margin,
                            xtol=np.nextafter(0.0, 1.0),
                        )
                    )
            assert mu_correction is not None
            result = integrate_at_mu(mu_correction, True)
            count = float(result.values.pop(_ELECTRON_COUNT).real)
            mu = mu_reference + mu_correction
            if mu_reference == 0:
                mu_reference, mu_correction = mu, 0.0
            self.mu = mu
            sampled = result.weights is not None
            band_result = BandKIntegrationResult(
                values=result.values,
                mu=mu,
                beta=beta,
                n_electrons=count,
                converged=result.converged
                and (
                    n_target is None
                    or abs(count - n_target) <= self.electron_count_atol
                ),
                quadrature_error=result.error,
                density_k=density_k if sampled else None,
                energies=energies if sampled else {},
                vectors=vectors if sampled else {},
                band_weights=band_weights if sampled else {},
                weights=result.weights,
                mu_reference=mu_reference,
                mu_correction=mu_correction,
            )

            if self.diagnostics is not None:
                self.diagnostics.emit(
                    "kintegration.result",
                    beta=beta,
                    mu=mu,
                    mu_reference=mu_reference,
                    mu_correction=mu_correction,
                    n_electrons=count,
                    quadrature_error=band_result.quadrature_error,
                    converged=band_result.converged,
                    use_midgap=self.use_midgap,
                    midgap_available=midgap_available,
                    midgap_accepted=midgap_accepted,
                    sampled_gap=sampled_gap,
                    filling_residual=(None if n_target is None else count - n_target),
                    electron_count_atol=self.electron_count_atol,
                )
            return band_result


def get_filling_edges(
    energies: Mapping[str, NDArray[np.float64]],
    weights: Sequence[float] | NDArray[np.float64],
    n_target: float,
    *,
    electron_count_atol: float,
) -> tuple[float, float]:
    """Find the occupied and empty edges at the target weighted band filling.

    energies contains arrays of shape (n_k, n_bands), one for each block.
    weights are nonnegative k-point weights and are normalized here.
    electron_count_atol is the absolute tolerance on the number of occupied
    states per cell when locating the filling boundary.

    If the target requires partial occupation of a level, both returned
    edges equal that level. Zero-weight points are omitted. These edges
    describe the supplied mesh; they do not establish a bulk band gap.
    """
    k_weights = np.asarray(weights, dtype=float)
    k_weights = k_weights / k_weights.sum()
    all_energies = np.concatenate([e.ravel() for e in energies.values()])
    all_weights = np.concatenate(
        [
            np.broadcast_to(k_weights[:, None], e.shape).ravel()
            for e in energies.values()
        ]
    )
    positive = all_weights > 0
    all_energies, all_weights = all_energies[positive], all_weights[positive]
    order = np.argsort(all_energies, kind="stable")
    levels, level_weights = all_energies[order], all_weights[order]
    count = np.cumsum(level_weights, dtype=np.longdouble)
    roundoff = 64 * np.finfo(float).eps * max(1, np.max(abs(levels)))
    boundaries = np.flatnonzero(
        (np.diff(levels) > roundoff)
        & (abs(count[:-1] - n_target) <= electron_count_atol)
    )
    if len(boundaries):
        boundary = int(boundaries[0])
        return float(levels[boundary]), float(levels[boundary + 1])
    boundary = min(int(np.searchsorted(count, n_target)), len(levels) - 1)
    return float(levels[boundary]), float(levels[boundary])


def _diagonalize(hamiltonian: BlockMatrixLike) -> tuple[BlockRealArray, BlockMatrix]:
    energies: BlockRealArray = {}
    vectors: BlockMatrix = {}
    for block, matrix in hamiltonian.items():
        e, vectors[block] = np.linalg.eigh(matrix)
        energies[block] = np.asarray(e, dtype=float)
    return energies, vectors
