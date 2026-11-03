"""Richardson extrapolation of k-space integrals."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from risb.diagnostics.output import DiagnosticObserver, stage
from risb.types import BlockMatrix, BlockRealArray

from .integrators import (
    BandKIntegrationResult,
    BandKIntegrator,
    BandObservables,
    Hamiltonian,
    IntegralValues,
    KIntegralError,
    KIntegrator,
)


@dataclass
class ExtrapolationResult:
    """Extrapolated value and error estimates.

    coefficients contains the weights applied to the input values.
    absolute_error is the error estimate for each entry; error_norm
    is its Frobenius norm.
    """

    value: NDArray[np.inexact]
    error_norm: float
    coefficients: NDArray[np.float64]
    absolute_error: NDArray[np.float64]


class TemperatureExtrapolatedBandKIntegrator:
    """Extrapolate k-integrals from finite smearing to the target temperature.

    Parameters
    ----------
    band_integrator : BandKIntegrator
        Calculates the band integrals at each supplied beta.
    betas : sequence of float
        Positive, finite, increasing inverse temperatures, all below the
        target beta. The spacing need not be uniform.
    atol, rtol : float
        Absolute and relative tolerances on the Euclidean norm of all
        integrated matrix entries and electron count. These entries are not
        rescaled. Mu is extrapolated separately and does not set convergence.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics. Debug output includes each
        beta, error estimate and stopping tolerance.

    Notes
    -----
    At each beta, the Hamiltonian is unchanged and mu is solved again if
    filling is fixed. beta is in inverse Hamiltonian-energy units; beta=infinity
    requests T=0.

    For a regular metal the Sommerfeld expansion has even powers of 1/beta,
    but partially occupied flat bands can introduce linear terms in both mu
    and the matrices. This class fits the latest three samples as a quadratic
    in 1/beta using richardson_extrapolate. A candidate must meet tolerance
    and agree with another extrapolation after a further cooling step.
    At least four samples are therefore needed for convergence.
    The error estimate includes the change between these extrapolations.
    Results include the Frobenius norm and largest absolute entry of this
    error for each matrix; diagnostics report both.
    It does not check the density matrix at each individual k-point: errors
    at different points can cancel in a sum. Too few k-points or a singular
    density of states can also make this estimate unreliable.

    Mu uses the same temperature dependence but is fitted separately.
    Its estimated error is returned as mu_extrapolation_error and does not
    control integral convergence. The fit keeps the reference and correction
    separate so that small changes in mu are not rounded away.

    The returned density matrices use the same extrapolation coefficients
    as the k-integrals. They retain fractional occupations at degenerate
    levels; the extrapolated mu alone does not determine those occupations.
    The wrapped BandKIntegrator controls how mu is found at each beta,
    including its use_midgap setting.

    References
    ----------
    The use of Richardson extrapolation for T=0 band filling follows the
    `AutoBZ.jl electron-density example
    <https://lxvm.github.io/AutoBZ.jl/stable/pages/demo/density/#T0-electron-density>`__.
    Here the fit uses 1/beta rather than 1/beta**2 to allow linear terms.

    """

    def __init__(
        self,
        band_integrator: BandKIntegrator,
        *,
        betas: Sequence[float] | NDArray[np.float64],
        atol: float,
        rtol: float,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        self.band_integrator = band_integrator
        self.kpoint_integrator: KIntegrator = band_integrator.kpoint_integrator
        self.betas = tuple(float(beta) for beta in betas)
        if not self.betas or any(not np.isfinite(b) or b <= 0 for b in self.betas):
            msg = "betas must be positive and finite."
            raise ValueError(msg)
        if any(a >= b for a, b in zip(self.betas, self.betas[1:], strict=False)):
            msg = "betas must be strictly increasing."
            raise ValueError(msg)
        if atol < 0 or rtol < 0 or not np.isfinite(atol + rtol):
            msg = "Extrapolation tolerances must be finite and nonnegative."
            raise ValueError(msg)
        self.atol, self.rtol = atol, rtol
        self.mu: float | None = None

    def integrate(
        self,
        hamiltonian: Hamiltonian,
        *,
        observables: BandObservables | None = None,
        beta: float | None = None,
        mu: float | None = None,
        n_target: float | None = None,
    ) -> BandKIntegrationResult:
        """Extrapolate k-integrals to beta, using the supplied finite-beta sequence.

        Sampling stops after a colder sample confirms the estimated accuracy.
        If the sequence ends before confirmation, converged is False even
        when the last candidate meets tolerance.
        """
        with stage(self.diagnostics, "kextrapolation"):
            beta = np.inf if beta is None else beta
            if np.isnan(beta) or beta <= self.betas[-1]:
                msg = "The target beta must exceed every sample beta."
                raise ValueError(msg)
            samples: list[BandKIntegrationResult] = []
            arrays: list[NDArray[np.inexact]] = []
            extrapolated: ExtrapolationResult
            converged = False
            target_mu: float
            mu_reference: float
            mu_correction: float
            mu_extrapolation_error: float
            previous_mu: tuple[float, float] | None = None
            previous: ExtrapolationResult | None = None
            previous_meets_tolerance = False
            previous_quadrature_error = 0.0
            temperature_extrapolation_errors: dict[Hashable, KIntegralError] = {}
            for sample_beta in self.betas:
                sample = self.band_integrator.integrate(
                    hamiltonian,
                    observables=observables,
                    beta=sample_beta,
                    mu=mu,
                    n_target=n_target,
                )
                if sample.weights is not None:
                    sample = replace(
                        sample,
                        density_k=None,
                        eigenvalues=samples[0].eigenvalues
                        if samples
                        else sample.eigenvalues,
                        eigenvectors=samples[0].eigenvectors
                        if samples
                        else sample.eigenvectors,
                    )
                samples.append(sample)
                arrays.append(
                    np.concatenate(
                        [
                            *(value.ravel() for value in sample.values.values()),
                            np.asarray([sample.n_electrons]),
                        ]
                    )
                )
                window_size = min(3, len(samples))
                temperatures = [1 / s.beta for s in samples[-window_size:]]
                candidate = richardson_extrapolate(
                    temperatures, arrays[-window_size:], target=1 / beta
                )
                mu_reference = sample.mu
                mu_candidate = richardson_extrapolate(
                    temperatures,
                    [
                        np.asarray(
                            s.mu - mu_reference
                            if s.mu_reference is None
                            else (s.mu_reference - mu_reference) + s.mu_correction
                        )
                        for s in samples[-window_size:]
                    ],
                    target=1 / beta,
                )
                mu_correction = float(mu_candidate.value)
                target_mu = mu_reference + mu_correction
                mu_extrapolation_error = (
                    np.inf
                    if previous_mu is None
                    else max(
                        mu_candidate.error_norm,
                        abs(
                            (mu_reference - previous_mu[0])
                            + (mu_correction - previous_mu[1])
                        ),
                    )
                )
                previous_mu = mu_reference, mu_correction
                candidate = replace(
                    candidate,
                    coefficients=np.pad(
                        candidate.coefficients, (len(samples) - window_size, 0)
                    ),
                )
                quadrature_error = sum(
                    abs(c) * s.quadrature_error
                    for c, s in zip(candidate.coefficients, samples, strict=True)
                    if s.quadrature_error is not None
                )
                tolerance = max(
                    self.atol, self.rtol * float(np.linalg.norm(candidate.value))
                )
                candidate_meets_tolerance = (
                    sample.converged
                    and len(samples) >= 3
                    and candidate.error_norm + quadrature_error <= tolerance
                )
                checked_error = (
                    np.full_like(candidate.absolute_error, np.inf)
                    if previous is None
                    else np.maximum(
                        candidate.absolute_error,
                        np.abs(candidate.value - previous.value),
                    )
                )
                extrapolated = replace(
                    candidate,
                    absolute_error=checked_error,
                    error_norm=float(np.linalg.norm(checked_error)),
                )
                confirmation_quadrature_error = (
                    quadrature_error + previous_quadrature_error
                )
                meets_tolerance = (
                    candidate_meets_tolerance
                    and previous_meets_tolerance
                    and extrapolated.error_norm + confirmation_quadrature_error
                    <= tolerance
                )
                previous = candidate
                previous_meets_tolerance = candidate_meets_tolerance
                previous_quadrature_error = quadrature_error
                temperature_extrapolation_errors = {}
                offset = 0
                for key, value in sample.values.items():
                    entry_errors = extrapolated.absolute_error[
                        offset : offset + value.size
                    ]
                    temperature_extrapolation_errors[key] = KIntegralError(
                        norm=float(np.linalg.norm(entry_errors)),
                        max_abs=float(np.max(entry_errors)),
                    )
                    offset += value.size
                if self.diagnostics is not None:
                    for key, error in temperature_extrapolation_errors.items():
                        self.diagnostics.emit(
                            "kextrapolation.matrix_sample",
                            observable=key,
                            beta=sample_beta,
                            error_norm=error.norm,
                            error_max_abs=error.max_abs,
                        )
                    self.diagnostics.emit(
                        "kextrapolation.sample",
                        beta=sample_beta,
                        smearing_width=1 / sample_beta,
                        target_beta=beta,
                        mu=sample.mu,
                        extrapolated_mu=target_mu,
                        extrapolated_mu_reference=mu_reference,
                        extrapolated_mu_correction=mu_correction,
                        n_electrons=sample.n_electrons,
                        temperature_extrapolation_error=extrapolated.error_norm,
                        electron_count_error=float(extrapolated.absolute_error[-1]),
                        mu_extrapolation_error=mu_extrapolation_error,
                        quadrature_error=(
                            None
                            if any(s.quadrature_error is None for s in samples)
                            else quadrature_error
                        ),
                        tolerance=tolerance,
                        candidate_error=candidate.error_norm,
                        candidate_meets_tolerance=candidate_meets_tolerance,
                        confirmed=meets_tolerance,
                        confirmation_quadrature_error=(
                            None
                            if any(s.quadrature_error is None for s in samples)
                            else confirmation_quadrature_error
                        ),
                        kpoint_converged=sample.converged,
                    )
                if not sample.converged:
                    break
                if meets_tolerance:
                    converged = True
                    break
            coefficients = extrapolated.coefficients
            values: IntegralValues = {
                key: sum(
                    (
                        c * sample.values[key]
                        for c, sample in zip(coefficients, samples, strict=True)
                    ),
                    np.zeros_like(samples[0].values[key]),
                )
                for key in samples[0].values
            }
            quadrature_errors = [sample.quadrature_error for sample in samples]
            quadrature_error = (
                None
                if any(e is None for e in quadrature_errors)
                else sum(
                    abs(c) * e
                    for c, e in zip(coefficients, quadrature_errors, strict=True)
                    if e is not None
                )
            )
            density_k: BlockMatrix | None = None
            occupations: BlockRealArray = {}
            if all(s.weights is not None for s in samples):
                occupations = {
                    block: sum(
                        (
                            c * s.occupations[block]
                            for c, s in zip(coefficients, samples, strict=True)
                        ),
                        np.zeros_like(samples[0].occupations[block]),
                    )
                    for block in samples[0].occupations
                }
                density_k = {
                    block: (v * occupations[block][:, None, :])
                    @ v.conj().swapaxes(-1, -2)
                    for block, v in samples[0].eigenvectors.items()
                }
            self.mu = target_mu
            result = replace(
                samples[-1],
                values=values,
                beta=beta,
                mu=self.mu,
                mu_reference=mu_reference,
                mu_correction=mu_correction,
                n_electrons=float(
                    sum(
                        c * s.n_electrons
                        for c, s in zip(coefficients, samples, strict=True)
                    )
                ),
                converged=converged and all(s.converged for s in samples),
                temperature_extrapolation_error=extrapolated.error_norm,
                mu_extrapolation_error=mu_extrapolation_error,
                temperature_extrapolation_errors=temperature_extrapolation_errors,
                quadrature_error=quadrature_error,
                betas=tuple(s.beta for s in samples),
                density_k=density_k,
                occupations=occupations,
            )

            if self.diagnostics is not None:
                for key, error in temperature_extrapolation_errors.items():
                    self.diagnostics.emit(
                        "kextrapolation.matrix_error",
                        observable=key,
                        beta=result.beta,
                        error_norm=error.norm,
                        error_max_abs=error.max_abs,
                    )
                self.diagnostics.emit(
                    "kextrapolation.result",
                    beta=result.beta,
                    betas=result.betas,
                    mu=result.mu,
                    mu_reference=result.mu_reference,
                    mu_correction=result.mu_correction,
                    n_electrons=result.n_electrons,
                    temperature_extrapolation_error=result.temperature_extrapolation_error,
                    electron_count_error=float(extrapolated.absolute_error[-1]),
                    mu_extrapolation_error=mu_extrapolation_error,
                    quadrature_error=result.quadrature_error,
                    converged=result.converged,
                )
            return result


def richardson_extrapolate(
    points: Sequence[float],
    values: Sequence[NDArray[np.inexact]],
    *,
    target: float = 0,
) -> ExtrapolationResult:
    """Estimate an array at target from values at decreasing points above it.

    Neville interpolation uses every supplied point. The error estimate
    compares the result with the extrapolation that omits the oldest point,
    with a roundoff floor. Its coefficients express the result as a weighted
    sum of the input values. A single point has an infinite error estimate.

    References
    ----------
    L. F. Richardson, Philos. Trans. R. Soc. A 210, 307-357 (1911),
    `doi:10.1098/rsta.1911.0009 <https://doi.org/10.1098/rsta.1911.0009>`__.

    The interpolation recurrence is also used by
    `Richardson.jl <https://github.com/JuliaMath/Richardson.jl/blob/master/src/Richardson.jl>`__.
    This NumPy implementation returns the extrapolation through all supplied
    points. The temperature wrapper chooses recent points and separately
    checks agreement with a colder extrapolation.

    """
    x = np.asarray(points, dtype=float) - target
    if len(x) != len(values) or len(x) == 0:
        msg = "Supply one point for each array value."
        raise ValueError(msg)
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) >= 0):
        msg = "Points must decrease strictly towards the target from above."
        raise ValueError(msg)
    if any(not np.all(np.isfinite(value)) for value in values):
        msg = "Cannot extrapolate nonfinite values."
        raise ValueError(msg)
    table = [np.array(value, copy=True) for value in values]
    coefficients = list(np.eye(len(x)))
    absolute_error = np.full(np.shape(values[0]), np.inf)
    for width in range(1, len(x)):
        for i in range(len(x) - width):
            j = i + width
            lower_order = table[i + 1].copy()
            ratio = x[i] / x[j]
            table[i] = table[i + 1] + (table[i + 1] - table[i]) / (ratio - 1)
            coefficients[i] = coefficients[i + 1] + (
                coefficients[i + 1] - coefficients[i]
            ) / (ratio - 1)
            if width == len(x) - 1:
                absolute_error = np.maximum(
                    np.abs(table[i] - lower_order),
                    np.finfo(float).eps
                    * np.abs(table[i])
                    * np.sum(abs(coefficients[i])),
                )
    return ExtrapolationResult(
        table[0],
        float(np.linalg.norm(absolute_error)),
        coefficients[0],
        np.asarray(absolute_error, dtype=float),
    )
