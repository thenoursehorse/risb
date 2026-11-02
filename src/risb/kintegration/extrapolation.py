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
    """Extrapolated value, total error and error of each array entry.

    coefficients contains the weights applied to the input values.
    absolute_error contains the estimated absolute error of each entry.
    """

    value: NDArray[np.inexact]
    error: float
    coefficients: NDArray[np.float64]
    absolute_error: NDArray[np.float64]


class TemperatureExtrapolatedKIntegrator:
    """Extrapolate k-integrals from finite smearing to the target temperature.

    Parameters
    ----------
    kintegrator : BandKIntegrator
        Calculates the band integrals at each supplied beta.
    betas : sequence of float
        Positive, finite, increasing inverse temperatures, all below the
        target beta. The spacing need not be uniform.
    atol, rtol : float
        Absolute and relative tolerances on the Euclidean norm of all
        integrated matrix entries, the electron count and mu. These entries
        are not rescaled.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics. Debug output includes each
        beta, error estimate and stopping tolerance.

    Notes
    -----
    At each beta, the Hamiltonian is unchanged and mu is solved again if
    filling is fixed. beta is in inverse Hamiltonian-energy units; beta=infinity
    requests T=0.

    For a regular metal the Sommerfeld expansion has even powers of 1/beta.
    This class extrapolates the latest three samples in beta**-2 using
    richardson_extrapolate. A candidate must meet tolerance and agree with
    another three-sample extrapolation after a further cooling step.
    At least four samples are therefore needed for convergence.
    The error estimate includes the change between these extrapolations.
    Results include the Frobenius norm and largest absolute entry of this
    error for each matrix; diagnostics report both.
    It does not check the density matrix at each individual k-point: errors
    at different points can cancel in a sum. Too few k-points or a singular
    density of states can also make this estimate unreliable.

    The returned density matrices use the same extrapolation coefficients
    as the k-integrals. The wrapped BandKIntegrator controls how mu is found
    at each beta, including its use_midgap setting.

    References
    ----------
    The use of Richardson extrapolation for T=0 band filling follows the
    `AutoBZ.jl electron-density example
    <https://lxvm.github.io/AutoBZ.jl/stable/pages/demo/density/#T0-electron-density>`__.

    """

    def __init__(
        self,
        kintegrator: BandKIntegrator,
        *,
        betas: Sequence[float] | NDArray[np.float64],
        atol: float,
        rtol: float,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        self.kintegrator = kintegrator
        self.kpoint_integrator: KIntegrator = kintegrator.kpoint_integrator
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
            previous: ExtrapolationResult | None = None
            previous_meets_tolerance = False
            previous_quadrature_error = 0.0
            temperature_errors: dict[Hashable, KIntegralError] = {}
            for sample_beta in self.betas:
                sample = self.kintegrator.integrate(
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
                        energies=samples[0].energies if samples else sample.energies,
                        vectors=samples[0].vectors if samples else sample.vectors,
                    )
                samples.append(sample)
                arrays.append(
                    np.concatenate(
                        [
                            *(value.ravel() for value in sample.values.values()),
                            np.asarray([sample.n_electrons, sample.mu]),
                        ]
                    )
                )
                window_size = min(3, len(samples))
                candidate = richardson_extrapolate(
                    [1 / s.beta**2 for s in samples[-window_size:]],
                    arrays[-window_size:],
                    target=1 / beta**2,
                )
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
                    and candidate.error + quadrature_error <= tolerance
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
                    error=float(np.linalg.norm(checked_error)),
                )
                confirmation_quadrature_error = (
                    quadrature_error + previous_quadrature_error
                )
                meets_tolerance = (
                    candidate_meets_tolerance
                    and previous_meets_tolerance
                    and extrapolated.error + confirmation_quadrature_error <= tolerance
                )
                previous = candidate
                previous_meets_tolerance = candidate_meets_tolerance
                previous_quadrature_error = quadrature_error
                temperature_errors = {}
                offset = 0
                for key, value in sample.values.items():
                    entry_errors = extrapolated.absolute_error[
                        offset : offset + value.size
                    ]
                    temperature_errors[key] = KIntegralError(
                        norm=float(np.linalg.norm(entry_errors)),
                        max_abs=float(np.max(entry_errors)),
                    )
                    offset += value.size
                if self.diagnostics is not None:
                    for key, error in temperature_errors.items():
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
                        n_electrons=sample.n_electrons,
                        temperature_error=extrapolated.error,
                        electron_count_error=float(extrapolated.absolute_error[-2]),
                        mu_error=float(extrapolated.absolute_error[-1]),
                        quadrature_error=(
                            None
                            if any(s.quadrature_error is None for s in samples)
                            else quadrature_error
                        ),
                        tolerance=tolerance,
                        candidate_error=candidate.error,
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
            band_weights: BlockRealArray = {}
            if all(s.weights is not None for s in samples):
                band_weights = {
                    block: sum(
                        (
                            c * s.band_weights[block]
                            for c, s in zip(coefficients, samples, strict=True)
                        ),
                        np.zeros_like(samples[0].band_weights[block]),
                    )
                    for block in samples[0].band_weights
                }
                density_k = {
                    block: (v * band_weights[block][:, None, :])
                    @ v.conj().swapaxes(-1, -2)
                    for block, v in samples[0].vectors.items()
                }
            self.mu = float(
                sum(c * s.mu for c, s in zip(coefficients, samples, strict=True))
            )
            result = replace(
                samples[-1],
                values=values,
                beta=beta,
                mu=self.mu,
                n_electrons=float(
                    sum(
                        c * s.n_electrons
                        for c, s in zip(coefficients, samples, strict=True)
                    )
                ),
                converged=converged and all(s.converged for s in samples),
                temperature_error=extrapolated.error,
                temperature_errors=temperature_errors,
                quadrature_error=quadrature_error,
                betas=tuple(s.beta for s in samples),
                density_k=density_k,
                band_weights=band_weights,
            )

            if self.diagnostics is not None:
                for key, error in temperature_errors.items():
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
                    n_electrons=result.n_electrons,
                    temperature_error=result.temperature_error,
                    electron_count_error=float(extrapolated.absolute_error[-2]),
                    mu_error=float(extrapolated.absolute_error[-1]),
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
