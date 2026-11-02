"""Richardson extrapolation of k-space integrals."""

from collections.abc import Sequence
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
    KIntegrator,
)


@dataclass
class ExtrapolationResult:
    """Extrapolated value, error estimate and coefficients of the input values."""

    value: NDArray[np.inexact]
    error: float
    coefficients: NDArray[np.float64]


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
    This class extrapolates in beta**-2 using richardson_extrapolate.
    The error estimate compares successive extrapolations of the k-sums.
    It does not check the density matrix at each individual k-point: errors
    at different points can cancel in a sum. Too few k-points or a singular
    density of states can also make this estimate unreliable.

    The returned density matrices use the same extrapolation coefficients
    as the k-integrals.

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

        Sampling stops when the estimated error meets tolerance. If every
        supplied beta is used without meeting tolerance, converged is False.
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
                extrapolated = richardson_extrapolate(
                    [1 / s.beta**2 for s in samples], arrays, target=1 / beta**2
                )
                q_error = sum(
                    abs(c) * s.quadrature_error
                    for c, s in zip(extrapolated.coefficients, samples, strict=True)
                    if s.quadrature_error is not None
                )
                tolerance = max(
                    self.atol, self.rtol * float(np.linalg.norm(extrapolated.value))
                )
                meets_tolerance = (
                    sample.converged
                    and len(samples) >= 3
                    and extrapolated.error + q_error <= tolerance
                )
                if self.diagnostics is not None:
                    self.diagnostics.emit(
                        "kextrapolation.sample",
                        beta=sample_beta,
                        smearing_width=1 / sample_beta,
                        target_beta=beta,
                        mu=sample.mu,
                        n_electrons=sample.n_electrons,
                        temperature_error=extrapolated.error,
                        quadrature_error=(
                            None
                            if any(s.quadrature_error is None for s in samples)
                            else q_error
                        ),
                        tolerance=tolerance,
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
            q_errors = [sample.quadrature_error for sample in samples]
            quadrature_error = (
                None
                if any(e is None for e in q_errors)
                else sum(
                    abs(c) * e
                    for c, e in zip(coefficients, q_errors, strict=True)
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
                quadrature_error=quadrature_error,
                betas=tuple(s.beta for s in samples),
                density_k=density_k,
                band_weights=band_weights,
            )

            if self.diagnostics is not None:
                self.diagnostics.emit(
                    "kextrapolation.result",
                    beta=result.beta,
                    betas=result.betas,
                    mu=result.mu,
                    n_electrons=result.n_electrons,
                    temperature_error=result.temperature_error,
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

    Neville interpolation removes successive terms in the expansion about
    target. The error estimate is the norm of the change from the preceding
    approximation, with a roundoff floor. Return the estimate with the
    smallest such change. Its coefficients express it as a weighted sum of
    the input values.

    References
    ----------
    L. F. Richardson, Philos. Trans. R. Soc. A 210, 307-357 (1911),
    `doi:10.1098/rsta.1911.0009 <https://doi.org/10.1098/rsta.1911.0009>`__.

    The sequence algorithm and error estimate follow
    `Richardson.jl <https://github.com/JuliaMath/Richardson.jl/blob/master/src/Richardson.jl>`__.
    The implementation here uses NumPy.

    """
    x = np.asarray(points, dtype=float) - target
    if len(x) != len(values) or len(x) == 0:
        msg = "Supply one point for each array value."
        raise ValueError(msg)
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) >= 0):
        msg = "Points must decrease strictly towards the target from above."
        raise ValueError(msg)
    table: list[NDArray[np.inexact]] = []
    coefficients: list[NDArray[np.float64]] = []
    best_value, best_error = np.asarray(values[0]), np.inf
    best_coefficients = np.eye(len(x))[0]
    for j, value in enumerate(values):
        if not np.all(np.isfinite(value)):
            msg = "Cannot extrapolate nonfinite values."
            raise ValueError(msg)
        table.append(np.array(value, copy=True))
        coefficients.append(np.eye(len(x))[j])
        for i in range(j - 1, -1, -1):
            old = table[i]
            ratio = x[i] / x[j]
            table[i] = table[i + 1] + (table[i + 1] - old) / (ratio - 1)
            coefficients[i] = coefficients[i + 1] + (
                coefficients[i + 1] - coefficients[i]
            ) / (ratio - 1)
            error = float(np.linalg.norm(table[i] - old))
            error = max(
                error,
                np.finfo(float).eps
                * float(np.linalg.norm(table[i]))
                * float(np.sum(np.abs(coefficients[i]))),
            )
            if error < best_error:
                best_value = table[i].copy()
                best_error = error
                best_coefficients = coefficients[i].copy()
    return ExtrapolationResult(best_value, float(best_error), best_coefficients)
