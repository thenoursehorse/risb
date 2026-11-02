"""Optional AutoBZCore backend for normalized Brillouin-zone integrals."""

import atexit
import json
import subprocess
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from risb.diagnostics.output import DiagnosticObserver, stage

from .integrators import KIntegrand, KIntegrationResult


class AutoBZIntegrator:
    """Calculate Brillouin-zone averages with AutoBZCore.

    Parameters
    ----------
    domain : sequence
        Coordinate bounds of the periodic cell, one pair per dimension.
        Results are divided by the cell volume.
    method : {'adaptive', 'automatic-uniform', 'uniform'}, optional
        Nested adaptive integration, a sequence of refined periodic
        trapezoidal meshes, or one fixed periodic mesh.
    atol, rtol : float, optional
        Tolerances on the norm of all integrated scalar and matrix entries.
    points_per_axis, max_points_per_axis : int, optional
        Starting and maximum numbers of k-points per direction for uniform
        refinement. Fixed uniform integration uses points_per_axis.
    max_evaluations : int, optional
        Evaluation budget passed to AutoBZCore.
    rate : float, optional
        Estimate of the exponential convergence rate used to choose
        successive uniform meshes.
    julia : str, optional
        Julia executable.
    project : str or pathlib.Path, optional
        Julia environment containing AutoBZCore and JSON. Packages are not
        installed automatically.
    diagnostics : risb.diagnostics.DiagnosticObserver, optional
        Reporter such as KIntegrationDiagnostics.

    Notes
    -----
    A Julia process runs AutoBZCore and requests values from the Python
    integrand at each k-point. It is reused between integrals. Communication
    adds overhead compared with evaluating the integrand entirely in Julia.
    Use a context manager or call close when finished.

    References
    ----------
    The integration methods are described in
    `SciPost Phys. 15, 062 (2023) <https://scipost.org/SciPostPhys.15.2.062>`__.

    """

    def __init__(
        self,
        domain: Sequence[tuple[float, float]],
        *,
        method: Literal["adaptive", "automatic-uniform", "uniform"] = "adaptive",
        atol: float = 1e-8,
        rtol: float = 1e-6,
        points_per_axis: int = 16,
        max_points_per_axis: int = 512,
        max_evaluations: int = 1_000_000,
        rate: float = 1,
        julia: str = "julia",
        project: str | Path | None = None,
        diagnostics: DiagnosticObserver | None = None,
    ) -> None:
        self.diagnostics = diagnostics
        bounds = np.asarray(domain, dtype=float)
        if (
            bounds.ndim != 2
            or bounds.shape[1] != 2
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])
        ):
            msg = "Supply finite, increasing coordinate bounds for each dimension."
            raise ValueError(msg)
        if method not in ("adaptive", "automatic-uniform", "uniform"):
            msg = "Unknown AutoBZ integration method."
            raise ValueError(msg)
        self.lower = bounds[:, 0]
        self.width = bounds[:, 1] - bounds[:, 0]
        self.settings = {
            "dimension": len(bounds),
            "method": method,
            "atol": atol,
            "rtol": rtol,
            "points": points_per_axis,
            "max_points": max_points_per_axis,
            "max_evaluations": max_evaluations,
            "rate": rate,
        }
        self.command = [julia, "--startup-file=no", "--threads=1"]
        if project is not None:
            self.command.append(f"--project={project}")
        self.command.append(str(Path(__file__).with_name("_autobz.jl")))
        self._process: subprocess.Popen[str] | None = None
        atexit.register(self.close)

    def __enter__(self) -> Self:
        """Return this k-integrator."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the Julia worker."""
        self.close()

    def integrate(
        self,
        integrand: KIntegrand | Mapping[Hashable, NDArray[np.inexact]],
    ) -> KIntegrationResult:
        """Integrate a function that can be evaluated at newly chosen k-points."""
        with stage(self.diagnostics, "autobz"):
            if not callable(integrand):
                msg = "AutoBZ requires an integrand callable at new k-points."
                raise TypeError(msg)
            probe = integrand((self.lower + self.width / 2)[None, :])
            shapes = {key: value.shape[1:] for key, value in probe.items()}
            size = sum(int(np.prod(shape)) for shape in shapes.values())
            if self._process is None:
                self._process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self._receive()
            evaluations = 1
            try:
                self._send(self.settings | {"size": size})
                while True:
                    message = self._receive()
                    if message["kind"] == "evaluate":
                        kpoints = (
                            self.lower
                            + self.width * np.asarray(message["kpoints"])[None, :]
                        )
                        values = integrand(kpoints)
                        packed = np.concatenate(
                            [values[key][0].ravel() for key in shapes]
                        )
                        self._send(
                            {"real": packed.real.tolist(), "imag": packed.imag.tolist()}
                        )
                        evaluations += 1
                        continue
                    if message["kind"] == "error":
                        raise RuntimeError(message["message"])
                    packed = np.asarray(message["real"]) + 1j * np.asarray(
                        message["imag"]
                    )
                    offset = 0
                    integrated = {}
                    for key, shape in shapes.items():
                        count = int(np.prod(shape))
                        integrated[key] = packed[offset : offset + count].reshape(shape)
                        offset += count
                    result = KIntegrationResult(
                        integrated,
                        error=message["error"],
                        converged=message["converged"],
                        n_evaluations=evaluations,
                    )
                    if self.diagnostics is not None:
                        self.diagnostics.emit(
                            "autobz.result",
                            method=self.settings["method"],
                            n_evaluations=evaluations,
                            quadrature_error=result.error,
                            converged=result.converged,
                        )
                    return result
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Stop the Julia worker."""
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()

    def _send(self, message: dict[str, Any]) -> None:
        assert self._process is not None
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self._process.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        assert self._process is not None
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            msg = "The AutoBZ Julia worker exited; check its environment and stderr."
            raise RuntimeError(msg)
        message: dict[str, Any] = json.loads(line)
        return message
