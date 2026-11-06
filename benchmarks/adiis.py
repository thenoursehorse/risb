"""Compare nonmonotone ADIIS-inspired mixing with BFGS and DIIS at T=0.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run python benchmarks/adiis.py --output _build/adiis/results.json

Restricted to the half-filled paramagnetic one-band Bethe model, B=1 or 3.
The model uses reduced-field energy gradients; proposals mix mapped fields.
No actual-energy line search is applied to ADIIS.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, override

import numpy as np
from fockit.algebra import c, cdag, hc, nc, zero
from fockit.hilbert import to_matrix
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.optimize import minimize, root

from risb import LatticeSolver, helpers
from risb.impurity_solvers import FockitImpuritySolver
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator
from risb.types import BlockMatrix

logger = logging.getLogger(__name__)
SPINS = ("up", "dn")
type Array = NDArray[np.float64]


@dataclass
class Sample:
    """One constrained energy evaluation and its updated embedding fields."""

    y: Array
    energy: float
    gradient: Array
    mapped: Array
    residual: float
    R: Array
    Lambda: Array
    double_occupancy: float
    z: float


class EnergyObjective(Protocol):
    """A reduced objective with a proposed field update."""

    def evaluate(self, y: Array) -> Sample:
        """Evaluate an admissible state and its mapped fields."""
        ...


class CountedImpurity(FockitImpuritySolver):
    """Count interacting ground-state solves."""

    solves = 0

    @override
    def solve(self) -> None:
        self.solves += 1
        super().solve()


class BetheObjective:
    """Reduced embedding-field energy with a dense adjoint response."""

    def __init__(self, bath_size: int, interaction: float = 1.0, nk: int = 400) -> None:
        self.bath_size = bath_size
        energy_grid = np.linspace(-1, 1, 100001)
        cdf = (
            0.5
            + (energy_grid * np.sqrt(1 - energy_grid**2) + np.arcsin(energy_grid))
            / np.pi
        )
        self.eps = np.interp((np.arange(nk) + 0.5) / nk, cdf, energy_grid)
        self.h_loc = interaction * (
            nc("up", 0) * nc("dn", 0) - 0.5 * (nc("up", 0) + nc("dn", 0))
        )
        self.impurity = CountedImpurity(
            self.h_loc,
            [(bl, 1) for bl in SPINS],
            bath_size=dict.fromkeys(SPINS, bath_size),
            eigensolver="eigh",
        )
        self.solver = LatticeSolver(
            dict.fromkeys(SPINS, self.eps[:, None, None]),
            [(bl, 1) for bl in SPINS],
            self.impurity,
            BandKIntegrator(WeightedKPointIntegrator(), beta=1e6, n_target=bath_size),
            symmetrizers=[paramagnetic],
            force_real=True,
            bath_size=[dict.fromkeys(SPINS, bath_size)],
        )
        reference = self.solver.solve(tol=1e-11, maxiter=500)
        if not reference.converged:
            msg = "Reference DIIS did not converge."
            raise RuntimeError(msg)
        self.reference_y = pack_fields(
            self.solver.D[0]["up"].real, self.solver.Lambda_c[0]["up"].real
        )
        if bath_size == 1:
            self.basis = np.zeros((0, 1, 1))
            self.band_initial = np.zeros(0)
        else:
            matrices = [
                np.diag([1.0, -1.0, 0.0]) / np.sqrt(2),
                np.diag([1.0, 1.0, -2.0]) / np.sqrt(6),
            ]
            for a, b in ((0, 1), (0, 2), (1, 2)):
                matrix = np.zeros((3, 3))
                matrix[a, b] = matrix[b, a] = 1 / np.sqrt(2)
                matrices.append(matrix)
            self.basis = np.array(matrices)
            _, rotation = np.linalg.eigh(self.solver.Lambda_c[0]["up"].real)
            signs = np.where(
                (rotation.T @ self.solver.D[0]["up"]).real.ravel() < 0, -1, 1
            )
            rotation *= signs
            self.band_initial = np.einsum(
                "aij,ij->a",
                self.basis,
                rotation.T @ self.solver.Lambda[0]["up"].real @ rotation,
            )
        basis = self.impurity.ensemble[0][0]
        self.fixed_h = np.asarray(to_matrix(self.h_loc, basis).toarray().real)
        self.hybrid = []
        self.bath = {}
        for a in range(bath_size):
            operator = sum((cdag(bl, 0) * c("bath_" + bl, a) for bl in SPINS), zero())
            self.hybrid.append(
                np.asarray(to_matrix(operator + hc(operator), basis).toarray().real)
            )
            for b in range(a, bath_size):
                operator = sum(
                    (c("bath_" + bl, a) * cdag("bath_" + bl, b) for bl in SPINS), zero()
                )
                if a != b:
                    operator += hc(operator)
                self.bath[a, b] = np.asarray(to_matrix(operator, basis).toarray().real)
        self.perturbations = (
            [self.hybrid[0]]
            if bath_size == 1
            else [
                self.hybrid[1],
                self.hybrid[0] + self.hybrid[2],
                -self.bath[0, 0] + self.bath[2, 2],
            ]
        )
        self.responses = 0
        self.band_evaluations = 0

    def bands(self, R: Array, target: Array) -> tuple[Array, Array, Array]:
        """Match the shared lattice density without interacting solves."""

        def state(coefficients: Array) -> tuple[Array, Array, Array]:
            self.band_evaluations += 1
            Lambda = np.einsum("a,aij->ij", coefficients, self.basis)
            values, vectors = np.linalg.eigh(
                self.eps[:, None, None] * (R @ R.T) + Lambda
            )
            occupations = np.zeros_like(values)
            occupations.ravel()[np.argsort(values.ravel())[: values.size // 2]] = 1
            density = (vectors * occupations[:, None, :]) @ vectors.transpose(0, 2, 1)
            return (
                Lambda,
                density.mean(axis=0),
                (self.eps[:, None, None] * density).mean(axis=0),
            )

        coefficients = self.band_initial
        if self.bath_size == 3:

            def residual(v: Array) -> Array:
                return np.asarray(
                    np.einsum("aij,ij->a", self.basis, state(v)[1] - target),
                    dtype=np.float64,
                )

            result = root(residual, coefficients, tol=1e-10)
            coefficients = result.x
        Lambda, density, kinetic = state(coefficients)
        if np.max(np.abs(density - target)) > 1e-8:
            msg = "Lattice target was not matched."
            raise RuntimeError(msg)
        return Lambda, density, kinetic

    def evaluate(self, y: Array) -> Sample:
        """Evaluate the constrained energy, full field residual, and gradient."""
        D, Lambda_c = unpack_fields(y)
        self.impurity.set_h_imp(dict.fromkeys(SPINS, Lambda_c), dict.fromkeys(SPINS, D))
        self.impurity.solve()
        density = self.impurity.get_rho_f("up").real
        cross = self.impurity.get_rho_cf("up").real
        R = helpers.get_r(cross, density).real
        Lambda, rho_qp, kinetic = self.bands(R, density)
        D_expected = helpers.get_d(rho_qp, (kinetic @ R).T).real
        C_expected = helpers.get_lambda_c(rho_qp, R, Lambda, D_expected).real
        residual = float(
            max(np.max(np.abs(D_expected - D)), np.max(np.abs(C_expected - Lambda_c)))
        )
        observable = np.zeros_like(self.fixed_h)
        for a, operator in enumerate(self.hybrid):
            observable += (D_expected[a, 0] - D[a, 0]) * operator
        for (a, b), operator in self.bath.items():
            observable += (C_expected[a, b] - Lambda_c[a, b]) * operator
        H = self.fixed_h.copy()
        for value, operator in zip(y, self.perturbations, strict=True):
            H += value * operator
        psi = self.impurity.ensemble[0][1][:, 0]
        if np.linalg.norm(H @ psi - self.impurity.gs_energy * psi) > 1e-8:
            msg = "Cached Hamiltonian disagrees with the impurity ground state."
            raise RuntimeError(msg)
        rhs = -(observable @ psi - psi * np.vdot(psi, observable @ psi))
        response = solve(
            H - self.impurity.gs_energy * np.eye(len(H)) + np.outer(psi, psi.conj()),
            rhs,
            assume_a="pos",
        )
        self.responses += 1
        gradient = np.array(
            [
                2 * np.vdot(response, operator @ psi).real
                for operator in self.perturbations
            ]
        )
        local = float(np.real(self.impurity.overlap(self.h_loc)))
        return Sample(
            y.copy(),
            float(2 * np.sum(kinetic * (R @ R.T)) + local),
            gradient,
            pack_fields(D_expected, C_expected),
            residual,
            R,
            Lambda,
            float(np.real(self.impurity.overlap(nc("up", 0) * nc("dn", 0)))),
            float(helpers.get_z(R, Lambda)[0, 0].real),
        )


def paramagnetic(matrices: list[BlockMatrix]) -> list[BlockMatrix]:
    """Average the spin blocks."""
    for matrix in matrices:
        mean = (matrix["up"] + matrix["dn"]) / 2
        matrix["up"] = mean
        matrix["dn"] = mean
    return matrices


def pack_fields(D: Array, Lambda_c: Array) -> Array:
    """Choose sorted bath levels and positive hybridizations."""
    levels, rotation = np.linalg.eigh(Lambda_c)
    hybrid = np.abs((rotation.T @ D).ravel())
    if len(levels) == 1:
        return np.array([hybrid[0]])
    return np.array(
        [hybrid[1], (hybrid[0] + hybrid[2]) / 2, (levels[2] - levels[0]) / 2]
    )


def unpack_fields(y: Array) -> tuple[Array, Array]:
    """Construct the particle-hole symmetric embedding fields."""
    if len(y) == 1:
        return y.reshape(1, 1), np.zeros((1, 1))
    return np.array([[y[1]], [y[0]], [y[1]]]), np.diag([-y[2], 0, y[2]])


def adiis_weights(history: list[Sample]) -> Array:
    """Minimize the stored ARH model using normalized squared coefficients."""
    points = np.array([sample.y for sample in history])
    gradients = np.array([sample.gradient for sample in history])
    displacements = points - points[-1]
    curvature = displacements @ (gradients - gradients[-1]).T
    curvature = (curvature + curvature.T) / 2
    linear = displacements @ gradients[-1]
    scale = max(float(np.max(np.abs(curvature))), float(np.max(np.abs(linear))), 1e-30)
    curvature, linear = curvature / scale, linear / scale

    def model(t: Array) -> tuple[float, Array]:
        norm = float(t @ t)
        weights = t**2 / norm
        derivative = linear + curvature @ weights
        value = linear @ weights + weights @ curvature @ weights / 2
        return float(value), 2 * t * (derivative - weights @ derivative) / norm

    candidates = list(np.eye(len(history)))
    for start in (np.ones(len(history)), np.r_[np.full(len(history) - 1, 0.1), 1.0]):
        result = minimize(
            model,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 500},
        )
        candidates.append(result.x**2 / (result.x @ result.x))
    return min(
        candidates,
        key=lambda weights: float(linear @ weights + weights @ curvature @ weights / 2),
    )


def run_adiis(
    objective: EnergyObjective,
    initial: Array,
    *,
    mixing: float = 1.0,
    maxiter: int = 300,
    history_size: int = 6,
) -> tuple[Sample, list[dict[str, float]]]:
    """Mix mapped fields without rejecting actual-energy increases."""
    history: list[Sample] = []
    trace = []
    y = initial.copy()
    sample = objective.evaluate(y)
    for iteration in range(maxiter):
        if iteration:
            sample = objective.evaluate(y)
        trace.append({"energy": sample.energy, "residual": sample.residual})
        if sample.residual < 1e-7:
            return sample, trace
        history = [*history, sample][-history_size:]
        weights = adiis_weights(history)
        inputs = sum(
            (w * s.y for w, s in zip(weights, history, strict=True)), np.zeros_like(y)
        )
        outputs = sum(
            (w * s.mapped for w, s in zip(weights, history, strict=True)),
            np.zeros_like(y),
        )
        y = inputs + mixing * (outputs - inputs)
    return sample, trace


def benchmark(
    bath_size: int, factors: Array, repeats: int, maxiter: int
) -> list[dict[str, Any]]:
    """Compare complete solvers from the same physical initial state."""
    problem = BetheObjective(bath_size)
    reference = problem.evaluate(problem.reference_y)
    initial = reference.y * factors
    start = problem.evaluate(initial)
    problem.band_initial = np.einsum("aij,ij->a", problem.basis, start.Lambda)
    finite_gradient = []
    for direction in np.eye(len(initial)):
        plus = problem.evaluate(initial + 1e-5 * direction).energy
        minus = problem.evaluate(initial - 1e-5 * direction).energy
        finite_gradient.append((plus - minus) / 2e-5)
    gradient_error = float(np.max(np.abs(start.gradient - finite_gradient)))
    if gradient_error > 1e-6:
        msg = f"Adjoint gradient check failed: {gradient_error}"
        raise RuntimeError(msg)
    records = []
    names = ["DIIS", "BFGS", "ADIIS", "ADIIS-half"]
    for repeat in range(repeats):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            problem.impurity.solves = problem.responses = problem.band_evaluations = 0
            trace: list[dict[str, float]] = []
            t0 = perf_counter()
            try:
                if name == "DIIS":
                    problem.solver.R = [dict.fromkeys(SPINS, start.R.copy())]
                    problem.solver.Lambda = [dict.fromkeys(SPINS, start.Lambda.copy())]
                    diis_result = problem.solver.solve(tol=1e-10, maxiter=maxiter)
                    y = pack_fields(
                        problem.solver.D[0]["up"].real,
                        problem.solver.Lambda_c[0]["up"].real,
                    )
                    optimizer_success = diis_result.converged
                    message = (
                        "converged" if diis_result.converged else "evaluation limit"
                    )
                elif name == "BFGS":

                    def value(y: Array) -> tuple[float, Array]:
                        sample = problem.evaluate(y)
                        return sample.energy, sample.gradient

                    bfgs_result = minimize(
                        value,
                        initial.copy(),
                        jac=True,
                        method="BFGS",
                        options={"gtol": 1e-8, "maxiter": maxiter},
                    )
                    y = bfgs_result.x
                    optimizer_success, message = (
                        bool(bfgs_result.success),
                        str(bfgs_result.message),
                    )
                else:
                    final, trace = run_adiis(
                        problem,
                        initial,
                        mixing=0.5 if name == "ADIIS-half" else 1.0,
                        maxiter=maxiter,
                    )
                    y = final.y
                    optimizer_success = final.residual < 1e-7
                    message = "converged" if optimizer_success else "evaluation limit"
                elapsed = perf_counter() - t0
                solves, responses, bands = (
                    problem.impurity.solves,
                    problem.responses,
                    problem.band_evaluations,
                )
                final = problem.evaluate(y)
                record = {
                    "B": bath_size,
                    "initial_factors": factors.tolist(),
                    "method": name,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "impurity_solves": solves,
                    "response_solves": responses,
                    "reduced_band_evaluations": None if name == "DIIS" else bands,
                    "optimizer_success": optimizer_success,
                    "message": message,
                    "physical_converged": final.residual < 1e-7,
                    "residual": final.residual,
                    "energy": final.energy,
                    "energy_error": abs(final.energy - reference.energy),
                    "docc_error": abs(
                        final.double_occupancy - reference.double_occupancy
                    ),
                    "z_error": abs(final.z - reference.z),
                    "gradient_check_error": gradient_error,
                    "uphill_steps": (
                        sum(
                            b["energy"] > a["energy"] + 1e-12
                            for a, b in pairwise(trace)
                        )
                        if trace
                        else None
                    ),
                    "largest_energy_rise": max(
                        (b["energy"] - a["energy"] for a, b in pairwise(trace)),
                        default=0.0,
                    )
                    if trace
                    else None,
                    "trace": trace,
                }
            except (RuntimeError, np.linalg.LinAlgError) as error:
                record = {
                    "B": bath_size,
                    "initial_factors": factors.tolist(),
                    "method": name,
                    "repeat": repeat,
                    "physical_converged": False,
                    "message": str(error),
                }
            records.append(record)
            logger.info(
                "%s",
                json.dumps(
                    {key: value for key, value in record.items() if key != "trace"}
                ),
            )
    return records


def main() -> None:
    """Run the local metallic restart comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("_build/adiis/results.json")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("risb").setLevel(logging.WARNING)
    records = []
    for bath_size in (1, 3):
        signs = np.array([1.0]) if bath_size == 1 else np.array([1.0, -1.0, 1.0])
        starts = [1 + perturbation * signs for perturbation in (0.03, 0.3)]
        if bath_size == 3:
            starts.append(np.array([0.5, 0.5, 1.5]))
        for factors in starts:
            records.extend(benchmark(bath_size, factors, args.repeats, args.maxiter))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
