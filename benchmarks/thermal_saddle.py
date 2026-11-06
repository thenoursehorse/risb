"""Small finite-temperature RISB saddle and nested-maximum experiments."""

import argparse
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from adiis import SPINS, Array
from adiis_hard import ThermalObjective, timeout
from fockit.algebra import c, cdag, hc, nc, zero
from fockit.hilbert import to_matrix
from scipy.special import expit

from risb import helpers

logger = logging.getLogger(__name__)


@dataclass
class ThermalSample:
    """A full quadratic/impurity stationarity evaluation."""

    x: Array
    y: Array
    value: float
    gradient: Array
    residual: float
    double_occupancy: float


class ThermalSaddle:
    """Real B=3 fields with independent quadratic and bath rotation gauges."""

    def __init__(self, temperature: float, nk: int = 400) -> None:
        setup = ThermalObjective(temperature, nk=nk, symmetric=False)
        self.beta = setup.beta
        self.eps = setup.eps
        self.impurity = setup.impurity
        self.impurity.eta = 200.0
        self.solver = setup.solver
        self.basis = np.array([np.diag(row) for row in np.eye(3)])
        levels, rotation = np.linalg.eigh(self.solver.Lambda_c[0]["up"].real)
        D = np.abs(rotation.T @ self.solver.D[0]["up"].real)
        self.reference_y = np.r_[D.ravel(), levels]
        levels, rotation = np.linalg.eigh(self.solver.Lambda[0]["up"].real)
        R = np.abs(rotation.T @ self.solver.R[0]["up"].real)
        self.reference_x = np.r_[R.ravel(), levels]
        self.inner_x = self.reference_x.copy()
        self.quadratic_evaluations = 0
        self.response_evaluations = 0
        self.quadratic_responses = 0
        self.budget = 200
        self.operator_cache: dict[int, Array] = {}
        self.impurity.solves = 0
        self.cached_y: Array | None = None
        self.cached_impurity: tuple[float, Array, float] | None = None
        self.transform = np.zeros((6, 6))
        self.transform[:3, :3] = np.eye(3)
        self.transform[3:, 3] = np.array([1, -1, 0]) / np.sqrt(2)
        self.transform[3:, 4] = np.array([1, 1, -2]) / np.sqrt(6)
        self.transform[3:, 5] = 1 / np.sqrt(3)

    def matrices(self, x: Array, y: Array) -> tuple[Array, Array, Array, Array]:
        """Unpack fields in their independent rotation gauges."""
        return (
            x[:3, None],
            np.einsum("a,aij->ij", x[3:], self.basis),
            y[:3, None],
            np.diag(y[3:]),
        )

    def quadratic(self, x: Array, y: Array) -> tuple[float, Array, float]:
        """Evaluate the quadratic functional and all its field derivatives."""
        self.quadratic_evaluations += 1
        R, Lambda, D, Lambda_c = self.matrices(x, y)
        values, vectors = np.linalg.eigh(self.eps[:, None, None] * (R @ R.T) + Lambda)
        density_k = (vectors * expit(-self.beta * values)[:, None, :]) @ (
            vectors.transpose(0, 2, 1)
        )
        density = density_k.mean(axis=0)
        kinetic = (self.eps[:, None, None] * density_k).mean(axis=0) @ R
        n0, cross, bath = helpers.get_h0_emb_densities(
            R, Lambda, D, Lambda_c, self.beta
        )
        value = -2 * np.mean(
            np.logaddexp(0, -self.beta * values).sum(axis=1)
        ) / self.beta - 2 * helpers.get_h0_emb_grand_potential(
            R, Lambda, D, Lambda_c, self.beta
        )
        gx = np.r_[
            4 * (kinetic - cross @ D).ravel(),
            2 * np.einsum("aij,ij->a", self.basis, density - n0),
        ].real
        gy = np.r_[-4 * (cross.T @ R).ravel(), -2 * np.diag(bath)].real
        error = max(np.max(np.abs(density - n0)), np.max(np.abs(kinetic - cross @ D)))
        return float(value), np.r_[gx, gy], float(error)

    def interacting(self, y: Array) -> tuple[float, Array, float]:
        """Evaluate one Gibbs impurity, reusing it for quadratic variations."""
        if self.cached_y is not None and np.array_equal(y, self.cached_y):
            assert self.cached_impurity is not None
            return self.cached_impurity
        self.impurity.set_h_imp(
            dict.fromkeys(SPINS, np.diag(y[3:])),
            dict.fromkeys(SPINS, y[:3, None]),
            mu=0,
        )
        if self.impurity.solves >= self.budget:
            msg = "Impurity solve budget exceeded"
            raise RuntimeError(msg)
        self.impurity.solve()
        bath = self.impurity.get_rho_f("up").real
        cross = self.impurity.get_rho_cf("up").real
        gradient = np.r_[4 * cross.ravel(), 2 * np.diag(bath)]
        docc = float(self.impurity.overlap(nc("up", 0) * nc("dn", 0)).real)
        self.cached_y = y.copy()
        self.cached_impurity = self.impurity.grand_potential, gradient, docc
        return self.cached_impurity

    def joint(self, x: Array, y: Array) -> ThermalSample:
        """Evaluate the coupled stationarity equations without an inner fit."""
        value, gradient, residual = self.quadratic(x, y)
        omega, impurity_gradient, docc = self.interacting(y)
        gradient[6:] += impurity_gradient
        R, Lambda, D, Lambda_c = self.matrices(x, y)
        _, cross, bath = helpers.get_h0_emb_densities(R, Lambda, D, Lambda_c, self.beta)
        residual = float(
            max(
                residual,
                np.max(np.abs(self.impurity.get_rho_f("up").real - bath)),
                np.max(np.abs(self.impurity.get_rho_cf("up").real - R.T @ cross)),
            )
        )
        return ThermalSample(
            x.copy(), y.copy(), value + omega, gradient, residual, docc
        )

    def reduced(self, y: Array) -> ThermalSample:
        """Eliminate quadratic fields on the continued gauge-fixed branch."""
        x = self.inner_x.copy()
        for _ in range(40):
            _, gradient, error = self.quadratic(x, y)
            if error < 1e-9:
                self.inner_x = x.copy()
                return self.joint(x, y)
            hessian = self.quadratic_hessian(x, y)[:6, :6]
            direction = newton_direction(hessian, gradient[:6])
            for scale in 0.5 ** np.arange(12):
                trial = x + scale * direction
                if np.linalg.norm(self.quadratic(trial, y)[1][:6]) < np.linalg.norm(
                    gradient[:6]
                ):
                    x = trial
                    break
            else:
                break
        error = self.quadratic(x, y)[2]
        msg = f"Quadratic inner matching failed: {error:.3g}"
        raise RuntimeError(msg)

    def quadratic_hessian(self, x: Array, y: Array) -> Array:
        """Differentiate quadratic band and reference Gibbs densities."""
        self.quadratic_responses += 1
        R, Lambda, D, Lambda_c = self.matrices(x, y)
        hessian = np.zeros((12, 12))
        directions = np.eye(3)[:, :, None]
        for epsilon in self.eps:
            derivatives = np.concatenate(
                (
                    epsilon * (directions @ R.T + R @ directions.transpose(0, 2, 1)),
                    self.basis,
                )
            )
            density, response = helpers.get_fermi_density_response(
                epsilon * (R @ R.T) + Lambda, derivatives, self.beta
            )
            hessian[:6, :6] += (
                2 * np.einsum("bij,aij->ab", response, derivatives).real / len(self.eps)
            )
            hessian[:3, :3] += 4 * epsilon * density.real / len(self.eps)
        derivatives = np.zeros((12, 6, 6))
        derivatives[:3, :3, 3:] = directions @ D.T
        derivatives[3:6, :3, :3] = self.basis
        derivatives[6:9, :3, 3:] = R @ directions.transpose(0, 2, 1)
        derivatives[:, 3:, :3] = derivatives[:, :3, 3:].transpose(0, 2, 1)
        derivatives[9:, 3:, 3:] = -np.array([np.diag(row) for row in np.eye(3)])
        density, response = helpers.get_fermi_density_response(
            helpers.get_h0_emb(R, Lambda, D, Lambda_c), derivatives, self.beta
        )
        hessian -= 2 * np.einsum("bij,aij->ab", response, derivatives).real
        hessian[:3, 6:9] -= 4 * density[:3, 3:].real
        hessian[6:9, :3] -= 4 * density[3:, :3].real
        return (hessian + hessian.T) / 2

    def hessian(self, sample: ThermalSample) -> Array:
        """Evaluate exact quadratic and complete-Gibbs spectral responses."""
        self.response_evaluations += 1
        hessian = self.quadratic_hessian(sample.x, sample.y)
        _, gradient, _ = self.interacting(sample.y)
        impurity_hessian = self.beta * np.outer(gradient, gradient)
        operators = []
        for a in range(3):
            operator = sum((cdag(bl, 0) * c("bath_" + bl, a) for bl in SPINS), zero())
            operators.append(operator + hc(operator))
        for a in range(3):
            operators.append(
                sum(
                    (c("bath_" + bl, a) * cdag("bath_" + bl, a) for bl in SPINS),
                    zero(),
                )
            )
        retained = 0
        for basis, states, weights in self.impurity.ensemble:
            retained += states.shape[1]
            if id(basis) not in self.operator_cache:
                self.operator_cache[id(basis)] = np.array(
                    [to_matrix(op, basis).toarray().real for op in operators]
                )
            rotated = states.conj().T @ self.operator_cache[id(basis)] @ states
            logarithms = np.log(weights)
            distance = logarithms[:, None] - logarithms[None, :]
            divided = np.divide(
                weights[:, None] - weights[None, :],
                distance,
                out=(weights[:, None] + weights[None, :]) / 2,
                where=np.abs(distance) > 1e-10,
            )
            impurity_hessian -= (
                self.beta
                * np.einsum("ij,aij,bij->ab", divided, rotated, rotated.conj()).real
            )
        if retained != 256:
            msg = f"Complete Gibbs response requires 256 states; retained {retained}"
            raise RuntimeError(msg)
        hessian[6:, 6:] += impurity_hessian
        return (hessian + hessian.T) / 2


def inverse_hessian(hessian: Array) -> Array:
    """Invert resolved signed curvature, leaving numerically flat modes fixed."""
    values, vectors = np.linalg.eigh(hessian)
    cutoff = max(1e-10, 1e-9 * float(np.max(np.abs(values))))
    inverse = np.divide(
        1, values, out=np.zeros_like(values), where=np.abs(values) > cutoff
    )
    return (vectors * inverse) @ vectors.T


def newton_direction(hessian: Array, gradient: Array) -> Array:
    """Limit a signed Newton step without changing the target equations."""
    direction = -inverse_hessian(hessian) @ gradient
    direction *= min(1.0, 0.25 / max(float(np.max(np.abs(direction))), 1e-30))
    return direction


def record_sample(record: dict[str, Any], sample: ThermalSample) -> None:
    """Record the full physical residual, potential, and stationarity error."""
    record.update(
        x=sample.x.tolist(),
        y=sample.y.tolist(),
        value=sample.value,
        residual=sample.residual,
        gradient_norm=float(np.max(np.abs(sample.gradient))),
        double_occupancy=sample.double_occupancy,
    )
    record["trace"].append(
        {key: record[key] for key in ("value", "residual", "gradient_norm")}
    )


def stationary(sample: ThermalSample) -> bool:
    """Require the original matrix equations as well as the field gradient."""
    return bool(sample.residual < 1e-7 and np.max(np.abs(sample.gradient)) < 1e-8)


def run_coupled(
    problem: ThermalSaddle,
    x: Array,
    y: Array,
    record: dict[str, Any],
    *,
    extragradient: bool,
) -> None:
    """Update every stationary field together, with no nested impurity fits."""
    sample = problem.joint(x, y)
    metric: Array | None = None
    for _ in range(100):
        record_sample(record, sample)
        if stationary(sample):
            record.update(stationary_converged=True, termination="stationary")
            return
        point = np.r_[sample.x, sample.y]
        norm = np.linalg.norm(sample.gradient)
        if not extragradient:
            direction = newton_direction(problem.hessian(sample), sample.gradient)
        else:
            metric = inverse_hessian(problem.hessian(sample))
            direction = -metric @ sample.gradient
            direction *= min(0.5, 0.25 / max(float(np.max(np.abs(direction))), 1e-30))
        for scale in 0.5 ** np.arange(12):
            trial = point + scale * direction
            predicted = problem.joint(trial[:6], trial[6:])
            if extragradient:
                assert metric is not None
                alpha = min(
                    0.5,
                    0.25 / max(float(np.max(np.abs(metric @ sample.gradient))), 1e-30),
                )
                trial = point - scale * alpha * metric @ predicted.gradient
                candidate = problem.joint(trial[:6], trial[6:])
            else:
                candidate = predicted
            if np.linalg.norm(candidate.gradient) < (1 - 1e-4 * scale) * norm:
                sample = candidate
                break
        else:
            msg = "Coupled stationarity line search stalled"
            raise RuntimeError(msg)
    record["termination"] = "iteration_limit"


def run_nested(
    problem: ThermalSaddle,
    x: Array,
    y: Array,
    record: dict[str, Any],
    *,
    split: str,
) -> None:
    """Maximize the reduced auxiliary block inside each outer minimum step."""
    problem.inner_x = x.copy()
    sample = problem.reduced(y)
    hessian = problem.hessian(sample)
    reduced = hessian[6:, 6:] - hessian[6:, :6] @ np.linalg.solve(
        hessian[:6, :6], hessian[:6, 6:]
    )
    if split == "common":
        transform = problem.transform
        nu = 5
    elif split == "PH":
        transform = np.zeros((6, 6))
        transform[:3, 0] = [0, 1, 0]
        transform[:3, 1] = np.array([1, 0, 1]) / np.sqrt(2)
        transform[3:, 2] = np.array([-1, 0, 1]) / np.sqrt(2)
        transform[:3, 3] = np.array([1, 0, -1]) / np.sqrt(2)
        transform[3:, 4] = np.array([1, -2, 1]) / np.sqrt(6)
        transform[3:, 5] = 1 / np.sqrt(3)
        nu = 3
    else:
        values, vectors = np.linalg.eigh(reduced)
        negative = values < -1e-8
        transform = np.column_stack((vectors[:, ~negative], vectors[:, negative]))
        nu = int(np.count_nonzero(~negative))
        record["initial_reduced_eigenvalues"] = values.tolist()
    record["maximizing_coordinates"] = 6 - nu
    rotation = np.zeros((12, 12))
    rotation[:6, :6] = np.eye(6)
    rotation[6:, 6:] = transform
    inner = np.r_[np.arange(6), np.arange(6 + nu, 12)]
    outer = np.arange(6, 6 + nu)
    u, v = (transform.T @ sample.y)[:nu], (transform.T @ sample.y)[nu:]

    def maximize(
        target: Array, initial_x: Array, initial_v: Array
    ) -> tuple[ThermalSample, Array]:
        point = np.r_[initial_x, initial_v]
        for _ in range(30):
            state = problem.joint(point[:6], transform @ np.r_[target, point[6:]])
            gradient = rotation.T @ state.gradient
            projected = rotation.T @ problem.hessian(state) @ rotation
            if np.max(np.abs(gradient[inner])) < 2e-9:
                error = problem.quadratic(state.x, state.y)[2]
                if error > 1e-8:
                    msg = f"Auxiliary quadratic matrix matching failed: {error:.3g}"
                    raise RuntimeError(msg)
                record["inner_matrix_residual"] = error
                block = projected[6:, 6:] - projected[6:, :6] @ np.linalg.solve(
                    projected[:6, :6], projected[:6, 6:]
                )
                largest = float(np.max(np.linalg.eigvalsh(block[nu:, nu:])))
                if largest >= -1e-8:
                    msg = f"Auxiliary stationary block is not strictly concave: {largest:.3g}"
                    raise RuntimeError(msg)
                record["auxiliary_largest_curvature"] = largest
                return state, projected
            direction = newton_direction(
                projected[np.ix_(inner, inner)], gradient[inner]
            )
            for scale in 0.5 ** np.arange(12):
                trial = point + scale * direction
                candidate = problem.joint(
                    trial[:6], transform @ np.r_[target, trial[6:]]
                )
                trial_gradient = rotation.T @ candidate.gradient
                if np.linalg.norm(trial_gradient[inner]) < np.linalg.norm(
                    gradient[inner]
                ):
                    point = trial
                    break
            else:
                msg = "Auxiliary stationarity line search stalled"
                raise RuntimeError(msg)
        msg = "Auxiliary stationarity iteration limit"
        raise RuntimeError(msg)

    for _ in range(50):
        sample, projected = maximize(u, sample.x, v)
        v = (transform.T @ sample.y)[nu:]
        record_sample(record, sample)
        schur = projected[np.ix_(outer, outer)] - (
            projected[np.ix_(outer, inner)]
            @ np.linalg.solve(
                projected[np.ix_(inner, inner)], projected[np.ix_(inner, outer)]
            )
        )
        minimum = float(np.min(np.linalg.eigvalsh(schur)))
        record["outer_smallest_curvature"] = minimum
        record["stationary_converged"] = stationary(sample)
        if minimum < -1e-7:
            msg = f"Remaining negative curvature after auxiliary maximum: {minimum:.3g}"
            raise RuntimeError(msg)
        if stationary(sample):
            record.update(
                minimum_converged=True,
                strict_minimum=minimum > 1e-7,
                termination="stationary_minimum",
            )
            return
        gradient = (rotation.T @ sample.gradient)[outer]
        direction = newton_direction(schur, gradient)
        for scale in 0.5 ** np.arange(10):
            trial_u = u + scale * direction
            predictor = np.r_[sample.x, v] - np.linalg.solve(
                projected[np.ix_(inner, inner)],
                projected[np.ix_(inner, outer)] @ (trial_u - u),
            )
            try:
                candidate, _ = maximize(trial_u, predictor[:6], predictor[6:])
            except RuntimeError as error:
                if problem.impurity.solves >= problem.budget:
                    raise
                record["rejected_inner_trials"] = (
                    record.get("rejected_inner_trials", 0) + 1
                )
                record["last_inner_failure"] = str(error)
                continue
            if candidate.value <= (
                sample.value + 1e-4 * scale * gradient @ direction + 1e-13
            ):
                u, sample = trial_u, candidate
                v = (transform.T @ sample.y)[nu:]
                break
        else:
            msg = "Outer minimum line search stalled"
            raise RuntimeError(msg)
    record["termination"] = "iteration_limit"


def run_root(
    problem: ThermalSaddle, x: Array, y: Array, record: dict[str, Any]
) -> None:
    """Compare default production root DIIS on the same initial matrices."""
    R, Lambda, D, Lambda_c = problem.matrices(x, y)
    for name, matrix in zip(
        ("R", "Lambda", "D", "Lambda_c"),
        (R, Lambda, D, Lambda_c),
        strict=True,
    ):
        setattr(problem.solver, name, [dict.fromkeys(SPINS, matrix.copy())])
    result = problem.solver.solve(tol=1e-10, maxiter=problem.budget)
    solves = problem.impurity.solves
    levels, rotation = np.linalg.eigh(problem.solver.Lambda[0]["up"].real)
    final_x = np.r_[np.abs(rotation.T @ problem.solver.R[0]["up"]).ravel(), levels]
    levels, rotation = np.linalg.eigh(problem.solver.Lambda_c[0]["up"].real)
    final_y = np.r_[np.abs(rotation.T @ problem.solver.D[0]["up"]).ravel(), levels]
    problem.cached_y = None
    problem.budget += 1
    sample = problem.joint(final_x, final_y)
    record_sample(record, sample)
    record.update(
        stationary_converged=stationary(sample),
        optimizer_converged=result.converged,
        solver_impurity_solves=solves,
        validation_impurity_solves=problem.impurity.solves - solves,
        termination="stationary" if stationary(sample) else "physical_check_failed",
    )


def run_method(
    problem: ThermalSaddle,
    x: Array,
    y: Array,
    method: str,
    *,
    budget: int = 200,
) -> dict[str, Any]:
    """Count every trial impurity solve, including rejected and inner steps."""
    problem.impurity.solves = 0
    problem.quadratic_evaluations = 0
    problem.response_evaluations = 0
    problem.quadratic_responses = 0
    problem.budget = budget
    problem.cached_y = None
    record: dict[str, Any] = {
        "method": method,
        "stationary_converged": False,
        "minimum_converged": False,
        "trace": [],
    }
    t0 = perf_counter()
    try:
        if method.startswith("nested"):
            run_nested(problem, x, y, record, split=method.split("-")[1])
        elif method == "root-DIIS":
            run_root(problem, x, y, record)
        else:
            run_coupled(
                problem,
                x,
                y,
                record,
                extragradient=method == "coupled-extragradient",
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
        record.update(termination="failed", error=str(error))
    record.update(
        seconds=perf_counter() - t0,
        impurity_solves=problem.impurity.solves,
        response_evaluations=problem.response_evaluations,
        quadratic_evaluations=problem.quadratic_evaluations,
        quadratic_responses=problem.quadratic_responses,
    )
    return record


def compare(
    temperature: float, perturbation: float, budget: int, seconds: float
) -> list[dict[str, Any]]:
    """Compare local B=3 restarts without hiding unsuccessful reductions."""
    problem = ThermalSaddle(temperature)
    reference = problem.joint(problem.reference_x, problem.reference_y)
    records = []
    for seed in (1, 2):
        generator = np.random.default_rng(seed)
        x = reference.x + perturbation * np.maximum(
            np.abs(reference.x), 0.2
        ) * generator.normal(size=6)
        y = reference.y + perturbation * np.maximum(
            np.abs(reference.y), 0.2
        ) * generator.normal(size=6)
        for method in (
            "root-DIIS",
            "nested-common",
            "nested-spectral",
            "nested-PH",
            "coupled-Newton",
            "coupled-extragradient",
        ):
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                record = run_method(problem, x, y, method, budget=budget)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            record.update(
                temperature=temperature,
                seed=seed,
                perturbation=perturbation,
                initial_x=x.tolist(),
                initial_y=y.tolist(),
                reference_value=reference.value,
                reference_docc=reference.double_occupancy,
            )
            if "value" in record:
                record["value_error"] = abs(record["value"] - reference.value)
                record["docc_error"] = abs(
                    record["double_occupancy"] - reference.double_occupancy
                )
            records.append(record)
            logger.info(
                "%s", json.dumps({k: v for k, v in record.items() if k != "trace"})
            )
    return records


def main() -> None:
    """Run bounded finite-temperature saddle experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("risb").setLevel(logging.WARNING)
    signal.signal(signal.SIGALRM, timeout)
    records = compare(args.temperature, args.perturbation, args.budget, args.seconds)
    output = args.output or Path("_build/adiis") / (
        f"thermal-saddle-T{args.temperature}-p{args.perturbation}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
