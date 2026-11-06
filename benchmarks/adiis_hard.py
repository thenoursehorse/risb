"""Small B=3 thermal reduced-energy experiments for the ADIIS prototype."""

import argparse
import json
import logging
import signal
from pathlib import Path
from time import perf_counter
from typing import Any, override

import numpy as np
from adiis import (
    SPINS,
    Array,
    BetheObjective,
    Sample,
    pack_fields,
    paramagnetic,
    run_adiis,
    unpack_fields,
)
from fockit.algebra import nc
from scipy.optimize import least_squares, minimize, root
from scipy.special import expit

from risb import ThermalLatticeSolver, helpers
from risb.impurity_solvers import FockitThermalImpuritySolver

logger = logging.getLogger(__name__)


class CountedThermal(FockitThermalImpuritySolver):
    """Count interacting Gibbs-state solves."""

    solves = 0

    @override
    def solve(self) -> None:
        self.solves += 1
        super().solve()


class ThermalObjective:
    """B=3 reduction in a fixed rotation gauge with a full Gibbs impurity."""

    def __init__(
        self, temperature: float, nk: int = 400, *, symmetric: bool = True
    ) -> None:
        self.symmetric = symmetric
        self.beta = 1 / temperature
        self.lower = (
            np.zeros(3) if symmetric else np.r_[np.zeros(3), np.full(3, -np.inf)]
        )
        grid = np.linspace(-1, 1, 100001)
        cdf = 0.5 + (grid * np.sqrt(1 - grid**2) + np.arcsin(grid)) / np.pi
        self.eps = np.interp((np.arange(nk) + 0.5) / nk, cdf, grid)
        h_loc = 2 * nc("up", 0) * nc("dn", 0) - nc("up", 0) - nc("dn", 0)
        self.impurity = CountedThermal(
            h_loc,
            [(bl, 1) for bl in SPINS],
            self.beta,
            eigensolver="eigh",
            method="lowest-scan",
            eta=14 * np.log(10),
            bath_size=dict.fromkeys(SPINS, 3),
        )
        self.solver = ThermalLatticeSolver(
            dict.fromkeys(SPINS, self.eps[:, None, None]),
            [(bl, 1) for bl in SPINS],
            self.impurity,
            beta=self.beta,
            mu=0,
            symmetrizers=[paramagnetic],
            force_real=True,
            bath_size=[dict.fromkeys(SPINS, 3)],
        )
        result = self.solver.solve(tol=1e-10, maxiter=200)
        physical = max(
            self.solver.bath_fit_error,
            *(
                float(np.max(np.abs(matrix)))
                for blocks in (self.solver.f1, self.solver.f2)
                for cluster in blocks
                for matrix in cluster.values()
            ),
        )
        if physical > 1e-7:
            msg = f"Thermal reference did not physically match: {physical}, {result.converged}"
            raise RuntimeError(msg)
        self.reference_y = self.pack(
            self.solver.D[0]["up"].real, self.solver.Lambda_c[0]["up"].real
        )
        self.inner_initial = self.pack(
            self.solver.R[0]["up"].real, self.solver.Lambda[0]["up"].real
        )
        self.map_tolerance = 1e-7
        self.map_outputs = False
        self.map_errors: list[float] = []
        self.responses = 0
        self.band_evaluations = 0

    def matrices(self, values: Array) -> tuple[Array, Array]:
        """Unpack fields in the chosen fixed rotation gauge."""
        if self.symmetric:
            return unpack_fields(values)
        return values[:3, None], np.diag(values[3:])

    def pack(self, coupling: Array, levels: Array) -> Array:
        """Sort levels and choose positive hybridizations."""
        if self.symmetric:
            return pack_fields(coupling, levels)
        eigenvalues, rotation = np.linalg.eigh(levels)
        return np.asarray(
            np.r_[np.abs((rotation.T @ coupling).ravel()), eigenvalues],
            dtype=np.float64,
        )

    def quadratic(
        self, x: Array, y: Array
    ) -> tuple[float, Array, Array, Array, Array, Array, Array, Array]:
        """Evaluate lattice/reference terms and their exact first gradients."""
        self.band_evaluations += 1
        R, Lambda = self.matrices(x)
        D, Lambda_c = self.matrices(y)
        values, vectors = np.linalg.eigh(self.eps[:, None, None] * (R @ R.T) + Lambda)
        occupation = expit(-self.beta * values)
        density_k = (vectors * occupation[:, None, :]) @ vectors.transpose(0, 2, 1)
        density = density_k.mean(axis=0)
        kinetic = (self.eps[:, None, None] * density_k).mean(axis=0) @ R
        n0, cross, bath = helpers.get_h0_emb_densities(
            R, Lambda, D, Lambda_c, self.beta
        )
        potential = float(
            -2 * np.mean(np.logaddexp(0, -self.beta * values).sum(axis=1)) / self.beta
            - 2 * helpers.get_h0_emb_grand_potential(R, Lambda, D, Lambda_c, self.beta)
        )
        gr = 4 * (kinetic - cross @ D)
        gl = 2 * (density - n0)
        gd = -4 * (cross.T @ R).ravel()
        gb = -2 * bath
        gx = (
            np.array([gr[1, 0], gr[0, 0] + gr[2, 0], -gl[0, 0] + gl[2, 2]])
            if self.symmetric
            else np.r_[gr.ravel(), np.diag(gl)]
        )
        gy = (
            np.array([gd[1], gd[0] + gd[2], -gb[0, 0] + gb[2, 2]])
            if self.symmetric
            else np.r_[gd, np.diag(gb)]
        )
        return potential, gx, gy, density, kinetic, n0.real, cross.real, bath.real

    def evaluate(self, y: Array) -> Sample:
        """Evaluate the reduced grand potential and every matching equation."""

        def inner(x: Array) -> Array:
            return self.quadratic(x, y)[1]

        result = root(inner, self.inner_initial, tol=1e-10)
        x = result.x

        potential, gx, gy, density, kinetic, n0, cross, bath = self.quadratic(x, y)
        R, Lambda = self.matrices(x)
        D, Lambda_c = self.matrices(y)
        lattice_error = float(
            max(np.max(np.abs(density - n0)), np.max(np.abs(kinetic - cross @ D)))
        )
        if lattice_error > 1e-8:
            msg = f"Thermal quadratic inner solve failed: {lattice_error:.3g}; |gx|={np.max(np.abs(gx)):.3g}"
            raise RuntimeError(msg)
        self.impurity.set_h_imp(
            dict.fromkeys(SPINS, Lambda_c), dict.fromkeys(SPINS, D), mu=0
        )
        self.impurity.solve()
        bath_imp = self.impurity.get_rho_f("up").real
        cf_imp = self.impurity.get_rho_cf("up").real
        gi = (
            np.array(
                [
                    4 * cf_imp[0, 1],
                    4 * (cf_imp[0, 0] + cf_imp[0, 2]),
                    2 * (-bath_imp[0, 0] + bath_imp[2, 2]),
                ]
            )
            if self.symmetric
            else np.r_[4 * cf_imp.ravel(), 2 * np.diag(bath_imp)]
        )
        residual = float(
            max(
                lattice_error,
                np.max(np.abs(bath_imp - bath)),
                np.max(np.abs(cf_imp - R.T @ cross)),
            )
        )
        mapped = y.copy()
        if self.map_outputs and residual >= self.map_tolerance:

            def impurity_matching(trial: Array) -> Array:
                trial_R, trial_L = self.matrices(trial)
                _, trial_cross, trial_bath = helpers.get_h0_emb_densities(
                    trial_R, trial_L, D, Lambda_c, self.beta
                )
                return np.concatenate(
                    (
                        (trial_bath.real - bath_imp).ravel(),
                        (trial_R.T @ trial_cross.real - cf_imp).ravel(),
                    )
                )

            qp = least_squares(
                impurity_matching,
                np.maximum(x, self.lower),
                bounds=(self.lower, np.inf),
                gtol=1e-12,
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=100,
            )
            self.map_errors.append(float(np.max(np.abs(impurity_matching(qp.x)))))

            def lattice_matching(trial: Array) -> Array:
                _, _, _, dn, ke, nr, cr, _ = self.quadratic(qp.x, trial)
                trial_D, _ = self.matrices(trial)
                return np.concatenate(((dn - nr).ravel(), (ke - cr @ trial_D).ravel()))

            fields = least_squares(
                lattice_matching,
                np.maximum(y, self.lower),
                bounds=(self.lower, np.inf),
                gtol=1e-12,
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=100,
            )
            self.map_errors.append(float(np.max(np.abs(lattice_matching(fields.x)))))
            mapped = self.pack(*self.matrices(fields.x))
        return Sample(
            y.copy(),
            self.impurity.grand_potential + potential,
            gy + gi,
            mapped,
            residual,
            R,
            Lambda,
            float(np.real(self.impurity.overlap(nc("up", 0) * nc("dn", 0)))),
            float(helpers.get_z(R, Lambda)[0, 0].real),
        )

    def observe(self, sample: Sample) -> dict[str, Any]:
        """Read thermodynamics and spectra without another impurity solve."""
        R, Lambda = sample.R, sample.Lambda
        D, Lambda_c = self.matrices(sample.y)
        values = np.linalg.eigvalsh(self.eps[:, None, None] * (R @ R.T) + Lambda)
        fields = [[dict.fromkeys(SPINS, matrix)] for matrix in (R, Lambda, D, Lambda_c)]
        entropy = helpers.get_thermal_entropy(
            eigenvalues=dict.fromkeys(SPINS, values),
            R=fields[0],
            Lambda=fields[1],
            D=fields[2],
            Lambda_c=fields[3],
            beta=self.beta,
            mu=0,
            impurity_entropy=[self.impurity.entropy],
        )
        frequencies = 1j * np.pi / self.beta * np.array([1, 3, 5])
        sigma = helpers.get_sigma_w(frequencies, R, Lambda).ravel()
        return {
            "grand_potential": sample.energy,
            "energy": sample.energy + entropy / self.beta,
            "entropy": entropy,
            "double_occupancy": sample.double_occupancy,
            "filling": float(2 * self.impurity.get_rho_c("up")[0, 0].real),
            "sigma_real": sigma.real.tolist(),
            "sigma_imag": sigma.imag.tolist(),
        }


def thermal_observation(problem: ThermalObjective) -> dict[str, Any]:
    """Observe the unprojected production state and all matching residuals."""
    solver, impurity = problem.solver, problem.impurity
    omega = helpers.get_thermal_grand_potential(
        eigenvalues=solver.eigenvalues_qp,
        R=solver.R,
        Lambda=solver.Lambda,
        D=solver.D,
        Lambda_c=solver.Lambda_c,
        beta=solver.beta,
        mu=solver.mu,
        impurity_grand_potential=[impurity.grand_potential],
    )
    entropy = helpers.get_thermal_entropy(
        eigenvalues=solver.eigenvalues_qp,
        R=solver.R,
        Lambda=solver.Lambda,
        D=solver.D,
        Lambda_c=solver.Lambda_c,
        beta=solver.beta,
        mu=solver.mu,
        impurity_entropy=[impurity.entropy],
    )
    residual = max(
        solver.bath_fit_error,
        *(
            float(np.max(np.abs(matrix)))
            for blocks in (solver.f1, solver.f2)
            for cluster in blocks
            for matrix in cluster.values()
        ),
    )
    sigma = helpers.get_sigma_w(
        1j * np.pi / solver.beta * np.array([1, 3, 5]),
        solver.R[0]["up"],
        solver.Lambda[0]["up"],
        solver.mu,
    ).ravel()
    return {
        "residual": residual,
        "grand_potential": omega,
        "entropy": entropy,
        "energy": omega + entropy / solver.beta,
        "double_occupancy": float(np.real(impurity.overlap(nc("up", 0) * nc("dn", 0)))),
        "filling": float(2 * impurity.get_rho_c("up")[0, 0].real),
        "sigma_real": sigma.real.tolist(),
        "sigma_imag": sigma.imag.tolist(),
    }


def run_case(case: str, maxiter: int, seconds: float) -> list[dict[str, Any]]:
    """Retain preparation failures, solver failures, and physical outcomes."""
    thermal = case.startswith("thermal")
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        if thermal:
            temperature = float(case.split("-")[1])
            problem: BetheObjective | ThermalObjective = ThermalObjective(
                temperature, symmetric=case.endswith("ph")
            )
        else:
            problem = BetheObjective(3, interaction=float(case.split("-")[1]))
        reference = problem.evaluate(problem.reference_y)
        reference_observation = (
            problem.observe(reference)
            if isinstance(problem, ThermalObjective)
            else {
                "energy": reference.energy,
                "double_occupancy": reference.double_occupancy,
                "z": reference.z,
            }
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
        return [{"case": case, "stage": "preparation", "error": str(error)}]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    base_inner = (
        problem.inner_initial.copy()
        if isinstance(problem, ThermalObjective)
        else problem.band_initial.copy()
    )
    records = []
    for seed in (1, 2):
        scale = np.maximum(np.abs(reference.y), 0.2)
        initial = reference.y + 0.15 * scale * np.random.default_rng(seed).normal(
            size=len(scale)
        )
        if isinstance(problem, ThermalObjective):
            problem.inner_initial = base_inner.copy()
            problem.map_outputs = False
        else:
            problem.band_initial = base_inner.copy()
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            start = problem.evaluate(initial)
            if isinstance(problem, ThermalObjective):
                problem.inner_initial = problem.pack(start.R, start.Lambda)
            else:
                problem.band_initial = np.einsum(
                    "aij,ij->a", problem.basis, start.Lambda
                )
            finite = []
            for direction in np.eye(len(initial)):
                plus = problem.evaluate(initial + 2e-5 * direction).energy
                minus = problem.evaluate(initial - 2e-5 * direction).energy
                finite.append((plus - minus) / 4e-5)
            gradient_error = float(np.max(np.abs(start.gradient - finite)))
            if gradient_error > 1e-5:
                msg = f"Reduced gradient failed its check: {gradient_error}"
                raise RuntimeError(msg)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
            records.append(
                {"case": case, "seed": seed, "stage": "initial", "error": str(error)}
            )
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        methods = ["DIIS-default", "BFGS", "ADIIS"]
        if isinstance(problem, ThermalObjective):
            methods.insert(1, "DIIS-no-anchor")
        for name in methods:
            problem.impurity.solves = problem.responses = problem.band_evaluations = 0
            if isinstance(problem, ThermalObjective):
                problem.map_outputs = name == "ADIIS"
                problem.map_errors = []
            record: dict[str, Any] = {
                "case": case,
                "seed": seed,
                "method": name,
                "initial": initial.tolist(),
                "gradient_check_error": gradient_error,
                "reference": reference_observation,
            }
            t0 = perf_counter()
            try:
                signal.setitimer(signal.ITIMER_REAL, seconds)
                trace: list[dict[str, float]] = []
                if name.startswith("DIIS"):
                    solver = problem.solver
                    solver.R = [dict.fromkeys(SPINS, start.R.copy())]
                    solver.Lambda = [dict.fromkeys(SPINS, start.Lambda.copy())]
                    D, Lambda_c = (
                        problem.matrices(initial)
                        if isinstance(problem, ThermalObjective)
                        else unpack_fields(initial)
                    )
                    solver.D = [dict.fromkeys(SPINS, D.copy())]
                    solver.Lambda_c = [dict.fromkeys(SPINS, Lambda_c.copy())]
                    result = solver.solve(
                        tol=1e-10,
                        maxiter=maxiter,
                        callbacks=[] if name == "DIIS-no-anchor" else None,
                    )
                    record.update(
                        seconds=perf_counter() - t0,
                        impurity_solves=problem.impurity.solves,
                        response_solves=problem.responses,
                        quadratic_evaluations=problem.band_evaluations,
                    )
                    record["optimizer_success"] = result.converged
                    if isinstance(problem, ThermalObjective):
                        record.update(thermal_observation(problem))
                    else:
                        final = problem.evaluate(
                            pack_fields(
                                solver.D[0]["up"].real, solver.Lambda_c[0]["up"].real
                            )
                        )
                        record.update(
                            residual=final.residual,
                            energy=final.energy,
                            double_occupancy=final.double_occupancy,
                            z=final.z,
                        )
                elif name == "BFGS":

                    def value(y: Array) -> tuple[float, Array]:
                        sample = problem.evaluate(y)
                        return sample.energy, sample.gradient

                    minimum = minimize(
                        value,
                        initial.copy(),
                        jac=True,
                        method="BFGS",
                        options={"gtol": 1e-8, "maxiter": maxiter},
                    )
                    record.update(
                        seconds=perf_counter() - t0,
                        impurity_solves=problem.impurity.solves,
                        response_solves=problem.responses,
                        quadratic_evaluations=problem.band_evaluations,
                    )
                    final = problem.evaluate(minimum.x)
                    record.update(
                        optimizer_success=bool(minimum.success),
                        message=str(minimum.message),
                    )
                    record.update(
                        problem.observe(final)
                        if isinstance(problem, ThermalObjective)
                        else {
                            "energy": final.energy,
                            "double_occupancy": final.double_occupancy,
                            "z": final.z,
                        }
                    )
                    record["residual"] = final.residual
                else:
                    final, trace = run_adiis(problem, initial, maxiter=maxiter)
                    record.update(
                        seconds=perf_counter() - t0,
                        impurity_solves=problem.impurity.solves,
                        response_solves=problem.responses,
                        quadratic_evaluations=problem.band_evaluations,
                    )
                    record.update(
                        optimizer_success=final.residual < 1e-7, residual=final.residual
                    )
                    record.update(
                        problem.observe(final)
                        if isinstance(problem, ThermalObjective)
                        else {
                            "energy": final.energy,
                            "double_occupancy": final.double_occupancy,
                            "z": final.z,
                        }
                    )
                record["physical_converged"] = record["residual"] < 1e-7
                record["trace"] = trace
                record["energy_error"] = abs(
                    record["energy"] - reference_observation["energy"]
                )
                record["docc_error"] = abs(
                    record["double_occupancy"] - reference.double_occupancy
                )
                if "sigma_imag" in record:
                    record["sigma_error"] = float(
                        max(
                            np.max(
                                np.abs(
                                    np.asarray(record[key]) - reference_observation[key]
                                )
                            )
                            for key in ("sigma_real", "sigma_imag")
                        )
                    )
            except (
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
                TimeoutError,
            ) as error:
                record.update(physical_converged=False, error=str(error))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if "seconds" not in record:
                record.update(
                    seconds=perf_counter() - t0,
                    impurity_solves=problem.impurity.solves,
                    response_solves=problem.responses,
                    quadratic_evaluations=problem.band_evaluations,
                )
            if isinstance(problem, ThermalObjective):
                record["largest_map_fit_error"] = max(problem.map_errors, default=0)
            records.append(record)
            logger.info(
                "%s",
                json.dumps(
                    {k: v for k, v in record.items() if k not in ("trace", "reference")}
                ),
            )
    return records


def timeout(_signum: int, _frame: Any) -> None:
    """Stop an individual small-case experiment at its wall-clock budget."""
    msg = "Wall-clock budget exceeded."
    raise TimeoutError(msg)


def main() -> None:
    """Run bounded small ghost cases without silently discarding failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "zero-2.4",
            "zero-2.8",
            "zero-3.0",
            "thermal-0.5-full",
            "thermal-1.0-full",
            "thermal-1.0-ph",
        ),
        required=True,
    )
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("risb").setLevel(logging.WARNING)
    signal.signal(signal.SIGALRM, timeout)
    records = run_case(args.case, args.maxiter, args.seconds)
    output = args.output or Path("_build/adiis") / (args.case + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
