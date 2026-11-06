"""Compare ADIIS and gradient DIIS on the same reduced RISB energy objective."""

import argparse
import json
import logging
import signal
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from adiis import (
    SPINS,
    Array,
    BetheObjective,
    Sample,
    adiis_weights,
    pack_fields,
    unpack_fields,
)
from adiis_hard import ThermalObjective, thermal_observation, timeout
from fockit.algebra import nc

from risb import helpers
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator
from risb.optimize.diis import extrapolate

logger = logging.getLogger(__name__)


def run_energy_hybrid(
    problem: BetheObjective | ThermalObjective,
    initial: Array,
    *,
    method: str = "hybrid",
    switch_tolerance: float = 1e-3,
    adiis_limit: int = 30,
    budget: int = 150,
) -> dict[str, Any]:
    """Change coefficient rules while retaining the reduced energy and map."""
    problem.impurity.solves = problem.responses = problem.band_evaluations = 0
    if isinstance(problem, ThermalObjective):
        problem.map_outputs = True
        problem.map_tolerance = 0.0
    history: list[Sample] = []
    trace: list[dict[str, Any]] = []
    phase = "energy-DIIS" if method == "DIIS" else "ADIIS"
    record: dict[str, Any] = {
        "method": method,
        "switch_tolerance": switch_tolerance if method == "hybrid" else None,
        "adiis_limit": adiis_limit,
        "budget": budget,
        "objective": "grand_potential"
        if isinstance(problem, ThermalObjective)
        else "energy",
        "adiis_proposals": 0,
        "diis_proposals": 0,
    }
    y = initial.copy()
    sample: Sample | None = None
    attempted = 0
    t0 = perf_counter()
    try:
        for evaluation in range(budget):
            attempted += 1
            sample = problem.evaluate(y)
            gradient_norm = float(np.max(np.abs(sample.gradient)))
            trace.append(
                {
                    "phase": phase,
                    "energy": sample.energy,
                    "gradient_norm": gradient_norm,
                    "residual": sample.residual,
                    "y": sample.y.tolist(),
                }
            )
            record.update(
                y=sample.y.tolist(),
                value=sample.energy,
                gradient_norm=gradient_norm,
                residual=sample.residual,
                double_occupancy=sample.double_occupancy,
                z=sample.z,
            )
            if gradient_norm < 1e-8 and sample.residual < 1e-7:
                record["physical_converged"] = True
                record["termination"] = "stationary_and_matched"
                break
            if evaluation + 1 == budget:
                record.update(physical_converged=False, termination="evaluation_limit")
                break
            history = [*history, sample][-6:]
            if (
                method == "hybrid"
                and phase == "ADIIS"
                and (gradient_norm < switch_tolerance or evaluation + 1 >= adiis_limit)
            ):
                phase = "energy-DIIS"
                record.update(
                    switch_evaluation=evaluation + 1,
                    switch_gradient=gradient_norm,
                    switch_residual=sample.residual,
                    switch_reason="gradient"
                    if gradient_norm < switch_tolerance
                    else "cycle_limit",
                )
            if phase == "ADIIS":
                weights = adiis_weights(history)
                y = sum(
                    (
                        weight * item.mapped
                        for weight, item in zip(weights, history, strict=True)
                    ),
                    np.zeros_like(initial),
                )
                record["adiis_proposals"] += 1
            else:
                y = extrapolate(
                    [item.y for item in history],
                    [item.mapped for item in history],
                    [item.gradient for item in history],
                )
                record["diis_proposals"] += 1
        else:
            record.update(physical_converged=False, termination="evaluation_limit")
        record.update(
            seconds=perf_counter() - t0,
            impurity_solves=problem.impurity.solves,
            response_solves=problem.responses,
            quadratic_evaluations=problem.band_evaluations,
        )
        if isinstance(problem, ThermalObjective) and sample is not None:
            record["observables"] = problem.observe(sample)
    except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
        record.update(
            physical_converged=False,
            termination="evaluation_failed",
            failure_phase=phase,
            error=str(error),
        )
    record.update(
        seconds=record.get("seconds", perf_counter() - t0),
        impurity_solves=problem.impurity.solves,
        response_solves=problem.responses,
        quadratic_evaluations=problem.band_evaluations,
        uphill_steps=sum(b["energy"] > a["energy"] + 1e-12 for a, b in pairwise(trace)),
        evaluations=len(trace),
        attempted_evaluations=attempted,
        trace=trace,
    )
    return record


def half_filled_occupations(values: Array, _beta: float, _mu: float) -> Array:
    """Fill half the equally weighted band states at zero temperature."""
    occupations = np.zeros_like(values)
    occupations.ravel()[np.argsort(values.ravel())[: values.size // 2]] = 1
    return occupations


def run_root_diis(
    problem: BetheObjective | ThermalObjective,
    initial: Array,
    start: Sample,
    *,
    budget: int = 150,
    sharp_bands: bool = True,
) -> dict[str, Any]:
    """Solve the original equations from the same physical initial state."""
    problem.impurity.solves = problem.responses = problem.band_evaluations = 0
    if isinstance(problem, ThermalObjective):
        problem.map_outputs = False
    else:
        problem.solver.band_integrator = (
            BandKIntegrator(
                WeightedKPointIntegrator(),
                smearing=half_filled_occupations,
                beta=np.inf,
                mu=0,
            )
            if sharp_bands
            else BandKIntegrator(
                WeightedKPointIntegrator(), beta=1e6, n_target=problem.bath_size
            )
        )
    solver = problem.solver
    D, Lambda_c = (
        problem.matrices(initial)
        if isinstance(problem, ThermalObjective)
        else unpack_fields(initial)
    )
    solver.R = [dict.fromkeys(SPINS, start.R.copy())]
    solver.Lambda = [dict.fromkeys(SPINS, start.Lambda.copy())]
    solver.D = [dict.fromkeys(SPINS, D.copy())]
    solver.Lambda_c = [dict.fromkeys(SPINS, Lambda_c.copy())]
    record: dict[str, Any] = {
        "method": "root-DIIS" if sharp_bands else "root-DIIS-smeared",
        "budget": budget,
    }
    t0 = perf_counter()
    try:
        result = solver.solve(tol=1e-10, maxiter=budget)
        record.update(
            seconds=perf_counter() - t0,
            impurity_solves=problem.impurity.solves,
            response_solves=problem.responses,
            quadratic_evaluations=problem.band_evaluations,
            optimizer_success=result.converged,
            optimizer_error=result.error,
        )
        residual = solver.self_consistency_error
        if isinstance(problem, ThermalObjective):
            observation = thermal_observation(problem)
            residual = observation["residual"]
            record.update(
                value=observation["grand_potential"],
                double_occupancy=observation["double_occupancy"],
                observables=observation,
            )
        else:
            kinetic = sum(
                np.trace(solver.lopsided_ke_qp[0][bl] @ solver.R[0][bl]).real
                for bl in SPINS
            )
            local = problem.impurity.overlap(problem.h_loc).real
            record.update(
                value=float(kinetic + local),
                double_occupancy=float(
                    problem.impurity.overlap(nc("up", 0) * nc("dn", 0)).real
                ),
                z=float(
                    helpers.get_z(solver.R[0]["up"], solver.Lambda[0]["up"])[0, 0].real
                ),
            )
        record.update(
            physical_converged=residual < 1e-7,
            residual=residual,
            termination="matched" if residual < 1e-7 else "physical_check_failed",
        )
        pack = problem.pack if isinstance(problem, ThermalObjective) else pack_fields
        final = problem.evaluate(
            pack(solver.D[0]["up"].real, solver.Lambda_c[0]["up"].real)
        )
        record.update(
            y=final.y.tolist(),
            gradient_norm=float(np.max(np.abs(final.gradient))),
            reduced_residual=final.residual,
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
        if "physical_converged" in record:
            record["validation_error"] = str(error)
        else:
            record.update(physical_converged=False, error=str(error))
    record.setdefault("seconds", perf_counter() - t0)
    record.setdefault("impurity_solves", problem.impurity.solves)
    record.setdefault("response_solves", problem.responses)
    record.setdefault("quadratic_evaluations", problem.band_evaluations)
    record["validation_impurity_solves"] = (
        problem.impurity.solves - record["impurity_solves"]
    )
    return record


def compare(case: str, budget: int, seconds: float) -> list[dict[str, Any]]:
    """Compare coefficient rules from identical reduced-energy starts."""
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        if case.startswith("thermal"):
            problem: BetheObjective | ThermalObjective = ThermalObjective(
                float(case.split("-")[1]), symmetric="-ph" in case
            )
        else:
            problem = BetheObjective(3, interaction=float(case.split("-")[1]))
        reference = problem.evaluate(problem.reference_y)
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
        initial = reference.y + 0.15 * np.maximum(np.abs(reference.y), 0.2) * (
            np.random.default_rng(seed).normal(size=len(reference.y))
        )
        if case.endswith("-distorted"):
            if case.startswith("thermal"):
                factors = (
                    np.array([1.3, 0.7, 1.3])
                    if seed == 1
                    else np.array([0.5, 0.5, 1.5])
                )
            else:
                factors = (
                    np.array([0.5, 0.5, 1.5])
                    if seed == 1
                    else np.array([1.6, 0.4, 1.6])
                )
            initial = reference.y * factors
        if isinstance(problem, ThermalObjective):
            problem.inner_initial = base_inner.copy()
            problem.map_outputs = False
        else:
            problem.band_initial = base_inner.copy()
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            start = problem.evaluate(initial)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, TimeoutError) as error:
            records.append(
                {"case": case, "seed": seed, "stage": "initial", "error": str(error)}
            )
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if isinstance(problem, ThermalObjective):
            problem.inner_initial = problem.pack(start.R, start.Lambda)
        else:
            problem.band_initial = np.einsum("aij,ij->a", problem.basis, start.Lambda)
        methods = [
            ("ADIIS", 1e-3),
            ("DIIS", 1e-3),
            ("hybrid", 1e-3),
            ("hybrid", 1e-4),
            ("root-DIIS", 1e-3),
        ]
        if isinstance(problem, BetheObjective):
            methods.append(("root-DIIS-smeared", 1e-3))
        for method, threshold in methods:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                if method.startswith("root-DIIS"):
                    record = run_root_diis(
                        problem,
                        initial,
                        start,
                        budget=budget,
                        sharp_bands=method == "root-DIIS",
                    )
                else:
                    record = run_energy_hybrid(
                        problem,
                        initial,
                        method=method,
                        switch_tolerance=threshold,
                        budget=budget,
                    )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            record.update(
                case=case,
                seed=seed,
                initial=initial.tolist(),
                reference_value=reference.energy,
                reference_docc=reference.double_occupancy,
            )
            if "value" in record:
                record["value_error"] = abs(record["value"] - reference.energy)
                record["docc_error"] = abs(
                    record["double_occupancy"] - reference.double_occupancy
                )
                record["same_reference_state"] = (
                    record["physical_converged"]
                    and record["value_error"] < 1e-10
                    and record["docc_error"] < 1e-7
                )
            records.append(record)
            logger.info(
                "%s",
                json.dumps(
                    {key: value for key, value in record.items() if key != "trace"}
                ),
            )
    return records


def main() -> None:
    """Run a bounded energy-minimization comparison with no saddle-root handoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "zero-1.0",
            "zero-2.4",
            "zero-2.8",
            "zero-3.0",
            "zero-1.0-distorted",
            "zero-2.8-distorted",
            "thermal-0.5-full",
            "thermal-1.0-full",
            "thermal-1.0-ph",
            "thermal-1.0-ph-distorted",
        ),
        required=True,
    )
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("risb").setLevel(logging.WARNING)
    signal.signal(signal.SIGALRM, timeout)
    records = compare(args.case, args.budget, args.seconds)
    output = args.output or Path("_build/adiis") / (
        "energy-hybrid-" + args.case + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
