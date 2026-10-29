# ruff: noqa: D100, D103
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from test_common import EXACT_THERMAL, bethe_h0_k, symmetrize_blocks
from test_thermal_latticesolver import ONE_BAND, hubbard

from risb import BaseLatticeSolver, LatticeSolver, ThermalLatticeSolver
from risb.diagnostics import (
    DiagnosticObserver,
    LatticeDiagnostics,
    ThermalLatticeDiagnostics,
)
from risb.impurity_solvers import FockitImpuritySolver, FockitThermalImpuritySolver
from risb.kweight import SmearingKWeight
from risb.optimize import FixedPointLoop
from risb.types import FixedPointProblem


def lattice(thermal: bool, diagnostics: DiagnosticObserver) -> BaseLatticeSolver:
    h_int = hubbard(2.0)
    h0_k = {bl: bethe_h0_k(100) for bl, _ in ONE_BAND}
    if thermal:
        return ThermalLatticeSolver(
            h0_k,
            ONE_BAND,
            FockitThermalImpuritySolver(h_int, ONE_BAND, beta=10, **EXACT_THERMAL),
            beta=10,
            mu=0,
            symmetrizers=[symmetrize_blocks],
            force_real=True,
            diagnostics=diagnostics,
        )
    return LatticeSolver(
        h0_k,
        ONE_BAND,
        FockitImpuritySolver(h_int, ONE_BAND),
        update_weights=SmearingKWeight(beta=40, n_target=1).update_weights,
        symmetrizers=[symmetrize_blocks],
        diagnostics=diagnostics,
    )


@pytest.mark.parametrize("thermal", [False, True])
def test_lattice_diagnostics_report_progress_without_controlling_the_solve(
    thermal: bool,
    tmp_path: Path,
) -> None:
    def stop(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        loop.stop()

    output = tmp_path / "lattice.out"
    output.write_text("previous run\n")
    reporter = ThermalLatticeDiagnostics if thermal else LatticeDiagnostics
    with reporter(level="debug", output=output) as diagnostics:
        solver = lattice(thermal, diagnostics)
        result = solver.solve(callbacks=[stop], self_consistency_tol=np.inf)
        assert result.cycles == 1
        report = output.read_text()
        assert report.startswith("previous run\n")
        for event in (
            "solve.started",
            "bands.finished",
            "impurity.finished",
            "residual.block",
            "cycle.summary",
            "optimizer.cycle",
            "solve.result",
        ):
            assert event in report
        assert "reason=callback" in report
        assert "anchor.set" not in report
        if isinstance(solver, ThermalLatticeSolver):
            assert "fit.result" in report
            assert "max_residual=" in report
            assert "total_cost=" in report
            assert "thermal.filling" in report
            solver.anchor()
            solver.one_cycle()
            solver.release()
            report = output.read_text()
            assert "anchor.set" in report
            assert "anchored=True" in report
            assert "anchor.released" in report
        solver.diagnostics = None
        solver.solve(callbacks=[stop], self_consistency_tol=np.inf)
        assert output.read_text() == report


def test_lattice_diagnostics_report_failure_and_leave_application_streams_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        sys.stdout.write("application output\n")
        raise RuntimeError

    output = StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    with (
        pytest.raises(RuntimeError),
        ThermalLatticeDiagnostics(output=output) as diagnostics,
    ):
        lattice(True, diagnostics).solve(callbacks=[fail], self_consistency_tol=np.inf)
    assert not output.closed
    assert "solve.failed" in output.getvalue()
    assert "application output" not in output.getvalue()
    assert capsys.readouterr().out == "application output\n"
    assert sys.stdout is stdout
    assert sys.stderr is stderr
