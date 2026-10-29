# ruff: noqa: D100, D103
# Also run on two ranks by test_block2.py, as
# mpirun -n 2 python -m mpi4py -m pytest tests/test_mpi_block2.py
from pathlib import Path
from typing import Any

import pytest
from test_block2 import (
    CUTOFF,
    assert_same_rho,
    assert_same_state,
    ghost_problem,
    solved,
)
from test_common import EXACT_THERMAL

from risb.helpers_fockit import N_op, S2_op
from risb.impurity_solvers import FockitImpuritySolver, FockitThermalImpuritySolver

try:
    from mpi4py import MPI
except (ImportError, RuntimeError) as error:
    # Without mpi4py, or without an MPI library for it to load
    pytest.skip(str(error), allow_module_level=True)
from risb.impurity_solvers import (
    Block2Diagnostics,
    Block2ImpuritySolver,
    Block2ThermalImpuritySolver,
)


def test_ground_state_with_sectors_split_over_ranks_is_the_exact_one(
    tmp_path: Path,
) -> None:
    problem = ghost_problem(1, 1)
    h_int, gf_struct, bath_size = problem[:3]
    expected = solved(
        FockitImpuritySolver(h_int, gf_struct, bath_size=bath_size), problem
    )
    output_dir = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    with Block2Diagnostics(
        level="debug",
        risb_output=output_dir / "risb.out",
        iprint=1,
        block2_output=output_dir / "block2.out",
    ) as diagnostics:
        solver = solved(
            Block2ImpuritySolver(
                h_int,
                gf_struct,
                bond_dim=50,
                symmetries={"s": None},
                bath_size=bath_size,
                comm=MPI.COMM_WORLD,
                cutoff=CUTOFF,
                diagnostics=diagnostics,
            ),
            problem,
        )
        assert_same_state(solver, expected)
        for Op in (N_op(["up", "dn"], 1), S2_op(["up", "dn"], 1)):
            assert solver.overlap(Op) == pytest.approx(expected.overlap(Op), abs=1e-8)
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    suffix = f".rank{rank}" if size > 1 else ""
    report = (output_dir / f"risb{suffix}.out").read_text()
    backend = (output_dir / f"block2{suffix}.out").read_text()
    assert f"rank={rank}/{size}" in report
    assert ("sector.selected" in report) == (rank == 0)
    assert "Sweep" in backend


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symmetries": ["s"], "method": "lowest", "eta": 40},
        {"symmetries": ["sz"], "method": "lowest-scan", "eta": 40},
        {"symmetries": [], "method": "purification"},
    ],
)
def test_thermal_state_with_sectors_split_over_ranks_is_the_exact_one(
    kwargs: dict[str, Any],
) -> None:
    problem = ghost_problem(1, 1)
    h_int, gf_struct, bath_size = problem[:3]
    expected = solved(
        FockitThermalImpuritySolver(
            h_int, gf_struct, 10, bath_size=bath_size, **EXACT_THERMAL
        ),
        problem,
    )
    solver = solved(
        Block2ThermalImpuritySolver(
            h_int,
            gf_struct,
            10,
            bond_dim=200,
            bath_size=bath_size,
            comm=MPI.COMM_WORLD,
            **kwargs,
            cutoff=CUTOFF,
        ),
        problem,
    )
    assert_same_rho(solver, expected)
    assert solver.energy == pytest.approx(expected.overlap(expected.h_imp), abs=1e-6)
    assert solver.grand_potential == pytest.approx(expected.grand_potential, abs=1e-6)
    for Op in (N_op(["up", "dn"], 1), S2_op(["up", "dn"], 1)):
        assert solver.overlap(Op) == pytest.approx(expected.overlap(Op), abs=1e-6)
