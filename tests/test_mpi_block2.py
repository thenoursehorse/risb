# ruff: noqa: D100, D103
# Also run on two ranks by test_block2.py, as
# mpirun -n 2 python -m mpi4py -m pytest tests/test_mpi_block2.py
from typing import Any

import pytest
from test_block2 import assert_same_rho, assert_same_state, ghost_problem, solved

from risb.helpers_fockit import N_op, S2_op
from risb.impurity_solvers import FockitImpuritySolver, FockitThermalImpuritySolver

try:
    from mpi4py import MPI
except (ImportError, RuntimeError) as error:
    # Without mpi4py, or without an MPI library for it to load
    pytest.skip(str(error), allow_module_level=True)
from risb.impurity_solvers import Block2ImpuritySolver, Block2ThermalImpuritySolver


def test_ground_state_with_sectors_split_over_ranks_is_the_exact_one() -> None:
    problem = ghost_problem(1, 1)
    h_int, gf_struct, bath_size = problem[:3]
    expected = solved(
        FockitImpuritySolver(h_int, gf_struct, bath_size=bath_size), problem
    )
    solver = solved(
        Block2ImpuritySolver(
            h_int,
            gf_struct,
            bond_dim=50,
            symmetries={"s": None},
            bath_size=bath_size,
            comm=MPI.COMM_WORLD,
        ),
        problem,
    )
    assert_same_state(solver, expected)
    for Op in (N_op(["up", "dn"], 1), S2_op(["up", "dn"], 1)):
        assert solver.overlap(Op) == pytest.approx(expected.overlap(Op), abs=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symmetries": ["s"], "method": "lowest", "eta": 40},
        {"symmetries": [], "method": "purification"},
    ],
)
def test_thermal_state_with_sectors_split_over_ranks_is_the_exact_one(
    kwargs: dict[str, Any],
) -> None:
    problem = ghost_problem(1, 1)
    h_int, gf_struct, bath_size = problem[:3]
    expected = solved(
        FockitThermalImpuritySolver(h_int, gf_struct, 10, bath_size=bath_size),
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
        ),
        problem,
    )
    assert_same_rho(solver, expected)
    assert solver.energy == pytest.approx(expected.overlap(expected.h_imp), abs=1e-6)
    assert solver.grand_potential == pytest.approx(expected.grand_potential, abs=1e-6)
    for Op in (N_op(["up", "dn"], 1), S2_op(["up", "dn"], 1)):
        assert solver.overlap(Op) == pytest.approx(expected.overlap(Op), abs=1e-6)
