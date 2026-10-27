# ruff: noqa: D100, D103
# Also run on two ranks by test_impurity_solvers.py, as
# mpirun -n 2 python -m mpi4py -m pytest tests/test_mpi_fockit.py
from typing import Literal

import numpy as np
import pytest
from fockit.algebra import nc

from risb.impurity_solvers import FockitThermalImpuritySolver

try:
    from mpi4py import MPI
except (ImportError, RuntimeError) as error:
    # Without mpi4py, or without an MPI library for it to load
    pytest.skip(str(error), allow_module_level=True)


@pytest.mark.parametrize("method", ["lowest", "lowest-scan"])
def test_thermal_state_with_sectors_split_over_ranks_is_that_of_one_process(
    method: Literal["lowest", "lowest-scan"],
) -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = {"up": 3, "dn": 3}
    Sz = nc("up", 0) - nc("dn", 0)
    for a in range(3):
        Sz += nc("bath_up", a) - nc("bath_dn", a)
    solvers = [
        FockitThermalImpuritySolver(
            2 * nc("up", 0) * nc("dn", 0),
            gf_struct,
            3.0,
            diagonal_symmetries=[Sz],
            method=method,
            eta=40,
            delta=4,
            bath_size=bath_size,
            comm=comm,
        )
        for comm in (None, MPI.COMM_WORLD)
    ]
    for solver in solvers:
        solver.set_h_imp(
            {bl: np.diag([-0.8, 0.1, 0.9]) for bl, _ in gf_struct},
            {bl: np.array([[0.3], [0.2], [0.4]]) for bl, _ in gf_struct},
            mu=0.3,
        )
        solver.solve()
    expected, solver = solvers
    for rho in ["rho_f", "rho_cf", "rho_c"]:
        for bl, _ in gf_struct:
            assert getattr(solver, rho)[bl] == pytest.approx(
                getattr(expected, rho)[bl], abs=1e-12
            )
    assert solver.grand_potential == pytest.approx(expected.grand_potential, abs=1e-12)
    assert solver.entropy == pytest.approx(expected.entropy, abs=1e-12)
    docc = nc("up", 0) * nc("dn", 0)
    assert solver.overlap(docc) == pytest.approx(expected.overlap(docc), abs=1e-12)
