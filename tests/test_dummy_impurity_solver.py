# ruff: noqa: D100, D103
import numpy as np
import pytest
from fockit.algebra import nc
from test_common import IMPURITY_SOLVERS, make_impurity_solver

from risb.impurity_solvers import DummyImpuritySolver
from risb.types import BlockMatrix


def swap_spin(A: BlockMatrix) -> BlockMatrix:
    return {"up": A["dn"], "dn": A["up"]}


@pytest.mark.parametrize("solver_name", IMPURITY_SOLVERS)
def test_dummy_returns_rotated_copy_of_referenced_solver(solver_name: str) -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = make_impurity_solver(
        solver_name, nc("up", 0) * nc("dn", 0), gf_struct
    )
    dummy = DummyImpuritySolver(impurity_solver, rotations=[swap_spin])

    # A magnetic field so the up and dn density matrices differ
    Lambda_c = {bl: np.array([[-0.5]]) for bl, _ in gf_struct}
    D = {bl: np.array([[-0.3333]]) for bl, _ in gf_struct}
    h0_loc_matrix = {"up": np.array([[-0.2]]), "dn": np.array([[0.2]])}
    for solver in [impurity_solver, dummy]:
        solver.set_h_imp(Lambda_c, D, h0_loc_matrix)
        solver.solve()

    assert impurity_solver.get_rho_f("up") != pytest.approx(
        impurity_solver.get_rho_f("dn")
    )
    for bl, bl_rotated in [("up", "dn"), ("dn", "up")]:
        assert dummy.get_rho_f(bl) == pytest.approx(
            impurity_solver.get_rho_f(bl_rotated)
        )
        assert dummy.get_rho_cf(bl) == pytest.approx(
            impurity_solver.get_rho_cf(bl_rotated)
        )
        assert dummy.get_rho_c(bl) == pytest.approx(
            impurity_solver.get_rho_c(bl_rotated)
        )
