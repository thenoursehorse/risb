# ruff: noqa: D100, D103
import numpy as np
import pytest
from fockit.algebra import nc

from risb.impurity_solvers import DummyImpuritySolver, FockitImpuritySolver
from risb.types import BlockMatrix


def swap_spin(A: BlockMatrix) -> BlockMatrix:
    return {"up": A["dn"], "dn": A["up"]}


def test_dummy_returns_rotated_copy_of_referenced_solver() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = FockitImpuritySolver(nc("up", 0) * nc("dn", 0), gf_struct)
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


def test_dummy_has_the_entropy_of_the_referenced_solver() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = FockitImpuritySolver(nc("up", 0) * nc("dn", 0), gf_struct, beta=2)
    dummy = DummyImpuritySolver(impurity_solver, rotations=[swap_spin])

    Lambda_c = {bl: np.array([[-0.5]]) for bl, _ in gf_struct}
    D = {bl: np.array([[-0.3333]]) for bl, _ in gf_struct}
    for solver in [impurity_solver, dummy]:
        solver.set_h_imp(Lambda_c, D)
        solver.solve()

    assert impurity_solver.entropy > 0.01
    assert dummy.entropy == pytest.approx(impurity_solver.entropy, abs=1e-14)
