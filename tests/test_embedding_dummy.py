# ruff: noqa: D100, D103
import numpy as np
import pytest
from triqs.operators import n

from risb.embedding import EmbeddingAtomDiag, EmbeddingDummy


def swap_spin(A):
    return {"up": A["dn"], "dn": A["up"]}


def test_dummy_returns_rotated_copy_of_referenced_solver():
    gf_struct = [("up", 1), ("dn", 1)]
    embedding = EmbeddingAtomDiag(n("up", 0) * n("dn", 0), gf_struct)
    dummy = EmbeddingDummy(embedding, rotations=[swap_spin])

    # A magnetic field so the up and dn density matrices differ
    Lambda_c = {bl: np.array([[-0.5]]) for bl, _ in gf_struct}
    D = {bl: np.array([[-0.3333]]) for bl, _ in gf_struct}
    h0_loc_matrix = {"up": np.array([[-0.2]]), "dn": np.array([[0.2]])}
    for e in [embedding, dummy]:
        e.set_h_emb(Lambda_c, D, h0_loc_matrix)
        e.solve()

    assert embedding.get_rho_f("up") != pytest.approx(embedding.get_rho_f("dn"))
    for bl, bl_rotated in [("up", "dn"), ("dn", "up")]:
        assert dummy.get_rho_f(bl) == pytest.approx(embedding.get_rho_f(bl_rotated))
        assert dummy.get_rho_cf(bl) == pytest.approx(embedding.get_rho_cf(bl_rotated))
        assert dummy.get_rho_c(bl) == pytest.approx(embedding.get_rho_c(bl_rotated))
