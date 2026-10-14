# ruff: noqa: D100, D103, PLC0415
from typing import Any

import numpy as np
import pytest
from test_common import build_two_orbital_complex_h0_k, random_unitary

from risb import LatticeSolver
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix


def test_self_energy_does_not_depend_on_the_quasiparticle_basis() -> None:
    pytest.importorskip("triqs")
    from triqs.gfs import MeshImFreq, MeshProduct
    from triqs.lattice.tight_binding import BravaisLattice, BrillouinZone, MeshBrZone

    from risb.helpers_triqs import (
        get_g0_k_w,
        get_g_k_w,
        get_g_qp_k_w,
        get_g_w_loc,
        get_sigma_w,
    )

    nkx = 6
    U, J = 3.0, 0.6
    Up = U - 2 * J
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * (1 - np.eye(2)),
        Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
        J_hund=J,
    )
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    h0_k = build_two_orbital_complex_h0_k(nkx=nkx)
    kweight = SmearingKWeight(beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        update_weights=kweight.update_weights,
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = 1.5 * np.eye(2)
    S.solve(tol=1e-10)
    assert kweight.mu is not None
    mu = kweight.mu

    k_mesh = MeshBrZone(BrillouinZone(BravaisLattice(units=np.eye(3))), n_k=nkx)
    iw_mesh = MeshImFreq(beta=40, statistic="Fermion", n_iw=32)
    k_iw_mesh = MeshProduct(k_mesh, iw_mesh)

    def green_functions(R: BlockMatrix, Lambda: BlockMatrix) -> tuple[Any, Any, Any]:
        Sigma_iw = get_sigma_w(
            mesh=iw_mesh,
            gf_struct=gf_struct,
            Lambda=Lambda,
            R=R,
            h0_loc=S.h0_loc_matrix[0],
            mu=mu,
        )
        G0_k_iw = get_g0_k_w(gf_struct=gf_struct, mesh=k_iw_mesh, h0_k=h0_k, mu=mu)
        G_qp_k_iw = get_g_qp_k_w(
            gf_struct=gf_struct,
            mesh=k_iw_mesh,
            h0_kin_k=S.h0_kin_k,
            Lambda=Lambda,
            R=R,
            mu=mu,
        )
        return (
            Sigma_iw,
            get_g_w_loc(get_g_k_w(g0_k_w=G0_k_iw, sigma_w=Sigma_iw)),
            get_g_w_loc(get_g_k_w(g_qp_k_w=G_qp_k_iw, R=R)),
        )

    u = random_unitary(2, seed=4)
    Sigma, G_dyson, G_qp = green_functions(S.R[0], S.Lambda[0])
    Sigma_rotated, G_dyson_rotated, G_qp_rotated = green_functions(
        {bl: u.conj().T @ S.R[0][bl] for bl, _ in gf_struct},
        {bl: u.conj().T @ S.Lambda[0][bl] @ u for bl, _ in gf_struct},
    )
    for bl, _ in gf_struct:
        assert Sigma_rotated[bl].data == pytest.approx(Sigma[bl].data, abs=1e-10)
        assert G_dyson[bl].data == pytest.approx(G_qp[bl].data, abs=1e-10)
        assert G_dyson_rotated[bl].data == pytest.approx(
            G_qp_rotated[bl].data, abs=1e-10
        )
