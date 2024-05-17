# ruff: noqa: T201, D100, D103
from itertools import product

import numpy as np
import pytest
from triqs.operators import Operator, c, c_dag, n
from triqs.operators.util.observables import N_op, S2_op

from risb.embedding import EmbeddingAtomDiag


def do_assert(
    subtests,
    rho_f,
    rho_cf,
    rho_c,
    gs_energy,
    N,
    S2,
    rho_f_expected,
    rho_cf_expected,
    rho_c_expected,
    gs_energy_expected,
    N_expected,
    S2_expected,
):
    abs = 1e-12
    with subtests.test(msg="rho_f"):
        for bl in rho_f:
            assert rho_f[bl] == pytest.approx(rho_f_expected, abs=abs)
    with subtests.test(msg="rho_cf"):
        for bl in rho_cf:
            assert rho_cf[bl] == pytest.approx(rho_cf_expected, abs=abs)
    with subtests.test(msg="rho_c"):
        for bl in rho_c:
            assert rho_c[bl] == pytest.approx(rho_c_expected, abs=abs)
    with subtests.test(msg="gs_energy"):
        assert gs_energy == pytest.approx(gs_energy_expected, abs=abs)
    with subtests.test(msg="N"):
        assert pytest.approx(N_expected, abs=abs) == N
    with subtests.test(ms="S2"):
        assert pytest.approx(S2_expected, abs=abs) == S2


@pytest.fixture()
def one_band():
    U = 1
    mu = U / 2.0  # half-filling
    n_orb = 1
    spin_names = ["up", "dn"]
    h_int = U * n("up", 0) * n("dn", 0)
    Lambda_c = {}
    D = {}
    h0_loc_mat = {}
    for bl in spin_names:
        Lambda_c[bl] = np.array([[-mu]])
        D[bl] = np.array([[-0.3333]])
        h0_loc_mat[bl] = np.array([[0]])
    return spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int


@pytest.fixture()
def one_band_expected():
    rho_f_expected = np.array([[0.5]])
    rho_cf_expected = np.array([[0.4681588161332029]])
    rho_c_expected = np.array([[0.5]])
    gs_energy_expected = -0.9619378905494498
    N_expected = 1.0
    S2_expected = 0.5066828353209953
    return (
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    )


@pytest.fixture()
def bilayer():
    U = 1
    V = 0.25
    J = 0
    mu = U / 2.0  # half-filling
    n_orb = 2
    spin_names = ["up", "dn"]
    h_int = Operator()
    for o in range(n_orb):
        h_int += U * n("up", o) * n("dn", o)
    for s1, s2 in product(spin_names, spin_names):
        h_int += 0.5 * J * c_dag(s1, 0) * c(s2, 0) * c_dag(s2, 1) * c(s1, 1)
    Lambda_c = {}
    D = {}
    h0_loc_mat = {}
    for bl in spin_names:
        Lambda_c[bl] = np.array([[-mu, -0.00460398], [-0.00460398, -mu]])
        D[bl] = np.array([[-2.59694448e-01, 0], [0, -2.59694448e-01]])
        h0_loc_mat[bl] = np.array([[0, V], [V, 0]])
    return spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int


@pytest.fixture()
def bilayer_expected():
    rho_f_expected = np.array([[0.5, -0.1999913941210893], [-0.1999913941210893, 0.5]])
    rho_cf_expected = np.array([[0.42326519677453511, 0], [0, 0.42326519677453511]])
    rho_c_expected = np.array([[0.5, -0.1836332097072352], [-0.1836332097072352, 0.5]])
    gs_energy_expected = -1.7429249197415944
    N_expected = 2.0
    S2_expected = 0.8247577338845973
    return (
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    )


@pytest.fixture()
def dh_trimer():
    # At two-thirds filling
    U = 1
    tk = 1
    n_orb = 3
    spin_names = ["up", "dn"]

    def hubb_N(tk, U, n_orb, spin_names):  # noqa: ARG001
        # hopping
        # phi = 2.0 * np.pi / n_orb
        # for a,m,mm,s in product(range(n_orb),range(n_orb),range(n_orb), spin_names):
        #    h0_loc += (-tk / n_orb) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * a * m) * np.exp(1j * phi * np.mod(a+1,n_orb) * mm)
        #    h0_loc += (-tk / n_orb) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * np.mod(a+1,n_orb) * m) * np.exp(1j * phi * a * mm)
        # hubbard U
        h_int = Operator()
        for m, mm, mmm in product(range(n_orb), range(n_orb), range(n_orb)):
            h_int += (
                (U / n_orb)
                * c_dag("up", m)
                * c("up", mm)
                * c_dag("dn", mmm)
                * c("dn", np.mod(m + mmm - mm, n_orb))
            )
        return h_int.real

    h_int = hubb_N(tk, U, n_orb, spin_names)
    Lambda_c = {}
    D = {}
    h0_loc_mat = {bl: np.zeros([n_orb, n_orb]) for bl in spin_names}
    for bl in spin_names:
        Lambda_c[bl] = np.array(
            [
                [-1.91730088, -0.0, -0.0],
                [-0.0, -1.69005946, -0.0],
                [-0.0, -0.0, -1.69005946],
            ]
        )
        D[bl] = np.array(
            [[-0.26504931, 0.0, 0.0], [0.0, -0.39631238, 0.0], [0.0, 0.0, -0.39631238]]
        )
        h0_loc_mat[bl][0, 0] = -2 * tk
        h0_loc_mat[bl][1, 1] = tk
        h0_loc_mat[bl][2, 2] = tk
    return spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int


@pytest.fixture()
def dh_trimer_expected():
    rho_f_expected = np.array(
        [
            [0.9932309740187902, 0.0, 0.0],
            [0.0, 0.5033842231804342, 0.0],
            [0.0, 0.0, 0.5033842231804342],
        ]
    )
    rho_cf_expected = np.array(
        [
            [0.0811187181751014, 0.0, 0.0],
            [0.0, 0.4910360103357626, 0.0],
            [0.0, 0.0, 0.4910360103357626],
        ]
    )
    rho_c_expected = np.array(
        [
            [0.9909259681893234, 0.0, 0.0],
            [0.0, 0.5045367260951683, 0.0],
            [0.0, 0.0, 0.5045367260951683],
        ]
    )
    gs_energy_expected = -9.555511743344764
    N_expected = 3.99999884075932
    S2_expected = 0.9171025003755656
    return (
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    )


@pytest.mark.parametrize(
    ("model", "model_expected"),
    [
        ("one_band", "one_band_expected"),
        ("bilayer", "bilayer_expected"),
        ("dh_trimer", "dh_trimer_expected"),
    ],
)
def test_solve(subtests, request, model, model_expected):
    model = request.getfixturevalue(model)
    model_expected = request.getfixturevalue(model_expected)
    spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int = model
    (
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    ) = model_expected
    gf_struct = [(bl, n_orb) for bl in spin_names]
    embedding = EmbeddingAtomDiag(h_int, gf_struct)
    embedding.set_h_emb(Lambda_c, D, h0_loc_mat)
    embedding.solve()
    rho_f = {}
    rho_cf = {}
    rho_c = {}
    for bl, _bl_size in gf_struct:
        rho_f[bl] = embedding.get_rho_f(bl)
        rho_cf[bl] = embedding.get_rho_cf(bl)
        rho_c[bl] = embedding.get_rho_c(bl)
    gs_energy = embedding.gs_energy
    NOp = N_op(spin_names, n_orb, off_diag=True)
    S2Op = S2_op(spin_names, n_orb, off_diag=True)
    N = embedding.overlap(NOp)
    S2 = embedding.overlap(S2Op)
    do_assert(
        subtests,
        rho_f,
        rho_cf,
        rho_c,
        gs_energy,
        N,
        S2,
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    )
