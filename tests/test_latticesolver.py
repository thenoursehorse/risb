# ruff: noqa: T201, D100, D103


import numpy as np
import pytest
from test_common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import Operator, n
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure

from risb import LatticeSolver
from risb.embedding import EmbeddingAtomDiag
from risb.kweight import SmearingKWeight

# FIXME add one_shot test


def do_assert(subtests, mu, Lambda, Z, mu_expected, Lambda_expected, Z_expected):
    n_clusters = len(Lambda)
    abs = 1e-10
    with subtests.test(msg="mu"):
        assert mu == pytest.approx(mu_expected, abs=abs)
    with subtests.test(msg="Lambda"):
        for i in range(n_clusters):
            for bl in Lambda[i]:
                assert Lambda[i][bl] == pytest.approx(Lambda_expected, abs=abs)
    with subtests.test(msg="Z"):
        for i in range(n_clusters):
            for bl in Z[i]:
                assert Z[i][bl] == pytest.approx(Z_expected, abs=abs)


@pytest.fixture()
def one_band():
    n_orb = 1
    spatial_dim = 3
    nkx = 10
    beta = 40
    U = 4
    mu = U / 2  # half-filling
    n_target = 1
    gf_struct = [(bl, n_orb) for bl in ["up", "dn"]]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = U * n("up", 0) * n("dn", 0)
    embedding = EmbeddingAtomDiag(h_int, gf_struct)
    # kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[2.0]])
    Z_expected = np.array([[0.437828801025]])
    return gf_struct, h0_k, embedding, kweight, mu, Lambda_expected, Z_expected


@pytest.fixture()
def bilayer():
    U = 4
    V = 0.25
    mu = U / 2.0  # half-filling
    n_target = 2
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ["up", "dn"]
    gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    for bl in h0_k:
        h0_k[bl][:, 0, 1] += V
        h0_k[bl][:, 1, 0] += V
    h_int = Operator()
    for o in range(n_orb):
        h_int += U * n("up", o) * n("dn", o)
    embedding = EmbeddingAtomDiag(h_int, gf_struct)
    # kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[2.0, 0.114569681915], [0.114569681915, 2.0]])
    Z_expected = np.array([[0.452846149446, 0], [0, 0.452846149446]])
    return gf_struct, h0_k, embedding, kweight, mu, Lambda_expected, Z_expected


@pytest.fixture()
def kanamori():
    coeff = 0.2
    U = 3
    J = coeff * U
    Up = U - 2 * J
    mu = 0.5 * U + 0.5 * Up + 0.5 * (Up - J)  # half-filling
    n_target = 2
    # mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2 # quarter-filling DMFT result
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ["up", "dn"]
    gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = h_int_kanamori(
        spin_names=spin_names,
        n_orb=n_orb,
        U=np.array([[0, Up - J], [Up - J, 0]]),
        Uprime=np.array([[U, Up], [Up, U]]),
        J_hund=J,
        off_diag=True,
    )
    embedding = EmbeddingAtomDiag(h_int, gf_struct)
    kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[3.0, 0.0], [0.0, 3.0]])
    Z_expected = np.array([[0.574940323948, 0.0], [0.0, 0.574940323948]])
    return gf_struct, h0_k, embedding, kweight, mu, Lambda_expected, Z_expected


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_diis_symmetrize(subtests, request, model):
    model = request.getfixturevalue(model)
    gf_struct, h0_k, embedding, kweight, mu_expected, Lambda_expected, Z_expected = (
        model
    )
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        embedding=[embedding],
        update_weights=kweight.update_weights,
        symmetries=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    S.solve()
    mu_calculated = kweight.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_diis_nosymmetrize(subtests, request, model):
    model = request.getfixturevalue(model)
    gf_struct, h0_k, embedding, kweight, mu_expected, Lambda_expected, Z_expected = (
        model
    )
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        embedding=[embedding],
        update_weights=kweight.update_weights,
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    S.solve()
    mu_calculated = kweight.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


@pytest.mark.parametrize(
    ("model", "root_method"),
    [
        ("one_band", "krylov"),
        ("bilayer", "krylov"),
        ("kanamori", "krylov"),
    ],
)
def test_scipy_root(subtests, request, model, root_method):
    model = request.getfixturevalue(model)
    gf_struct, h0_k, embedding, kweight, mu_expected, Lambda_expected, Z_expected = (
        model
    )
    from scipy.optimize import root as root_fun

    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        embedding=[embedding],
        update_weights=kweight.update_weights,
        symmetries=[symmetrize_blocks],
        root=root_fun,
        return_x_new=False,
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    S.solve(method=root_method, tol=1e-12)
    mu_calculated = kweight.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )
