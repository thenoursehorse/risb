# ruff: noqa: D100, D103


import numpy as np
import pytest
from fockit.algebra import nc, zero
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.optimize import root as root_fun
from test_common import (
    IMPURITY_SOLVERS,
    build_cubic_h0_k,
    make_impurity_solver,
    symmetrize_blocks,
)

from risb import LatticeSolver
from risb.helpers_fockit import h_int_kanamori
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix, GfStruct, ImpuritySolver

# FIXME add one_shot test

# Every test runs once per impurity solver, through the model fixtures
pytestmark = pytest.mark.usefixtures("solver_name")


@pytest.fixture(params=IMPURITY_SOLVERS)
def solver_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


type Model = tuple[
    GfStruct,
    dict[str, NDArray[np.float64]],
    ImpuritySolver,
    SmearingKWeight,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]


def do_assert(
    subtests: pytest.Subtests,
    mu: float | None,
    Lambda: list[BlockMatrix],
    Z: list[BlockMatrix],
    mu_expected: float,
    Lambda_expected: NDArray[np.float64],
    Z_expected: NDArray[np.float64],
) -> None:
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


@pytest.fixture
def one_band(solver_name: str) -> Model:
    n_orb = 1
    spatial_dim = 3
    nkx = 10
    beta = 40
    U = 4
    mu = U / 2  # half-filling
    n_target = 1
    gf_struct = [(bl, n_orb) for bl in ["up", "dn"]]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = U * nc("up", 0) * nc("dn", 0)
    impurity_solver = make_impurity_solver(solver_name, h_int, gf_struct)
    # kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[2.0]])
    Z_expected = np.array([[0.437828801025]])
    return gf_struct, h0_k, impurity_solver, kweight, mu, Lambda_expected, Z_expected


@pytest.fixture
def bilayer(solver_name: str) -> Model:
    U = 4
    V = 0.25
    mu = U / 2.0  # half-filling
    n_target = 2
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    for bl in h0_k:
        h0_k[bl][:, 0, 1] += V
        h0_k[bl][:, 1, 0] += V
    h_int = zero()
    for o in range(n_orb):
        h_int += U * nc("up", o) * nc("dn", o)
    impurity_solver = make_impurity_solver(solver_name, h_int, gf_struct)
    # kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[2.0, 0.114569681915], [0.114569681915, 2.0]])
    Z_expected = np.array([[0.452846149446, 0], [0, 0.452846149446]])
    return gf_struct, h0_k, impurity_solver, kweight, mu, Lambda_expected, Z_expected


@pytest.fixture
def kanamori(solver_name: str) -> Model:
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
    gf_struct = [(bl, n_orb) for bl in spin_names]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = h_int_kanamori(
        spin_names=spin_names,
        n_orb=n_orb,
        U=np.array([[0, Up - J], [Up - J, 0]]),
        Uprime=np.array([[U, Up], [Up, U]]),
        J_hund=J,
    )
    impurity_solver = make_impurity_solver(solver_name, h_int, gf_struct)
    kweight = SmearingKWeight(beta=beta, mu=mu)
    kweight = SmearingKWeight(beta=beta, n_target=n_target)
    Lambda_expected = np.array([[3.0, 0.0], [0.0, 3.0]])
    Z_expected = np.array([[0.574940323948, 0.0], [0.0, 0.574940323948]])
    return gf_struct, h0_k, impurity_solver, kweight, mu, Lambda_expected, Z_expected


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_diis_symmetrize(
    subtests: pytest.Subtests, request: pytest.FixtureRequest, model: str
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        kweight,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        update_weights=kweight.update_weights,
        symmetrizers=[symmetrize_blocks],
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
def test_diis_nosymmetrize(
    subtests: pytest.Subtests, request: pytest.FixtureRequest, model: str
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        kweight,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
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
def test_scipy_root(
    subtests: pytest.Subtests,
    request: pytest.FixtureRequest,
    model: str,
    root_method: str,
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        kweight,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)

    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        update_weights=kweight.update_weights,
        symmetrizers=[symmetrize_blocks],
        root=root_fun,
        return_x_new=False,
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    result = S.solve(method=root_method, options={"fatol": 1e-10})
    assert isinstance(result, OptimizeResult)
    assert result.success, result.message
    mu_calculated = kweight.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


def test_three_orbital_kanamori_is_mott_insulator(solver_name: str) -> None:
    n_orb = 3
    U = 3
    J = 0.2 * U
    Up = U - 2 * J
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    off_diag = 1 - np.eye(n_orb)
    h_int = h_int_kanamori(
        spin_names,
        n_orb,
        U=(Up - J) * off_diag,
        Uprime=U * np.eye(n_orb) + Up * off_diag,
        J_hund=J,
    )
    S = LatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct, nkx=10, spatial_dim=3),
        gf_struct=gf_struct,
        impurity_solvers=make_impurity_solver(solver_name, h_int, gf_struct),
        update_weights=SmearingKWeight(beta=40, n_target=n_orb).update_weights,
        symmetrizers=[symmetrize_blocks],
    )
    S.solve(tol=1e-8)
    assert S.optimize.success
    for bl in S.Z[0]:
        assert S.Z[0][bl] == pytest.approx(np.zeros((n_orb, n_orb)), abs=1e-6)
