# ruff: noqa: D100, D103


import numpy as np
import pytest
from fockit.algebra import OpSum, nc, zero
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.optimize import root as root_fun
from test_common import (
    IMPURITY_SOLVERS,
    build_cubic_h0_k,
    build_two_orbital_complex_h0_k,
    make_impurity_solver,
    one_band_cubic_h0_k,
    random_unitary,
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
    # Lambda and Z depend on the gauge of the quasiparticles; their eigenvalues do not
    with subtests.test(msg="Lambda"):
        for i in range(n_clusters):
            for bl in Lambda[i]:
                assert np.linalg.eigvalsh(Lambda[i][bl]) == pytest.approx(
                    np.linalg.eigvalsh(Lambda_expected), abs=abs
                )
    with subtests.test(msg="Z"):
        for i in range(n_clusters):
            for bl in Z[i]:
                assert np.linalg.eigvalsh(Z[i][bl]) == pytest.approx(
                    np.linalg.eigvalsh(Z_expected), abs=abs
                )


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


# In the atomic limit the self-consistency equations need not be satisfied
@pytest.mark.filterwarnings("ignore:The RISB self-consistency:RuntimeWarning")
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


def two_orbital_complex_lattice(
    solver_name: str, h_int: OpSum
) -> tuple[LatticeSolver, SmearingKWeight]:
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    kweight = SmearingKWeight(beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=build_two_orbital_complex_h0_k(),
        gf_struct=gf_struct,
        impurity_solvers=make_impurity_solver(solver_name, h_int, gf_struct),
        update_weights=kweight.update_weights,
    )
    return S, kweight


def test_quasiparticle_basis_of_the_starting_guess_does_not_change_the_physics(
    solver_name: str,
) -> None:
    U, J = 3.0, 0.6
    Up = U - 2 * J
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * (1 - np.eye(2)),
        Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
        J_hund=J,
    )
    results = []
    for u in [np.eye(2), random_unitary(2, seed=1), random_unitary(2, seed=2)]:
        S, kweight = two_orbital_complex_lattice(solver_name, h_int)
        for bl in ["up", "dn"]:
            S.Lambda[0][bl] = 1.5 * np.eye(2)
            S.R[0][bl] = u.conj().T
        S.solve(tol=1e-10)
        assert S.optimize.success
        results.append((kweight.mu, S.Z[0]["up"]))
    for mu, weight in results[1:]:
        assert mu == pytest.approx(results[0][0], abs=1e-8)
        assert weight == pytest.approx(results[0][1], abs=1e-8)


def test_rotated_copy_of_the_non_interacting_solution_is_converged(
    solver_name: str,
) -> None:
    S, _ = two_orbital_complex_lattice(solver_name, zero())
    u = random_unitary(2, seed=3)
    for bl in ["up", "dn"]:
        S.R[0][bl] = u.conj().T
    S.solve(tol=1e-10)
    assert S.optimize.success
    assert S.optimize.n <= 2
    for bl in ["up", "dn"]:
        assert S.Z[0][bl] == pytest.approx(np.eye(2), abs=1e-8)


def test_completely_full_orbital_decouples_and_leaves_the_one_band_solution(
    solver_name: str,
) -> None:
    # Orbital 0 is a completely full band with no hopping to orbital 1, so
    # with only density interactions orbital 1 is the one-band model, shifted
    # in energy. The full orbital has no bath partner in the impurity problem
    spin_names = ["up", "dn"]
    gf_struct = [(bl, 2) for bl in spin_names]
    h_int = h_int_kanamori(
        spin_names, 2, U=2 * (1 - np.eye(2)), Uprime=2 * np.ones((2, 2)), J_hund=0
    )
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=10, spatial_dim=3)
    for bl in h0_k:
        h0_k[bl] = h0_k[bl] + np.diag([-2.0, 2.0])
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=make_impurity_solver(solver_name, h_int, gf_struct),
        update_weights=SmearingKWeight(beta=40, n_target=3).update_weights,
        symmetrizers=[symmetrize_blocks],
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = np.diag([0.0, 6.0])
    S.solve(tol=1e-10)

    gf_struct_one = [(bl, 1) for bl in spin_names]
    one_band = LatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct_one, nkx=10, spatial_dim=3),
        gf_struct=gf_struct_one,
        impurity_solvers=make_impurity_solver(
            solver_name, 2 * nc("up", 0) * nc("dn", 0), gf_struct_one
        ),
        update_weights=SmearingKWeight(beta=40, n_target=1).update_weights,
    )
    for bl, _ in gf_struct_one:
        one_band.Lambda[0][bl] = np.array([[1.0]])
    one_band.solve(tol=1e-10)

    assert S.optimize.success
    assert S.self_consistency_error < 1e-8
    for bl, _ in gf_struct:
        assert np.linalg.eigvalsh(S.Z[0][bl]) == pytest.approx(
            [0, one_band.Z[0][bl][0, 0].real], abs=1e-6
        )


def test_two_site_cell_with_a_cluster_per_site_gives_the_one_site_solution(
    solver_name: str, subtests: pytest.Subtests
) -> None:
    U = 4.0
    spin_names = ["up", "dn"]

    def solve(
        h0_k: NDArray[np.complex128],
        gf_struct: list[GfStruct],
        projectors: list[BlockMatrix] | None = None,
        gf_struct_mapping: list[dict[str, str]] | None = None,
    ) -> LatticeSolver:
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0_k),
            gf_struct=gf_struct,
            impurity_solvers=[
                make_impurity_solver(
                    solver_name, U * nc(gf[0][0], 0) * nc(gf[1][0], 0), gf
                )
                for gf in gf_struct
            ],
            update_weights=SmearingKWeight(
                beta=40, n_target=len(gf_struct)
            ).update_weights,
            projectors=projectors,
            gf_struct_mapping=gf_struct_mapping,
            check_projectors=projectors is not None,
        )
        for i in range(S.n_clusters):
            for bl in S.Lambda[i]:
                S.Lambda[i][bl] = np.array([[U / 2]])
        S.solve(tol=1e-12)
        return S

    one_site = solve(
        one_band_cubic_h0_k(6, supercell=False), [[(bl, 1) for bl in spin_names]]
    )
    projectors = [
        dict.fromkeys(spin_names, np.array([[1.0, 0.0]])),
        dict.fromkeys(spin_names, np.array([[0.0, 1.0]])),
    ]
    h0_k = one_band_cubic_h0_k(6, supercell=True)
    with subtests.test("same blocks as h0_k"):
        S = solve(h0_k, [[(bl, 1) for bl in spin_names]] * 2, projectors)
        for i in range(2):
            assert S.Z[i]["up"] == pytest.approx(one_site.Z[0]["up"], abs=1e-10)
    with subtests.test("blocks renamed with gf_struct_mapping"):
        names = [{bl: f"{bl}_{site}" for bl in spin_names} for site in ["A", "B"]]
        S = solve(
            h0_k,
            [[(names[i][bl], 1) for bl in spin_names] for i in range(2)],
            [
                {names[i][bl]: P[bl] for bl in spin_names}
                for i, P in enumerate(projectors)
            ],
            [{names[i][bl]: bl for bl in spin_names} for i in range(2)],
        )
        for i in range(2):
            assert S.Z[i][names[i]["up"]] == pytest.approx(
                one_site.Z[0]["up"], abs=1e-10
            )


def test_check_projectors_rejects_projectors_that_break_the_assumptions(
    solver_name: str, subtests: pytest.Subtests
) -> None:
    spin_names = ["up", "dn"]
    h0_k = dict.fromkeys(spin_names, one_band_cubic_h0_k(4, supercell=True))
    gf_struct = [[(bl, 1) for bl in spin_names]] * 2
    impurity_solvers = [
        make_impurity_solver(solver_name, nc("up", 0) * nc("dn", 0), gf)
        for gf in gf_struct
    ]
    site_a = np.array([[1.0, 0.0]])
    bad = {
        "missing an orbital": [site_a, site_a * 0],
        "overlapping": [site_a, site_a],
        "not orthonormal": [2 * site_a, np.array([[0.0, 1.0]])],
    }
    for case, (P_a, P_b) in bad.items():
        projectors = [dict.fromkeys(spin_names, P_a), dict.fromkeys(spin_names, P_b)]
        with subtests.test(case):
            with pytest.raises(ValueError, match="projectors"):
                LatticeSolver(
                    h0_k=h0_k,
                    gf_struct=gf_struct,
                    impurity_solvers=impurity_solvers,
                    update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
                    projectors=projectors,
                    check_projectors=True,
                )
            # Not checked by default
            LatticeSolver(
                h0_k=h0_k,
                gf_struct=gf_struct,
                impurity_solvers=impurity_solvers,
                update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
                projectors=projectors,
            )
