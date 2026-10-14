# ruff: noqa: D100, D103


import numpy as np
import pytest
from fockit.algebra import OpSum, nc, zero
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from scipy.optimize import root as root_fun
from test_common import (
    IMPURITY_SOLVERS,
    bethe_h0_k,
    build_cubic_h0_k,
    build_two_orbital_complex_h0_k,
    make_impurity_solver,
    one_band_cubic_h0_k,
    random_unitary,
    symmetrize_blocks,
)

from risb import LatticeSolver, helpers
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
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
def test_three_orbital_kanamori_is_a_paramagnetic_mott_insulator(
    solver_name: str,
) -> None:
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
    # The ground state of the impurity is a spin multiplet, and the symmetrizer
    # averages it into the same density in each spin
    assert S.rho_c[0]["up"] == pytest.approx(S.rho_c[0]["dn"], abs=1e-10)


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


def test_check_projectors_rejects_only_projectors_that_break_the_assumptions(
    solver_name: str, subtests: pytest.Subtests
) -> None:
    spin_names = ["up", "dn"]
    h0 = one_band_cubic_h0_k(4, supercell=True)
    h0_k = dict.fromkeys(spin_names, h0)
    n_k = h0.shape[0]
    site_a = np.array([[1.0, 0.0]])
    site_b = np.array([[0.0, 1.0]])
    # The same rotation of both sites at every k-point
    rotations = np.array([random_unitary(2, seed=k) for k in range(n_k)])

    def lattice(projectors: list[NDArray[np.inexact]], check: bool) -> None:
        gf_struct = [[(bl, 1) for bl in spin_names]] * len(projectors)
        LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=[
                make_impurity_solver(solver_name, nc("up", 0) * nc("dn", 0), gf)
                for gf in gf_struct
            ],
            update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
            projectors=[dict.fromkeys(spin_names, P) for P in projectors],
            check_projectors=check,
        )

    not_orthonormal_at_one_k = np.broadcast_to(site_a, (n_k, 1, 2)).copy()
    not_orthonormal_at_one_k[3] *= 2
    bad = {
        "overlapping": [site_a, site_a],
        "not orthonormal": [2 * site_a, site_b],
        "not orthonormal at one k-point": [not_orthonormal_at_one_k, site_b],
    }
    for case, projectors in bad.items():
        with subtests.test(case):
            with pytest.raises(ValueError, match="projectors"):
                lattice(projectors, check=True)
            # Not checked by default
            lattice(projectors, check=False)
    good = {
        "an orbital in no cluster": [site_a],
        "depending on k": [site_a @ rotations, site_b @ rotations],
    }
    for case, projectors in good.items():
        with subtests.test(case):
            lattice(projectors, check=True)


def test_orbital_in_no_cluster_keeps_its_band(solver_name: str) -> None:
    # Site 1 in no cluster must give the same as site 1 a cluster without
    # interactions, where RISB is exact
    spin_names = ["up", "dn"]
    h0_k = dict.fromkeys(spin_names, one_band_cubic_h0_k(6, supercell=True))
    site_a = dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))
    site_b = dict.fromkeys(spin_names, np.array([[0.0, 1.0]]))
    h_ints = [4 * nc("up", 0) * nc("dn", 0), zero()]

    def solve(projectors: list[BlockMatrix]) -> LatticeSolver:
        gf_struct = [[(bl, 1) for bl in spin_names]] * len(projectors)
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=[
                make_impurity_solver(solver_name, h_ints[i], gf)
                for i, gf in enumerate(gf_struct)
            ],
            update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
            projectors=projectors,
        )
        for i in range(S.n_clusters):
            for bl in spin_names:
                S.Lambda[i][bl] = np.array([[2.0 if i == 0 else 0.0]])
        S.solve(tol=1e-12)
        return S

    one_cluster = solve([site_a])
    two_clusters = solve([site_a, site_b])
    assert one_cluster.optimize.success
    for bl in spin_names:
        assert one_cluster.Z[0][bl] == pytest.approx(two_clusters.Z[0][bl], abs=1e-8)
        assert two_clusters.Z[1][bl] == pytest.approx(np.eye(1), abs=1e-8)


def test_projectors_that_depend_on_k_give_the_same_solution(solver_name: str) -> None:
    # The lattice in a basis rotated differently at every k-point, with the
    # projectors rotated with it, is the same problem
    spin_names = ["up", "dn"]
    h0 = one_band_cubic_h0_k(6, supercell=True)
    h0[:, 0, 0] *= 1.3  # Inequivalent sites
    rotations = np.array([random_unitary(2, seed=k) for k in range(h0.shape[0])])
    sites = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]

    def solve(
        h0: NDArray[np.complex128], projectors: list[NDArray[np.inexact]]
    ) -> LatticeSolver:
        gf_struct = [[(bl, 1) for bl in spin_names]] * 2
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0),
            gf_struct=gf_struct,
            impurity_solvers=[
                make_impurity_solver(solver_name, 4 * nc("up", 0) * nc("dn", 0), gf)
                for gf in gf_struct
            ],
            update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
            projectors=[dict.fromkeys(spin_names, P) for P in projectors],
            check_projectors=True,
        )
        for i in range(2):
            for bl in spin_names:
                S.Lambda[i][bl] = np.array([[2.0]])
        S.solve(tol=1e-12)
        return S

    fixed = solve(h0, sites)
    rotated = solve(
        rotations @ h0 @ np.swapaxes(rotations.conj(), -1, -2),
        [P @ np.swapaxes(rotations.conj(), -1, -2) for P in sites],
    )
    assert rotated.optimize.success
    for i in range(2):
        for bl in spin_names:
            assert rotated.Z[i][bl] == pytest.approx(fixed.Z[i][bl], abs=1e-8)
            assert rotated.Lambda[i][bl] == pytest.approx(fixed.Lambda[i][bl], abs=1e-8)


def test_constant_double_counting_is_a_shift_of_the_local_energies(
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
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    potential = np.array([[0.3, 0.1], [0.1, -0.2]])
    calls: list[tuple[int, BlockMatrix]] = []

    def double_counting(i: int, rho_c: BlockMatrix) -> BlockMatrix:
        calls.append((i, rho_c))
        return dict.fromkeys(rho_c, potential)

    def solve(
        h0_k: dict[str, NDArray[np.complex128]], with_hook: bool
    ) -> LatticeSolver:
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=make_impurity_solver(solver_name, h_int, gf_struct),
            update_weights=SmearingKWeight(beta=40, n_target=2).update_weights,
            double_counting=double_counting if with_hook else None,
        )
        for bl, _ in gf_struct:
            S.Lambda[0][bl] = 1.5 * np.eye(2)
        S.solve(tol=1e-12)
        return S

    h0_k = build_two_orbital_complex_h0_k()
    hook = solve(h0_k, with_hook=True)
    shifted = solve({bl: h - potential for bl, h in h0_k.items()}, with_hook=False)
    assert hook.optimize.success
    for bl, _ in gf_struct:
        assert hook.Z[0][bl] == pytest.approx(shifted.Z[0][bl], abs=1e-8)
        assert hook.Lambda[0][bl] == pytest.approx(shifted.Lambda[0][bl], abs=1e-8)
    # Called every cycle with the cluster index and a matrix for each block
    assert len(calls) == hook.optimize.n + 1
    assert all(i == 0 and set(rho) == {"up", "dn"} for i, rho in calls)


def test_ghost_risb_without_interactions_is_the_non_interacting_solution(
    solver_name: str,
) -> None:
    if solver_name != "fockit":
        pytest.skip("only FockitImpuritySolver has bath orbitals of their own size")
    B = 3
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = [{bl: B for bl, _ in gf_struct}]
    h0_k = {bl: bethe_h0_k(200) for bl, _ in gf_struct}
    impurity_solver = FockitImpuritySolver(zero(), gf_struct, bath_size=bath_size[0])
    kweight = SmearingKWeight(beta=200, n_target=1 + (B - 1))
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        update_weights=kweight.update_weights,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=bath_size,
    )
    S.solve(tol=1e-10)
    assert S.self_consistency_error < 1e-8
    assert kweight.mu is not None
    quasiparticle_weight = helpers.get_z(S.R[0]["up"], S.Lambda[0]["up"], kweight.mu)
    assert quasiparticle_weight == pytest.approx(np.eye(1), abs=1e-8)
    assert impurity_solver.overlap(nc("up", 0) * nc("dn", 0)) == pytest.approx(
        0.25, abs=1e-8
    )
    # Kinetic energy sum_k eps_k R^dag f(H_qp) R of both spins, -4 / (3 pi) for
    # the half-filled semicircular density of states
    amplitude = np.einsum("a,kan->kn", S.R_full["up"][:, 0], S.bloch_vector_qp["up"])
    kinetic = 2 * np.sum(
        S.kweights["up"] * np.abs(amplitude) ** 2 * h0_k["up"][:, 0, 0, None]
    )
    assert kinetic == pytest.approx(-4 / (3 * np.pi), abs=1e-4)


@pytest.mark.parametrize(
    ("U", "energy", "double_occupancy", "Z"),
    [(2.4, -0.061, 0.054, 0.17), (3.5, -0.037, 0.0108, 0.0)],
)
def test_ghost_risb_gives_the_published_energy_of_the_hubbard_model(
    solver_name: str, U: float, energy: float, double_occupancy: float, Z: float
) -> None:
    # Half-filled Hubbard model on the Bethe lattice with three quasiparticles
    # per spin, a metal and a Mott insulator. Values read off Figs. 3 and 4 of
    # Lee, Lanata, and Kotliar, Phys. Rev. B 107, L121104 (2023)
    if solver_name != "fockit":
        pytest.skip("only FockitImpuritySolver has bath orbitals of their own size")
    B = 3
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = [{bl: B for bl, _ in gf_struct}]
    h0_k = {bl: bethe_h0_k(200) for bl, _ in gf_struct}
    h_int = U * nc("up", 0) * nc("dn", 0) - U / 2 * (nc("up", 0) + nc("dn", 0))
    Sz = nc("up", 0) - nc("dn", 0)
    for a in range(B):
        Sz += nc("bath_up", a) - nc("bath_dn", a)
    # A thermal impurity averages the two ground states of the Mott insulator
    impurity_solver = FockitImpuritySolver(
        h_int, gf_struct, diagonal_symmetries={Sz: 0}, beta=200, bath_size=bath_size[0]
    )
    kweight = SmearingKWeight(beta=200, n_target=1 + (B - 1))
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        update_weights=kweight.update_weights,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=bath_size,
    )
    S.solve(tol=1e-9)
    assert S.n_ghost_electrons == pytest.approx(B - 1)
    assert S.self_consistency_error < 1e-8
    assert S.rho_c[0]["up"] == pytest.approx(np.array([[0.5]]), abs=1e-8)
    # Kinetic energy sum_k eps_k R^dag f(H_qp) R of both spins
    amplitude = np.einsum("a,kan->kn", S.R_full["up"][:, 0], S.bloch_vector_qp["up"])
    kinetic = 2 * np.sum(
        S.kweights["up"] * np.abs(amplitude) ** 2 * h0_k["up"][:, 0, 0, None]
    )
    docc = impurity_solver.overlap(nc("up", 0) * nc("dn", 0))
    assert docc == pytest.approx(double_occupancy, abs=1e-3)
    assert kinetic + U * docc == pytest.approx(energy, abs=1e-3)
    assert kweight.mu is not None
    quasiparticle_weight = helpers.get_z(S.R[0]["up"], S.Lambda[0]["up"], kweight.mu)
    assert quasiparticle_weight == pytest.approx(np.array([[Z]]), abs=1e-2)
