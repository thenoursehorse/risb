# ruff: noqa: D100, D103
from itertools import product
from typing import Any

import numpy as np
import pytest
from fockit.algebra import OpSum, c, cdag, nc, zero
from numpy.typing import NDArray
from test_common import IMPURITY_SOLVERS, make_impurity_solver, overlap

from risb.helpers_fockit import N_op, S2_op
from risb.impurity_solvers import FockitImpuritySolver
from risb.types import BlockMatrix

type Model = tuple[
    list[str],
    int,
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
    dict[str, NDArray[np.float64]],
    OpSum,
]
type Expected = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float, float
]


def do_assert(
    subtests: pytest.Subtests,
    rho_f: BlockMatrix,
    rho_cf: BlockMatrix,
    rho_c: BlockMatrix,
    gs_energy: float,
    N: complex,
    S2: complex,
    rho_f_expected: NDArray[np.float64],
    rho_cf_expected: NDArray[np.float64],
    rho_c_expected: NDArray[np.float64],
    gs_energy_expected: float,
    N_expected: float,
    S2_expected: float,
) -> None:
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
    with subtests.test(msg="S2"):
        assert pytest.approx(S2_expected, abs=abs) == S2


@pytest.fixture
def one_band() -> Model:
    U = 1
    mu = U / 2.0  # half-filling
    n_orb = 1
    spin_names = ["up", "dn"]
    h_int = U * nc("up", 0) * nc("dn", 0)
    Lambda_c: dict[str, NDArray[np.float64]] = {}
    D: dict[str, NDArray[np.float64]] = {}
    h0_loc_mat: dict[str, NDArray[np.float64]] = {}
    for bl in spin_names:
        Lambda_c[bl] = np.array([[-mu]])
        D[bl] = np.array([[-0.3333]])
        h0_loc_mat[bl] = np.array([[0]])
    return spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int


@pytest.fixture
def one_band_expected() -> Expected:
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


@pytest.fixture
def bilayer() -> Model:
    U = 1
    V = 0.25
    J = 0
    mu = U / 2.0  # half-filling
    n_orb = 2
    spin_names = ["up", "dn"]
    h_int = zero()
    for o in range(n_orb):
        h_int += U * nc("up", o) * nc("dn", o)
    for s1, s2 in product(spin_names, spin_names):
        h_int += 0.5 * J * cdag(s1, 0) * c(s2, 0) * cdag(s2, 1) * c(s1, 1)
    Lambda_c: dict[str, NDArray[np.float64]] = {}
    D: dict[str, NDArray[np.float64]] = {}
    h0_loc_mat: dict[str, NDArray[np.float64]] = {}
    for bl in spin_names:
        Lambda_c[bl] = np.array([[-mu, -0.00460398], [-0.00460398, -mu]])
        D[bl] = np.array([[-2.59694448e-01, 0], [0, -2.59694448e-01]])
        h0_loc_mat[bl] = np.array([[0, V], [V, 0]])
    return spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int


@pytest.fixture
def bilayer_expected() -> Expected:
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


@pytest.fixture
def dh_trimer() -> Model:
    # At two-thirds filling
    U = 1
    tk = 1
    n_orb = 3
    spin_names = ["up", "dn"]

    def hubb_N(tk: float, U: float, n_orb: int, spin_names: list[str]) -> OpSum:  # noqa: ARG001
        # hopping
        # phi = 2.0 * np.pi / n_orb
        # for a,m,mm,s in product(range(n_orb),range(n_orb),range(n_orb), spin_names):
        #    h0_loc += (-tk / n_orb) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * a * m) * np.exp(1j * phi * np.mod(a+1,n_orb) * mm)
        #    h0_loc += (-tk / n_orb) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * np.mod(a+1,n_orb) * m) * np.exp(1j * phi * a * mm)
        # hubbard U
        h_int = zero()
        for m, mm, mmm in product(range(n_orb), range(n_orb), range(n_orb)):
            h_int += (
                (U / n_orb)
                * cdag("up", m)
                * c("up", mm)
                * cdag("dn", mmm)
                * c("dn", (m + mmm - mm) % n_orb)
            )
        return h_int

    h_int = hubb_N(tk, U, n_orb, spin_names)
    Lambda_c: dict[str, NDArray[np.float64]] = {}
    D: dict[str, NDArray[np.float64]] = {}
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


@pytest.fixture
def dh_trimer_expected() -> Expected:
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


@pytest.mark.parametrize("solver_name", IMPURITY_SOLVERS)
@pytest.mark.parametrize(
    ("model", "model_expected"),
    [
        ("one_band", "one_band_expected"),
        ("bilayer", "bilayer_expected"),
        ("dh_trimer", "dh_trimer_expected"),
    ],
)
def test_solve(
    subtests: pytest.Subtests,
    request: pytest.FixtureRequest,
    solver_name: str,
    model: str,
    model_expected: str,
) -> None:
    spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int = request.getfixturevalue(model)
    (
        rho_f_expected,
        rho_cf_expected,
        rho_c_expected,
        gs_energy_expected,
        N_expected,
        S2_expected,
    ) = request.getfixturevalue(model_expected)
    gf_struct = [(bl, n_orb) for bl in spin_names]
    impurity_solver = make_impurity_solver(solver_name, h_int, gf_struct)
    impurity_solver.set_h_imp(Lambda_c, D, h0_loc_mat)
    impurity_solver.solve()
    rho_f: BlockMatrix = {}
    rho_cf: BlockMatrix = {}
    rho_c: BlockMatrix = {}
    for bl, _bl_size in gf_struct:
        rho_f[bl] = impurity_solver.get_rho_f(bl)
        rho_cf[bl] = impurity_solver.get_rho_cf(bl)
        rho_c[bl] = impurity_solver.get_rho_c(bl)
    gs_energy = impurity_solver.gs_energy
    N = overlap(impurity_solver, N_op(spin_names, n_orb))
    S2 = overlap(impurity_solver, S2_op(spin_names, n_orb))
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


def test_fockit_solver_rejects_non_hermitian_hamiltonian() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = FockitImpuritySolver(cdag("up", 0) * c("dn", 0), gf_struct)
    one = {bl: np.array([[1.0]]) for bl, _ in gf_struct}
    impurity_solver.set_h_imp(one, one)
    with pytest.raises(ValueError, match="not Hermitian"):
        impurity_solver.solve()


def sz_op(n_orb: int) -> OpSum:
    Sz = zero()
    for o in range(n_orb):
        for up, dn in [("up", "dn"), ("bath_up", "bath_dn")]:
            Sz += 0.5 * (nc(up, o) - nc(dn, o))
    return Sz


def spin_swap(n_orb: int) -> dict[tuple[str, int], tuple[str, int]]:
    swap = {"up": "dn", "dn": "up", "bath_up": "bath_dn", "bath_dn": "bath_up"}
    return {(bl, o): (swap[bl], o) for bl in swap for o in range(n_orb)}


@pytest.mark.parametrize("option", ["eigh", "Sz", "Sz and spin swap"])
@pytest.mark.parametrize(
    ("model", "model_expected", "swap_eigenvalue"),
    [
        ("one_band", "one_band_expected", -1),
        ("bilayer", "bilayer_expected", 1),
        ("dh_trimer", "dh_trimer_expected", -1),
    ],
)
def test_fockit_options_give_the_same_ground_state(
    subtests: pytest.Subtests,
    request: pytest.FixtureRequest,
    option: str,
    model: str,
    model_expected: str,
    swap_eigenvalue: int,
) -> None:
    spin_names, n_orb, Lambda_c, D, h0_loc_mat, h_int = request.getfixturevalue(model)
    expected = request.getfixturevalue(model_expected)
    gf_struct = [(bl, n_orb) for bl in spin_names]
    all_options: dict[str, dict[str, Any]] = {
        "eigh": {"method": "eigh"},
        "Sz": {"diagonal_symmetries": {sz_op(n_orb): 0}},
        "Sz and spin swap": {
            "diagonal_symmetries": {sz_op(n_orb): 0},
            "permutation_symmetries": [(spin_swap(n_orb), swap_eigenvalue)],
        },
    }
    impurity_solver = FockitImpuritySolver(h_int, gf_struct, **all_options[option])
    impurity_solver.set_h_imp(Lambda_c, D, h0_loc_mat)
    impurity_solver.solve()
    do_assert(
        subtests,
        impurity_solver.rho_f,
        impurity_solver.rho_cf,
        impurity_solver.rho_c,
        impurity_solver.gs_energy,
        impurity_solver.overlap(N_op(spin_names, n_orb)),
        impurity_solver.overlap(S2_op(spin_names, n_orb)),
        *expected,
    )


def test_particle_hole_sector_gives_the_same_ground_state() -> None:
    U = 4.0
    gf_struct = [("up", 1), ("dn", 1)]
    h_int = U * nc("up", 0) * nc("dn", 0)
    # Particle-hole symmetric at half-filling
    Lambda_c = {bl: np.array([[0.0]]) for bl, _ in gf_struct}
    D = {bl: np.array([[-0.4]]) for bl, _ in gf_struct}
    h0_loc_mat = {bl: np.array([[-U / 2]]) for bl, _ in gf_struct}
    particle_hole = [c(bl, 0) + cdag(bl, 0) for bl in ["up", "dn"]] + [
        c(bl, 0) - cdag(bl, 0) for bl in ["bath_up", "bath_dn"]
    ]
    plain = FockitImpuritySolver(h_int, gf_struct)
    symmetric = FockitImpuritySolver(
        h_int, gf_struct, onsite_symmetries=[(particle_hole, -1)]
    )
    for impurity_solver in [plain, symmetric]:
        impurity_solver.set_h_imp(Lambda_c, D, h0_loc_mat)
        impurity_solver.solve()
    assert symmetric.gs_energy == pytest.approx(plain.gs_energy, abs=1e-12)
    for bl, _ in gf_struct:
        assert symmetric.rho_f[bl] == pytest.approx(plain.rho_f[bl], abs=1e-12)
        assert symmetric.rho_cf[bl] == pytest.approx(plain.rho_cf[bl], abs=1e-12)
        assert symmetric.rho_c[bl] == pytest.approx(plain.rho_c[bl], abs=1e-12)


def test_fockit_solver_rejects_symmetries_h_imp_does_not_have() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    Lambda_c = {bl: np.array([[-0.5]]) for bl, _ in gf_struct}
    D = {bl: np.array([[-0.3]]) for bl, _ in gf_struct}

    # A magnetic field breaks the spin swap
    field = {"up": np.array([[-0.2]]), "dn": np.array([[0.2]])}
    impurity_solver = FockitImpuritySolver(
        nc("up", 0) * nc("dn", 0),
        gf_struct,
        diagonal_symmetries={sz_op(1): 0},
        permutation_symmetries=[(spin_swap(1), -1)],
    )
    impurity_solver.set_h_imp(Lambda_c, D, field)
    with pytest.raises(ValueError, match="not symmetric"):
        impurity_solver.solve()

    # A spin flip breaks Sz conservation
    spin_flip = cdag("up", 0) * c("dn", 0) + cdag("dn", 0) * c("up", 0)
    impurity_solver = FockitImpuritySolver(
        nc("up", 0) * nc("dn", 0) + 0.3 * spin_flip,
        gf_struct,
        diagonal_symmetries={sz_op(1): 0},
    )
    impurity_solver.set_h_imp(Lambda_c, D)
    with pytest.raises(ValueError, match="does not conserve"):
        impurity_solver.solve()


def test_fockit_solver_warns_about_a_degenerate_ground_state() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = FockitImpuritySolver(
        nc("up", 0) * nc("dn", 0), gf_struct, degenerate="warn"
    )
    Lambda_c = {bl: np.array([[-0.5]]) for bl, _ in gf_struct}

    # Without hybridization the ground state is degenerate
    impurity_solver.set_h_imp(Lambda_c, {bl: np.array([[0.0]]) for bl, _ in gf_struct})
    with pytest.warns(RuntimeWarning, match="degenerate"):
        impurity_solver.solve()

    # With hybridization it is not
    impurity_solver.set_h_imp(Lambda_c, {bl: np.array([[-0.3]]) for bl, _ in gf_struct})
    impurity_solver.solve()
