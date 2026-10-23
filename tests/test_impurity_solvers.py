# ruff: noqa: D100, D103
from itertools import product
from typing import Any, Literal

import numpy as np
import pytest
from fockit.algebra import OpSum, c, cdag, nc, zero
from fockit.hilbert import FermionSpace, build_basis, to_matrix
from numpy.typing import NDArray
from test_common import (
    SOLVERS,
    build_two_orbital_complex_h0_k,
    make_impurity_solver,
    tolerance,
)

from risb import LatticeSolver
from risb.helpers_fockit import N_op, S2_op, h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver, FockitThermalImpuritySolver
from risb.kweight import SmearingKWeight
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
    abs: float = 1e-12,
) -> None:
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

    # Hubbard U
    h_int = zero()
    for m, mm, mmm in product(range(n_orb), range(n_orb), range(n_orb)):
        h_int += (
            (U / n_orb)
            * cdag("up", m)
            * c("up", mm)
            * cdag("dn", mmm)
            * c("dn", (m + mmm - mm) % n_orb)
        )
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


@pytest.mark.parametrize("solver", SOLVERS)
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
    model: str,
    model_expected: str,
    solver: str,
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
    impurity_solver = make_impurity_solver(solver, h_int, gf_struct)
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
    N = impurity_solver.overlap(N_op(spin_names, n_orb))
    S2 = impurity_solver.overlap(S2_op(spin_names, n_orb))
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
        tolerance(solver),
    )


@pytest.mark.parametrize("solver", SOLVERS[:2])
def test_solver_rejects_non_hermitian_hamiltonian(solver: str) -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = make_impurity_solver(
        solver, cdag("up", 0) * c("dn", 0), gf_struct
    )
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


@pytest.mark.parametrize("solver", SOLVERS)
def test_decoupled_extra_bath_orbitals_leave_the_impurity_unchanged(
    solver: str,
) -> None:
    U = 4.0
    gf_struct = [("up", 1), ("dn", 1)]
    h_int = U * nc("up", 0) * nc("dn", 0)
    h0_loc_mat = {bl: np.array([[-U / 2]]) for bl, _ in gf_struct}
    one_bath = make_impurity_solver(solver, h_int, gf_struct)
    one_bath.set_h_imp(
        {bl: np.array([[0.1]]) for bl, _ in gf_struct},
        {bl: np.array([[-0.4]]) for bl, _ in gf_struct},
        h0_loc_mat,
    )
    one_bath.solve()
    # The bath term is f Lambda_c f^dag, so a level of +10 (-10) fills (empties)
    # the orbital, and the half-filled sector keeps one electron per spin for
    # the impurity and the coupled bath orbital
    three_baths = make_impurity_solver(
        solver, h_int, gf_struct, bath_size={bl: 3 for bl, _ in gf_struct}
    )
    three_baths.set_h_imp(
        {bl: np.diag([0.1, 10.0, -10.0]) for bl, _ in gf_struct},
        {bl: np.array([[-0.4], [0.0], [0.0]]) for bl, _ in gf_struct},
        h0_loc_mat,
    )
    three_baths.solve()
    for bl, _ in gf_struct:
        assert three_baths.rho_c[bl] == pytest.approx(
            one_bath.rho_c[bl], abs=tolerance(solver)
        )
        assert three_baths.rho_cf[bl] == pytest.approx(
            np.array([[one_bath.rho_cf[bl][0, 0], 0, 0]]), abs=tolerance(solver)
        )
        assert three_baths.rho_f[bl] == pytest.approx(
            np.diag([one_bath.rho_f[bl][0, 0], 0, 1]), abs=tolerance(solver)
        )


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
    with pytest.raises(ValueError, match="does not have the permutation"):
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


def test_thermal_density_matrices_average_a_degenerate_ground_state() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    impurity_solver = FockitImpuritySolver(
        nc("up", 0) * nc("dn", 0), gf_struct, beta=1000
    )
    # Without hybridization the ground state is the four states with one
    # electron on the impurity and one in the bath, 0.5 below the others
    impurity_solver.set_h_imp(
        {bl: np.array([[-0.5]]) for bl, _ in gf_struct},
        {bl: np.array([[0.0]]) for bl, _ in gf_struct},
    )
    impurity_solver.solve()
    assert impurity_solver.entropy == pytest.approx(np.log(4), abs=1e-12)
    for bl, _ in gf_struct:
        assert impurity_solver.rho_c[bl] == pytest.approx(np.array([[0.5]]), abs=1e-12)
        assert impurity_solver.rho_f[bl] == pytest.approx(np.array([[0.5]]), abs=1e-12)
        assert impurity_solver.rho_cf[bl] == pytest.approx(np.array([[0.0]]), abs=1e-12)


def test_eigsh_and_eigh_give_the_same_thermal_density_matrices() -> None:
    # One orbital with six bath orbitals, one at zero energy and weakly
    # coupled, has degenerate low-lying levels. Its sector has 3432 states,
    # more than are diagonalized as a dense matrix. Whether eigsh alone misses
    # a state of a degenerate level depends on rounding, and here it does
    gf_struct = [("up", 1), ("dn", 1)]
    U = 2.4
    h_int = U * nc("up", 0) * nc("dn", 0) - U / 2 * (nc("up", 0) + nc("dn", 0))
    rng = np.random.default_rng(34)
    levels = np.sort(rng.normal(size=6))
    levels[3] = 0
    D = 0.4 * rng.normal(size=(6, 1))
    D[3] = 0.01
    solvers = {}
    methods: list[Literal["eigsh", "eigh"]] = ["eigsh", "eigh"]
    for method in methods:
        impurity_solver = FockitImpuritySolver(
            h_int, gf_struct, method=method, beta=200, bath_size={"up": 6, "dn": 6}
        )
        impurity_solver.set_h_imp(
            {bl: np.diag(levels) for bl, _ in gf_struct}, {bl: D for bl, _ in gf_struct}
        )
        impurity_solver.solve()
        solvers[method] = impurity_solver
    assert solvers["eigsh"].entropy == pytest.approx(solvers["eigh"].entropy, abs=1e-10)
    for bl, _ in gf_struct:
        for rho in ["rho_f", "rho_cf", "rho_c"]:
            got = getattr(solvers["eigsh"], rho)[bl]
            expected = getattr(solvers["eigh"], rho)[bl]
            assert got == pytest.approx(expected, abs=1e-10)


def test_complex_two_orbital_model_reaches_the_mott_state_with_a_thermal_impurity() -> (
    None
):
    # In the Mott state the bath orbitals decouple. With the impurity problem
    # at the temperature of the quasiparticles they are partly filled, close
    # to the quasiparticles but not equal to them
    U = 5
    J = 0.2 * U
    Up = U - 2 * J
    off_diag = 1 - np.eye(2)
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * off_diag,
        Uprime=U * np.eye(2) + Up * off_diag,
        J_hund=J,
    )
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    beta = 40
    S = LatticeSolver(
        h0_k=build_two_orbital_complex_h0_k(),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct, beta=beta),
        update_weights=SmearingKWeight(beta=beta, n_target=2).update_weights,
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = (0.5 * U + Up - 0.5 * J) * np.eye(2)
    # In the atomic limit the bath decouples, and the self-consistency
    # equations are not satisfied
    with pytest.warns(RuntimeWarning, match="self-consistency"):
        result = S.solve(tol=1e-8)
    assert result.converged
    for bl, _ in gf_struct:
        assert S.Z[0][bl] == pytest.approx(np.zeros((2, 2)), abs=1e-6)


def thermal_average_over_the_fock_space(
    impurity_solver: FockitThermalImpuritySolver, h: OpSum, beta: float
) -> tuple[dict[str, BlockMatrix], float]:
    # The density matrices and grand potential from all eigenstates of h in the
    # whole Fock space of the impurity problem
    modes = [
        (bl, i)
        for bl, size in impurity_solver.gf_struct + impurity_solver.gf_struct_bath
        for i in range(size)
    ]
    basis = build_basis({FermionSpace(): modes})
    energies, vectors = np.linalg.eigh(to_matrix(h, basis).toarray())
    weights = np.exp(-beta * (energies - energies[0]))
    rho = (vectors * (weights / np.sum(weights))) @ vectors.conj().T

    def average(op: OpSum) -> complex:
        return complex(np.trace(rho @ to_matrix(op, basis).toarray()))

    expected: dict[str, BlockMatrix] = {"rho_f": {}, "rho_cf": {}, "rho_c": {}}
    for (bl, size), (bl_bath, n_bath) in zip(
        impurity_solver.gf_struct, impurity_solver.gf_struct_bath, strict=True
    ):
        expected["rho_f"][bl] = np.array(
            [
                [average(c(bl_bath, b) * cdag(bl_bath, a)) for b in range(n_bath)]
                for a in range(n_bath)
            ]
        )
        expected["rho_cf"][bl] = np.array(
            [
                [average(cdag(bl, alpha) * c(bl_bath, a)) for a in range(n_bath)]
                for alpha in range(size)
            ]
        )
        expected["rho_c"][bl] = np.array(
            [
                [average(cdag(bl, alpha) * c(bl, b)) for b in range(size)]
                for alpha in range(size)
            ]
        )
    omega = energies[0] - np.log(np.sum(weights)) / beta
    return expected, float(omega)


def test_grand_canonical_impurity_is_the_thermal_state_over_every_particle_number() -> (
    None
):
    # Both spins in one block, so the complex local terms can flip the spin,
    # and a chemical potential away from half filling
    beta, mu = 5.0, 0.4
    gf_struct = [("s", 2)]
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Lambda_c = {"s": 0.5 * (A + A.conj().T)}
    D = {"s": 0.5 * (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))}
    h0_loc = {"s": np.array([[0.2, 0.3 - 0.1j], [0.3 + 0.1j, -0.1]])}
    impurity_solver = FockitThermalImpuritySolver(
        2 * nc("s", 0) * nc("s", 1), gf_struct, beta=beta, bath_size={"s": 4}
    )
    # The chemical potential is on the physical and bath orbitals
    impurity_solver.set_h_imp(Lambda_c, D, h0_loc)
    h = impurity_solver.h_imp
    for i in range(2):
        h -= mu * nc("s", i)
    for a in range(4):
        h -= mu * nc("bath_s", a)
    impurity_solver.set_h_imp(Lambda_c, D, h0_loc, mu=mu)
    impurity_solver.solve()

    expected, omega = thermal_average_over_the_fock_space(impurity_solver, h, beta)
    for rho in ["rho_f", "rho_cf", "rho_c"]:
        assert getattr(impurity_solver, rho)["s"] == pytest.approx(
            expected[rho]["s"], abs=1e-10
        )
    assert impurity_solver.grand_potential == pytest.approx(omega, abs=1e-10)
    assert len(impurity_solver.ensemble) > 2


def test_grand_canonical_impurity_keeps_every_value_of_a_diagonal_symmetry() -> None:
    # Sz splits the Hamiltonian into blocks, and the average is over all of
    # them
    beta, mu = 3.0, 0.3
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = {"up": 3, "dn": 3}
    Lambda_c = {bl: np.diag([-0.8, 0.1, 0.9]) for bl, _ in gf_struct}
    D = {bl: np.array([[0.3], [0.2], [0.4]]) for bl, _ in gf_struct}
    h0_loc = {"up": np.array([[-0.2]]), "dn": np.array([[0.1]])}
    Sz = nc("up", 0) - nc("dn", 0)
    for a in range(3):
        Sz += nc("bath_up", a) - nc("bath_dn", a)
    impurity_solver = FockitThermalImpuritySolver(
        2 * nc("up", 0) * nc("dn", 0),
        gf_struct,
        diagonal_symmetries=[Sz],
        beta=beta,
        bath_size=bath_size,
    )
    impurity_solver.set_h_imp(Lambda_c, D, h0_loc, mu=mu)
    impurity_solver.solve()

    expected, omega = thermal_average_over_the_fock_space(
        impurity_solver, impurity_solver.h_imp, beta
    )
    for rho in ["rho_f", "rho_cf", "rho_c"]:
        for bl, _ in gf_struct:
            assert getattr(impurity_solver, rho)[bl] == pytest.approx(
                expected[rho][bl], abs=1e-10
            )
    assert impurity_solver.grand_potential == pytest.approx(omega, abs=1e-10)


@pytest.mark.parametrize("solver", ["fockit", "fockit thermal", *SOLVERS[1:]])
def test_a_second_solve_with_new_terms_matches_a_new_solver(solver: str) -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = {"up": 3, "dn": 3}
    Sz = 0.5 * (nc("up", 0) - nc("dn", 0))
    for a in range(3):
        Sz += 0.5 * (nc("bath_up", a) - nc("bath_dn", a))
    swap = {"up": "dn", "dn": "up", "bath_up": "bath_dn", "bath_dn": "bath_up"}
    spin_swap = {
        (bl, o): (swap[bl], o) for bl in swap for o in range(3) if o < 1 or "bath" in bl
    }

    def new_solver(h_int: OpSum) -> Any:
        if solver == "fockit thermal":
            return FockitThermalImpuritySolver(
                h_int, gf_struct, 3.0, diagonal_symmetries=[Sz], bath_size=bath_size
            )
        if solver == "fockit":
            return FockitImpuritySolver(
                h_int,
                gf_struct,
                diagonal_symmetries={Sz: 0},
                permutation_symmetries=[(spin_swap, 1)],
                bath_size=bath_size,
            )
        return make_impurity_solver(solver, h_int, gf_struct, bath_size=bath_size)

    def terms(seed: int) -> tuple[BlockMatrix, BlockMatrix]:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(3, 3))
        D = 0.4 * rng.normal(size=(3, 1))
        return {bl: 0.5 * (A + A.T) for bl in swap if "bath" not in bl}, {
            bl: D for bl in swap if "bath" not in bl
        }

    h_int = 2 * nc("up", 0) * nc("dn", 0)
    impurity_solver = new_solver(h_int)
    impurity_solver.set_h_imp(*terms(0), mu=0.1)
    impurity_solver.solve()
    impurity_solver.set_h_int(3 * h_int)
    impurity_solver.set_h_imp(*terms(1), mu=0.2)
    impurity_solver.solve()

    fresh = new_solver(3 * h_int)
    fresh.set_h_imp(*terms(1), mu=0.2)
    fresh.solve()
    for rho in ["rho_f", "rho_cf", "rho_c"]:
        for bl, _ in gf_struct:
            assert getattr(impurity_solver, rho)[bl] == pytest.approx(
                getattr(fresh, rho)[bl], abs=tolerance(solver)
            )


@pytest.mark.parametrize("symmetry", ["layer swap", "onsite phases"])
def test_grand_canonical_impurity_keeps_every_eigenvalue_of_a_symmetry(
    symmetry: str,
) -> None:
    # Two orbitals per spin, each with one bath orbital. The layer swap
    # exchanges the orbitals and their bath orbitals. The onsite phases
    # multiply each particle in orbital 1 or its bath orbital by i
    beta, mu = 2.0, 0.2
    spins = ["up", "dn"]
    gf_struct = [(s, 2) for s in spins]
    bath_size = dict.fromkeys(spins, 2)
    rng = np.random.default_rng(5)
    a, b, d, e = rng.normal(size=4)
    if symmetry == "layer swap":
        Lambda_c = np.array([[a, b], [b, a]])
        D = 0.4 * np.array([[d, e], [e, d]])
    else:
        Lambda_c = np.diag([a, b])
        D = 0.4 * np.diag([d, e])
    h_int = zero()
    for o in range(2):
        h_int += 2.0 * nc("up", o) * nc("dn", o)
    h_int += 0.5 * (nc("up", 0) + nc("dn", 0)) * (nc("up", 1) + nc("dn", 1))
    Sz = zero()
    for s, sign in [("up", 0.5), ("dn", -0.5)]:
        for o in range(2):
            Sz += sign * (nc(s, o) + nc("bath_" + s, o))
    kw: dict[str, Any] = {}
    if symmetry == "layer swap":
        swap = {
            (bl, o): (bl, 1 - o)
            for bl in [*spins, "bath_up", "bath_dn"]
            for o in range(2)
        }
        kw["permutation_symmetries"] = [swap]
    else:
        modes = [(bl, 1) for bl in [*spins, "bath_up", "bath_dn"]]
        kw["onsite_symmetries"] = [
            [c(*m) * cdag(*m) + 1j * cdag(*m) * c(*m) for m in modes]
        ]
    impurity_solver = FockitThermalImpuritySolver(
        h_int, gf_struct, beta, diagonal_symmetries=[Sz], bath_size=bath_size, **kw
    )
    impurity_solver.set_h_imp(
        dict.fromkeys(spins, Lambda_c), dict.fromkeys(spins, D), mu=mu
    )
    impurity_solver.solve()

    expected, omega = thermal_average_over_the_fock_space(
        impurity_solver, impurity_solver.h_imp, beta
    )
    for rho in ["rho_f", "rho_cf", "rho_c"]:
        for bl in spins:
            assert getattr(impurity_solver, rho)[bl] == pytest.approx(
                expected[rho][bl], abs=1e-10
            )
    assert impurity_solver.grand_potential == pytest.approx(omega, abs=1e-10)


def test_grand_canonical_impurity_rejects_symmetries_it_cannot_split_by() -> None:
    gf_struct = [("up", 1), ("dn", 1)]
    Sz = 0.5 * (nc("up", 0) - nc("dn", 0) + nc("bath_up", 0) - nc("bath_dn", 0))
    swap = {"up": "dn", "dn": "up", "bath_up": "bath_dn", "bath_dn": "bath_up"}
    spin_swap = {(bl, 0): (other, 0) for bl, other in swap.items()}
    # The spin swap changes Sz
    with pytest.raises(ValueError, match="Use one or the other"):
        FockitThermalImpuritySolver(
            zero(),
            gf_struct,
            1.0,
            diagonal_symmetries=[Sz],
            permutation_symmetries=[spin_swap],
        )
    # A cyclic shift and a reflection of three sites do not commute
    shift = {("s", i): ("s", (i + 1) % 3) for i in range(3)}
    shift |= {("bath_s", i): ("bath_s", (i + 1) % 3) for i in range(3)}
    reflect = {("s", i): ("s", (3 - i) % 3) for i in range(3)}
    reflect |= {("bath_s", i): ("bath_s", (3 - i) % 3) for i in range(3)}
    with pytest.raises(ValueError, match="commute"):
        FockitThermalImpuritySolver(
            zero(), [("s", 3)], 1.0, permutation_symmetries=[shift, reflect]
        )
