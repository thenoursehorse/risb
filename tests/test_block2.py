# ruff: noqa: D100, D103
from itertools import product
from typing import Any

import numpy as np
import pytest
from fockit.algebra import OpSum, c, cdag, hc, nc, zero
from numpy.typing import NDArray

from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver

pytest.importorskip("pyblock2")
from risb.impurity_solvers import Block2ImpuritySolver

type Problem = tuple[
    OpSum,
    list[tuple[str, int]],
    dict[str, int],
    dict[str, NDArray[np.floating[Any]]],
    dict[str, NDArray[np.floating[Any]]],
    dict[str, NDArray[np.floating[Any]]],
]


def kanamori(n_orb: int, U: float, J: float) -> OpSum:
    Up, off = U - 2 * J, 1 - np.eye(n_orb)
    return h_int_kanamori(
        ["up", "dn"],
        n_orb,
        U=(Up - J) * off,
        Uprime=U * np.eye(n_orb) + Up * off,
        J_hund=J,
    )


def ghost_problem(n_orb: int, B: int) -> Problem:
    # Each orbital couples to its own B ghosts, as at a RISB solution
    lam = np.diag(np.linspace(-0.8, 0.8, B)) if B > 1 else np.zeros((1, 1))
    d = np.random.default_rng(1).uniform(0.3, 1.0, size=(B, 1))
    spins = ("up", "dn")
    return (
        kanamori(n_orb, 3.0, 0.6),
        [(bl, n_orb) for bl in spins],
        dict.fromkeys(spins, B * n_orb),
        dict.fromkeys(spins, np.kron(lam, np.eye(n_orb))),
        dict.fromkeys(spins, np.kron(d, np.eye(n_orb))),
        dict.fromkeys(spins, np.diag(np.linspace(-0.1, 0.1, n_orb))),
    )


def solved(solver: Any, problem: Problem) -> Any:
    _, _, _, Lambda_c, D, h0_loc = problem
    solver.set_h_imp(Lambda_c, D, h0_loc, mu=1.5)
    solver.solve()
    return solver


def assert_same_state(solver: Any, expected: FockitImpuritySolver) -> None:
    assert solver.gs_energy == pytest.approx(expected.gs_energy, abs=1e-8)
    for bl, _ in expected.gf_struct:
        assert solver.rho_f[bl] == pytest.approx(expected.rho_f[bl], abs=1e-6)
        assert solver.rho_cf[bl] == pytest.approx(expected.rho_cf[bl], abs=1e-6)
        assert solver.rho_c[bl] == pytest.approx(expected.rho_c[bl], abs=1e-6)


@pytest.mark.parametrize(
    "symmetries", [{}, {"sz": None}, {"sz": 0}, {"s": None}, {"s": 0}]
)
def test_ground_state_of_a_ghost_impurity_is_the_exact_one(
    symmetries: dict[str, Any],
) -> None:
    problem = ghost_problem(2, 3)
    h_int, gf_struct, bath_size = problem[:3]
    expected = solved(
        FockitImpuritySolver(h_int, gf_struct, bath_size=bath_size), problem
    )
    solver = Block2ImpuritySolver(
        h_int, gf_struct, bond_dim=200, symmetries=symmetries, bath_size=bath_size
    )
    assert_same_state(solved(solver, problem), expected)


def test_state_of_a_spin_multiplet_is_the_lowest_with_its_spin_projection() -> None:
    problem = ghost_problem(2, 1)
    h_int, gf_struct, bath_size = problem[:3]
    Sz = zero()
    for bl, i in product(("up", "bath_up"), range(2)):
        Sz += 0.5 * (nc(bl, i) - nc(bl.replace("up", "dn"), i))
    expected = solved(
        FockitImpuritySolver(
            h_int, gf_struct, diagonal_symmetries={Sz: 1}, bath_size=bath_size
        ),
        problem,
    )
    solver = solved(
        Block2ImpuritySolver(
            h_int,
            gf_struct,
            bond_dim=200,
            symmetries={"s": None, "sz": 1},
            bath_size=bath_size,
        ),
        problem,
    )
    assert solver.sector == {"s": 1, "sz": 1}
    assert_same_state(solver, expected)


def test_spin_symmetry_of_a_spin_polarized_impurity_is_rejected() -> None:
    h_int, gf_struct, bath_size, Lambda_c, D, h0_loc = ghost_problem(2, 1)
    solver = Block2ImpuritySolver(
        h_int, gf_struct, symmetries={"s": 0}, bath_size=bath_size
    )
    solver.set_h_imp(
        Lambda_c, D, {"up": h0_loc["up"] + 0.1, "dn": h0_loc["dn"]}, mu=1.5
    )
    with pytest.raises(ValueError, match="spin-up and spin-down"):
        solver.solve()


@pytest.mark.parametrize("sz", [None, 1])
def test_total_spin_of_a_triplet_is_that_of_the_multiplet_or_of_its_state(
    sz: int | None,
) -> None:
    problem = ghost_problem(2, 1)
    h_int, gf_struct, bath_size = problem[:3]
    solver = solved(
        Block2ImpuritySolver(
            h_int,
            gf_struct,
            bond_dim=200,
            symmetries={"s": 1, "sz": sz},
            bath_size=bath_size,
        ),
        problem,
    )
    S_plus, Sz = zero(), zero()
    for bl, i in product(("", "bath_"), range(2)):
        S_plus += cdag(bl + "up", i) * c(bl + "dn", i)
        Sz += 0.5 * (nc(bl + "up", i) - nc(bl + "dn", i))
    S2 = S_plus * hc(S_plus) + Sz * Sz - Sz
    assert solver.overlap(S2) == pytest.approx(2, abs=1e-8)
    assert solver.overlap(Sz) == pytest.approx(sz or 0, abs=1e-8)
