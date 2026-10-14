# ruff: noqa: D100, D103
import runpy
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pytest

# hubbard.py calls plt.show(), which must not open a window
mpl.use("Agg")

examples_dir = Path(__file__).parent.parent / "examples"

# Reference values were obtained by running the examples. The tolerances are
# looser than the tolerance each example solves to.


def run_example(name: str) -> dict[str, Any]:
    return runpy.run_path(str(examples_dir / f"{name}.py"))


def assert_g_constructions_agree(g: dict[str, Any]) -> None:
    for bl, gf in g["G_iw_loc"]:
        assert gf.data == pytest.approx(g["G_iw_loc2"][bl].data, abs=1e-10)


def test_bilayer_hubbard_example() -> None:
    g = run_example("bilayer_hubbard")
    S = g["S"]
    abs = 1e-5
    Z_expected = 0.4528461698 * np.eye(2)
    Lambda_expected = np.array([[2.0000000442, 0.1145696892], [0.1145696892, 2.0]])

    assert S.optimize.success
    assert g["total_number"] == pytest.approx(g["n_target"], abs=abs)
    for bl in ["up", "dn"]:
        assert S.Z[0][bl] == pytest.approx(Z_expected, abs=abs)
        assert S.Lambda[0][bl] == pytest.approx(Lambda_expected, abs=abs)


def test_green_functions_example() -> None:
    pytest.importorskip("triqs")
    g = run_example("green_functions")
    abs = 1e-5

    assert g["S"].optimize.success
    assert g["G_qp_iw_loc"].total_density().real == pytest.approx(
        g["n_target"], abs=abs
    )
    assert_g_constructions_agree(g)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_hubbard_example() -> None:
    g = run_example("hubbard")
    Z = np.array(g["Z"])
    abs = 1e-5
    # Z for U = 0, 0.5, ..., 4.5, then the Mott insulator (Z = 0) up to U = 10
    Z_expected = [
        *[1.0, 0.9913303433, 0.9653170682, 0.9219446542, 0.8611766236],
        *[0.7829278438, 0.6869832278, 0.5727149025, 0.4378282795, 0.2680755960],
        *[0.0] * 11,
    ]

    assert g["S"].optimize.success
    assert g["total_number"] == pytest.approx(
        [g["n_target"]] * len(g["U_arr"]), abs=abs
    )
    assert Z[0] == pytest.approx(1.0)
    assert np.all(np.diff(Z) <= 1e-12)
    assert np.all((Z >= -1e-12) & (Z <= 1 + 1e-12))
    assert pytest.approx(Z_expected, abs=abs) == Z


def test_kagome_hubbard_example() -> None:
    g = run_example("kagome_hubbard")
    S = g["S"]
    impurity_solvers = g["impurity_solvers"]
    abs = 1e-7
    Z_expected = 0.4767757344
    Lambda_expected = 5.0000000047

    assert S.optimize.success
    assert g["total_number"] == pytest.approx(1.0, abs=abs)
    for i in [1, 2]:
        for bl in ["up", "dn"]:
            assert impurity_solvers[i].get_rho_f(bl) == pytest.approx(
                impurity_solvers[0].get_rho_f(bl)
            )
            assert impurity_solvers[i].get_rho_cf(bl) == pytest.approx(
                impurity_solvers[0].get_rho_cf(bl)
            )
            assert impurity_solvers[i].get_rho_c(bl) == pytest.approx(
                impurity_solvers[0].get_rho_c(bl)
            )
    for i in range(3):
        for bl in ["up", "dn"]:
            assert S.Z[i][bl] == pytest.approx(Z_expected, abs=abs)
            assert S.Lambda[i][bl] == pytest.approx(Lambda_expected, abs=abs)


# The decorated honeycomb is solved in the basis of the trimer orbitals and in
# the basis of the sites, which must give the same physics
decorated_honeycomb_Z = {"A": 0.8080352562, "E1": 0.5911964903, "E2": 0.5911964903}
decorated_honeycomb_Lambda = {
    "A": 1.7912505067,
    "E1": 4.0374422190,
    "E2": 4.0374422190,
}


def test_decorated_honeycomb_trimer_basis_example() -> None:
    g = run_example("decorated_honeycomb_trimer_basis")
    S = g["S"]
    abs = 1e-3
    Z_expected = decorated_honeycomb_Z
    Lambda_expected = decorated_honeycomb_Lambda

    assert S.optimize.success
    for i in range(2):
        for spin in ["up", "dn"]:
            for irrep in ["A", "E1", "E2"]:
                bl = f"{spin}_{irrep}"
                assert S.Z[i][bl] == pytest.approx(Z_expected[irrep], abs=abs)
                assert S.Lambda[i][bl] == pytest.approx(Lambda_expected[irrep], abs=abs)


def test_decorated_honeycomb_site_basis_example() -> None:
    g = run_example("decorated_honeycomb_site_basis")
    S = g["S"]
    abs = 1e-3

    assert S.optimize.success
    # In the basis of the sites the eigenvalues are those of the trimer orbitals
    for i in range(2):
        for bl in ["up", "dn"]:
            assert np.linalg.eigvalsh(S.Z[i][bl]) == pytest.approx(
                sorted(decorated_honeycomb_Z.values()), abs=abs
            )
            assert np.linalg.eigvalsh(S.Lambda[i][bl]) == pytest.approx(
                sorted(decorated_honeycomb_Lambda.values()), abs=abs
            )
