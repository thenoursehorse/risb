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


def test_bilayer_hubbard_example() -> None:
    g = run_example("bilayer_hubbard")
    S = g["S"]
    abs = 1e-5
    Z_expected = 0.4528461698 * np.eye(2)
    Lambda_expected = np.array([[2.0000000442, 0.1145696892], [0.1145696892, 2.0]])

    assert g["result"].converged
    assert g["total_number"] == pytest.approx(g["n_target"], abs=abs)
    for bl in ["up", "dn"]:
        assert S.Z[0][bl] == pytest.approx(Z_expected, abs=abs)
        assert S.Lambda[0][bl] == pytest.approx(Lambda_expected, abs=abs)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_green_functions_example() -> None:
    g = run_example("green_functions")

    assert g["result"].converged
    for bl in g["spin_names"]:
        assert g["G_iw_loc"][bl] == pytest.approx(g["G_iw_loc2"][bl], abs=1e-10)
    # The broadened spectral function loses a little weight outside the window
    assert g["spectral_weight"] == pytest.approx(
        np.trace(g["S"].Z[0]["up"]).real, abs=1e-2
    )


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_hubbard_example() -> None:
    g = run_example("hubbard")
    Z = np.array(g["Z"])
    abs = 1e-5
    # Z for U = 0, 0.25, ..., 4.5, then the Mott insulator (Z = 0) up to U = 10
    Z_expected = [
        *[1.0, 0.9978326492, 0.9913303439, 0.9804922917, 0.9653170509],
        *[0.9458023203, 0.9219445792, 0.8937387168, 0.8611766037, 0.8242460707],
        *[0.7829278442, 0.7371909225, 0.6869832177, 0.6322137113, 0.5727147565],
        *[0.5081565302, 0.4378278163, 0.3599486367, 0.2680772341],
        *[0.0] * 22,
    ]

    assert g["result"].converged
    assert g["total_number"] == pytest.approx(
        [g["n_target"]] * len(g["U_arr"]), abs=abs
    )
    assert Z[0] == pytest.approx(1.0)
    assert np.all(np.diff(Z) <= 1e-12)
    assert np.all((Z >= -1e-12) & (Z <= 1 + 1e-12))
    assert pytest.approx(Z_expected, abs=abs) == Z


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_ghost_hubbard_example() -> None:
    g = run_example("ghost_hubbard")
    abs = 1e-5
    # Z and the double occupancy with ghosts (B = 3) for U = 0.25, 0.5, ..., 4.
    # From U = 3 the Mott insulator has Z = 0 and a finite double occupancy
    Z_expected = [
        *[0.9828587257, 0.9335192175, 0.8576008216, 0.7626891499, 0.6565703649],
        *[0.5459098687, 0.4356290090, 0.3288919206, 0.2274567441, 0.1321248421],
        *[0.0427705197],
        *[0.0] * 5,
    ]
    double_occupancy_expected = [
        *[0.2291687023, 0.2083695887, 0.1876338643, 0.1669907041, 0.1464660719],
        *[0.1260815425, 0.1058536780, 0.0857947282, 0.0659146729, 0.0462217009],
        *[0.0266570860, 0.0153136203, 0.0128325000, 0.0109265580, 0.0094258480],
        *[0.0082205005],
    ]

    assert pytest.approx(Z_expected, abs=abs) == np.real(g["Z"])
    assert pytest.approx(double_occupancy_expected, abs=abs) == np.real(
        g["double_occupancy"]
    )


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_thermal_hubbard_example() -> None:
    g = run_example("thermal_hubbard")
    abs = 1e-5
    # The double occupancy, energy, and entropy / ln 2 with ghosts (B = 3) at
    # U = 2 for T = 0.01 to 10, as in Fig. 7 of arXiv:2603.20559
    double_occupancy_expected = [
        *[0.0849524266, 0.0825713937, 0.0753048635, 0.0626243025, 0.0525707223],
        *[0.0522091678, 0.0639272726, 0.0957314110, 0.1418369021, 0.1831492320],
        *[0.2111147038, 0.2278877658, 0.2375207828],
    ]
    energy_expected = [
        *[-1.0891486653, -1.0872363347, -1.0826509174, -1.0758012530, -1.0700102089],
        *[-1.0624332916, -1.0334844309, -0.9506608737, -0.8207845979, -0.6997277650],
        *[-0.6164968626, -0.5663074849, -0.5374323035],
    ]
    entropy_expected = [
        *[0.1890074057, 0.3837777771, 0.6629373631, 0.8962471676, 1.0123272783],
        *[1.0911158204, 1.2587548475, 1.5353717992, 1.7867265049, 1.9211853513],
        *[1.9736620496, 1.9915185186, 1.9973022638],
    ]

    assert pytest.approx(double_occupancy_expected, abs=abs) == g["double_occupancy"]
    assert pytest.approx(energy_expected, abs=abs) == g["energy"]
    assert pytest.approx(entropy_expected, abs=abs) == np.array(g["entropy"]) / np.log(
        2
    )


def test_kagome_hubbard_example() -> None:
    g = run_example("kagome_hubbard")
    S = g["S"]
    impurity_solvers = g["impurity_solvers"]
    abs = 1e-7
    Z_expected = 0.4767757344
    Lambda_expected = 5.0000000047

    assert g["result"].converged
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

    assert g["result"].converged
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

    assert g["result"].converged
    # In the basis of the sites the eigenvalues are those of the trimer orbitals
    for i in range(2):
        for bl in ["up", "dn"]:
            assert np.linalg.eigvalsh(S.Z[i][bl]) == pytest.approx(
                sorted(decorated_honeycomb_Z.values()), abs=abs
            )
            assert np.linalg.eigvalsh(S.Lambda[i][bl]) == pytest.approx(
                sorted(decorated_honeycomb_Lambda.values()), abs=abs
            )
