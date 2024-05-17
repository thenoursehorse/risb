# ruff: noqa: T201, D100, D103

from pathlib import Path

import h5py
import numpy as np
import pytest

from risb import helpers

# FIXME finish helpers: lambda, r, f1, f2

abs = 1e-12
helpers_filename = Path(__file__).parent / Path("data_helpers.h5")


def test_get_h_qp(subtests):
    with h5py.File(helpers_filename, "r") as f:
        h0_k = f["h0_k"][:]
        eig_expected = f["eig"][:]
        vec_expected = f["vec"][:]
    R = np.zeros(shape=(2, 2))
    np.fill_diagonal(R, 1)
    Lambda = np.zeros(shape=(2, 2))
    np.fill_diagonal(Lambda, 0.5)
    h_qp = helpers.get_h_qp(R, Lambda, h0_k)
    eig, vec = np.linalg.eigh(h_qp)
    with subtests.test(msg="eigenvalues"):
        assert eig == pytest.approx(eig_expected, abs=abs)
    with subtests.test(msg="eigenvectors"):
        assert vec == pytest.approx(vec_expected, abs=abs)


def test_get_h0_kin_k_R():
    R = np.zeros(shape=(2, 2))
    np.fill_diagonal(R, 1)
    with h5py.File(helpers_filename, "r") as f:
        h0_k = f["h0_k"][:]
        vec = f["vec"][:]
        h0_k_R_expected = f["h0_R"][:]
    h0_k_R = helpers.get_h0_kin_k_R(R, h0_k, vec)
    assert h0_k_R == pytest.approx(h0_k_R_expected, abs=abs)


def test_get_ke():
    with h5py.File(helpers_filename, "r") as f:
        h0_k_R = f["h0_R"][:]
        vec = f["vec"][:]
        wks = f["wks"][:]
    ke = helpers.get_ke(h0_k_R, vec, wks)
    ke_expected = np.array([[-0.36035732126514364, 0], [0, -0.36035732126514364]])
    assert ke == pytest.approx(ke_expected, abs=abs)


def test_get_rho_qp():
    with h5py.File(helpers_filename, "r") as f:
        vec = f["vec"][:]
        wks = f["wks"][:]
    rho_qp = helpers.get_rho_qp(vec, wks)
    rho_qp_expected = np.array([[0.30667105643085796, 0], [0, 0.30667105643085796]])
    assert rho_qp == pytest.approx(rho_qp_expected, abs=abs)


def test_get_d():
    rho_qp = np.array([[0.19618454, 0.0], [0.0, 0.19618454]])
    ke = np.array([[-0.13447044, 0.0], [0.0, -0.13447044]])
    D = helpers.get_d(rho_qp, ke)
    D_expected = np.array([[-0.33862284815908383, 0.0], [0.0, -0.33862284815908383]])
    assert pytest.approx(D_expected, abs=abs) == D


def test_get_lambda_c():
    Lambda = np.array([[0.5, 0.0], [0.0, 0.5]])
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_qp = np.array([[0.19618454, 0.0], [0.0, 0.19618454]])
    D = np.array([[-0.33862285, 0.0], [0.0, -0.33862285]])
    Lambda_c = helpers.get_lambda_c(rho_qp, R, Lambda, D)
    Lambda_c_expected = np.array(
        [[0.018138135818154377, 0.0], [0.0, 0.018138135818154377]]
    )
    assert Lambda_c == pytest.approx(Lambda_c_expected, abs=abs)


def test_get_lambda():
    # R = np.zeros(shape=(2,2))
    # np.fill_diagonal(R, 1.0)
    # Lambda = helpers.get_lambda(R, D, Lambda_c, rho_f)
    pass


def test_get_R():
    # R = helpers.get_r(rho_cf, rho_f)
    pass


def test_get_f1():
    # R = np.zeros(shape=(2,2))
    # np.fill_diagonal(R, 1.0)
    # f1 = helpers.get_f1(rho_cf, rho_qp, R)
    pass


def test_get_f2():
    # f2 = helpers.get_f2(rho_f, rho_qp)
    pass
