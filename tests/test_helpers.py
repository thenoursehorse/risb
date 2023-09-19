#!/usr/bin/env python

import numpy as np
from pytest import approx
from common import build_cubic_h0_k, build_block_mf_matrices
from triqs.operators import *
from risb import helpers
from risb.kweight import SmearingKWeight

beta = 10
gf_struct=[('up',2),('dn',2)]

# FIXME eig, vec to h5 and load ref to test
def test_get_hqp(subtests):
    h0_k = build_cubic_h0_k(gf_struct=gf_struct)
    R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
    for block,_ in gf_struct:
        eig, vec = helpers.get_h_qp(R[block], Lambda[block], h0_k[block])
    #with subtests.test(msg="eigenvalues"):
    #    for bl,_ in gf_struct:
    #        assert eig[bl] == approx(eig_expected, abs=1e-8)
    #with subtests.test(msg="eigenvectors"):
    #    for bl,_ in gf_struct:
    #        assert vec[bl] == approx(vec_expected, abs=1e-8)
        
# FIXME eig, R, vec, and wks in h5 and load ref to test
def test_get_ke():
    h0_k = build_cubic_h0_k(gf_struct=gf_struct)
    R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
    for bl,_ in gf_struct:
        np.fill_diagonal(Lambda[bl], 0.5)
    eig = dict()
    vec = dict()
    for bl,_ in gf_struct:
        eig[bl], vec[bl] = helpers.get_h_qp(R[bl], Lambda[bl], h0_k[bl])
    kweight = SmearingKWeight(beta=beta, mu=0)
    wks = kweight.update_weights(eig)
    h0_R = dict()
    ke = dict()
    for block,_ in gf_struct:
        h0_R[bl] = helpers.get_h0_R(R[bl], h0_k[bl], vec[bl])
        ke[bl] = helpers.get_ke(h0_R[bl], vec[bl], wks[bl])
    #for bl,_ in gf_struct:
    #    assert ke[bl] == approx(ke_expected, abs=1e-8)
        
# FIXME vec and wks in h5 and load ref to test
def test_get_pdensity():
    h0_k = build_cubic_h0_k(gf_struct=gf_struct)
    R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
    for bl,_ in gf_struct:
        np.fill_diagonal(Lambda[bl], 0.5)
    eig = dict()
    vec = dict()
    for bl,_ in gf_struct:
        eig[bl], vec[bl] = helpers.get_h_qp(R[bl], Lambda[bl], h0_k[bl])
    kweight = SmearingKWeight(beta=beta, mu=0)
    wks = kweight.update_weights(eig)
    pdensity = dict()
    for bl,_ in gf_struct:
        pdensity[bl] = helpers.get_pdensity(vec[bl], wks[bl])
    #for bl,_ in gf_struct:
    #    assert pdensity[bl] == approx(pdensity_expected, abs=1e-8)

def test_get_d():
    pdensity = np.array([[0.19618454, 0.        ],
                         [0.        , 0.19618454]])
    ke = np.array([[-0.13447044,  0.        ],
                   [ 0.        , -0.13447044]])
    D = helpers.get_d(pdensity, ke)
    D_expected = np.array([[-0.33862285,  0.        ],
                           [ 0.        , -0.33862285]])
    assert D == approx(D_expected, abs=1e-8)
        
def test_get_lambda_c(): 
    Lambda = np.array([[0.5, 0. ],
                       [ 0., 0.5]])
    R = np.array([[1., 0. ],
                  [0., 1.]])
    pdensity = np.array([[0.19618454, 0.        ],
                         [0.        , 0.19618454]])
    D = np.array([[-0.33862285,  0.        ],
                  [ 0.        , -0.33862285]])
    Lambda_c = dict()
    Lambda_c = helpers.get_lambda_c(pdensity, R, Lambda, D)
    Lambda_c_expected = np.array([[ 0.01813814, -0.        ],
                                  [-0.        ,  0.01813814]])
    assert Lambda_c == approx(Lambda_c_expected, abs=1e-8)