#!/usr/bin/env python

import numpy as np
import unittest
from common import build_cubic_h0_k, build_block_mf_matrices
from triqs.operators import *
from risb import helpers
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag

beta = 10
gf_struct=[('up',2),('dn',2)]

class tests(unittest.TestCase):

    # FIXME eig, vec to h5 and load ref to test
    def test_get_hqp(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            eig, vec = helpers.get_h_qp(R[block], Lambda[block], h0_k[block])
        
    # FIXME eig, R, vec, and wks in h5 and load ref to test
    def test_get_ke(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            np.fill_diagonal(Lambda[block], 0.5)
        eig = dict()
        vec = dict()
        for block,_ in gf_struct:
            eig[block], vec[block] = helpers.get_h_qp(R[block], Lambda[block], h0_k[block])
        kweight = SmearingKWeight(beta=beta, mu=0)
        wks = kweight.update_weights(eig)
        h0_R = dict()
        ke = dict()
        for block,_ in gf_struct:
            h0_R[block] = helpers.get_h0_R(R[block], h0_k[block], vec[block])
            ke[block] = helpers.get_ke(h0_R[block], vec[block], wks[block])
        
    # FIXME vec and wks in h5 and load ref to test
    def test_get_pdensity(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            np.fill_diagonal(Lambda[block], 0.5)
        eig = dict()
        vec = dict()
        for block,_ in gf_struct:
            eig[block], vec[block] = helpers.get_h_qp(R[block], Lambda[block], h0_k[block])
        kweight = SmearingKWeight(beta=beta, mu=0)
        wks = kweight.update_weights(eig)
        pdensity = dict()
        for block,_ in gf_struct:
            pdensity[block] = helpers.get_pdensity(vec[block], wks[block])

    def test_get_d(self):
        pdensity = {'up': np.array([[0.19618454, 0.        ],
                                    [0.        , 0.19618454]]), 
                    'dn': np.array([[0.19618454, 0.        ],
                                    [0.        , 0.19618454]])}
        ke = {'up': np.array([[-0.13447044,  0.        ],
                              [ 0.        , -0.13447044]]),
              'dn': np.array([[-0.13447044,  0.        ],
                              [ 0.        , -0.13447044]])}
        D = dict()
        for block,_ in gf_struct:
            D[block] = helpers.get_d(pdensity[block], ke[block])
        D_expected = {'up': np.array([[-0.33862285,  0.        ],
                                      [ 0.        , -0.33862285]]), 
                      'dn': np.array([[-0.33862285,  0.       ],
                                      [ 0.        , -0.33862285 ]])}
        for block,_ in gf_struct:
            np.testing.assert_allclose(D_expected[block], D[block], rtol=0, atol=1e-8)
        
    def test_get_lambda_c(self): 
        Lambda = {'up': np.array([[0.5, 0. ],
                                  [ 0., 0.5]]), 
                  'dn': np.array([[0.5, 0.  ],
                                  [ 0., 0.5]])}
        R = {'up': np.array([[1., 0. ],
                             [0., 1.]]), 
             'dn': np.array([[1., 0.  ],
                             [0., 1.]])}
        pdensity = {'up': np.array([[0.19618454, 0.        ],
                                    [0.        , 0.19618454]]), 
                    'dn': np.array([[0.19618454, 0.        ],
                                    [0.        , 0.19618454]])}
        D = {'up': np.array([[-0.33862285,  0.        ],
                             [ 0.        , -0.33862285]]), 
             'dn': np.array([[-0.33862285,  0.       ],
                             [ 0.        , -0.33862285 ]])}
        Lambda_c = dict()
        for block,_ in gf_struct:
            Lambda_c[block] = helpers.get_lambda_c(pdensity[block], R[block], Lambda[block], D[block])
        Lambda_c_expected = {'up': np.array([[ 0.01813814, -0.        ],
                                             [-0.        ,  0.01813814]]),
                             'dn': np.array([[ 0.01813814, -0.        ],
                                             [-0.        ,  0.01813814]])}
        for block,_ in gf_struct:
            np.testing.assert_allclose(Lambda_c_expected[block], Lambda_c[block], rtol=0, atol=1e-8)

    # FIXME nf, mcf, and nc in h5 and load ref to test
    def test_solve_emb(self):
        Lambda_c = {'up': np.array([[ 0.01813814, -0.        ],
                                    [-0.        ,  0.01813814]]),
                    'dn': np.array([[ 0.01813814, -0.        ],
                                    [-0.        ,  0.01813814]])}
        D = {'up': np.array([[-0.33862285,  0.        ],
                             [ 0.        , -0.33862285]]), 
             'dn': np.array([[-0.33862285,  0.       ],
                             [ 0.        , -0.33862285 ]])}
        h_loc = n('up',0) * n('dn',0)
        embedding = EmbeddingAtomDiag(h_loc, gf_struct) 
        embedding.set_h_emb(Lambda_c, D)
        embedding.solve()
        Nf = dict()
        Nc = dict()
        Mcf = dict()
        for bl, bl_size in gf_struct:
            Nf[bl] = embedding.get_nf(bl)
            Nc[bl] = embedding.get_nc(bl)
            Mcf[bl] = embedding.get_mcf(bl)

if __name__ == '__main__':
    unittest.main()
