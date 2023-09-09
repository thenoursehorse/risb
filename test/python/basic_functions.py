#!/usr/bin/env python

import numpy as np
import unittest
from common import build_cubic_h0_k, build_block_mf_matrices
from triqs.operators import *
from risb import sc
from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding_atom_diag import EmbeddingAtomDiag

beta = 10
gf_struct=[('up',2),('dn',2)]

class tests(unittest.TestCase):

    def test_get_hqp(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            eig, vec = sc.get_h_qp(R[block], Lambda[block], h0_k[block])
        
    def test_get_ke(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            np.fill_diagonal(Lambda[block], 0.5)
        eig = dict()
        vec = dict()
        for block,_ in gf_struct:
            eig[block], vec[block] = sc.get_h_qp(R[block], Lambda[block], h0_k[block])
        sumk = SmearingKWeight(beta=beta, mu=0)
        wks = sumk.update_weights(eig)
        h0_R = dict()
        ke = dict()
        for block,_ in gf_struct:
            h0_R[block] = sc.get_h0_R(R[block], h0_k[block], vec[block])
            ke[block] = sc.get_ke(h0_R[block], vec[block], wks[block])
        
    def test_get_pdensity(self):
        h0_k = build_cubic_h0_k(gf_struct=gf_struct)
        R, Lambda = build_block_mf_matrices(gf_struct=gf_struct)
        for block,_ in gf_struct:
            np.fill_diagonal(Lambda[block], 0.5)
        eig = dict()
        vec = dict()
        for block,_ in gf_struct:
            eig[block], vec[block] = sc.get_h_qp(R[block], Lambda[block], h0_k[block])
        sumk = SmearingKWeight(beta=beta, mu=0)
        wks = sumk.update_weights(eig)
        pdensity = dict()
        for block,_ in gf_struct:
            pdensity[block] = sc.get_pdensity(vec[block], wks[block])

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
            D[block] = sc.get_d(pdensity[block], ke[block])
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
            Lambda_c[block] = sc.get_lambda_c(pdensity[block], R[block], Lambda[block], D[block])
        Lambda_c_expected = {'up': np.array([[ 0.01813814, -0.        ],
                                             [-0.        ,  0.01813814]]),
                             'dn': np.array([[ 0.01813814, -0.        ],
                                             [-0.        ,  0.01813814]])}
        for block,_ in gf_struct:
            np.testing.assert_allclose(Lambda_c_expected[block], Lambda_c[block], rtol=0, atol=1e-8)

    def test_solve_emb(self):
        #fops_local = [(s,o) for s,o in product(('up','dn'),list(range(1,2)))]
        Lambda_c = {'up': np.array([[ 0.01813814, -0.        ],
                                    [-0.        ,  0.01813814]]),
                    'dn': np.array([[ 0.01813814, -0.        ],
                                    [-0.        ,  0.01813814]])}
        D = {'up': np.array([[-0.33862285,  0.        ],
                             [ 0.        , -0.33862285]]), 
             'dn': np.array([[-0.33862285,  0.       ],
                             [ 0.        , -0.33862285 ]])}
        h_loc = n('up',0) * n('dn',0)
        emb_solver = EmbeddingAtomDiag(h_loc, gf_struct) 
        emb_solver.set_h_emb(Lambda_c, D)
        emb_solver.solve()
        Nf = dict()
        Mcf = dict()
        for block,_ in gf_struct:
            Nf[block] = emb_solver.get_nf(block)
            Mcf[block] = emb_solver.get_mcf(block)

        #print("TEST", Nf)
        #print("TEST2", Mcf)
        #print("D", D)
        #print("Lambda_c", Lambda_c)
        #print(null)
        
        #for block,_ in gf_struct:
        #    assert_arrays_are_close(Nf_expected[block], Nf[block], 1e-8)
        #    assert_arrays_are_close(Mcf_expected[block], Mcf[block], 1e-8)

if __name__ == '__main__':
    unittest.main()
