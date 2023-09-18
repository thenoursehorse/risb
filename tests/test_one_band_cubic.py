#!/usr/bin/env python

import numpy as np
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import n
from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag
        
mu_expected = 2
Lambda_expected = np.array([[2.0]])
Z_expected = np.array([[0.437828801025]])

def setup_problem():
    n_orb = 1
    spatial_dim = 3
    nkx = 10
    beta = 40
    gf_struct = [ (bl, n_orb) for bl in ['up', 'dn'] ]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    U = 4
    h_loc = U * n('up',0) * n('dn',0)
    mu = U / 2
    embedding = EmbeddingAtomDiag(h_loc, gf_struct)
    kweight = SmearingKWeight(beta=beta, mu=mu)
    return gf_struct, h0_k, embedding, kweight        

class tests(unittest.TestCase):
 
    def test_diis_symmetrize(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights,
                          symmetries=[symmetrize_blocks])
        S.solve()
            
        mu_calculated = kweight.mu                
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)

    def test_diis_nosymmetrize(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights)
        S.solve()
 
        mu_calculated = kweight.mu
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)
    
    def test_scipy_root(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        from scipy.optimize import root as root_fun
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights,
                          symmetries=[symmetrize_blocks],
                          root=root_fun,
                          return_x_new = False)
        S.solve()
 
        mu_calculated = kweight.mu
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)
        
if __name__ == '__main__':
    unittest.main()
