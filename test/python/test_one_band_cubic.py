#!/usr/bin/env python

import numpy as np
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import n
from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        n_orb = 1
        spatial_dim = 3
        nkx = 10
        beta = 40

        block_names = ['up','dn']
        gf_struct = [ (bl, n_orb) for bl in block_names ]
        
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
        
        U = 4
        h_loc = U * n('up',0) * n('dn',0)
        mu = U / 2
        
        embedding = EmbeddingAtomDiag(h_loc, gf_struct)
        kweight = SmearingKWeight(beta=beta, mu=mu)
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          kweight=kweight,
                          symmetries=[symmetrize_blocks])
        
        S.solve()
            
        mu_calculated = kweight.mu
        mu_expected = mu
        Lambda_expected = np.array([[2.0]])
        Z_expected = np.array([[0.437828801025]])
                
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl in block_names:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)
        
if __name__ == '__main__':
    unittest.main()
