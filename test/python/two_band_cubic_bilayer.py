#!/usr/bin/env python

import numpy as np
from itertools import product
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import *
from triqs.operators.util.op_struct import set_operator_structure

from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding_atom_diag import EmbeddingAtomDiag

class tests(unittest.TestCase):
 
    def test_hubbard_bilayer_half(self):
        n_orb = 2
        spatial_dim = 3
        nkx = 10
        beta = 40
        num_cycles = 25
        
        U = 4
        V = 0.25
        J = 0
        mu = U / 2.0 # half-filling

        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
             
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
  
        h_loc = Operator()
        for o in range(n_orb):
            h_loc += U * n("up",o) * n("dn",o)
        for s in spin_names:
            h_loc += V * ( c_dag(s,0)*c(s,1) + c_dag(s,1)*c(s,0) )
        for s1,s2 in product(spin_names,spin_names):
            h_loc += 0.5 * J * c_dag(s1,0) * c(s2,0) * c_dag(s2,1) * c(s1,1)
        
        emb_solver = EmbeddingAtomDiag(h_loc, gf_struct)
        kweight_solver = SmearingKWeight(beta=beta, mu=mu)
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          emb_solver=emb_solver,
                          kweight_solver=kweight_solver,
                          symmetries=[symmetrize_blocks])
        
        # First guess for Lambda will have mu on the diagonal
        for bl,_ in S.gf_struct:
            np.fill_diagonal(S.Lambda[bl], mu)

        S.solve()
        
        mu_calculated = 0
        for bl in block_names:
            mu_calculated += np.trace(S.Lambda[bl]) / (n_orb * len(spin_names))
        mu_expected = mu
        Z_expected = np.array([[0.452846432,0],[0,0.452846432]])
        Lambda_expected = np.array([[mu_expected,0.11456981],[0.11456981,mu_expected]])
        
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl in block_names:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
