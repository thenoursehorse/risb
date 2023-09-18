#!/usr/bin/env python

import numpy as np
from itertools import product
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import Operator, c_dag, c, n
from triqs.operators.util.op_struct import set_operator_structure

from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag
    
U = 4
V = 0.25
J = 0
mu = U / 2.0 # half-filling
        
mu_expected = 2.0
Lambda_expected = np.array([[2.0, 0.114569681915],[0.114569681915, 2.0]])
Z_expected = np.array([[0.452846149446, 0],[0, 0.452846149446]])

def setup_problem():
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ['up','dn']
    gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_loc = Operator()
    for o in range(n_orb):
        h_loc += U * n("up",o) * n("dn",o)
    for s in spin_names:
        h_loc += V * ( c_dag(s,0)*c(s,1) + c_dag(s,1)*c(s,0) )
    for s1,s2 in product(spin_names,spin_names):
        h_loc += 0.5 * J * c_dag(s1,0) * c(s2,0) * c_dag(s2,1) * c(s1,1)
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
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=5e-5)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=5e-5)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=5e-5)
    
    def test_diis_nosymmetrize(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights)
        S.solve()
        
        mu_calculated = kweight.mu
        # Lower tolerance beause no symmetrize is more error prone
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-4)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-4)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-4)
    
    def test_scipy_root(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        from scipy.optimize import root as root_fun
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights,
                          symmetries=[symmetrize_blocks],
                          root=root_fun,
                          return_x_new=False)
        # First guess for Lambda will have mu on the diagonal
        if mu is not None:
            for bl,_ in S.gf_struct:
                np.fill_diagonal(S.Lambda[bl], mu)
        S.solve(method='broyden1')

        mu_calculated = kweight.mu
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=5e-5)
        for bl, bl_size in gf_struct:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=5e-5)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=5e-5)
    
if __name__ == '__main__':
    unittest.main()
