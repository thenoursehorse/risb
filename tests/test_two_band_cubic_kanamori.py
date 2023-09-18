#!/usr/bin/env python

import numpy as np
from itertools import product
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import Operator, c_dag, c, n
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag
    
filling = 'half'
coeff = 0.2
U = 3
J = coeff * U
Up = U - 2*J
    
# Expression of mu for half and quarter filling
if filling == 'half':
    mu = 0.5*U + 0.5*Up + 0.5*(Up-J)
    n_target = None
# This gives the wrong mu for RISB, because it is the DMFT result.
elif filling == 'quarter':
    #mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2
    mu = None
    n_target = 1
        
mu_expected = mu
Lambda_expected = np.array([[3.0, 0.0],[0.0, 3.0]])
Z_expected = np.array([[0.574940323948, 0.0],[0.0, 0.574940323948]])

def hubb_kanamori(U, Up, J, spin_names=['up','dn']):
    s_up = spin_names[0]
    s_dn = spin_names[1]
    n_orb = 2
    h_loc = Operator()
    for m in range(n_orb):
        h_loc += U * n(s_up,m) * n(s_dn,m)
    for s,ss in product(spin_names, spin_names):
        h_loc += Up * n(s,0) * n(ss,1)
    for s in spin_names:
        h_loc -= J * n(s,0) * n(s,1)
    h_loc += J * c_dag(s_up,0) * c_dag(s_dn,1) * c(s_dn,0) * c(s_up,1)
    h_loc += J * c_dag(s_dn,0) * c_dag(s_up,1) * c(s_up,0) * c(s_dn,1)
    h_loc += J * c_dag(s_up,0) * c_dag(s_dn,0) * c(s_dn,1) * c(s_up,1)
    h_loc += J * c_dag(s_up,1) * c_dag(s_dn,1) * c(s_dn,0) * c(s_up,0)
    return h_loc

def setup_problem():
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
                
    spin_names = ['up','dn']
    gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
         
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
 
    h_loc = h_int_kanamori(spin_names=spin_names,
                           n_orb=n_orb,
                           U=np.array([[0, Up-J], [Up-J, 0]]),
                           Uprime=np.array([[U, Up], [Up, U]]),
                           J_hund=J,
                           off_diag=True)
    
    embedding = EmbeddingAtomDiag(h_loc, gf_struct)
    kweight = SmearingKWeight(beta=beta, mu=mu, n_target=n_target)
    return gf_struct, h0_k, embedding, kweight        

class tests(unittest.TestCase):
 
    def test_diis_symmetrize(self):
        gf_struct, h0_k, embedding, kweight = setup_problem()
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          embedding=embedding,
                          update_weights=kweight.update_weights,
                          symmetries=[symmetrize_blocks])
        if mu is not None:
            for bl,_ in S.gf_struct:
                np.fill_diagonal(S.Lambda[bl], mu)
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
        if mu is not None:
            for bl,_ in S.gf_struct:
                np.fill_diagonal(S.Lambda[bl], mu)
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
