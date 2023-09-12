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

class tests(unittest.TestCase):
 
    def test_hubbard_kanamori_half(self):
        n_orb = 2
        spatial_dim = 3
        nkx = 10
        beta = 40
        filling = 'half'
                
        # Note that one can instead do four blocks
        block_names = ['up','dn']
        spin_names = block_names
        gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
         
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
 
        coeff = 0.2
        U = 3
        J = coeff * U
        Up = U - 2*J
        h_loc = h_int_kanamori(spin_names=block_names,
                               n_orb=n_orb,
                               U=np.array([[0, Up-J], [Up-J, 0]]),
                               Uprime=np.array([[U, Up], [Up, U]]),
                               J_hund=J,
                               off_diag=True)
                        
        # Expression of mu for half and quarter filling
        if filling == 'half':
            mu = 0.5*U + 0.5*Up + 0.5*(Up-J)
            n_target = None
        # This gives the wrong mu for RISB, because it is the DMFT result.
        elif filling == 'quarter':
            #mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2
            mu = None
            n_target = 1
        
        emb_solver = EmbeddingAtomDiag(h_loc, gf_struct)
        kweight_solver = SmearingKWeight(beta=beta, mu=mu, n_target=n_target)
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          emb_solver=emb_solver,
                          kweight_solver=kweight_solver,
                          symmetries=[symmetrize_blocks])
        
        # First guess for Lambda will have mu on the diagonal
        if mu is not None:
            for bl,_ in S.gf_struct:
                np.fill_diagonal(S.Lambda[bl], mu)
        
        S.solve()
        
        mu_calculated = kweight_solver.mu
        mu_expected = mu
        Lambda_expected = np.array([[3.0, 0.0],[0.0, 3.0]])
        Z_expected = np.array([[0.574940323948, 0.0],[0.0, 0.574940323948]])
                
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl in block_names:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
