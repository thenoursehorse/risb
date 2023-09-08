#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb.embedding_atom_diag import *

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
            
        emb_solver = EmbeddingAtomDiag(gf_struct)
        
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        
        h_loc = Operator()
        for o in range(n_orb):
            h_loc += U * n("up",o) * n("dn",o)
        for s in spin_names:
            h_loc += V * ( c_dag(s,0)*c(s,1) + c_dag(s,1)*c(s,0) )
        for s1,s2 in product(spin_names,spin_names):
            h_loc += 0.5 * J * c_dag(s1,0) * c(s2,0) * c_dag(s2,1) * c(s1,1)

        # First guess for Lambda will have mu on the diagonal
        for bl in block_names:
            np.fill_diagonal(Lambda[bl], mu)
        
        for cycle in range(num_cycles):
            Lambda, R, norm = inner_cycle(emb_solver=emb_solver,
                                          h0_k=h0_k,
                                          h_loc=h_loc,
                                          R=R, 
                                          Lambda=Lambda, 
                                          block_names=block_names,
                                          beta=beta,
                                          mu=mu)
            if norm < 1e-6:
                break

        Z = dict()
        for bl in block_names:
            Z[bl] = np.dot(R[bl], R[bl])

        mu_calculated = 0
        for bl in block_names:
            mu_calculated += np.trace(Lambda[bl]) / (n_orb * len(spin_names))
        mu_expected = mu
        #Z_expected = np.array([[0.853098,0],[0,0.853098]])
        #Lambda_expected = np.array([[mu_expected,0.213962],[0.213962,mu_expected]])
        Z_expected = np.array([[0.452846432,0],[0,0.452846432]])
        Lambda_expected = np.array([[mu_expected,0.11456981],[0.11456981,mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for bl in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[bl], 1e-6)
            assert_arrays_are_close(Z_expected, Z[bl], 1e-6)

if __name__ == '__main__':
    unittest.main()
