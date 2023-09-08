#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb.embedding_atom_diag import *

class tests(unittest.TestCase):
 
    def test_hubbard_kanamori_half(self):
        n_orbs = 2
        spatial_dim = 3
        nkx = 6
        beta = 10
        num_cycles = 25
        filling = 'half' #'quarter'
        
        # Note that one can instead do four blocks
        block_names = ['up','dn']
        gf_struct = [ (block, n_orbs) for block in block_names ]
        
        emb_solver = EmbeddingAtomDiag(gf_struct)
        
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
        nk = h0_k['up'].shape[0]

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        pdensity = deepcopy(Lambda)
        ke = deepcopy(Lambda)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)
                        
        coeff = 0.2
        U = 1
        J = coeff * U
        h_loc = h_int_kanamori(spin_names=block_names,
                               n_orb=n_orbs,
                               U=np.array([[0,U-3*J],[U-3*J,0]]),
                               Uprime=np.array([[U,U-2*J],[U-2*J,U]]),
                               J_hund=J,
                               off_diag=True)

        # Expression of mu for half and quarter filling
        if filling == 'half':
            mu = 0.5*U + 0.5*(U-2*J) + 0.5*(U-3*J)
        # This gives the wrong mu for RISB, because it is the DMFT result.
        elif filling == 'quarter':
            mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2 

        # First guess for Lambda will have mu on the diagonal
        for b in block_names:
            np.fill_diagonal(Lambda[b], mu)
        
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
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])

        mu_calculated = 0
        for block in block_names:
            mu_calculated += np.trace(Lambda[block]) / (n_orbs * len(block_names))
        mu_expected = mu
        Z_expected = np.array([[0.77862,0],[0,0.77862]])
        Lambda_expected = np.array([[mu_expected,0],[0,mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-6)
            assert_arrays_are_close(Z_expected, Z[b], 1e-6)

if __name__ == '__main__':
    unittest.main()
