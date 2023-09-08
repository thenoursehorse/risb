#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb.embedding_atom_diag import *

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
        num_cycles = 25
        filling = 'half'
        
        # Note that one can instead do four blocks
        block_names = ['up','dn']
        spin_names = block_names
        gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)
        
        emb_solver = EmbeddingAtomDiag(gf_struct)
        
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)

        [R, Lambda] = build_block_mf_matrices(gf_struct)
                        
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
            fixed = 'mu'
            N_target = 2
         # This gives the wrong mu for RISB, because it is the DMFT result.
        elif filling == 'quarter':
            mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2
            fixed = 'filling'
            N_target = 1


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
                                          mu=mu,
                                          N_target=N_target,
                                          fixed=fixed)
            if norm < 1e-8:
                break

        print("cycles:", cycle)

        Z = dict()
        for bl in block_names:
            Z[bl] = np.dot(R[bl], R[bl])

        mu_calculated = 0
        for bl in block_names:
            mu_calculated += np.trace(Lambda[bl]) / (n_orb * len(block_names))
        mu_expected = mu
        #Z_expected = np.array([[0.77862,0],[0,0.77862]])
        Z_expected = np.array([[0.57494033,0],[0,0.57494033]])
        Lambda_expected = np.array([[mu_expected,0],[0,mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for bl in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[bl], 1e-6)
            assert_arrays_are_close(Z_expected, Z[bl], 1e-6)

if __name__ == '__main__':
    unittest.main()
