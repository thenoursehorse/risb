#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb.embedding_atom_diag import *

class tests(unittest.TestCase):
 
    def test_hubbard_kanamori_half(self):
        orb_dim = 2
        spatial_dim = 3
        nkx = 6
        beta = 10
        num_cycles = 25;
        filling = 'half' #'quarter'
        
        orb_names = [1, 2]
        spin_names = ["up","dn"]
        block_names = spin_names
        # Note that one can instead do four blocks
        gf_struct = [ ["up", [1, 2]], ["dn", [1,2]] ]
        gf_struct = set_operator_structure(spin_names,orb_names,True)
            
        emb_solver = EmbeddingAtomDiag(gf_struct)

        dispersion = build_cubic_dispersion(nkx,orb_dim,spatial_dim)
        nk = dispersion.shape[0]

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)

        coeff = 0.2
        U = 1
        J = coeff * U
        h_loc = h_int_kanamori(spin_names,orb_names,
                               np.array([[0,U-3*J],[U-3*J,0]]),
                               np.array([[U,U-2*J],[U-2*J,U]]),
                               J,True)

        # Expression of mu for half and quarter filling
        if filling == 'half':
            mu = 0.5*U + 0.5*(U-2*J) + 0.5*(U-3*J)
        # This gives the wrong mu for RISB, because it is the DMFT result.
        elif filling == 'quarter':
            mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2 

        #eprint("U =", U, "J_rat =", coeff)
        #eprint("mu =", mu)
       
        # First guess for Lambda will have mu on the diagonal
        for block in block_names:
            Lambda[block] = np.eye(Lambda[block].shape[0]) * mu
        
        for cycle in range(num_cycles):

            # Symmetrize
            symmetrize(R,block_names)
            symmetrize(Lambda,block_names)

            norm = 0
            R_old = deepcopy(R)
            Lambda_old = deepcopy(Lambda)

            for b in block_names:
                eig, vec = sc.get_h_qp(R[b], Lambda[b], dispersion)
                disp_R = sc.get_disp_R(R[b], dispersion, vec)
                wks = fermi_fnc(eig, beta, mu) / nk

                pdensity = sc.get_pdensity(vec, wks)
                ke = sc.get_ke(disp_R, vec, wks)

                D[b] = sc.get_d(pdensity, ke)
                Lambda_c[b] = sc.get_lambda_c(pdensity, R[b], Lambda[b], D[b])

            emb_solver.set_h_emb(h_loc, Lambda_c, D)
            emb_solver.solve()

            for b in block_names:
                Nf = emb_solver.get_nf(b)
                Mcf = emb_solver.get_mcf(b)

                Lambda[b] = sc.get_lambda(R[b], D[b], Lambda_c[b], Nf)
                R[b] = sc.get_r(Mcf, Nf)

                norm += np.linalg.norm(R[b] - R_old[b])
                norm += np.linalg.norm(Lambda[b] - Lambda_old[b])

            if norm < 1e-6:
                break

        Z = dict()
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])
        #eprint("cycles =", cycle, "norm =", norm)
        #eprint("Z =", Z)
        #eprint("Lambda =", Lambda)

        mu_calculated = 0
        for block in block_names:
            mu_calculated += np.trace(Lambda[block]) / (len(orb_names) * len(block_names))
        mu_expected = mu
        Z_expected = np.array([[0.77862,0],[0,0.77862]])
        Lambda_expected = np.array([[mu_expected,0],[0,mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-6)
            assert_arrays_are_close(Z_expected, Z[b], 1e-6)

if __name__ == '__main__':
    unittest.main()
