#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from risb.embedding_atom_diag import *

class tests(unittest.TestCase):
 
    def test_hubbard_bilayer_half(self):
        orb_dim = 2
        spatial_dim = 3
        nkx = 6
        beta = 10
        num_cycles = 25;
        
        orb_names = [1, 2]
        spin_names = ["up","dn"]
        block_names = spin_names
        gf_struct = [ ["up", [1, 2]], ["dn", [1,2]] ]
        gf_struct = set_operator_structure(block_names,orb_names,True)
            
        emb_solver = EmbeddingAtomDiag(gf_struct)

        dispersion = build_cubic_dispersion(nkx,orb_dim,spatial_dim)
        nk = dispersion.shape[-1]

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)

        U = 1
        V = 0.25
        J = 0
        mu = U / 2.0 # half-filling
        
        h_loc = Operator()
        for o in orb_names:
            h_loc += U * n("up",o) * n("dn",o)
        for s in spin_names:
            h_loc += V * ( c_dag(s,1)*c(s,2) + c_dag(s,2)*c(s,1) )
        for s1,s2 in product(spin_names,spin_names):
            h_loc += 0.5 * J * c_dag(s1,1) * c(s2,1) * c_dag(s2,2) * c(s1,1)

        #eprint("U =", U, "V =", V, "J =", J)

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
                eig, vec = sc.get_h_qp(R[b], Lambda[b], dispersion, mu)
                disp_R = sc.get_disp_R(R[b], dispersion, vec)
                wks = fermi_fnc(eig, beta) / nk

                pdensity = sc.get_pdensity(vec, wks)
                ke = sc.get_ke(disp_R, vec, wks)

                D[b] = sc.get_d(pdensity, ke)
                Lambda_c[b] = sc.get_lambda_c(pdensity, R[b], Lambda[b], D[b])

            emb_solver.set_h_emb(h_loc, Lambda_c, D, mu)
            emb_solver.solve()

            for b in block_names:
                Nf = emb_solver.get_nf(b)
                Mcf = emb_solver.get_mcf(b)
                Nc = emb_solver.get_nc(b)

                Lambda[b] = sc.get_lambda(R[b], D[b], Lambda_c[b], Nf)
                R[b] = sc.get_r(Mcf, Nf)

                norm += np.linalg.norm(R[b] - R_old[b])
                norm += np.linalg.norm(Lambda[b] - Lambda_old[b])

            if norm < 1e-6:
                break

        #eprint("D =", D)
        #eprint("Lambda_c =", Lambda_c)
        #eprint("Nf =", Nf)
        #eprint("Mcf =", Mcf)
        #eprint("Nc =", Nc)

        Z = dict()
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])
        #e, v = np.linalg.eigh(Z["up"])
        #eprint("cycles =", cycle, "norm =", norm)
        #eprint("Z =", Z)
        #eprint("Z+- =",e)
        #eprint("Lambda =", Lambda)
        #eprint("mu =", mu)

        mu_calculated = 0
        for block in block_names:
            mu_calculated += np.trace(Lambda[block]) / (len(orb_names) * len(spin_names))
        mu_expected = mu
        Z_expected = np.array([[0.853098,0],[0,0.853098]])
        Lambda_expected = np.array([[mu_expected,0.213962],[0.213962,mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-6)
            assert_arrays_are_close(Z_expected, Z[b], 1e-6)

if __name__ == '__main__':
    unittest.main()
