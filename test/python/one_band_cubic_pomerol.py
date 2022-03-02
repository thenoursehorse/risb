#!/usr/bin/env python

from common import *
from risb.embedding_pomerol import *

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        beta = 10
        num_cycles = 25;

        spin_names = ["up","dn"]
        gf_struct = [ ["up", [0]], ["dn", [0]] ] # pomerol requires starting indices from 0
            
        emb_solver = EmbeddingPomerol(gf_struct)

        dispersion = build_cubic_dispersion()
        nk = dispersion.shape[4]

        U = 1.5
        h_loc = U * n("up",0) * n("dn",0)
        mu = U / 2 # half-filling
        
        [R, Lambda] = build_block_mf_matrices(gf_struct)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)

        # First guess for Lambda will have mu on the diagonal
        for block in spin_names:
            Lambda[block] = np.eye(Lambda[block].shape[0]) * mu
        
        for cycle in range(num_cycles):

            norm = 0
            R_old = deepcopy(R)
            Lambda_old = deepcopy(Lambda)

            for b in spin_names:
                eig, vec = sc.get_h_qp(R[b], Lambda[b], dispersion[0,0,...], mu)
                disp_R = sc.get_disp_R(R[b], dispersion[0,0,...])
                wks = fermi_fnc(eig, beta) / nk
                
                pdensity = sc.get_pdensity(vec, wks)
                ke = sc.get_ke(vec, wks, pdensity, disp_R)

                D[b] = sc.get_d(pdensity, ke)
                Lambda_c[b] = sc.get_lambda_c(pdensity, R[b], Lambda[b], D[b])

            emb_solver.set_h_emb(h_loc, Lambda_c, D, mu)
            emb_solver.solve()
            
            for b in spin_names:
                Nf = emb_solver.get_nf(b)
                Mcf = emb_solver.get_mcf(b)
            
                Lambda[b] = sc.get_lambda(R[b], D[b], Lambda_c[b], Nf)
                R[b] = sc.get_r(Mcf, Nf)
            
                norm += np.linalg.norm(R[b] - R_old[b])
                norm += np.linalg.norm(Lambda[b] - Lambda_old[b])

            if norm < 1e-6:
                break
        
        eprint("cycles =", cycle, "norm =", norm)
        eprint("R =")
        eprint(R)
        eprint("Lambda =")
        eprint(Lambda)
        
        mu_calculated = 0
        for block in spin_names:
            mu_calculated += np.sum(Lambda[block]) / 2.0
        mu_expected = U / 2.
        R_expected = np.array([[0.861617]])
        Lambda_expected = np.array([[mu_expected]])
        
        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in spin_names:
            assert_arrays_are_close(R_expected, R[b], 1e-6)
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-6)
                
if __name__ == '__main__':
    unittest.main()
