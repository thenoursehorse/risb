#!/usr/bin/env python

from common import *
from embedding_ed import *

import os
filename = 'results_one_band'
if not os.path.exists(filename):
    os.makedirs(filename)

class tests(unittest.TestCase):
 
    def test_seemicircular_gf_half_filling(self):

        # Parameters of the model
        t = 0.5
        beta = 10.0
        n_loops = 10

        [fops_loc, orb_names, spin_names] = build_fops_local(1)
        [fops_bath, fops_emb] = get_embedding_space(fops_loc)
        dim = len(orb_names) * len(spin_names)
        
        [R, Lambda] = build_mf_matrices(len(orb_names)*len(spin_names))
        g_iw = GfImFreq(indices = spin_names, beta = beta)
        g_iw['up','up'] << SemiCircular(2*t)
        g_iw['dn','dn'] << SemiCircular(2*t)
        g0_iw = GfImFreq(indices = ['up','dn'], beta = beta)
        g0_iw['up','up'] << SemiCircular(2*t)
        g0_iw['dn','dn'] << SemiCircular(2*t)

        for U in [0.5]: #, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
            h_loc = 0
            for orb in orb_names:
                h_loc += U * n(spin_names[0],orb) * n(spin_names[1],orb)
            mu = U / 2

            eprint("U = ", U)
            
            for i in range(n_loops):
                R[0,0] = 0.5 * (R[0,0] + R[1,1])
                R[1,1] = R[0,0]
                R[0,1] = 0
                R[1,0] = 0
                
                Lambda[0,0] = 0.5 * (Lambda[0,0] + Lambda[1,1])
                Lambda[1,1] = Lambda[0,0]
                Lambda[0,1] = 0
                Lambda[1,0] = 0

                R_old = deepcopy(R)
                Lambda_old = deepcopy(Lambda)

                # Calculate new sigma
                sigma_iw = get_sigma_z(g_iw,R,Lambda,mu);

                # Calculate new g_iw:
                #for g in g_iw:
                g_iw << inverse( inverse(g0_iw) - sigma_iw )

                # symmetrize
                g_iw['up','up'] = 0.5 * (g_iw['up','up'] + g_iw['dn','dn'] )
                g_iw['dn','dn'] = g_iw['up','up']
                g_iw['up','dn'][:] = 0.0
                g_iw['dn','up'][:] = 0.0

                # Calcualte new g0_iw
                #for g0 in g0_iw:
                g0_iw << inverse( iOmega_n + mu - t**2 * g_iw )

                # Calculate hybridization function
                delta_iw = get_delta_z(g0_iw)

                # RISB self-consistent part to calculate new sigma_iw
                pdensity = get_pdensity_gf(g_iw, R)
                ke = get_ke_gf(g_iw, delta_iw, R)
                
                D = get_d(pdensity, ke)
                Lambda_c = get_lambda_c(pdensity, R, Lambda, D)
                
                #h_emb = get_h_emb(h_loc, D, Lambda_c, fops_loc, fops_bath)

                emb_solver = EmbeddingEd(h_loc, fops_loc)
                emb_solver.set_h_emb(Lambda_c, D)
                emb_solver.solve()
                
                Nf = emb_solver.get_nf()
                Mcf = emb_solver.get_mcf()
                
                Lambda = get_lambda(R, D, Lambda_c, Nf)
                R = get_r(Mcf, Nf)
                
                norm = 0
                norm += np.linalg.norm(R - R_old)
                norm += np.linalg.norm(Lambda - Lambda_old)

                # Save iteration in archive
                with HDFArchive(filename + "/half-U%.2f.h5"%U) as A:
                    A['G-%i'%i] = g_iw
                    A['Sigma-%i'%i] = sigma_iw


                if norm < 1e-6:
                    break
            
            eprint("cycles =", i, "norm =", norm)
            eprint("R =")
            eprint(R)
            eprint("Lambda =")
            eprint(Lambda)
        
        mu_calculated = np.sum(Lambda) / dim
        mu_expected = U/2.
        R_expected = np.array([[0.987918,0],[0,0.987918]])
        Lambda_expected = np.array([[0.25,0],[0,0.25]])
        
        assert are_close(mu_calculated, mu_expected, 1e-3), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        assert_arrays_are_close(R_expected, R, 1e-3)
        assert_arrays_are_close(Lambda_expected, Lambda, 1e-3)
                
if __name__ == '__main__':
    unittest.main()
