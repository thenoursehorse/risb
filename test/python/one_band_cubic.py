#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        nkx = 6
        beta = 10
        num_cycles = 25;

        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = [ ['up', [0]], ['dn', [0]] ]
        
        emb_solver = EmbeddingAtomDiag(gf_struct)

        dispersion, bl = build_cubic_dispersion(nkx = nkx, return_bl = True)
        nk = dispersion.shape[0]
 
        [R, Lambda] = build_block_mf_matrices(gf_struct)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)
        
        for U in [1.5]: #np.arange(0,3,0.1):
            h_loc = U * n('up',0) * n('dn',0)
            mu = U / 2

            for b in block_names:
                for inner in Lambda[b]:
                    inner = mu
            
            for cycle in range(num_cycles):

                symmetrize(R,block_names)
                symmetrize(Lambda,block_names)

                norm = 0
                R_old = deepcopy(R)
                Lambda_old = deepcopy(Lambda)

                for b in spin_names:
                    eig, vec = sc.get_h_qp(R[b], Lambda[b], dispersion)
                    disp_R = sc.get_disp_R(R[b], dispersion, vec)
                    wks = fermi_fnc(eig, beta, mu) / nk

                    pdensity = sc.get_pdensity(vec, wks)
                    ke = sc.get_ke(disp_R, vec, wks)

                    D[b] = sc.get_d(pdensity, ke)
                    Lambda_c[b] = sc.get_lambda_c(pdensity, R[b], Lambda[b], D[b])

                emb_solver.set_h_emb(h_loc,Lambda_c, D)
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
            
        mu_calculated = 0
        for block in block_names:
            mu_calculated += np.sum(Lambda[block]) / 2.0
        mu_expected = U/2.
        R_expected = np.array([[0.861617]])
        Lambda_expected = np.array([[mu_expected]])
        
        # Test green's function density
#        nw = 10*beta
#        mesh_k = MeshBrZone(BrillouinZone(bl), nkx)
#        mesh_iw = MeshImFreq(beta= beta, S = "Fermion", n_max = nw)
#        mesh_k_iw = MeshProduct(mesh_k, mesh_iw)
#
#        mu_matrix = mu * np.eye(R['up'].shape[0])
#
#        g0_k_iw = Gf(mesh = mesh_k_iw, indices = [0])
#        g0_iw = Gf(mesh = mesh_iw, indices = [0])
#        for k,kay in enumerate(g0_k_iw.mesh.components[0]):
#            g0_iw << inverse( iOmega_n  - dispersion[...,k] + mu_matrix)
#            g0_k_iw[kay,:].data[:] = g0_iw.data
#        
#        for iw in mesh_iw:
#            g0_iw[iw] = 0.0
#            for k in mesh_k:
#                g0_iw[iw] += g0_k_iw[k,iw] / nk
#
#        sigma_iw = sc.get_sigma_z(mesh_iw,R['up'],Lambda['up'],mu)
#        g_k_iw = sc.get_g_k_z(g0_k_iw, sigma_iw)
#      
#        g_iw = g0_iw.copy()
#        for iw in mesh_iw:
#            g_iw[iw] = 0.0
#            for k in mesh_k:
#                g_iw[iw] += g_k_iw[k,iw] / nk
#
#        gqp_k_iw = sc.get_gqp_k_z(g0_k_iw, R['up'], Lambda['up'], dispersion, mu)
#        gqp_iw = g0_iw.copy()
#        for iw in mesh_iw:
#            gqp_iw[iw] = 0.0
#            for k in mesh_k:
#                gqp_iw[iw] += gqp_k_iw[k,iw] / nk
#        
#        g_k_iw2 = sc.get_g_k_z2(gqp_k_iw, R['up'])
#        g_iw2 = g0_iw.copy()
#        for iw in mesh_iw:
#            g_iw2[iw] = 0.0
#            for k in mesh_k:
#                g_iw2[iw] += g_k_iw[k,iw] / nk
#
#        g0_iw2 = dyson(G_iw = g_iw, Sigma_iw = sigma_iw)
#                
#        Nf = emb_solver.get_nf('up')
#        Nc = emb_solver.get_nc('up')
#
#        eprint("pdensity =", pdensity)
#        eprint("Nf =", Nf)
#        eprint("Nc =", Nc)
#        eprint("gqp_iw.density() =", gqp_iw.density())
#        
#        eprint("g0_iw.density() =", g0_iw.density())
#        eprint("g0_iw2.density() =", g0_iw.density())
#        eprint("g_iw.density() =", g_iw.density())
#        eprint("g_iw2.density() =", g_iw2.density())
#        eprint("Rgqp_iw.density()R =", np.dot(np.dot(R['up'], gqp_iw.density()), R['up']) )
        
        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in block_names:
            assert_arrays_are_close(R_expected, R[b], 1e-6)
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-6)
                
if __name__ == '__main__':
    unittest.main()
