#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        n_orb = 1
        spatial_dim = 3
        nkx = 10
        beta = 40
        num_cycles = 25

        block_names = ['up','dn']
        gf_struct = [ (bl, n_orb) for bl in block_names ]
        
        emb_solver = EmbeddingAtomDiag(gf_struct)

        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)

        [R, Lambda] = build_block_mf_matrices(gf_struct)

        U = 4
        h_loc = U * n('up',0) * n('dn',0)
        mu = U / 2

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

        print("Lambda", Lambda)
        print("R", R)
            
        mu_calculated = 0
        for bl in block_names:
            mu_calculated += np.sum(Lambda[bl]) / 2.0
        mu_expected = U/2.
        #R_expected = np.array([[0.861617]])
        R_expected = np.array([[0.66168702]])
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
#            g0_iw << inverse( iOmega_n  - h0_k['up'][...,k] + mu_matrix)
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
#        gqp_k_iw = sc.get_gqp_k_z(g0_k_iw, R['up'], Lambda['up'], h0_k, mu)
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
        for bl in block_names:
            assert_arrays_are_close(Lambda_expected, Lambda[bl], 1e-6)
            assert_arrays_are_close(R_expected, R[bl], 1e-6)
                
if __name__ == '__main__':
    unittest.main()
