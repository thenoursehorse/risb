#!/usr/bin/env python

import numpy as np
import unittest
from common import build_cubic_h0_k, symmetrize_blocks
from triqs.operators import n
from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        n_orb = 1
        spatial_dim = 3
        nkx = 10
        beta = 40

        block_names = ['up','dn']
        gf_struct = [ (bl, n_orb) for bl in block_names ]
        
        h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
        
        U = 4
        h_loc = U * n('up',0) * n('dn',0)
        mu = U / 2
        
        emb_solver = EmbeddingAtomDiag(h_loc, gf_struct)
        kweight_solver = SmearingKWeight(beta=beta, mu=mu)
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          emb_solver=emb_solver,
                          kweight_solver=kweight_solver,
                          symmetries=[symmetrize_blocks])
        
        S.solve()
            
        mu_calculated = kweight_solver.mu
        mu_expected = mu
        Lambda_expected = np.array([[2.0]])
        Z_expected = np.array([[0.437828801025]])
                
        np.testing.assert_allclose(mu_calculated, mu_expected, rtol=0, atol=1e-6)
        for bl in block_names:
            np.testing.assert_allclose(Lambda_expected, S.Lambda[bl], rtol=0, atol=1e-6)
            np.testing.assert_allclose(Z_expected, S.Z[bl], rtol=0, atol=1e-6)
        
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
  
if __name__ == '__main__':
    unittest.main()
