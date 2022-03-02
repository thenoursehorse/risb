#!/usr/bin/env python

from common import *
from risb.embedding_atom_diag import *
from risb.solver import Solver

class tests(unittest.TestCase):
 
    def test_cubic_gf_half_filling(self):
        spatial_dim = 2
        t = 0.5 / spatial_dim
        beta = 10
        n_iw = 10*beta
        n_loops = 25
        nkx = 6

        gf_struct = [ ['up', [0]], ['dn', [0]] ]
            
        emb_solver = EmbeddingAtomDiag(gf_struct)
        S = Solver(beta = beta, gf_struct = gf_struct, n_iw = n_iw, emb_solver = emb_solver)

        dispersion, bl = build_cubic_dispersion(nkx = nkx, return_bl = True)

        mesh_k = MeshBrillouinZone(BrillouinZone(bl), nkx)
        mesh_iw = S.G_iw.mesh
        mesh_k_iw = MeshProduct(mesh_k, mesh_iw)
        nk = len(mesh_k)

        g0_iw = Gf(mesh = mesh_iw, target_shape=S.R['up'].shape)
        #G0_iw = BlockGf(name_list = S.block_names, block_list = (g0,g0), make_copies = True)
        g_iw = g0_iw.copy()

        for U in [1.5]: #np.arange(0,3,0.1):
            h_loc = U * n('up',0) * n('dn',0)
            mu = U / 2
            eprint("U =", U, "mu =", mu)
            
            for block in S.block_names:
                for inner in S.Lambda[block]:
                    inner = mu
            
            # Non-interacting lattice Green's function
            mu_matrix = mu*np.eye(S.R['up'].shape[0])
            g0_k_iw = Gf(mesh = mesh_k_iw, target_shape=g0_iw.target_shape)
            for k,kay in enumerate(g0_k_iw.mesh.components[0]):
                g0_iw << inverse( iOmega_n  - dispersion[...,k] + mu_matrix )
                g0_k_iw[kay,:].data[:] = g0_iw.data

            for i in range(n_loops):
                
                # Symmetrize
                symmetrize(S.R,S.block_names)
                symmetrize(S.Lambda,S.block_names)

                norm = 0
                R_old = deepcopy(S.R)
                Lambda_old = deepcopy(S.Lambda)

                # Using local self-energy for lattice Green's function
                g_k_iw = sc.get_g_k_z(g0_k_iw, S.Sigma_iw['up'])
                
                # Integrate lattice Green's function
                for iw in mesh_iw:
                    g_iw[iw] = 0.0
                    for k in mesh_k:
                        g_iw[iw] += g_k_iw[k,iw] / nk

                for name, g in S.G_iw:
                    g << g_iw

                # Calculate new G0_iw of impurity:
                S.G0_iw = dyson(G_iw = S.G_iw, Sigma_iw = S.Sigma_iw)

                S.solve(h_int = h_loc, mu = mu)
         
                for block in S.block_names:
                    norm += np.linalg.norm(S.R[block] - R_old[block])
                    norm += np.linalg.norm(S.Lambda[block] - Lambda_old[block])

                if norm < 1e-6:
                    break
            
            eprint("cycles =", i, "norm =", norm)
            eprint("R =", S.R)
            eprint("Lambda =", S.Lambda)
            eprint("pdensity =", S.pdensity)
            eprint("G0_iw.density() =", S.G0_iw.density())
            eprint("Gqp_iw.density() =", S.Gqp_iw.density())
            eprint("G_iw.density() =", S.G_iw.density())
       
        mu_calculated = 0
        for block in S.block_names:
            mu_calculated += np.trace(S.Lambda[block]) / 2.0
        mu_expected = U/2.
        #R_expected = np.array([[0.987918]])
        #Lambda_expected = np.array([[mu_expected]])
        #
        assert are_close(mu_calculated, mu_expected, 1e-3), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        #for b in S.block_names:
        #    assert_arrays_are_close(R_expected, R[b], 1e-3)
        #    assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-3)

if __name__ == '__main__':
    unittest.main()
