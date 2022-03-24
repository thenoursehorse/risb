#!/usr/bin/env python

from common import *
from embedding_ed import *
from kint import Tetras

class tests(unittest.TestCase):
 
    def test_hubbard_half_filling(self):
        orb_dim = 1
        beta = 10 # inverse temperature
        N_elec = 1 # electron filling
        nkx = 20
        num_cycles = 100

        [fops_loc, orb_names, spin_names] = build_fops_local(orb_dim)
        [fops_bath, fops_emb] = get_embedding_space(fops_loc)
        dim = len(orb_names) * len(spin_names)
        
        spatial_dim = 3
        #t = 0.5 / float(spatial_dim)
        t = 1.0
        kintegrator = Tetras(nkx,nkx,nkx)
        mesh = kintegrator.getMesh
        mesh_num = mesh.shape[0]
        G = np.zeros([spatial_dim, spatial_dim])
        np.fill_diagonal(G, 2. * np.pi)
        dispersion = np.zeros([1,1,dim,dim,mesh_num])
        for k in range(mesh_num):
            kay = np.dot(G.T, mesh[k,:])
            for a in range(dim):
                dispersion[:,:,a,a,k] = -2. * t * np.sum(np.cos(kay))
        nk = dispersion.shape[4]

        [R, Lambda] = build_mf_matrices(len(orb_names)*len(spin_names))
         
        for U in [12]: #;np.arange(0,17+0.1,1):
            h_loc = 0
            for orb in orb_names:
                h_loc += U * n(spin_names[0],orb) * n(spin_names[1],orb)
            mu = U / 2

            #eprint("U = ", U)
            
            for a in range(dim):
                Lambda[a,a] = mu
        
            for cycle in range(num_cycles):
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

                #h_qp = get_h_qp([R], [Lambda], dispersion, mu)
                #wks = fermi_fnc(h_qp.val, beta) / nk
                h_qp = get_h_qp([R], [Lambda], dispersion)
                kintegrator.setEks(h_qp.val)
                kintegrator.setEF(mu)
                #kintegrator.setEF_fromFilling(N_elec)
                wks = kintegrator.getWs;

                ke = get_ke([R], dispersion, h_qp, wks)

                pdensity = get_pdensity(0, h_qp, wks)
                D = get_d(pdensity, ke[0])
                Lambda_c = get_lambda_c(pdensity, R, Lambda, D)

                #eprint("D =", D)
                #eprint("Lambda_c =", Lambda_c)
                
                #h_emb = get_h_emb(h_loc, D, Lambda_c, fops_loc, fops_bath)
                #emb_solver = EmbeddingAtomDiag(h_emb, fops_emb, fops_loc, fops_bath)
                emb_solver = EmbeddingEd(h_loc, fops_loc)
                emb_solver.set_h_emb(Lambda_c, D)
                Ec = emb_solver.solve()

                Nf = emb_solver.get_nf()
                Mcf = emb_solver.get_mcf()
                
                Lambda = get_lambda(R, D, Lambda_c, Nf)
                R = get_r(Mcf, Nf)
                
                norm = 0
                norm += np.linalg.norm(R - R_old)
                norm += np.linalg.norm(Lambda - Lambda_old)

                #eprint("Ec =", Ec)
                #eprint("Nf =", Nf)
                #eprint("Mcf =", Mcf)

                if norm < 1e-6:
                    break
            
            #eprint("cycles =", cycle, "norm =", norm)
            #eprint("R =")
            #eprint(R)
            #eprint("Lambda =")
            #eprint(Lambda)
        
        mu_calculated = np.sum(Lambda) / dim
        mu_expected = U/2.
        R_expected = np.array([[0.986323,0],[0,0.986323]])
        Lambda_expected = np.array([[0.25,0],[0,0.25]])
        
        assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        #assert_arrays_are_close(R_expected, R, 1e-6)
        #assert_arrays_are_close(Lambda_expected, Lambda, 1e-6)
                
if __name__ == '__main__':
    unittest.main()
