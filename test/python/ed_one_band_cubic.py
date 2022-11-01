#!/usr/bin/env python

from common import *
from embedding_ed import *
from kint import Tetras

class tests(unittest.TestCase):

    def test_hubbard_half_filling(self):
        orb_dim = 1
        N_elec = 1
        nkx = 6
        spatial_dim = 3
        num_cycles = 25

        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = [ ['up', [0]], ['dn', [0]] ]

        # set up the kint integrator on a 3D lattice
        kintegrator = Tetras(nkx,nkx,nkx)
        mesh = kintegrator.getMesh
        mesh_num = mesh.shape[0]

        # set up the reciprocal lattice vectors
        G = np.zeros([spatial_dim, spatial_dim])
        np.fill_diagonal(G, 2. * np.pi)

        # build the dispersion relation
        t = 0.5 / float(spatial_dim)
        dispersion = np.zeros([orb_dim,orb_dim,mesh_num])
        for k in range(mesh_num):
            kay = np.dot(G.T, mesh[k,:]) # rotate into cartesian basis
            for a in range(orb_dim):
                dispersion[a,a,k] = -2. * t * np.sum(np.cos(kay)) # e_k
        nk = dispersion.shape[-1] # number of lattice sites nkx*nkx*nkx

        # Set up the mean-field matrices
        [R, Lambda] = build_block_mf_matrices(gf_struct)
        D = deepcopy(Lambda)
        Lambda_c = deepcopy(Lambda)

        for U in [1]: #;np.arange(0,17+0.1,1):
            h_loc = U * n('up',0) * n('dn',0)
            mu = U / 2.

            # set up solver with symmetry blocks determined from h_loc
            emb_solver = EmbeddingEd(h_loc,gf_struct)
            psiS_size = emb_solver.get_psiS_size()

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
                    # build h_qp
                    eig, vec = sc.get_h_qp(R[b], Lambda[b], dispersion)
                    disp_R = sc.get_disp_R(R[b], dispersion, vec)

                    # solve h_qp
                    kintegrator.setEks(np.transpose(eig))
                    #kintegrator.setEF(mu)
                    kintegrator.setEF_fromFilling(N_elec / len(block_names))
                    mu_calculated = kintegrator.getEF
                    wks = np.transpose(kintegrator.getWs)

                    # get orbital occupations and kinetic energy of f electrons
                    pdensity = sc.get_pdensity(vec, wks)
                    ke = sc.get_ke(disp_R, vec, wks)

                    # get the hybridization and bath for the impurity problem
                    D[b] = sc.get_d(pdensity, ke)
                    Lambda_c[b] = sc.get_lambda_c(pdensity, R[b], Lambda[b], D[b])

                # set h_emb and solve impurity problem
                emb_solver.set_h_emb(Lambda_c, D)
                Ec = emb_solver.solve()

                for b in block_names:
                    # get the density matrix of the f-electrons, the hybridization, and the c-electrons
                    Nf = emb_solver.get_nf(b)
                    Mcf = emb_solver.get_mcf(b)

                    # get new lambda and r
                    Lambda[b] = np.real( sc.get_lambda(R[b], D[b], Lambda_c[b], Nf) )
                    R[b] = np.real( sc.get_r(Mcf, Nf) )

                    # convergence condition
                    norm += np.linalg.norm(R[b] - R_old[b])
                    norm += np.linalg.norm(Lambda[b] - Lambda_old[b])

                if norm < 1e-6:
                    break

        mu_expected = U/2.
        R_expected = np.array([[0.927500]])
        Lambda_expected = np.array([[mu_expected]])

        assert are_close(mu_calculated, mu_expected, 1e-5), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        for b in block_names:
            assert_arrays_are_close(R_expected, R[b], 1e-5)
            assert_arrays_are_close(Lambda_expected, Lambda[b], 1e-5)

if __name__ == '__main__':
    unittest.main()