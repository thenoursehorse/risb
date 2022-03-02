#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from risb.embedding_atom_diag import *
from triqs.atom_diag import trace_rho_op
from kint import Tetras

def build_ghost_dispersion(t = 1.0, nkx = 6, orb_dim = 2, spatial_dim = 2):
        # The k-space integrator
        kintegrator = Tetras(nkx,nkx,1,1) # shift away from high symmetry points
        mesh = kintegrator.getMesh
        mesh_num = mesh.shape[0]

        # Unit cell lattice vectors and Bravai lattice vectors
        R = np.eye(spatial_dim).T
        G = 2.0*np.pi*np.linalg.inv(R)

        dispersion = np.zeros([orb_dim,orb_dim,mesh_num])

        for k,a,b in product(range(mesh_num),range(orb_dim),range(orb_dim)):
            kay = np.dot(G.T, mesh[k,:])
            if (a == 0) and (b == 0):
                dispersion[a,b,k] = -2.0 * t * np.sum(np.cos(kay)) / spatial_dim # only hopping to physical orbital
            else:
                dispersion[a,b,k] = 0
        return (dispersion,kintegrator)

class tests(unittest.TestCase):
 
    def test_ghost(self):
        orb_dim = 3 # 2 ghost orbitals
        nkx = 6
        #beta = 10
        num_cycles = 1;
        N_elec = 1
        
        orb_names = range(orb_dim)
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names,orb_names, True)
            
        emb_solver = EmbeddingAtomDiag(gf_struct)

        [dispersion, kintegrator] = build_ghost_dispersion(nkx = nkx, orb_dim = orb_dim)

        R, Lambda = build_block_mf_matrices(gf_struct)
        D = deepcopy(R)
        Lambda_c = deepcopy(Lambda)

        for U in [1]: #in np.arange(0,5,0.1):
            mu = U / 2
            h_loc = U*n('up',0)*n('dn',0) + (mu-U/2)*(n('up',0) + n('dn',0))

            eprint("U =", U, "#orbs =", orb_dim)

            # First guess for Lambda is the quadratic terms in h_loc
            #for b in block_names:
            #    for inner in Lambda[b]:
            #        inner = mu
            
            for cycle in range(num_cycles):

                # Symmetrize
                symmetrize(R,block_names)
                symmetrize(Lambda,block_names)
                    
                norm = 0
                R_old = deepcopy(R)
                Lambda_old = deepcopy(Lambda)

                #eprint("R =", R)
                #eprint("Lambda =", Lambda)

                for b, block in enumerate(['up']):

                    eig, vec = sc.get_h_qp(R[block], Lambda[block], dispersion) #, mu)
                    disp_R = sc.get_disp_R(R[block], dispersion, vec)

                    kintegrator.setEks(np.transpose(eig))
                    #kintegrator.setEF_fromFilling(N_elec / len(block_names)) # divide by number of blocks
                    #mu = kintegrator.getEF
                    kintegrator.setEF(0)
                    wks = np.transpose(kintegrator.getWs)
                    #wks = fermi_fnc(eig, beta) / nk
                    
                    pdensity = sc.get_pdensity(vec, wks)
                    ke = np.real( sc.get_ke(disp_R, vec, wks) )
 
                    D[block] = sc.get_d(pdensity, ke)
                    Lambda_c[block] = sc.get_lambda_c(pdensity, R[block], Lambda[block], D[block])

                Lambda_c['dn'] = Lambda_c['up']
                D['dn'] = D['up']

                emb_solver.set_h_emb(h_loc, Lambda_c, D)#, mu)
                emb_solver.solve()

                for b, block in enumerate(['up']):
                    Nf = emb_solver.get_nf(block)
                    Mcf = emb_solver.get_mcf(block)

                    Lambda[block] = sc.get_lambda(R[block], D[block], Lambda_c[block], Nf)
                    R[block] = sc.get_r(Mcf, Nf)
                    
                    norm += np.linalg.norm(R[block] - R_old[block])
                    norm += np.linalg.norm(Lambda[block] - Lambda_old[block])

                Lambda['dn'] = Lambda['up']
                R['dn'] = R['up']

                if norm < 1e-6:
                    break

            N = N_op(spin_names,orb_names,off_diag=True)
            S2 = S2_op(spin_names,orb_names,off_diag=True)
            S2_avg = trace_rho_op(emb_solver.get_dm(), S2, emb_solver.get_ad())

            Nf = emb_solver.get_nf('up')
            Nc = emb_solver.get_nc('up')

            Z = dict()
            for block in block_names:
                Z[block] = np.dot(R[block], R[block])
            eprint("cycles =", cycle, "norm =", norm)
            eprint("mu =", mu)
            eprint("R =", R)
            eprint("Lambda =", Lambda)
            eprint("pdensity =", pdensity)
            eprint("D = ", D)
            eprint("Lambda_c =", Lambda_c)
            eprint("Nf =", Nf)
            eprint("Nc =", Nc)

if __name__ == '__main__':
    unittest.main()
