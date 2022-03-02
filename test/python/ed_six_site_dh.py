#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from embedding_ed import EmbeddingEd
from kint import Tetras

def build_dh_dispersion(tg = 0.5, nkx = 60):
        na = 1
        orb_dim = 6

        # The k-space integrator
        kintegrator = Tetras(nkx,nkx,1,1) # shift away from M high symmetry point
        mesh = kintegrator.getMesh
        mesh_num = mesh.shape[0]

        # Unit cell lattice vectors and Bravai lattice vectors
        R1 = ( 3.0/2.0, np.sqrt(3.0)/2.0)
        R2 = ( 3.0/2.0, -np.sqrt(3.0)/2.0)
        R = np.array((R1, R2)).T
        G = 2.0*np.pi*np.linalg.inv(R)

        # Vectors to inter-triangle nearest neighbors
        d0 = ( 1.0, 0.0 )
        d1 = ( -0.5, np.sqrt(3.0)/2.0 )
        d2 = ( -0.5, -np.sqrt(3.0)/2.0 )
        
        dispersion = np.zeros([mesh_num,na,na,orb_dim,orb_dim],dtype=complex)

        for k,i,j,m,mm in product(range(mesh_num),range(na),range(na),range(orb_dim),range(orb_dim)):
            kay = np.dot(G.T, mesh[k,:])
            if (i == j):
                if (m == 1) and (mm == 4):
                    dispersion[k,i,j,m,mm] = -tg * ( np.exp(1j * np.dot(kay,d1)) )
                elif (m == 2) and (mm == 5):
                    dispersion[k,i,j,m,mm] = -tg * ( np.exp(1j * np.dot(kay,d2)) )
                elif (m == 4) and (mm == 1):
                    dispersion[k,i,j,m,mm] = -tg * ( np.exp(-1j * np.dot(kay,d1)) )
                elif (m == 5) and (mm == 2):
                    dispersion[k,i,j,m,mm] = -tg * ( np.exp(-1j * np.dot(kay,d2)) )
                else:
                    dispersion[k,i,j,m,mm] = 0
            else:
                dispersion[k,i,j,m,mm] = 0

        return (dispersion,kintegrator)

def get_h_qp2(R, Lambda, dispersion, mu=0):
    mesh_num = dispersion.shape[0]
    na = dispersion.shape[1]
    orb_dim = dispersion.shape[3]
    
    h_qp = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)

    for i,j in product(range(na),range(na)):
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        h_qp[the_slice] = np.matmul( R, np.matmul(dispersion[:,i,j,...], R.conj().T) )
        #h_qp[the_slice] = np.einsum('ac,kcd,db->kab', R, dispersion[:,i,j,...], R.conj().T)
        if i == j:
            h_qp[the_slice] += Lambda - mu*np.eye(Lambda.shape[0])
    
    eig, vec = np.linalg.eigh(h_qp)
    return (eig, vec)

def get_disp_R2(R, dispersion, vec):
    mesh_num = dispersion.shape[0]
    na = dispersion.shape[1]
    orb_dim = dispersion.shape[3]
    
    disp_R = np.zeros(shape=(mesh_num,na*orb_dim,na*orb_dim), dtype=complex)
    for i,j in product(range(na),range(na)):
        the_slice = np.index_exp[:, i*orb_dim:(i+1)*orb_dim, j*orb_dim:(j+1)*orb_dim]
        disp_R[the_slice] = np.matmul(dispersion[:,i,j,...], R.conj().T)
        #disp_R[the_slice] = np.einsum('kac,cb->kab',dispersion[:,i,j,...], R.conj().T)
    #A = np.einsum('kac,kcb->kab', disp_R, vec)
    return np.matmul(disp_R, vec) # Right multiply into eigenbasis of quasiparticles

class tests(unittest.TestCase):
 
    def test_dh_two_third(self):
        orb_dim = 6
        tk = 1.0
        tg = 0.5
        nkx = 60
        #beta = 10
        num_cycles = 200;
        N_elec = 8
        
        orb_names = [0, 1, 2, 3, 4, 5]
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names,orb_names, True)

        [dispersion, kintegrator] = build_dh_dispersion(tg,nkx)

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        [D, Lambda_c] = build_block_mf_matrices(gf_struct)

        U = 0
        h_loc = Operator()
        for orb in orb_names:
            h_loc += U * n("up",orb) * n("dn",orb)
        for sp in spin_names:
            h_loc -= tk * (c_dag(sp,0) * c(sp,1) + c_dag(sp,1) * c(sp,2) + c_dag(sp,2) * c(sp,0)
                         + c_dag(sp,3) * c(sp,4) + c_dag(sp,4) * c(sp,5) + c_dag(sp,5) * c(sp,3));
            h_loc -= tk * (c_dag(sp,1) * c(sp,0) + c_dag(sp,2) * c(sp,1) + c_dag(sp,0) * c(sp,2)
                         + c_dag(sp,4) * c(sp,3) + c_dag(sp,5) * c(sp,4) + c_dag(sp,3) * c(sp,5));
            h_loc -= tg * (c_dag(sp,0)*c(sp,3) + c_dag(sp,3)*c(sp,0));

        '''

        eprint("U =", U, "tk =", tk, "tg =", tg)

        eprint("R =", R)
        eprint("Lambda =", Lambda)
        eprint("h_loc =", h_loc)
        
        emb_solver = EmbeddingEd(h_loc, gf_struct)

        psiS_size = emb_solver.get_psiS_size()
        eprint("psiS_Size = ", psiS_size)
        
        # First guess for Lambda is the quadratic terms in h_loc
        #for block in block_names:
        #    Lambda[block] = np.array([[-2,0,0],[0,1,0],[0,0,1]])
        
        for cycle in range(num_cycles):
            eprint("cycle =", cycle)

            # Symmetrize
            #symmetrize(R,block_names)
            #symmetrize(Lambda,block_names)
                
            norm = 0
            R_old = deepcopy(R)
            Lambda_old = deepcopy(Lambda)

            #for b, block in enumerate(block_names):
            for b, block in enumerate(['up']):

                # python
                eig, vec = sc.get_h_qp2([R[block]], [Lambda[block]], dispersion) #, mu)
                disp_R = sc.get_disp_R2([R[block]], dispersion, vec)
                
                kintegrator.setEks(np.transpose(eig)) # python
                kintegrator.setEF_fromFilling(N_elec / len(block_names)) # divide by number of blocks
                mu = kintegrator.getEF
                wks = np.transpose(kintegrator.getWs); #python
                
                # python
                pdensity = sc.get_pdensity(vec, wks)
                ke = np.real( sc.get_ke(disp_R, vec, wks) )
                
                # Set non-diagonal elements to zero
                #pdensity = np.diag(np.diag(pdensity))
                #ke = np.diag(np.diag(ke))

                D[block] = sc.get_d(pdensity, ke)
                Lambda_c[block] = sc.get_lambda_c(pdensity, R[block], Lambda[block], D[block])

            Lambda_c['dn'] = Lambda_c['up']
            D['dn'] = D['up']

            emb_solver.set_h_emb(Lambda_c, D)
            ec = emb_solver.solve(ncv = min(30, psiS_size), max_iter = 1000, tolerance = 0)
            #ec = emb_solver.solve()

            #for b, block in enumerate(block_names):
            for b, block in enumerate(['up']):
                Nf = emb_solver.get_nf(block)
                Mcf = emb_solver.get_mcf(block)
                Nc = emb_solver.get_nc(block)
                
                # Set non-diagonal elements to zero
                #Nf = np.diag(np.diag(Nf))
                #Mcf = np.diag(np.diag(Mcf))
                #Nc = np.diag(np.diag(Nc))
                
                Lambda[block] = sc.get_lambda(R[block], D[block], Lambda_c[block], Nf)
                R[block] = sc.get_r(Mcf, Nf)
                
                # Set non-diagonal elements to zero
                #Lambda[block] = np.diag(np.diag(Lambda[block]))
                #R[block] = np.diag(np.diag(R[block]))
                
                norm += np.linalg.norm(R[block] - R_old[block])
                norm += np.linalg.norm(Lambda[block] - Lambda_old[block])

            Lambda['dn'] = Lambda['up']
            R['dn'] = R['up']

            if norm < 1e-3:
                break

        eprint("D =", D)
        eprint("Lambda_c =", Lambda_c)
        eprint("Nf =", Nf)
        eprint("Mcf =", Mcf)
        eprint("Nc =", Nc)
        eprint("N_elec =", 2. * np.trace(Nc)) # for each spin

        #NOp = N_op(spin_names,orb_names,off_diag=True)
        #S2Op = S2_op(spin_names,orb_names,off_diag=True)
        #N = emb_solver.overlap(NOp)
        #S2 = emb_solver.overlap(S2Op)

        Z = dict()
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])
        eprint("cycles =", cycle, "norm =", norm)
        eprint("Z =", Z)
        eprint("Lambda =", Lambda)
        eprint("mu =", mu)
        #eprint("N =", N )
        #eprint("S2 =", S2)
        #eprint("S =", np.sqrt(S2 + 0.25) - 0.5 )

        #mu_calculated = 0
        #for block in block_names:
        #    mu_calculated += np.trace(Lambda[block]) / (len(orb_names) * len(block_names))
        #mu_expected = mu
        #Z_expected = np.array([[0.780708,0,0],[0,0.780708,0],[0,0,0.780708]])
        #Lambda_expected = np.array([[mu_expected,0,0],[0,mu_expected,0],[0,0,mu_expected]])

        #assert are_close(mu_calculated, mu_expected, 1e-6), "mu_calculated = {0}, mu_expected = {1}".format(mu_calculated,mu_expected)
        #for block in block_names:
        #    assert_arrays_are_close(Lambda_expected, Lambda[block], 1e-6)
        #    assert_arrays_are_close(Z_expected, Z[block], 1e-6)

        '''

if __name__ == '__main__':
    unittest.main()
