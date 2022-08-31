#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from risb.embedding_atom_diag import *
#from triqs.atom_diag import trace_rho_op
#from kint import Tetras

def build_dh_dispersion(tg = 0.5, nkx = 18):
        na = 2
        orb_dim = 3
        phi = 2.0 * np.pi / 3.0

        # The k-space integrator if using linear tetrahedron from kint
        #kintegrator = Tetras(nkx,nkx,1,1) # shift away from M high symmetry point
        #mesh = kintegrator.getMesh

        # Build shifted 2D mesh
        kintegrator = None
        mesh = np.empty(shape=(nkx*nkx, 2))
        for idx,coords in enumerate(zip(range(nkx), range(nkx))):
            mesh[idx,0] = coords[0]/nkx + 0.5/nkx
            mesh[idx,1] = coords[1]/nkx + 0.5/nkx
        
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
            if (i == 0) and (j == 1):
                dispersion[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(1j * np.dot(kay,d0)) 
                                                     + np.exp(1j * np.dot(kay,d1)) * np.exp(1j * phi * (mm-m)) 
                                                     + np.exp(1j * (np.dot(kay,d2)))*np.exp(1j * 2.0 * phi * (mm-m)) )
            elif (i == 1) and (j == 0):
                dispersion[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(-1j * np.dot(kay,d0)) 
                                                     + np.exp(-1j * np.dot(kay,d1)) * np.exp(-1j *phi * (m-mm)) 
                                                     + np.exp(-1j * np.dot(kay,d2)) * np.exp(-1j * 2.0 * phi * (m-mm)) )
            else:
                dispersion[k,i,j,m,mm] = 0

        return (dispersion,kintegrator)

def hubb_N(tk, U, orbs, spins):
    N = len(orbs)
    phi = 2.0 * np.pi / N
    h_loc = Operator()

    #for m,s in product(orbs,spins):
    #    h_loc += -2.0 * tk * c_dag(s,m) * c(s,m) * np.cos(phi * m)

    # -tk * sum_a ( c_dag(s,a) * c(s,a+1) + c_dag(s,a+1) * c_(s,a) )   // np.mod(a+1)
    # c_dag(s,a) = sum_m b_dag(s,m) * np.exp(-1j * phi * a * m)
    # c(s,a) = sum_m b(s,m) * np.exp(1j * phi * a * m)
    for a,m,mm,s in product(orbs,orbs,orbs,spins):
        h_loc += (-tk / N) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * a * m) * np.exp(1j * phi * np.mod(a+1,N) * mm)
        h_loc += (-tk / N) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * np.mod(a+1,N) * m) * np.exp(1j * phi * a * mm)
    
    for m,mm,mmm in product(orbs,orbs,orbs):
        h_loc += (U / N) * c_dag("up",m) * c("up",mm) * c_dag("dn",mmm) * c("dn",np.mod(m+mmm-mm,N))
    
    return h_loc.real


# FIXME add possible rotation
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
        orb_dim = 3
        tk = 1.0
        tg = 0.5
        nkx = 18
        beta = 10
        num_cycles = 5
        N_elec = 8
        mu = 1
        
        orb_names = [0, 1, 2]
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names,orb_names, True)
            
        emb_solver = EmbeddingAtomDiag(gf_struct)

        [dispersion, kintegrator] = build_dh_dispersion(tg,nkx)
        #dispersion = np.transpose(dispersion, (1,2,3,4,0)) # c++

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        [D, Lambda_c] = build_block_mf_matrices(gf_struct)

        U = 1.
        h_loc = hubb_N(tk, U, orb_names, spin_names)

        eprint("U =", U, "tk =", tk)

        # First guess for Lambda is the quadratic terms in h_loc
        for block in block_names:
            Lambda[block] = np.array([[-2,0,0],[0,1,0],[0,0,1]])
        
        for cycle in range(num_cycles):

            # Symmetrize
            #symmetrize(R,block_names)
            #symmetrize(Lambda,block_names)
                
            norm = 0
            R_old = deepcopy(R)
            Lambda_old = deepcopy(Lambda)

            #for b, block in enumerate(block_names):
            for b, block in enumerate(['up']):

                # python
                eig, vec = sc.get_h_qp2([R[block],R[block]], [Lambda[block],Lambda[block]], dispersion, mu)
                disp_R = sc.get_disp_R2([R[block],R[block]], dispersion, vec)
                
                # c++
                #h_qp = get_h_qp( [R[block],R[block]], [Lambda[block],Lambda[block]], dispersion )
                #disp_R = get_disp_R([R[block],R[block]], dispersion, h_qp)[0:3,:,:] # project onto one of the triangles in the unit cell
                #eig = h_qp.val
                #vec = h_qp.vec[0:3,:,:]
                #vec_dag = h_qp.vec_dag[:,0:3,:]

                # Using kint linear tetrahedron
                #kintegrator.setEks(np.transpose(eig)) # python
                ##kintegrator.setEks(eig) # c++
                #kintegrator.setEF_fromFilling(N_elec / len(block_names)) # divide by number of blocks
                #mu = kintegrator.getEF
                #wks = np.transpose(kintegrator.getWs); #python
                ##wks = kintegrator.getWs; #c++

                # Using fermi smear
                wks = fermi_fnc(eig, beta) / nkx**2
                
                # python
                pdensity = sc.get_pdensity(vec[:,0:3,:], wks) # Project onto one of the triangles in the unit cell
                ke = np.real( sc.get_ke(disp_R[:,0:3,:], vec[:,0:3,:], wks) )
                
                # c++
                #pdensity = get_pdensity(vec, vec_dag, wks)
                #ke = np.real( get_ke(disp_R, vec_dag, wks) )

                # Set non-diagonal elements to zero
                pdensity = np.diag(np.diag(pdensity))
                ke = np.diag(np.diag(ke))

                eprint("pdensity =", pdensity)

                D[block] = sc.get_d(pdensity, ke)
                Lambda_c[block] = sc.get_lambda_c(pdensity, R[block], Lambda[block], D[block])

            Lambda_c['dn'] = Lambda_c['up']
            D['dn'] = D['up']
                
            emb_solver.set_h_emb(h_loc, Lambda_c, D) #, mu)
            emb_solver.solve()
        
            N = N_op(spin_names,orb_names,off_diag=True)
            #eprint("N =", trace_rho_op(emb_solver.get_dm(), N, emb_solver.get_ad()) )
            eprint("N =", emb_solver.overlap(N))

            #for b, block in enumerate(block_names):
            for b, block in enumerate(['up']):
                Nf = emb_solver.get_nf(block)
                Mcf = emb_solver.get_mcf(block)
                Nc = emb_solver.get_nc(block)
                
                # Set non-diagonal elements to zero
                Nf = np.diag(np.diag(Nf))
                Mcf = np.diag(np.diag(Mcf))
                Nc = np.diag(np.diag(Nc))
        
                eprint("D =", D)
                eprint("Lambda_c =", Lambda_c)
                eprint("Nf =", Nf)
                eprint("Mcf =", Mcf)
                eprint("Nc =", Nc)
                
                Lambda[block] = sc.get_lambda(R[block], D[block], Lambda_c[block], Nf)
                R[block] = sc.get_r(Mcf, Nf)
                
                # Set non-diagonal elements to zero
                Lambda[block] = np.diag(np.diag(Lambda[block]))
                R[block] = np.diag(np.diag(R[block]))
                
                norm += np.linalg.norm(R[block] - R_old[block])
                norm += np.linalg.norm(Lambda[block] - Lambda_old[block])

            Lambda['dn'] = Lambda['up']
            R['dn'] = R['up']

            if norm < 1e-6:
                break

        eprint("D =", D)
        eprint("Lambda_c =", Lambda_c)
        eprint("Nf =", Nf)
        eprint("Mcf =", Mcf)
        eprint("Nc =", Nc)

        N = N_op(spin_names,orb_names,off_diag=True)
        S2 = S2_op(spin_names,orb_names,off_diag=True)
        #S2_avg = trace_rho_op(emb_solver.get_dm(), S2, emb_solver.get_ad())
        S2_avg = emb_solver.overlap(S2)


        Z = dict()
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])
        eprint("cycles =", cycle, "norm =", norm)
        eprint("Z =", Z)
        eprint("Lambda =", Lambda)
        eprint("mu =", mu)
        #eprint("N =", trace_rho_op(emb_solver.get_dm(), N, emb_solver.get_ad()) )
        eprint("N =", emb_solver.overlap(N))
        eprint("S2 =", S2_avg)
        eprint("S =", np.sqrt(S2_avg + 0.25) - 0.5 )

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

if __name__ == '__main__':
    unittest.main()
