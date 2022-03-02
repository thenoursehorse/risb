#!/usr/bin/env python

from common import *
from triqs.operators.util.hamiltonians import h_int_kanamori
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from embedding_ed import EmbeddingEd
from kint import Tetras

from triqs.utility.dichotomy import dichotomy
from scipy import optimize

def impurity_fit(n_elec_target, emb_solver, Lambda_c, D, block_names):
    
    def f_n_elec(dmu):
        # shift the bath levels
        for block in block_names:
            Lambda_c[block] -= dmu * np.eye(Lambda_c[block].shape[0])

        # solve the impurity
        psiS_size = emb_solver.get_psiS_size()
        emb_solver.set_h_emb(Lambda_c, D)
        emb_solver.solve(ncv = min(30,psiS_size), max_iter = 1000, tolerance= 0)

        # get total number of c-electrons in this fragment
        n_elec_calc = 0
        for block in block_names:
            n_elec_calc += np.trace(emb_solver.get_nc(block))

        # add back to lambda_c for next guess
        for block in block_names:
            Lambda_c[block] += dmu * np.eye(Lambda_c[block].shape[0])

        return n_elec_calc - n_elec_target

    # find correct dmu shift using scipy (this will use secant method)
    sol = optimize.root_scalar(f_n_elec, x0 = 0, x1 = 0.2, xtol = 1e-4, maxiter = 1000)
    dmu = sol.root

    # find correct dmu shift using triqs
    # x = dmu, f(x) = n_elec_calc - n_elec_target  calculated from impurity
    #(dmu, n_elec_calc) = dichotomy(function = f_n_elec, x_init = 0.0, y_value = 0.0, precision_on_y = 1e-4, delta_x = 0.05, 
    #                               x_name = "dmu", y_name = "n_elec_calc - n_elec_target")
    
    # subtract off the correct dmu
    for block in block_names:
        Lambda_c[block] -= dmu * np.eye(Lambda_c[block].shape[0])

def build_dh_dispersion(tg = 0.5, nkx = 60):
        na = 2
        orb_dim = 3
        phi = 2.0 * np.pi / 3.0

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
        nkx = 60
        #beta = 10
        num_cycles = 200;
        N_elec = 8
        
        orb_names = [0, 1, 2]
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names,orb_names, True)

        [dispersion, kintegrator] = build_dh_dispersion(tg,nkx)
        #dispersion = np.transpose(dispersion, (1,2,3,4,0)) # c++

        [R, Lambda] = build_block_mf_matrices(gf_struct)
        [D, Lambda_c] = build_block_mf_matrices(gf_struct)

        U = 1
        h_loc = hubb_N(tk, U, orb_names, spin_names)

        emb_solver = EmbeddingEd(h_loc, gf_struct)

        eprint("U =", U, "tk =", tk, "tg =", tg)

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
                eig, vec = sc.get_h_qp2([R[block],R[block]], [Lambda[block],Lambda[block]], dispersion) #, mu)
                disp_R = sc.get_disp_R2([R[block],R[block]], dispersion, vec)
                
                # c++
                #h_qp = get_h_qp( [R[block],R[block]], [Lambda[block],Lambda[block]], dispersion )
                #disp_R = get_disp_R([R[block],R[block]], dispersion, h_qp)[0:3,:,:] # project onto one of the triangles in the unit cell
                #eig = h_qp.val
                #vec = h_qp.vec[0:3,:,:]
                #vec_dag = h_qp.vec_dag[:,0:3,:]

                kintegrator.setEks(np.transpose(eig)) # python
                #kintegrator.setEks(eig) # c++
                kintegrator.setEF_fromFilling(N_elec / len(block_names)) # divide by number of blocks
                mu = kintegrator.getEF
                wks = np.transpose(kintegrator.getWs); #python
                #wks = kintegrator.getWs; #c++

                #wks = fermi_fnc(eig, beta) / nk
                
                # python
                pdensity = sc.get_pdensity(vec[:,0:3,:], wks) # Project onto one of the triangles in the unit cell
                ke = np.real( sc.get_ke(disp_R[:,0:3,:], vec[:,0:3,:], wks) )
                
                # c++
                #pdensity = get_pdensity(vec, vec_dag, wks)
                #ke = np.real( get_ke(disp_R, vec_dag, wks) )

                # Set non-diagonal elements to zero
                pdensity = np.diag(np.diag(pdensity))
                ke = np.diag(np.diag(ke))

                D[block] = sc.get_d(pdensity, ke)
                Lambda_c[block] = sc.get_lambda_c(pdensity, R[block], Lambda[block], D[block])

            Lambda_c['dn'] = Lambda_c['up']
            D['dn'] = D['up']

            #if cycle < 100:
            #    emb_solver.set_h_emb(Lambda_c, D)
            #    psiS_size = emb_solver.get_psiS_size()  # 136: for insulating solutions the number of Lanczos vectors has to be high
            #    ec = emb_solver.solve(ncv = min(30, psiS_size), max_iter = 10000, tolerance = 0) # this seems to be high enough
            #    # For metallic 4 is enough. For points near the critical point maybe it isn't. Have to play around.
            
            #else:
            # this will set h_emb, solve, and find the correct mu
            impurity_fit(N_elec / 2., emb_solver, Lambda_c, D, block_names) # 2 impurities per unit cell

            #for b, block in enumerate(block_names):
            for b, block in enumerate(['up']):
                Nf = emb_solver.get_nf(block)
                Mcf = emb_solver.get_mcf(block)
                Nc = emb_solver.get_nc(block)
                
                # Set non-diagonal elements to zero
                Nf = np.diag(np.diag(Nf))
                Mcf = np.diag(np.diag(Mcf))
                Nc = np.diag(np.diag(Nc))
                
                Lambda[block] = sc.get_lambda(R[block], D[block], Lambda_c[block], Nf)
                R[block] = sc.get_r(Mcf, Nf)
                
                # Set non-diagonal elements to zero
                Lambda[block] = np.diag(np.diag(Lambda[block]))
                R[block] = np.diag(np.diag(R[block]))
                
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
        eprint("N_elec =", 2. * 2. * np.trace(Nc)) # two impurities, for each spin

        NOp = N_op(spin_names,orb_names,off_diag=True)
        S2Op = S2_op(spin_names,orb_names,off_diag=True)
        N = emb_solver.overlap(NOp)
        S2 = emb_solver.overlap(S2Op)

        Z = dict()
        for block in block_names:
            Z[block] = np.dot(R[block], R[block])
        eprint("cycles =", cycle, "norm =", norm)
        eprint("Z =", Z)
        eprint("Lambda =", Lambda)
        eprint("mu =", mu)
        eprint("N =", N )
        eprint("S2 =", S2)
        eprint("S =", np.sqrt(S2 + 0.25) - 0.5 )

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
