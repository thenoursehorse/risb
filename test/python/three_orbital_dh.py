#!/usr/bin/env python

from common import *
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from risb.embedding_atom_diag import *

def build_dh_h0_k(tg=0.5, nkx=18, block_names=['up','dn']):
    na = 2
    orb_dim = 3
    phi = 2.0 * np.pi / 3.0

    # Build shifted 2D mesh
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
        
    h0_k = np.zeros([mesh_num,na,na,orb_dim,orb_dim],dtype=complex)

    for k,i,j,m,mm in product(range(mesh_num),range(na),range(na),range(orb_dim),range(orb_dim)):
        kay = np.dot(G.T, mesh[k,:])
        if (i == 0) and (j == 1):
            h0_k[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(1j * np.dot(kay,d0)) 
                               + np.exp(1j * np.dot(kay,d1)) * np.exp(1j * phi * (mm-m)) 
                               + np.exp(1j * (np.dot(kay,d2)))*np.exp(1j * 2.0 * phi * (mm-m)) )
        elif (i == 1) and (j == 0):
            h0_k[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(-1j * np.dot(kay,d0)) 
                               + np.exp(-1j * np.dot(kay,d1)) * np.exp(-1j *phi * (m-mm)) 
                               + np.exp(-1j * np.dot(kay,d2)) * np.exp(-1j * 2.0 * phi * (m-mm)) )
        else:
            h0_k[k,i,j,m,mm] = 0

    h0_k_out = dict()
    for bl in block_names:
        h0_k_out[bl] = h0_k
    return h0_k_out

def hubb_N(tk, U, spin_names):
    n_orbs = 3
    phi = 2.0 * np.pi / n_orbs
    h_loc = Operator()

    spin_up = spin_names[0]
    spin_dn = spin_names[1]

    for a,m,mm,s in product(range(n_orbs), range(n_orbs), range(n_orbs), spin_names):
        h_loc += (-tk / float(n_orbs)) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * a * m) * np.exp(1j * phi * np.mod(a+1,n_orbs) * mm)
        h_loc += (-tk / float(n_orbs)) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * np.mod(a+1,n_orbs) * m) * np.exp(1j * phi * a * mm)
    
    for m,mm,mmm in product(range(n_orbs), range(n_orbs), range(n_orbs)):
        h_loc += (U / float(n_orbs)) * c_dag(spin_up,m) * c(spin_up,mm) * c_dag(spin_dn,mmm) * c(spin_dn,np.mod(m+mmm-mm,n_orbs))
    
    return h_loc.real

class tests(unittest.TestCase):
 
    def test_dh_two_third(self):
        n_orbs = 3
        nkx = 18
        beta = 40
        num_cycles = 5

        tk = 1.0
        tg = 0.5
        U = 3
        fixed = 'density'
        N_target = 8
        
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names, n_orbs, off_diag=True)
        
        emb_solver = EmbeddingAtomDiag(gf_struct)

        h0_k = build_dh_h0_k(tg, nkx, block_names)

        [R, Lambda] = build_block_mf_matrices(gf_struct)

        h_loc = hubb_N(tk, U, spin_names)

        # First guess for Lambda is the quadratic terms in h_loc
        for bl in block_names:
            Lambda[bl] = np.array([[-2,0,0],[0,1,0],[0,0,1]])

        pdensity = deepcopy(R)
        Lambda_c = deepcopy(R)
        D = deepcopy(R)

        P = np.zeros(shape=(6,3))
        P[0,0] = 1
        P[1,1] = 1
        P[2,2] = 1
        print(P)
        
        n_k = h0_k['up'].shape[0]
        for cycle in range(num_cycles):

            norm = 0
            R_old = deepcopy(R)
            Lambda_old = deepcopy(Lambda)

            eig = dict()
            vec = dict()
            for bl in block_names:
                eig[bl], vec[bl] = sc.get_h_qp2([R[bl],R[bl]], [Lambda[bl],Lambda[bl]], h0_k[bl])
  
            if fixed == 'density':
                mu = update_mu(eig, N_target, beta, n_k)

            for bl in ['up']:
                disp_R = sc.get_h0_R2([R[bl],R[bl]], h0_k[bl], vec[bl])
                wks = fermi_fnc(eig[bl], beta, mu) / n_k

                vec[bl] = np.einsum('ij,kjl->kil', P.conj().T, vec[bl])
                disp_R = np.einsum('ij,kjl->kil', P.conj().T, disp_R)
                pdensity = sc.get_pdensity(vec[bl], wks) # Project onto one of the triangles in the unit cell
                ke = np.real( sc.get_ke(disp_R, vec[bl], wks) )
                
                #pdensity = sc.get_pdensity(vec[bl][:,0:3,:], wks) # Project onto one of the triangles in the unit cell
                #ke = np.real( sc.get_ke(disp_R[:,0:3,:], vec[bl][:,0:3,:], wks) )
                
                # Set non-diagonal elements to zero
                pdensity = np.diag(np.diag(pdensity))
                ke = np.diag(np.diag(ke))

                D[bl] = sc.get_d(pdensity, ke)
                Lambda_c[bl] = sc.get_lambda_c(pdensity, R[bl], Lambda[bl], D[bl])

            Lambda_c['dn'] = Lambda_c['up']
            D['dn'] = D['up']
                
            emb_solver.set_h_emb(h_loc, Lambda_c, D)
            emb_solver.solve()
        
            for bl in ['up']:
                Nf = emb_solver.get_nf(bl)
                Mcf = emb_solver.get_mcf(bl)
                Nc = emb_solver.get_nc(bl)
                
                # Set non-diagonal elements to zero
                Nf = np.diag(np.diag(Nf))
                Mcf = np.diag(np.diag(Mcf))
                Nc = np.diag(np.diag(Nc))
        
                Lambda[bl] = sc.get_lambda(R[bl], D[bl], Lambda_c[bl], Nf)
                R[bl] = sc.get_r(Mcf, Nf)
                
                # Set non-diagonal elements to zero
                Lambda[bl] = np.diag(np.diag(Lambda[bl]))
                R[bl] = np.diag(np.diag(R[bl]))
                
                norm += np.linalg.norm(R[bl] - R_old[bl])
                norm += np.linalg.norm(Lambda[bl] - Lambda_old[bl])

            Lambda['dn'] = Lambda['up']
            R['dn'] = R['up']

            if norm < 1e-6:
                break
            

        eprint("Nf =", Nf)
        eprint("Mcf =", Mcf)
        eprint("Nc =", Nc)

        N = N_op(spin_names, n_orbs, off_diag=True)
        S2 = S2_op(spin_names, n_orbs, off_diag=True)
        S2_avg = emb_solver.overlap(S2)

        Z = dict()
        for bl in block_names:
            Z[bl] = np.dot(R[bl], R[bl])
        eprint("cycles =", cycle, "norm =", norm)
        eprint("Z =", Z)
        eprint("Lambda =", Lambda)
        eprint("mu =", mu)
        eprint("N =", emb_solver.overlap(N))
        eprint("S2 =", S2_avg)
        eprint("S =", np.sqrt(S2_avg + 0.25) - 0.5 )

        print(null)

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
