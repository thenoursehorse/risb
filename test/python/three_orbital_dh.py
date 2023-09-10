#!/usr/bin/env python

import numpy as np
from itertools import product
import unittest
from common import symmetrize_blocks
from triqs.operators import Operator, c_dag, c
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding_atom_diag import EmbeddingAtomDiag

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
        n_cycles = 5
        n_target = 8

        tk = 1.0
        tg = 0.5
        U = 3
        
        spin_names = ['up','dn']
        block_names = spin_names
        gf_struct = set_operator_structure(spin_names, n_orbs, off_diag=True)

        h0_k = build_dh_h0_k(tg, nkx, block_names)
        h_loc = hubb_N(tk, U, spin_names)
        
        emb_solver = EmbeddingAtomDiag(h_loc, gf_struct)
        kweight_solver = SmearingKWeight(beta=beta, n_target=n_target)
        S = LatticeSolver(h0_k=h0_k,
                          gf_struct=gf_struct,
                          emb_solver=emb_solver,
                          kweight_solver=kweight_solver,
                          symmetries=[symmetrize_blocks])

        # First guess for Lambda is the quadratic terms in h_loc
        for bl in block_names:
            S.Lambda[bl] = np.array([[-2,0,0],[0,1,0],[0,0,1]])
        
        # Set up projectors onto a triangle
        P = np.zeros(shape=(3,6))
        P[0,0] = 1
        P[1,1] = 1
        P[2,2] = 1
        print(P)
        
        S.solve(n_cycles=n_cycles)
        
        #eig[bl], vec[bl] = sc.get_h_qp2([R[bl],R[bl]], [Lambda[bl],Lambda[bl]], h0_k[bl])
        #disp_R = sc.get_h0_R2([R[bl],R[bl]], h0_k[bl], vec[bl])
        #wks = fermi_fnc(eig[bl], beta, mu) / n_k
        #vec[bl] = np.einsum('ij,kjl->kil', P, vec[bl])
        #disp_R = np.einsum('ij,kjl->kil', P, disp_R)
        ##pdensity = sc.get_pdensity(vec[bl][:,0:3,:], wks) # Project onto one of the triangles in the unit cell
        ##ke = np.real( sc.get_ke(disp_R[:,0:3,:], vec[bl][:,0:3,:], wks) )
        
        with np.printoptions(suppress=True, precision=12):
            print("mu:", kweight_solver.mu)
            print("Lambda:", S.Lambda)
            print("Z:", S.Z)
            print("Nf =", S.Nf)
            print("Mcf =", S.Mcf)
            print("Nc =", S.Nc)
        print(null)
                
        N = N_op(spin_names, n_orbs, off_diag=True)
        S2 = S2_op(spin_names, n_orbs, off_diag=True)
        S2_avg = emb_solver.overlap(S2)

        print("Z =", S.Z)
        print("Lambda =", S.Lambda)
        print("mu =", kweight_solver.mu)
        print("N =", emb_solver.overlap(N))
        print("S2 =", S2_avg)
        print("S =", np.sqrt(S2_avg + 0.25) - 0.5 )

        print(null)

if __name__ == '__main__':
    unittest.main()
