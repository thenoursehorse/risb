from __future__ import print_function
import sys
import unittest
import numpy as np
from itertools import product
from triqs.gf import *
from triqs.operators import *
from triqs.utility.comparison_tests import *
from triqs.lattice.tight_binding import *
from h5 import *
from copy import deepcopy
import risb.sc_cycle as sc

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def are_close(a,b,tol=1e-10):
  return abs(a-b)<tol

def set_approx_zero(A, tol=1e-10):
    idx = np.abs(A) < tol
    A[idx] = 0

def fermi_fnc(eks, beta = 10, mu = 0):
    e = eks - mu
    return np.exp(-beta*e*(e > 0))/(1 + np.exp(-beta*np.abs(e)))

def symmetrize(A,block_names):
    A_sym = 0
    for b in block_names:
        A_sym += A[b] / len(A)
    for b in block_names:
        A[b] = A_sym

def build_cubic_h0_k(gf_struct=[('up',1),('dn',1)], nkx=6, spatial_dim=2):
    t = - 0.5 / float(spatial_dim)
    for _,bsize in gf_struct:
        orbitals_dim = bsize
    for _,bsize in gf_struct:
        if bsize != orbitals_dim:
            raise ValueError('Each block must have the same number of orbitals !')
        
    orbital_positions=[(0,0,0)]*orbitals_dim

    # Cubic lattice
    units = np.eye(spatial_dim)
    
    hoppings = {}
    for i in range(spatial_dim):
        hoppings[ tuple((units[:,i]).astype(int)) ] = np.eye(orbitals_dim) * t
        hoppings[ tuple((-units[:,i]).astype(int)) ] = np.eye(orbitals_dim) * t
    tbl = TBLattice(units=units, hoppings=hoppings, orbital_positions=orbital_positions)

    bl = BravaisLattice(units=units)
    bz = BrillouinZone(bl)
    mk = MeshBrZone(bz, nkx)

    h0_k = BlockGf(mesh=mk, gf_struct=gf_struct)
    for block,_ in gf_struct:
        h0_k[block] << tbl.fourier(mk)

    # Take it out of Gf structure to just get values
    h0_out = dict()
    for block,_ in gf_struct:
        h0_out[block] = h0_k[block].data

    return h0_out
    
def build_mf_matrices(orb_dim = 1):
    R = np.zeros([orb_dim, orb_dim])
    np.fill_diagonal(R, 1.)
    Lambda = np.zeros([orb_dim, orb_dim])
    return (R, Lambda)

def build_block_mf_matrices(gf_struct=[('up',1),('dn',1)]):
    R = dict()
    Lambda = dict()
    for bname,bsize in gf_struct:
        R[bname] = np.zeros((bsize,bsize))
        Lambda[bname] = np.zeros((bsize,bsize))
        np.fill_diagonal(R[bname], 1)
    return (R,Lambda)

def build_fops_local(orb_dim = 1):
    orb_names = list(range(1,orb_dim+1))
    spin_names = ['up','dn']
    fops = [(s,o) for s,o in product(spin_names,orb_names)]
    return (fops, orb_names, spin_names)

def inner_cycle(emb_solver, 
                h0_k,
                h_loc,
                R, 
                Lambda, 
                block_names=['up','dn'],
                beta=10,
                mu=0):
        
        nk = h0_k['up'].shape[0]
        
        symmetrize(R, block_names)
        symmetrize(Lambda, block_names)

        norm = 0
        R_old = deepcopy(R)
        Lambda_old = deepcopy(Lambda)

        pdensity = dict()
        ke = dict()
        D = dict()
        Lambda_c = dict()
        for b in block_names:
            eig, vec = sc.get_h_qp(R[b], Lambda[b], h0_k[b].data)
            h0_R = sc.get_h0_R(R[b], h0_k[b], vec)
            wks = fermi_fnc(eig, beta, mu) / nk

            pdensity[b] = sc.get_pdensity(vec, wks)
            ke[b] = sc.get_ke(h0_R, vec, wks)
        
            D[b] = sc.get_d(pdensity[b], ke[b]).real # FIXME
            Lambda_c[b] = sc.get_lambda_c(pdensity[b], R[b], Lambda[b], D[b])
        
        symmetrize(D, block_names)
        symmetrize(Lambda_c, block_names)

        emb_solver.set_h_emb(h_loc, Lambda_c, D)
        emb_solver.solve()

        Nf = dict()
        Mcf = dict()
        for b in block_names:
            Nf[b] = emb_solver.get_nf(b)
            Mcf[b] = emb_solver.get_mcf(b)
        
        symmetrize(Nf, block_names)
        symmetrize(Mcf, block_names)

        for b in block_names:
            Lambda[b] = sc.get_lambda(R[b], D[b], Lambda_c[b], Nf[b])
            R[b] = sc.get_r(Mcf[b], Nf[b])

            norm += np.linalg.norm(R[b] - R_old[b])
            norm += np.linalg.norm(Lambda[b] - Lambda_old[b])

        return Lambda, R, norm

