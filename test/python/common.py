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
from scipy.optimize import brentq
from copy import deepcopy
import risb.sc_cycle as sc

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def are_close(a, b, tol=1e-10):
  return abs(a-b)<tol

def set_approx_zero(A, tol=1e-10):
    idx = np.abs(A) < tol
    A[idx] = 0

def fermi_fnc(eks, beta=10, mu=0):
    e = eks - mu
    return np.exp(-beta*e*(e > 0))/(1 + np.exp(-beta*np.abs(e)))

def symmetrize(A, block_names):
    A_sym = 0
    for b in block_names:
        A_sym += A[b] / len(A)
    for b in block_names:
        A[b] = A_sym

def build_cubic_h0_k(gf_struct=[('up',1),('dn',1)], nkx=6, spatial_dim=2, t=1):
    t_scaled = -t / float(spatial_dim)
    for _, bsize in gf_struct:
        n_orb = bsize
    for _, bsize in gf_struct:
        if bsize != n_orb:
            raise ValueError('Each block must have the same number of orbitals !')
        
    orbital_positions=[(0,0,0)]*n_orb

    # Cubic lattice
    units = np.eye(spatial_dim)
    
    hoppings = {}
    for i in range(spatial_dim):
        hoppings[ tuple((units[:,i]).astype(int)) ] = np.eye(n_orb) * t_scaled
        hoppings[ tuple((-units[:,i]).astype(int)) ] = np.eye(n_orb) * t_scaled
    tbl = TBLattice(units=units, hoppings=hoppings, orbital_positions=orbital_positions)

    bl = BravaisLattice(units=units)
    bz = BrillouinZone(bl)
    mk = MeshBrZone(bz, nkx)

    h0_k = BlockGf(mesh=mk, gf_struct=gf_struct)
    for bl, _ in gf_struct:
        h0_k[bl] << tbl.fourier(mk)

    # Take it out of Gf structure to just get values
    h0_out = dict()
    for bl, _ in gf_struct:
        h0_out[bl] = h0_k[bl].data

    return h0_out
    
def build_mf_matrices(orb_dim = 1):
    R = np.zeros([orb_dim, orb_dim])
    np.fill_diagonal(R, 1.)
    Lambda = np.zeros([orb_dim, orb_dim])
    return (R, Lambda)

def build_block_mf_matrices(gf_struct=[('up',1),('dn',1)]):
    R = dict()
    Lambda = dict()
    for bl, bsize in gf_struct:
        R[bl] = np.zeros((bsize,bsize))
        Lambda[bl] = np.zeros((bsize,bsize))
        np.fill_diagonal(R[bl], 1)
    return (R,Lambda)

def build_fops_local(orb_dim = 1):
    orb_names = list(range(1,orb_dim+1))
    spin_names = ['up','dn']
    fops = [(s,o) for s,o in product(spin_names,orb_names)]
    return (fops, orb_names, spin_names)

def update_mu(energies, N_target, beta, n_k):
    e_min = np.inf
    e_max = -np.inf
    for en in energies.values():
        bl_min = en.min()
        bl_max = en.max()
        if bl_min < e_min:
            e_min = bl_min
        if bl_max > e_max:
            e_max = bl_max
            
    def target_function(mu):
        N = 0
        for en in energies.values():
            N += np.sum(fermi_fnc(en, beta, mu)) / n_k
        return N - N_target
    mu = brentq(target_function, e_min, e_max)
    return mu

def inner_cycle(emb_solver, 
                h0_k,
                h_loc,
                R, 
                Lambda, 
                block_names=['up','dn'],
                beta=10,
                mu=0,
                symmetrize_fnc=symmetrize,
                fixed='mu',
                N_target=None):
        
        n_k = h0_k['up'].shape[0]
        
        if symmetrize_fnc is not None:
            symmetrize_fnc(R, block_names)
            symmetrize_fnc(Lambda, block_names)

        norm = 0
        R_old = deepcopy(R)
        Lambda_old = deepcopy(Lambda)

        pdensity = dict()
        ke = dict()
        D = dict()
        Lambda_c = dict()
        eig = dict()
        vec = dict()
        for bl in block_names:
            eig[bl], vec[bl] = sc.get_h_qp(R[bl], Lambda[bl], h0_k[bl].data)
            
        if fixed == 'density':
            mu = update_mu(eig, N_target, beta, n_k)

        for bl in block_names:
            h0_R = sc.get_h0_R(R[bl], h0_k[bl], vec[bl])
            wks = fermi_fnc(eig[bl], beta, mu) / n_k

            pdensity[bl] = sc.get_pdensity(vec[bl], wks)
            ke[bl] = sc.get_ke(h0_R, vec[bl], wks)
        
            D[bl] = sc.get_d(pdensity[bl], ke[bl]).real # FIXME
            Lambda_c[bl] = sc.get_lambda_c(pdensity[bl], R[bl], Lambda[bl], D[bl])
        
        if symmetrize_fnc is not None:
            symmetrize_fnc(D, block_names)
            symmetrize_fnc(Lambda_c, block_names)

        emb_solver.set_h_emb(h_loc, Lambda_c, D)
        emb_solver.solve()

        Nf = dict()
        Mcf = dict()
        for bl in block_names:
            Nf[bl] = emb_solver.get_nf(bl)
            Mcf[bl] = emb_solver.get_mcf(bl)
        
        if symmetrize_fnc is not None:
            symmetrize_fnc(Nf, block_names)
            symmetrize_fnc(Mcf, block_names)

        for bl in block_names:
            Lambda[bl] = sc.get_lambda(R[bl], D[bl], Lambda_c[bl], Nf[bl])
            R[bl] = sc.get_r(Mcf[bl], Nf[bl])

            norm += np.linalg.norm(R[bl] - R_old[bl])
            norm += np.linalg.norm(Lambda[bl] - Lambda_old[bl])

        return Lambda, R, norm

