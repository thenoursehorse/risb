from __future__ import print_function
import sys
import unittest
import numpy as np
from itertools import product
from triqs.gf import *
from triqs.operators import *
from triqs.utility.comparison_tests import *
from triqs.lattice.tight_binding import *
#from triqs.arrays.block_matrix import *
from h5 import *
from copy import deepcopy
from risb.functions import *
import risb.sc_cycle as sc

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def are_close(a,b,tol=1e-10):
  return abs(a-b)<tol

def set_approx_zero(A, tol=1e-10):
    idx = np.abs(A) < tol
    A[idx] = 0

def fermi_fnc(eks, beta = 10, mu = 0):
    return 1. / (np.exp(beta * (eks - mu)) + 1)

def symmetrize(A,block_names):
    A_sym = 0
    for b in block_names:
        A_sym += A[b] / len(A)
    for b in block_names:
        A[b] = A_sym
    #return A

def build_cubic_dispersion(nkx = 6, orb_dim = 1, spatial_dim = 2, return_bl = False):
    na = 1
    t = - 0.5 / float(spatial_dim)

    # Cubic lattice
    units = np.eye(spatial_dim)
    bl = BravaisLattice(units = units, orbital_positions= [ (0,0) ] ) # only do one orbital because all will be the same
    
    hop = {}
    for i in range(spatial_dim):
        hop[ tuple((units[:,i]).astype(int)) ] = [[t]]
        hop[ tuple((-units[:,i]).astype(int)) ] = [[t]]
    
    tb = TightBinding(bl, hop)
    
    energies = energies_on_bz_grid(tb, nkx)
    nk = energies.shape[1]

    di = np.diag_indices(orb_dim)
    dispersion = np.zeros([orb_dim, orb_dim, nk])
    dispersion[di[0],di[1],:] = energies[None,:]
    
    if return_bl:
        return dispersion, bl
    else:
        return dispersion

def build_mf_matrices(orb_dim = 1):
    R = np.zeros([orb_dim, orb_dim])
    np.fill_diagonal(R, 1.)
    Lambda = np.zeros([orb_dim, orb_dim])
    return (R, Lambda)

def build_block_mf_matrices(gf_struct = [ ["up",[1]], ["dn",[1]] ]):
    R = dict()
    Lambda = dict()
    for block,ind in gf_struct:
        R[block] = np.zeros((len(ind),len(ind)))
        Lambda[block] = np.zeros((len(ind),len(ind)))
        np.fill_diagonal(R[block], 1)

    #R_block = []
    #Lambda_block = []
    #names = []
    #for block in gf_struct:
    #    dim = len(block[1])
    #    names.append(block[0])
    #    
    #    R1 = np.zeros([dim,dim])
    #    np.fill_diagonal(R1,1)
    #    R_block.append(R1)
    #    
    #    Lambda_block.append(np.zeros([dim,dim]))
    #
    #R = BlockMatrix(names,R_block)
    #Lambda = BlockMatrix(names,Lambda_block)
    return (R,Lambda)

def build_fops_local(orb_dim = 1):
    orb_names = list(range(1,orb_dim+1))
    spin_names = ['up','dn']
    fops = [(s,o) for s,o in product(spin_names,orb_names)]
    return (fops, orb_names, spin_names)

