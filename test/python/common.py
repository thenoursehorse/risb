#from __future__ import print_function
import numpy as np
from triqs.gf import *
from triqs.lattice.tight_binding import *
#from h5 import *

#def eprint(*args, **kwargs):
#    print(*args, file=sys.stderr, **kwargs)

def symmetrize_blocks(A):
    A_sym = 0
    for bl in A:
        A_sym += A[bl] / len(A)
    for bl in A:
        A[bl] = A_sym
    return A

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
    
#def build_mf_matrices(orb_dim = 1):
#    R = np.zeros([orb_dim, orb_dim])
#    np.fill_diagonal(R, 1.)
#    Lambda = np.zeros([orb_dim, orb_dim])
#    return (R, Lambda)

def build_block_mf_matrices(gf_struct=[('up',1),('dn',1)]):
    R = dict()
    Lambda = dict()
    for bl, bsize in gf_struct:
        R[bl] = np.zeros((bsize,bsize))
        Lambda[bl] = np.zeros((bsize,bsize))
        np.fill_diagonal(R[bl], 1)
    return (R,Lambda)

#def build_fops_local(orb_dim = 1):
#    orb_names = list(range(1,orb_dim+1))
#    spin_names = ['up','dn']
#    fops = [(s,o) for s,o in product(spin_names,orb_names)]
#    return (fops, orb_names, spin_names)