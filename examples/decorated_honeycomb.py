import numpy as np
from itertools import product

from triqs.operators import Operator, c_dag, c
from triqs.operators.util.op_struct import set_operator_structure
from triqs.operators.util.observables import S2_op
from triqs.operators.util.observables import N_op

from risb import LatticeSolver
from risb.kweight import SmearingKWeight
from risb.embedding import EmbeddingAtomDiag
from risb.helpers import get_h0_kin_k, block_to_full
from risb.helpers_triqs import get_h0_loc

def get_h0_k(tg=0.5, tk=1.0, nkx=18, spin_names=['up','dn']):
    na = 2 # Break up unit cell into 2 clusters
    n_orb = 3 # Number of orbitals/sites per cluster
    phi = 2.0 * np.pi / 3.0 # bloch factor for transforming to trimer orbital basis
    n_k = nkx**2 # total number of k-points

    # Build shifted 2D mesh
    mesh = np.empty(shape=(n_k, 2))
    for idx,coords in enumerate(zip(range(nkx), range(nkx))):
        mesh[idx,0] = coords[0]/nkx + 0.5/nkx
        mesh[idx,1] = coords[1]/nkx + 0.5/nkx

    # Unit cell lattice vectors
    R1 = ( 3.0/2.0, np.sqrt(3.0)/2.0)
    R2 = ( 3.0/2.0, -np.sqrt(3.0)/2.0)
    R = np.array((R1, R2)).T

    # Bravais lattice vectors
    G = 2.0*np.pi*np.linalg.inv(R)

    # Vectors to inter-triangle nearest neighbors
    d0 = ( 1.0, 0.0 )
    d1 = ( -0.5, np.sqrt(3.0)/2.0 )
    d2 = ( -0.5, -np.sqrt(3.0)/2.0 )
        
    h0_k = np.zeros([n_k, na, na, n_orb, n_orb], dtype=complex)

    # Construct in inequivalent block matrix structure  
    for k,i,j,m,mm in product(range(n_k), range(na), range(na), range(n_orb), range(n_orb)):
        kay = np.dot(G.T, mesh[k,:])

        # Dispersion terms between clusters
        if (i == 0) and (j == 1):
            h0_k[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(1j * np.dot(kay,d0)) 
                               + np.exp(1j * np.dot(kay,d1)) * np.exp(1j * phi * (mm-m)) 
                               + np.exp(1j * (np.dot(kay,d2)))*np.exp(1j * 2.0 * phi * (mm-m)) )
        elif (i == 1) and (j == 0):
            h0_k[k,i,j,m,mm] = -(tg/3.0) * ( np.exp(-1j * np.dot(kay,d0)) 
                               + np.exp(-1j * np.dot(kay,d1)) * np.exp(-1j *phi * (m-mm)) 
                               + np.exp(-1j * np.dot(kay,d2)) * np.exp(-1j * 2.0 * phi * (m-mm)) )
            
        # Local terms on a cluster
        elif (i == j) and (m == mm):
            if m == 0:
                h0_k[k,i,j,m,mm] = - 2.0 * tk
            else:
                h0_k[k,i,j,m,mm] = tk

        else:
            continue

    # Get rid of the inequivalent block structure
    h0_k_out = dict()
    for bl in spin_names:
        h0_k_out[bl] = block_to_full( h0_k )
    return h0_k_out

def get_hubb_N(spin_names, U=0, tk=0, n_orb=3):
    phi = 2.0 * np.pi / n_orb # bloch factor for transforming to molecular orbital basis

    spin_up = spin_names[0]
    spin_dn = spin_names[1]
    
    h_loc = Operator()
    for a,m,mm,s in product(range(n_orb), range(n_orb), range(n_orb), spin_names):
        h_loc += (-tk / float(n_orb)) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * a * m) * np.exp(1j * phi * np.mod(a+1,n_orb) * mm)
        h_loc += (-tk / float(n_orb)) * c_dag(s,m) * c(s,mm) * np.exp(-1j * phi * np.mod(a+1,n_orb) * m) * np.exp(1j * phi * a * mm)
    
    for m,mm,mmm in product(range(n_orb), range(n_orb), range(n_orb)):
        h_loc += (U / float(n_orb)) * c_dag(spin_up,m) * c(spin_up,mm) * c_dag(spin_dn,mmm) * c(spin_dn,np.mod(m+mmm-mm,n_orb))
    
    return h_loc.real

# Setup problem and gf_struct for each inequivalent trimer cluster A, B
n_clusters = 2
n_orb = 3
spin_names = ['up','dn']
gf_struct = [set_operator_structure(spin_names, n_orb, off_diag=True) for _ in range(n_clusters)]

# Setup non-interacting Hamiltonian matrix on the lattice
tg = 0.5
nkx = 6
h0_k = get_h0_k(tg=tg, nkx=nkx, spin_names=spin_names)

# Make projectors onto each trimer cluster
projectors = [dict(), dict()]
for i in range(n_clusters):
    for bl_sub, bl_sub_size in gf_struct[i]:
        projectors[i][bl_sub] =  np.eye(n_clusters*n_orb)[i*bl_sub_size:(bl_sub_size+i*bl_sub_size),:]

# Get the non-interacting kinetic Hamiltonian matrix on the lattice
h0_kin_k = get_h0_kin_k(h0_k, projectors)

# Get the non-interacting local operator terms
h0_loc = [get_h0_loc(h0_k=h0_k, P=P) for P in projectors]

# Get the local interaction operator terms
U = 0
h_int = [get_hubb_N(spin_names=spin_names, U=U) for _ in range(n_clusters)]

# Define the local Hamiltonian
h_loc = [h0_loc[i] + h_int[i] for i in range(n_clusters)]

# Set up class to work out k-space integration weights
beta = 40 # inverse temperature
n_target = 8 # 2/3rds filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)

# Set up embedding solvers 
embedding = [EmbeddingAtomDiag(h_loc[i], gf_struct[i]) for i in range(n_clusters)]

# Setup RISB solver class  
#S = LatticeSolver(h0_k=h0_k,
#                  gf_struct=gf_struct,
#                  embedding=embedding,
#                  update_weights=kweight.update_weights)

# Solve
#S.solve(tol=1e-6)
 
# Average number of particles on a cluster
#NOp = N_op(spin_names, n_orb, off_diag=True)
#N = [e.overlap(NOp) for e in embedding]

# Effective total spin of a cluster
#S2Op = S2_op(spin_names, n_orb, off_diag=True)
#S2 = [e.overlap(S2Op) for e in embedding]