# ruff: noqa: T201, D100, D103
from itertools import product

import numpy as np
from triqs.operators import n
from triqs.operators.util.observables import N_op, S2_op
from triqs.operators.util.op_struct import set_operator_structure

from risb import LatticeSolver
from risb.embedding import EmbeddingAtomDiag, EmbeddingDummy
from risb.kweight import SmearingKWeight


def get_h0_k(t=1, nkx=18, spin_names=None):
    if spin_names is None:
        spin_names = ["up", "dn"]
    n_orb = 3

    # Build shifted 2D mesh
    n_k = nkx**2
    mesh = np.empty(shape=(n_k, 2))
    for idx, coords in enumerate(product(range(nkx), range(nkx))):
        mesh[idx, 0] = coords[0] / nkx + 0.5 / nkx
        mesh[idx, 1] = coords[1] / nkx + 0.5 / nkx

    # Unit cell lattice vectors
    R1 = (1.0, 0)
    R2 = (0.5, np.sqrt(3.0) / 2.0)
    R = np.array((R1, R2)).T

    # Bravais lattice vectors
    G = 2.0 * np.pi * np.linalg.inv(R).T

    h0_k = np.zeros([n_k, n_orb, n_orb], dtype=complex)
    for k in range(n_k):
        kay = np.dot(G, mesh[k, :])
        k1 = kay[0]
        k2 = 0.5 * kay[0] + 0.5 * np.sqrt(3) * kay[1]
        k3 = -0.5 * kay[0] + 0.5 * np.sqrt(3) * kay[1]

        h0_k[k][...] = np.array(
            [
                [0, -2 * t * np.cos(0.5 * k1), -2 * t * np.cos(0.5 * k2)],
                [-2 * t * np.cos(0.5 * k1), 0, -2 * t * np.cos(0.5 * k3)],
                [-2 * t * np.cos(0.5 * k2), -2 * t * np.cos(0.5 * k3), 0],
            ]
        )

    h0_k_out = {}
    for bl in spin_names:
        h0_k_out[bl] = h0_k
    return h0_k_out


# Setup problem and gf_struct. There are three inequivalent clusters with one
# orbital per cluster.
n_clusters = 3
n_orb = 1
spin_names = ["up", "dn"]
gf_struct = [
    set_operator_structure(spin_names, n_orb, off_diag=True) for _ in range(n_clusters)
]

# Setup non-interacting Hamiltonian matrix on the lattice
h0_k = get_h0_k(t=1, nkx=18, spin_names=spin_names)
# For testing because on mac it returns nans
for bl in h0_k:
    idx = np.where(np.isnan(h0_k[bl]))
    h0_k[bl][idx] = 0

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 3  # half-filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)

# Define the local interaction
U = 10
h_int = [U * n("up", 0) * n("dn", 0) for _ in range(n_clusters)]

# Set up embedding solvers
# Assuming all of the sites are equivalent it is sufficient to just solve for
# one site and copy and paste onto the other sites
embedding = [EmbeddingAtomDiag(h_int[0], gf_struct[0])]
embedding.append(EmbeddingDummy(embedding[0]))
embedding.append(EmbeddingDummy(embedding[0]))

# Setup projectors onto the three correlated spaces in a unit cell
# Dictionaries of projectors onto each cluster
P_A = {bl: np.array([[1, 0, 0]]) for bl in spin_names}
P_B = {bl: np.array([[0, 1, 0]]) for bl in spin_names}
P_C = {bl: np.array([[0, 0, 1]]) for bl in spin_names}
projectors = [P_A, P_B, P_C]


# Enforce paramagnetism. This is not really necessary because the primitive
# unit cell will not magnetically order. But it helps to stabalize the
# solver in the insulating state
def force_paramagnetic(A):
    n_clusters = len(A)
    for i in range(n_clusters):
        A[i]["up"] = 0.5 * (A[i]["up"] + A[i]["dn"])
        A[i]["dn"] = A[i]["up"]
    return A


# Setup RISB solver class
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    embedding=embedding,
    update_weights=kweight.update_weights,
    projectors=projectors,
    symmetries=[force_paramagnetic],
)

# Solve
S.solve(tol=1e-8)

# Average number of particles on a cluster
total_number_Op = N_op(spin_names, n_orb, off_diag=True)
total_number = [e.overlap(total_number_Op) for e in embedding]

# Effective total spin of a cluster
total_spin_Op = S2_op(spin_names, n_orb, off_diag=True)
total_spin = [e.overlap(total_spin_Op) for e in embedding]

# Print out some interesting observables
with np.printoptions(formatter={"float": "{: 0.4f}".format}):
    print("Observables on each cluster:")
    print(f"Quasiaprticle weight Z = \n{S.Z[0]['up']}")
    print(f"Correlation potential Lambda = \n{S.Lambda[0]['up']}")
    print(f"Density matrix rho_qp = \n{S.rho_qp[0]['up']}")
    print(f"Density matrix rho_f = \n{embedding[0].rho_f['up']}")
    print(f"Density matrix rho_c = \n{embedding[0].get_rho_c('up')}")
    print(f"Density matrix rho_cf = \n{embedding[0].rho_cf['up']}")
    print(f"Bath coupling matrix Lambda_c = \n{S.Lambda_c[0]['up']}")
    print(f"Hybridization matrix D = \n{S.D[0]['up']}")
    print(f"Number of partices on cluster N = \n{total_number[0] : 0.4f}")
    print(
        f"Effective spin of cluster S = \n{  -0.5 + 0.5 * np.sqrt( 1 + 4*np.array(total_spin[0]) ) : 0.4f}"
    )
