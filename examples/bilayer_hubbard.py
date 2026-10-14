# ruff: noqa: D100
import logging
from itertools import product

import numpy as np
from fockit.algebra import nc, zero

from risb import LatticeSolver
from risb.helpers_fockit import N_op, S2_op
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Set to logging.INFO to see each iteration of the solver
logging.getLogger("risb").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Number of orbitals and spin structure
n_orb = 2
spin_names = ["up", "dn"]
gf_struct = [(bl, n_orb) for bl in spin_names]

# Non-interacting cubic dispersion on a Gamma-centred k-mesh
nkx = 10  # nkx**3 total number of k-points
t = -1.0 / 3.0  # hopping amplitude
V = 0.25  # Bilayer hopping between orbitals on same site
k_mesh = 2 * np.pi * np.array(list(product(range(nkx), repeat=3))) / nkx
dispersion = 2 * t * np.sum(np.cos(k_mesh), axis=1)
h0_k = {
    bl: dispersion[:, None, None] * np.eye(n_orb) + np.array([[0, V], [V, 0]])
    for bl in spin_names
}

# Hubbard interactions
U = 4
h_int = zero()
for o in range(n_orb):
    h_int += U * nc("up", o) * nc("dn", o)

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 2  # half-filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)
# or fix mu = U/2 -> kweight = SmearingKWeight(beta=beta, mu=mu)

# (Optional) set up function to symmetrize mean-field matrices
# def symmetries(A):
#    n_clusters = len(A)
#    A_sym = [0 for i in range(n_clusters)]
#    for i in range(n_clusters):
#        for bl in A[i]:
#            A_sym[i] += A[i][bl] / len(A[i])
#        for bl in A[i]:
#            A[i][bl] = A_sym[i]
#    return A

# The impurity problem is symmetric under swapping the two layers, in every
# block including the bath orbitals ("bath_" + block). The solver finds the
# lowest state with the given eigenvalue of the swap, which here (+1) contains
# the ground state.
layer_swap = {
    (bl, o): (bl, 1 - o)
    for bl in spin_names + ["bath_" + bl for bl in spin_names]
    for o in range(n_orb)
}

# Set up the impurity solver
impurity_solver = FockitImpuritySolver(
    h_int, gf_struct, permutation_symmetries=[(layer_swap, 1)]
)

# Setup RISB solver class
# gf_struct and the impurity solvers are given for each cluster. In this case
# there is only one cluster, so they are passed directly instead of as lists.
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solver,
    update_weights=kweight.update_weights,
)
# symmetrizers=[symmetries])

# (Optional) Initialize R and Lambda matrices
# for bl, bl_size in gf_struct:
#    np.fill_diagonal(S.R[bl], 1)
#    np.fill_diagonal(S.Lambda[bl], 0)

# Solve
S.solve(tol=1e-6)

# Average number of particles on a cluster
total_number = impurity_solver.overlap(N_op(spin_names, n_orb))

# Effective total spin of a cluster
total_spin = impurity_solver.overlap(S2_op(spin_names, n_orb))

# Print out some interesting observables
# with np.printoptions(formatter={'float': '{: 0.4f}'.format}):
with np.printoptions(precision=4, suppress=True):
    for i in range(S.n_clusters):
        for bl, Z in S.Z[i].items():
            logger.info(f"Quasiaprticle weight Z[{bl}] = \n{Z}")
        for bl, Lambda in S.Lambda[i].items():
            logger.info(f"Correlation potential Lambda[{bl}] = \n{Lambda}")
        logger.info(f"Number of partices per cluster N = \n{total_number:0.4f}")
        logger.info(
            f"Effective spin of a cluster S = \n{-0.5 + 0.5 * np.sqrt(1 + 4 * total_spin): 4f}"
        )
