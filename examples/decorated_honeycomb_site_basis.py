# ruff: noqa: D100, D103
import logging
from itertools import product

import numpy as np
from fockit.algebra import nc
from numpy.typing import NDArray

from risb import LatticeSolver
from risb.helpers import block_to_full, get_h0_kin_k
from risb.helpers_fockit import get_h0_loc
from risb.impurity_solvers import DummyImpuritySolver, FockitImpuritySolver
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix, ImpuritySolver

logging.basicConfig(level=logging.INFO, format="%(message)s")
# Set to logging.INFO to see each iteration of the solver
logging.getLogger("risb").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_h0_k(
    tg: float = 0.5,
    tk: float = 1.0,
    nkx: int = 18,
    spin_names: list[str] | None = None,
) -> dict[str, NDArray[np.inexact]]:
    if spin_names is None:
        spin_names = ["up", "dn"]
    na = 2  # Break up unit cell into 2 clusters
    n_orb = 3  # Number of orbitals/sites per cluster
    n_k = nkx**2  # total number of k-points

    # Build Gamma-centred 2D mesh
    mesh = np.empty(shape=(n_k, 2))
    for idx, coords in enumerate(product(range(nkx), range(nkx))):
        mesh[idx, 0] = coords[0] / nkx
        mesh[idx, 1] = coords[1] / nkx

    # Unit cell lattice vectors
    R1 = (3.0 / 2.0, np.sqrt(3.0) / 2.0)
    R2 = (3.0 / 2.0, -np.sqrt(3.0) / 2.0)
    R = np.array((R1, R2)).T

    # Bravais lattice vectors
    G = 2.0 * np.pi * np.linalg.inv(R).T

    # Vectors to inter-triangle nearest neighbors
    d0 = (1.0, 0.0)
    d1 = (-0.5, np.sqrt(3.0) / 2.0)
    d2 = (-0.5, -np.sqrt(3.0) / 2.0)
    d_vec = [d0, d1, d2]

    h0_k = np.zeros([n_k, na, na, n_orb, n_orb], dtype=complex)

    # Construct in inequivalent block matrix structure
    for k, a in product(range(n_k), range(n_orb)):
        kay = np.dot(G, mesh[k, :])

        # Site a of one triangle hops to site a of the other along d_a
        h0_k[k, 0, 1, a, a] = -tg * np.exp(1j * kay @ d_vec[a])
        h0_k[k, 1, 0, a, a] = -tg * np.exp(-1j * kay @ d_vec[a])

        # Hopping between the sites of a triangle
        for i, b in product(range(na), range(n_orb)):
            if b != a:
                h0_k[k, i, i, a, b] = -tk

    # Get rid of the inequivalent block structure
    h0_k_out: dict[str, NDArray[np.inexact]] = {}
    for bl in spin_names:
        h0_k_out[bl] = block_to_full(h0_k)
    return h0_k_out


# Setup problem and gf_struct for each inequivalent trimer cluster
n_clusters = 2
n_orb = 3
spin_names = ["up", "dn"]

# Setup non-interacting Hamiltonian matrix on the lattice
tg = 0.5
nkx = 18
h0_k = get_h0_k(tg=tg, nkx=nkx, spin_names=spin_names)

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 8  # 2/3rds filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)

# Set up gf_structure of clusters: the three sites of a triangle in each block
gf_struct = [[(bl, n_orb) for bl in spin_names] for _ in range(n_clusters)]

# Make projectors onto each trimer cluster
projectors: list[BlockMatrix] = [
    {
        bl: np.eye(n_clusters * n_orb)[i * n_orb : (i + 1) * n_orb, :]
        for bl in spin_names
    }
    for i in range(n_clusters)
]

# Get the non-interacting kinetic Hamiltonian matrix on the lattice
h0_kin_k = get_h0_kin_k(h0_k, projectors)

# Get the non-interacting local operator terms
h0_loc = [get_h0_loc(h0_k=h0_k, P=P) for P in projectors]

# Get the local interaction operator terms
U = 4
h_int = [
    U * sum(nc("up", a) * nc("dn", a) for a in range(n_orb)) for _ in range(n_clusters)
]

# Define the local Hamiltonian
h_loc = [h0_loc[i] + h_int[i] for i in range(n_clusters)]

# The trimer is symmetric under a C3 rotation. In the basis of the sites the
# rotation relabels site a as site a + 1, on every physical and bath orbital
# ("bath_" + block), so it is a permutation. The solver finds the lowest state
# with the given eigenvalue of the rotation, which here (1) contains the ground
# state.
c3_rotation = {
    (bl, a): (bl, (a + 1) % n_orb)
    for bl in spin_names + ["bath_" + bl for bl in spin_names]
    for a in range(n_orb)
}

# Set up impurity solvers
impurity_solvers: list[ImpuritySolver] = [
    FockitImpuritySolver(
        h_loc[0], gf_struct[0], permutation_symmetries=[(c3_rotation, 1)]
    )
]
for _ in range(n_clusters - 1):
    impurity_solvers.append(DummyImpuritySolver(impurity_solvers[0]))

# The matrices that rotate the sites of a triangle by 0, 1 and 2 steps
rotations = [np.roll(np.eye(n_orb), n, axis=0) for n in range(n_orb)]


def symmetries(A: list[BlockMatrix]) -> list[BlockMatrix]:
    n_clusters = len(A)
    # Paramagnetic
    for i in range(n_clusters):
        A[i]["up"] = 0.5 * (A[i]["up"] + A[i]["dn"])
        A[i]["dn"] = A[i]["up"]
    # Symmetric under the C3 rotation. The impurity problem only has the
    # rotation if the matrices that couple to the bath do
    for i in range(n_clusters):
        for bl in spin_names:
            A[i][bl] = np.mean([P @ A[i][bl] @ P.T for P in rotations], axis=0)
    return A


# Setup RISB solver class. The matrices are real in the basis of the sites
# because of time-reversal symmetry
S = LatticeSolver(
    h0_k=h0_kin_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solvers,
    update_weights=kweight.update_weights,
    projectors=projectors,
    symmetrizers=[symmetries],
    force_real=True,
)

# Solve
S.solve(tol=1e-4)

# Print out some interesting observables
with np.printoptions(formatter={"float": "{: 0.4f}".format}):
    for i in range(S.n_clusters):
        logger.info(f"Cluster {i}:")
        for bl, Z in S.Z[i].items():
            logger.info(f"Quasiaprticle weight Z[{bl}] = \n{Z}")
        for bl, Lambda in S.Lambda[i].items():
            logger.info(f"Correlation potential Lambda[{bl}] = \n{Lambda}")
