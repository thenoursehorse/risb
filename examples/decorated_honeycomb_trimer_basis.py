# ruff: noqa: D100, D103
import logging
from itertools import product

import numpy as np
from fockit.algebra import OpSum, c, cdag, zero
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
    phi = 2.0 * np.pi / 3.0  # bloch factor for transforming to trimer orbital basis
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
    for k, i, j, m, mm in product(
        range(n_k), range(na), range(na), range(n_orb), range(n_orb)
    ):
        kay = np.dot(G, mesh[k, :])

        # Dispersion terms between clusters
        if (i == 0) and (j == 1):
            for a in range(n_orb):
                h0_k[k, i, j, m, mm] += (
                    -(tg / 3.0)
                    * np.exp(1j * kay @ d_vec[a])
                    * np.exp(1j * phi * (mm - m) * a)
                )
        elif (i == 1) and (j == 0):
            for a in range(n_orb):
                h0_k[k, i, j, m, mm] += (
                    -(tg / 3.0)
                    * np.exp(-1j * kay @ d_vec[a])
                    * np.exp(-1j * phi * (m - mm) * a)
                )
        # Local terms on a cluster
        elif (i == j) and (m == mm):
            h0_k[k, i, j, m, mm] = -2.0 * tk * np.cos(m * phi)
        else:
            continue

    # Get rid of the inequivalent block structure
    h0_k_out: dict[str, NDArray[np.inexact]] = {}
    for bl in spin_names:
        h0_k_out[bl] = block_to_full(h0_k)
    return h0_k_out


def get_hubb_trimer(spin_names: list[str], U: float = 0, tk: float = 0) -> OpSum:
    n_orb = 3
    phi = (
        2.0 * np.pi / n_orb
    )  # bloch factor for transforming to molecular orbital basis
    block_map = {0: "A", 1: "E1", 2: "E2"}
    orb_map = {0: 0, 1: 0, 2: 0}

    def get_c(s: str, m: int, dagger: bool) -> OpSum:
        if dagger:
            return cdag(s + "_" + block_map[m], orb_map[m])
        return c(s + "_" + block_map[m], orb_map[m])

    spin_up = spin_names[0]
    spin_dn = spin_names[1]

    h_loc = zero()
    for a, m, mm, s in product(range(n_orb), range(n_orb), range(n_orb), spin_names):
        h_loc += (
            (-tk / float(n_orb))
            * get_c(s, m, True)
            * get_c(s, mm, False)
            * np.exp(-1j * phi * a * m)
            * np.exp(1j * phi * np.mod(a + 1, n_orb) * mm)
        )
        h_loc += (
            (-tk / float(n_orb))
            * get_c(s, m, True)
            * get_c(s, mm, False)
            * np.exp(-1j * phi * np.mod(a + 1, n_orb) * m)
            * np.exp(1j * phi * a * mm)
        )

    for m, mm, mmm in product(range(n_orb), range(n_orb), range(n_orb)):
        h_loc += (
            (U / float(n_orb))
            * get_c(spin_up, m, True)
            * get_c(spin_up, mm, False)
            * get_c(spin_dn, mmm, True)
            * get_c(spin_dn, (m + mmm - mm) % n_orb, False)
        )

    return h_loc


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

# Set up gf_structure of clusters
gf_struct_molecule = [
    ("up_A", 1),
    ("up_E1", 1),
    ("up_E2", 1),
    ("dn_A", 1),
    ("dn_E1", 1),
    ("dn_E2", 1),
]
gf_struct_molecule_mapping = {
    "up_A": "up",
    "up_E1": "up",
    "up_E2": "up",
    "dn_A": "dn",
    "dn_E1": "dn",
    "dn_E2": "dn",
}
gf_struct = [gf_struct_molecule for _ in range(n_clusters)]
gf_struct_mapping = [gf_struct_molecule_mapping for _ in range(n_clusters)]

# Make projectors onto each trimer cluster
projectors: list[BlockMatrix] = [{} for i in range(n_clusters)]
for i in range(n_clusters):
    projectors[i]["up_A"] = np.eye(n_clusters * n_orb)[0 + i * n_orb : 1 + i * n_orb, :]
    projectors[i]["dn_A"] = np.eye(n_clusters * n_orb)[0 + i * n_orb : 1 + i * n_orb, :]
    projectors[i]["up_E1"] = np.eye(n_clusters * n_orb)[
        1 + i * n_orb : 2 + i * n_orb, :
    ]
    projectors[i]["dn_E1"] = np.eye(n_clusters * n_orb)[
        1 + i * n_orb : 2 + i * n_orb, :
    ]
    projectors[i]["up_E2"] = np.eye(n_clusters * n_orb)[
        2 + i * n_orb : 3 + i * n_orb, :
    ]
    projectors[i]["dn_E2"] = np.eye(n_clusters * n_orb)[
        2 + i * n_orb : 3 + i * n_orb, :
    ]

# Get the non-interacting kinetic Hamiltonian matrix on the lattice
h0_kin_k = get_h0_kin_k(h0_k, projectors, gf_struct_mapping=gf_struct_mapping)

# Get the non-interacting local operator terms
h0_loc = [
    get_h0_loc(h0_k=h0_k, P=P, gf_struct_mapping=gf_struct_mapping[i])
    for i, P in enumerate(projectors)
]

# Get the local interaction operator terms
U = 4
h_int = [get_hubb_trimer(spin_names=spin_names, U=U) for i in range(n_clusters)]

# Define the local Hamiltonian
h_loc = [h0_loc[i] + h_int[i] for i in range(n_clusters)]

# The trimer is symmetric under a C3 rotation. In the basis of the A, E1, E2
# orbitals the rotation multiplies an orbital by w^q, with w = exp(2 pi i / 3)
# and q = 0, 1, 2, so it is a product of on-site factors on every physical and
# bath orbital ("bath_" + block). The solver finds the lowest state with the
# given eigenvalue of the rotation, which here (w^0 = 1) contains the ground
# state.
w = np.exp(2j * np.pi / 3)
charge = {"A": 0, "E1": 1, "E2": 2}
c3_factors = []
for bl, _ in gf_struct_molecule:
    for mode in [(bl, 0), ("bath_" + bl, 0)]:
        q = charge[bl.split("_")[1]]
        if q != 0:
            c3_factors.append(c(*mode) * cdag(*mode) + w**q * cdag(*mode) * c(*mode))

# Set up impurity solvers
impurity_solvers: list[ImpuritySolver] = [
    FockitImpuritySolver(h_loc[0], gf_struct[0], onsite_symmetries=[(c3_factors, 1)])
]
for _ in range(n_clusters - 1):
    impurity_solvers.append(DummyImpuritySolver(impurity_solvers[0]))


def symmetries(A: list[BlockMatrix]) -> list[BlockMatrix]:
    n_clusters = len(A)
    # Paramagnetic
    for i in range(n_clusters):
        A[i]["up_A"] = 0.5 * (A[i]["up_A"] + A[i]["dn_A"])
        A[i]["dn_A"] = A[i]["up_A"]
        A[i]["up_E1"] = 0.5 * (A[i]["up_E1"] + A[i]["dn_E1"])
        A[i]["dn_E1"] = A[i]["up_E1"]
        A[i]["up_E2"] = 0.5 * (A[i]["up_E2"] + A[i]["dn_E2"])
        A[i]["dn_E2"] = A[i]["up_E2"]
    # E1 = E2.conj()
    for i in range(n_clusters):
        A[i]["up_E2"] = A[i]["up_E1"].conj().T
        A[i]["dn_E2"] = A[i]["dn_E1"].conj().T
    return A


# Setup RISB solver class
S = LatticeSolver(
    h0_k=h0_kin_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solvers,
    update_weights=kweight.update_weights,
    projectors=projectors,
    symmetrizers=[symmetries],
    gf_struct_mapping=gf_struct_mapping,
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
