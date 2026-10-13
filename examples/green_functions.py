# ruff: noqa: D100
import logging

import numpy as np
from fockit.algebra import nc, zero
from triqs.gfs import MeshImFreq, MeshProduct
from triqs.lattice.tight_binding import (
    BravaisLattice,
    BrillouinZone,
    MeshBrZone,
    TBLattice,
)

from risb import LatticeSolver
from risb.helpers_triqs import (
    get_g0_k_w,
    get_g_k_w,
    get_g_qp_k_w,
    get_g_w_loc,
    get_sigma_w,
)
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

# Non-interacting cubic dispersion on lattice, built using TRIQS
nkx = 10  # nkx**3 total number of k-points
t = -1.0 / 3.0  # hopping amplitude
V = 0.25  # Bilayer hopping between orbitals on same site
units = np.eye(3)  # lattice vectors, a1, a2, a3 on a cube
hoppings = {}
for i in range(3):
    hoppings[tuple((units[:, i]).astype(int))] = np.eye(n_orb) * t
    hoppings[tuple((-units[:, i]).astype(int))] = np.eye(n_orb) * t
    hoppings[(0, 0, 0)] = np.array([[0, V], [V, 0]])
tbl = TBLattice(units=units, hoppings=hoppings, orbital_positions=[(0, 0, 0)] * n_orb)
bl = BravaisLattice(units=units)
bz = BrillouinZone(bl)
mk = MeshBrZone(bz, nkx)
h0_k = {}
for bl, _ in gf_struct:
    h0_k[bl] = tbl.fourier(mk).data

# Hubbard interactions
U = 4
h_int = zero()
for o in range(n_orb):
    h_int += U * nc("up", o) * nc("dn", o)

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 2  # half-filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)

# Set up the impurity solver
impurity_solver = FockitImpuritySolver(h_int, gf_struct)

# Setup RISB solver class
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solver,
    update_weights=kweight.update_weights,
)

# Solve
S.solve(tol=1e-6)

# Some different ways to construct some Green's functions

# The k-space and frequency mesh of the problem
iw_mesh = MeshImFreq(beta=beta, statistic="Fermion", n_iw=64)
k_iw_mesh = MeshProduct(mk, iw_mesh)

assert kweight.mu is not None
mu = kweight.mu

# Gf constructed from local self-energy
G0_k_iw = get_g0_k_w(gf_struct=gf_struct, mesh=k_iw_mesh, h0_k=h0_k, mu=mu)
Sigma_iw = get_sigma_w(
    mesh=iw_mesh,
    gf_struct=gf_struct,
    Lambda=S.Lambda[0],
    R=S.R[0],
    h0_loc=S.h0_loc_matrix[0],
    mu=mu,
)
G_k_iw = get_g_k_w(g0_k_w=G0_k_iw, sigma_w=Sigma_iw)

# Gf constructed from quasiparticle Gf
G_qp_k_iw = get_g_qp_k_w(
    gf_struct=gf_struct,
    mesh=k_iw_mesh,
    h0_kin_k=S.h0_kin_k,
    Lambda=S.Lambda[0],
    R=S.R[0],
    mu=mu,
)
G_k_iw2 = get_g_k_w(g_qp_k_w=G_qp_k_iw, R=S.R[0])

# Local Green's functions integrated over k
G0_iw_loc = get_g_w_loc(
    G0_k_iw
)  # this is using the correlated chemical potential, so will not have right filling
G_qp_iw_loc = get_g_w_loc(G_qp_k_iw)
G_iw_loc = get_g_w_loc(G_k_iw)
G_iw_loc2 = get_g_w_loc(G_k_iw2)

# Filling of physical electron scales with Z
logger.info(f"Filling G0: {G0_iw_loc.total_density().real:.4f}")
logger.info(f"Filling G_qp: {G_qp_iw_loc.total_density().real:.4f}")
logger.info(f"Filling G: {G_iw_loc.total_density().real:.4f}")
logger.info(f"Filling G2: {G_iw_loc2.total_density().real:.4f}")
