# ruff: noqa: D100
import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from fockit.algebra import nc, zero

from risb import LatticeSolver, helpers
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

# Bilayer cubic dispersion on a Gamma-centred k-mesh
nkx = 10  # nkx**3 total number of k-points
t = -1.0 / 3.0  # hopping amplitude
V = 0.25  # hopping between the two orbitals of a site
k_mesh = 2 * np.pi * np.array(list(product(range(nkx), repeat=3))) / nkx
dispersion = 2 * t * np.sum(np.cos(k_mesh), axis=1)
h0 = dispersion[:, None, None] * np.eye(n_orb) + np.array([[0, V], [V, 0]])
h0_k = dict.fromkeys(spin_names, h0)

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
assert kweight.mu is not None
mu = kweight.mu

# R and Lambda in the blocks of the lattice. S.R_full and S.Lambda_full are
# from the start of the last cycle, one step behind S.R and S.Lambda
R_full, Lambda_full = helpers.upfold_r_lambda(S.R, S.Lambda, S.h0_kin_k)

# Green's functions at Matsubara frequencies, indexed as k, frequency, orb, orb
iw = 1j * (2 * np.arange(64) + 1) * np.pi / beta

# From Dyson's equation with the self-energy of the cluster
G0_k_iw = helpers.get_g0_k_w(iw, h0_k, mu)
Sigma_iw = {
    bl: helpers.get_sigma_w(iw, S.R[0][bl], S.Lambda[0][bl], mu, S.h0_loc_matrix[0][bl])
    for bl in spin_names
}
G_k_iw = {
    bl: np.linalg.inv(np.linalg.inv(G0_k_iw[bl]) - Sigma_iw[bl][None])
    for bl in spin_names
}

# From the quasiparticles, R^dag G_qp R
G_k_iw2 = helpers.get_g_k_w(iw, R_full, Lambda_full, S.h0_kin_k, mu)

# Local Green's functions, averaged over k
G_iw_loc = helpers.get_g_w_loc(G_k_iw)[0]
G_iw_loc2 = helpers.get_g_w_loc(G_k_iw2)[0]

# Local spectral function on real frequencies, broadened by eta
eta = 0.05
w = np.linspace(-10, 10, 4001)
G_w_loc = helpers.get_g_w_loc(
    helpers.get_g_k_w(w + 1j * eta, R_full, Lambda_full, S.h0_kin_k, mu)
)[0]
A_w = {bl: -np.trace(G_w_loc[bl], axis1=1, axis2=2).imag / np.pi for bl in spin_names}

# The spectral weight is Tr R^dag R, not the number of orbitals, because the
# Hubbard bands are not in the Green's function of RISB without ghosts
spectral_weight = np.sum(A_w["up"]) * (w[1] - w[0])
logger.info(
    f"Spectral weight {spectral_weight:.4f}, Tr R^dag R "
    f"{np.trace(S.Z[0]['up']).real:.4f}"
)

plt.plot(w, A_w["up"], "-k")
plt.xlabel(r"$\omega$")
plt.ylabel(r"$A(\omega)$")
plt.show()
