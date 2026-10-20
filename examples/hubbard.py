# ruff: noqa: D100, D103
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from fockit.algebra import OpSum, nc, zero

from risb import LatticeSolver
from risb.helpers_fockit import N_op, S2_op
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix


def hubbard(U: float, n_orb: int) -> OpSum:
    h_int = zero()
    for o in range(n_orb):
        h_int += U * nc("up", o) * nc("dn", o)
    return h_int


# Number of orbitals and spin structure
n_orb = 1
spin_names = ["up", "dn"]
gf_struct = [(bl, n_orb) for bl in spin_names]

# Non-interacting cubic dispersion on a Gamma-centred k-mesh
nkx = 10  # nkx**3 total number of k-points
t = -1.0 / 3.0  # hopping amplitude
k_mesh = 2 * np.pi * np.array(list(product(range(nkx), repeat=3))) / nkx
dispersion = 2 * t * np.sum(np.cos(k_mesh), axis=1)
h0_k = {bl: dispersion[:, None, None] * np.eye(n_orb) for bl in spin_names}

# Hubbard interaction
h_int = hubbard(U=0, n_orb=n_orb)

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 1  # half-filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)


# Symmetrize spin blocks to be the same (paramagnetism)
def force_paramagnetic(A: list[BlockMatrix]) -> list[BlockMatrix]:
    # Paramagnetic
    A[0]["up"] = 0.5 * (A[0]["up"] + A[0]["dn"])
    A[0]["dn"] = A[0]["up"]
    return A


# The impurity problem conserves spin, so the solver can be restricted to the
# states with as many up as down electrons, counting the bath orbitals
# ("bath_" + block). The solver finds the lowest state in the chosen sector,
# which here contains the ground state.
spin_imbalance = zero()
for o in range(n_orb):
    spin_imbalance += nc("up", o) + nc("bath_up", o) - nc("dn", o) - nc("bath_dn", o)

# Set up the impurity solver
impurity_solver = FockitImpuritySolver(
    h_int, gf_struct, diagonal_symmetries={spin_imbalance: 0}
)

# Setup RISB solver class
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solver,
    update_weights=kweight.update_weights,
    symmetrizers=[force_paramagnetic],
    force_real=True,
)

# Some observables
total_number_Op = N_op(spin_names, n_orb)
total_spin_Op = S2_op(spin_names, n_orb)
Z = []
total_number = []  # Avg particle number per site
total_spin = []  # Total spin per site

U_arr = np.arange(0, 10 + 0.1, 0.25)
for U in U_arr:
    impurity_solver.set_h_int(hubbard(U=U, n_orb=n_orb))

    # Solve
    result = S.solve(tol=1e-6)

    total_number.append(impurity_solver.overlap(total_number_Op))
    total_spin.append(impurity_solver.overlap(total_spin_Op))
    Z.append(S.Z[0]["up"][0, 0])


fig, axis = plt.subplots(1, 2)
axis[0].plot(U_arr, Z, "-ok")
axis[0].set_xlabel(r"$U$")
axis[0].set_ylabel(r"$Z$")
axis[1].plot(U_arr, -0.5 + 0.5 * np.sqrt(1 + 4 * np.array(total_spin)), "-ok")
axis[1].set_xlabel(r"$U$")
axis[1].set_ylabel(r"$\mathcal{S}$")
plt.show()
