# ruff: noqa: T201, D100, D103
import matplotlib.pyplot as plt
import numpy as np
from triqs.gf import MeshImFreq, MeshProduct
from triqs.lattice.tight_binding import (
    BravaisLattice,
    BrillouinZone,
    MeshBrZone,
    TBLattice,
)
from triqs.operators import Operator, n
from triqs.operators.util.observables import N_op, S2_op
from triqs.operators.util.op_struct import set_operator_structure

from risb import LatticeSolver
from risb.embedding import EmbeddingAtomDiag
from risb.helpers_triqs import (
    get_g0_k_w,
    get_g_k_w,
    get_g_qp_k_w,
    get_g_w_loc,
    get_sigma_w,
)
from risb.kweight import SmearingKWeight


def hubbard(U, n_orb):
    h_int = Operator()
    for o in range(n_orb):
        h_int += U * n("up", o) * n("dn", o)
    return h_int


# Number of orbitals and spin structure
n_orb = 1
spin_names = ["up", "dn"]
gf_struct = set_operator_structure(spin_names, n_orb, off_diag=True)

# Non-interacting cubic dispersion on lattice, built using TRIQS
nkx = 10  # nkx**3 total number of k-points
t = -1.0 / 3.0  # hopping amplitude
units = np.eye(3)  # lattice vectors, a1, a2, a3 on a cube
hoppings = {}
for i in range(3):
    hoppings[tuple((units[:, i]).astype(int))] = np.eye(n_orb) * t
    hoppings[tuple((-units[:, i]).astype(int))] = np.eye(n_orb) * t
tbl = TBLattice(units=units, hoppings=hoppings, orbital_positions=[(0, 0, 0)] * n_orb)
bl = BravaisLattice(units=units)
bz = BrillouinZone(bl)
mk = MeshBrZone(bz, nkx)
h0_k = {}
for bl, _ in gf_struct:
    h0_k[bl] = tbl.fourier(mk).data

# Hubbard interaction
h_int = hubbard(U=0, n_orb=n_orb)

# Set up class to work out k-space integration weights
beta = 40  # inverse temperature
n_target = 1  # half-filling
kweight = SmearingKWeight(beta=beta, n_target=n_target)


# Symmetrize spin blocks to be the same (paramagnetism)
def force_paramagnetic(A):
    # Paramagnetic
    A[0]["up"] = 0.5 * (A[0]["up"] + A[0]["dn"])
    A[0]["dn"] = A[0]["up"]
    return A


# Set up class to solve embedding problem
embedding = EmbeddingAtomDiag(h_int, gf_struct)

# Setup RISB solver class
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    embedding=embedding,
    update_weights=kweight.update_weights,
    symmetries=[force_paramagnetic],
    force_real=True,
)

# Some observables
total_number_Op = N_op(spin_names, n_orb, off_diag=True)
total_spin_Op = S2_op(spin_names, n_orb, off_diag=True)
Z = []
total_number = []  # Avg particle number per site
total_spin = []  # Total spin per site

U_arr = np.arange(0, 10 + 0.1, 0.5)
for U in U_arr:
    embedding.set_h_int(hubbard(U=U, n_orb=n_orb))

    # Solve
    S.solve(tol=1e-6)

    total_number.append(embedding.overlap(total_number_Op))
    total_spin.append(embedding.overlap(total_spin_Op))
    Z.append(S.Z[0]["up"][0, 0])

    if np.abs(U - 3.5) < 1e-10:
        # Some different ways to construct some Green's functions
        mu = kweight.mu

        # The k-space and frequency mesh of the problem
        iw_mesh = MeshImFreq(beta=beta, S="Fermion", n_max=64)
        k_iw_mesh = MeshProduct(mk, iw_mesh)

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

        # Local Gf integrated over k
        G0_iw_loc = get_g_w_loc(
            G0_k_iw
        )  # this is using the correlated chemical potential, so will not have right filling
        G_qp_iw_loc = get_g_w_loc(G_qp_k_iw)
        G_iw_loc = get_g_w_loc(G_k_iw)
        G_iw_loc2 = get_g_w_loc(G_k_iw2)

        # Filling of physical electron scales with Z
        print(f"Filling G0: {G0_iw_loc.total_density().real:.4f}")
        print(f"Filling G_qp: {G_qp_iw_loc.total_density().real:.4f}")
        print(f"Filling G: {G_iw_loc.total_density().real:.4f}")
        print(f"Filling G2: {G_iw_loc2.total_density().real:.4f}")
        print(f"Filling Z:: {S.Z[0]['up'][0,0]:.4f}")

fig, axis = plt.subplots(1, 2)
axis[0].plot(U_arr, Z, "-ok")
axis[0].set_xlabel(r"$U$")
axis[0].set_ylabel(r"$Z$")
axis[1].plot(U_arr, -0.5 + 0.5 * np.sqrt(1 + 4 * np.array(total_spin)), "-ok")
axis[1].set_xlabel(r"$U$")
axis[1].set_ylabel(r"$\mathcal{S}$")
plt.show()
