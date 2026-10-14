# ruff: noqa: D100, D103
import matplotlib.pyplot as plt
import numpy as np
from fockit.algebra import OpSum, nc
from numpy.typing import NDArray

from risb import LatticeSolver, helpers
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight
from risb.types import BlockMatrix


def bethe_energies(n_k: int) -> NDArray[np.float64]:
    # Energies at equal steps of the cumulative distribution of the
    # semicircular density of states of half bandwidth 1, so that equal weights
    # at each k give the density of states
    energies = np.linspace(-1, 1, 100001)
    cdf = 0.5 + (energies * np.sqrt(1 - energies**2) + np.arcsin(energies)) / np.pi
    return np.interp((np.arange(n_k) + 0.5) / n_k, cdf, energies)


def hubbard(U: float) -> OpSum:
    # Particle-hole symmetric at half-filling
    return U * nc("up", 0) * nc("dn", 0) - 0.5 * U * (nc("up", 0) + nc("dn", 0))


# Symmetrize spin blocks to be the same (paramagnetism)
def force_paramagnetic(A: list[BlockMatrix]) -> list[BlockMatrix]:
    A[0]["up"] = 0.5 * (A[0]["up"] + A[0]["dn"])
    A[0]["dn"] = A[0]["up"]
    return A


def solve(B: int, U_arr: NDArray[np.float64]) -> tuple[list[float], list[float]]:
    spin_names = ["up", "dn"]
    gf_struct = [(bl, 1) for bl in spin_names]
    h0_k = {bl: bethe_energies(4000)[:, None, None] for bl in spin_names}

    # B quasiparticles for the one orbital of each spin, which is the number of
    # bath orbitals of the impurity problem. B = 1 is RISB without ghosts
    bath_size = dict.fromkeys(spin_names, B)

    # The impurity problem conserves spin, counting the bath orbitals
    spin_imbalance = nc("up", 0) - nc("dn", 0)
    for a in range(B):
        spin_imbalance += nc("bath_up", a) - nc("bath_dn", a)
    impurity_solver = FockitImpuritySolver(
        hubbard(0),
        gf_struct,
        diagonal_symmetries={spin_imbalance: 0},
        bath_size=bath_size,
    )

    kweight = SmearingKWeight(beta=2000, n_target=1)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        update_weights=kweight.update_weights,
        symmetrizers=[force_paramagnetic],
        force_real=True,
        bath_size=[bath_size],
    )
    # The quasiparticle bands hold the electrons of the ghosts as well
    kweight.n_target = 1 + S.n_ghost_electrons

    Z = []
    double_occupancy = []
    for U in U_arr:
        impurity_solver.set_h_int(hubbard(U))
        S.solve(tol=1e-8)
        assert kweight.mu is not None
        # With ghosts R^dag R is not the quasiparticle weight
        Z_matrix = helpers.get_z(S.R[0]["up"], S.Lambda[0]["up"], kweight.mu)
        Z.append(float(np.real(Z_matrix[0, 0])))
        docc = impurity_solver.overlap(nc("up", 0) * nc("dn", 0))
        double_occupancy.append(float(np.real(docc)))
    return Z, double_occupancy


U_arr = np.arange(0.25, 4 + 0.01, 0.25)
fig, axis = plt.subplots(1, 2)
for B, marker in [(1, "-ok"), (3, "-sr")]:
    Z, double_occupancy = solve(B, U_arr)
    axis[0].plot(U_arr, Z, marker, label=f"B = {B}")
    axis[1].plot(U_arr, double_occupancy, marker, label=f"B = {B}")
axis[0].set_xlabel(r"$U$")
axis[0].set_ylabel(r"$Z$")
axis[1].set_xlabel(r"$U$")
axis[1].set_ylabel(r"$\langle n_{\uparrow} n_{\downarrow} \rangle$")
axis[1].legend()
plt.show()
