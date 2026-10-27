# ruff: noqa: D100, D103
import matplotlib.pyplot as plt
import numpy as np
from fockit.algebra import OpSum, nc
from numpy.typing import NDArray

from risb import ThermalLatticeSolver, helpers
from risb.impurity_solvers import FockitThermalImpuritySolver
from risb.types import BlockMatrix


def bethe_energies(n_k: int) -> NDArray[np.float64]:
    # Energies at equal steps of the cumulative distribution of the
    # semicircular density of states of half bandwidth 1, so that equal weights
    # at each k give the density of states
    energies = np.linspace(-1, 1, 100001)
    cdf = 0.5 + (energies * np.sqrt(1 - energies**2) + np.arcsin(energies)) / np.pi
    return np.interp((np.arange(n_k) + 0.5) / n_k, cdf, energies)


def hubbard(U: float) -> OpSum:
    # Particle-hole symmetric at half-filling, so mu = 0
    return U * nc("up", 0) * nc("dn", 0) - 0.5 * U * (nc("up", 0) + nc("dn", 0))


# Symmetrize spin blocks to be the same (paramagnetism)
def force_paramagnetic(A: list[BlockMatrix]) -> list[BlockMatrix]:
    A[0]["up"] = 0.5 * (A[0]["up"] + A[0]["dn"])
    A[0]["dn"] = A[0]["up"]
    return A


U, B, mu = 2.0, 3, 0.0
spin_names = ["up", "dn"]
gf_struct = [(bl, 1) for bl in spin_names]
h0_k = {bl: bethe_energies(1000)[:, None, None] for bl in spin_names}
bath_size = dict.fromkeys(spin_names, B)

T_arr = np.geomspace(0.01, 10, 13)
double_occupancy = []
energy = []
entropy = []
start = None
for T in T_arr:
    # The thermal state of the impurity problem. It keeps every state whose
    # Boltzmann weight is above exp(-40) times that of the lowest state
    impurity_solver = FockitThermalImpuritySolver(
        hubbard(U),
        gf_struct,
        bath_size=bath_size,
        beta=1 / T,
        method="lowest-scan",
        eta=40,
    )
    S = ThermalLatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        beta=1 / T,
        mu=mu,
        symmetrizers=[force_paramagnetic],
        force_real=True,
        bath_size=[bath_size],
    )
    # Start from the solution at the previous temperature
    if start is not None:
        S.R, S.Lambda = start
    S.solve(tol=1e-10)
    start = (S.R, S.Lambda)

    docc = impurity_solver.overlap(nc("up", 0) * nc("dn", 0))
    double_occupancy.append(float(np.real(docc)))
    omega = helpers.get_thermal_grand_potential(
        energies=S.energies_qp,
        R=S.R,
        Lambda=S.Lambda,
        D=S.D,
        Lambda_c=S.Lambda_c,
        beta=S.beta,
        mu=mu,
        impurity_grand_potential=[impurity_solver.grand_potential],
    )
    s = helpers.get_thermal_entropy(
        energies=S.energies_qp,
        R=S.R,
        Lambda=S.Lambda,
        D=S.D,
        Lambda_c=S.Lambda_c,
        beta=S.beta,
        mu=mu,
        impurity_entropy=[impurity_solver.entropy],
    )
    # E - mu N per site
    energy.append(omega + T * s)
    entropy.append(s)

fig, axis = plt.subplots(1, 3)
axis[0].semilogx(T_arr, double_occupancy, "-sr")
axis[0].set_ylabel(r"$\langle n_{\uparrow} n_{\downarrow} \rangle$")
axis[1].semilogx(T_arr, energy, "-sr")
axis[1].set_ylabel(r"$E - \mu N$")
axis[2].semilogx(T_arr, np.array(entropy) / np.log(2), "-sr")
axis[2].set_ylabel(r"$S / \ln 2$")
for ax in axis:
    ax.set_xlabel(r"$T$")
plt.show()
