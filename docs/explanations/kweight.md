# About k-integration

:::{seealso}
[Using `BandKIntegrator`](../how-to/kintegration.md).
:::

Let $U_k$ diagonalize the quasiparticle Hamiltonian, with eigenvalues
$\epsilon^{\mathrm{qp}}_{kn}$. At inverse temperature $\beta$, the occupations are

$$
f_{kn} = \frac{1}{e^{\beta(\epsilon^{\mathrm{qp}}_{kn}-\mu)} + 1}.
$$

The quasiparticle density matrix at each k-point is

$$
n^{\mathrm{qp}}_k = U_k\,\operatorname{diag}(f_{kn})\,U_k^\dagger.
$$

On a finite mesh, Brillouin-zone averages take the form

$$
I = \sum_k w_k A_k n^{\mathrm{qp}}_k,
\qquad \sum_k w_k = 1.
$$

The k-point weights $w_k$ are $1/N_k$ on a uniform full-zone mesh. On an
irreducible mesh they include the symmetry multiplicities. They contain no
occupation factors. The combined weights $W_{kn} = w_k f_{kn}$ depend on both
k-point and band.

`BandKIntegrationResult.occupations` stores $f_{kn}$, and `weights` stores $w_k$.
`LatticeSolver.weighted_occupations_qp` stores their product $W_{kn}$.
Gaussian and Methfessel-Paxton smearing replace the Fermi-Dirac function with
the corresponding occupation function.

`BandKIntegrator` diagonalizes the Hamiltonian, calculates occupations and,
at fixed filling, solves for the chemical potential. Its `kpoint_integrator`
performs the Brillouin-zone average: `WeightedKPointIntegrator` uses a supplied
mesh, while `AutoBZIntegrator` chooses integration points.

`TemperatureExtrapolatedBandKIntegrator` extrapolates the finite-temperature
integrals. On a supplied mesh, the returned occupations and density matrices
use the same extrapolation coefficients. They cannot in general be recovered
by evaluating a Fermi function at the extrapolated chemical potential.
