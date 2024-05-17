# About k-integration

:::{seealso}
[Using `SmearingKWeight`](../how-to/kweight.md).
:::

{{RISB}} requires integrating many mean-field matrices as a function of $k$.
The way to do this that generalizes to
many kinds of $k$-space integration methods is to
find the weight of the integral at each $k$-point. This is, e.g., how
linear tetrahedron works and smearing methods work.

The reference energy for the integration weights in {{RISB}} are the
eigenenergies of the single-particle Hamiltonian $\hat{H}^{\mathrm{qp}}$,
which change at every iteration of the self-consistent process.
The quasiparticle Hamiltonian is given by

All of the integrals are in the thermodynamic limit and take the form

$$
I = \lim_{\mathcal{N} \rightarrow \infty}
\frac{1}{\mathcal{N}} \sum_k \mathbf{A}_k f(\mathbf{H}^{\mathrm{qp}}_k),
$$

where $\mathcal{N}$ is the number of unit cells, $\mathbf{A}_k$ is a matrix
of a generic function of $k$, and $f(\mathbf{H}^{\mathrm{qp}})$ is the
Fermi-Dirac distribution. The meaning of $f(\mathbf{H}^{\mathrm{qp}}_k)$ is
specifically the operation

$$
\mathbf{U}^{}_k \mathbf{U}^{\dagger}_k f(\mathbf{H}^{\mathrm{qp}}_k)
\mathbf{U}^{}_k \mathbf{U}^{\dagger}_k
= \mathbf{U}^{}_k f(\xi^{\mathrm{qp}}_{kn}) \mathbf{U}^{\dagger}_k,
$$

where $\mathbf{U}_k$ is the matrix representation of the unitary that
diagonalizes $\hat{H}^{\mathrm{qp}}_k$, $\xi^{\mathrm{qp}}_{kn}$ are
the eigenenergies (bands) of $\hat{H}^{\mathrm{qp}}$, and
$f(\mathbf{\xi}^{\mathrm{qp}}_{kn})$ is a diagonal matrix of the Fermi-Dirac
distribution for each quasiparticle band $n$.

The integral can be converted to a series of finite $k$-points, with an
appropriate integration weight such that the integral now takes the form

$$
I = \sum_k A_k w(\xi^{\mathrm{qp}}_{kn}).
$$

Most $k$-space integration methods can be reduced to different approximations
to choose the weighting function $w(\xi^{\mathrm{qp}}_{kn})$.

All of our `Solver` classes
require a function that takes the quasiparticle eigenenergies
$\xi_{kn}^{\mathrm{qp}}$ and returns the weights $w(\xi_{kn})$.
