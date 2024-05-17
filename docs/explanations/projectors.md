# About projectors

:::{seealso}
[Using projectors](../how-to/projectors.md).
:::

Sometimes one might want to project onto a correlated subspace
$\mathcal{C}_i = \{ |\chi_{Rm} \rangle_i \}$ from a larger set of states
$\mathcal{H} = \{ |\Psi_{k \nu} \rangle \}$, where $R$ labels a reference
unit cell, $i$ labels inequivalent subspaces in the reference cell, $m$ labels
one of the $M_i$ correlated orbitals in $\mathcal{C}_i$, $k$ labels a reciprocal
lattice vector $\vec{k}$, and $\nu$ labels one of the $N$ orbitals in
$\mathcal{H}$. It is not a requirement that $N$ is the same for each $k$.

At each $k$, the projection onto $\mathcal{C}_i$ can be encoded into
$M_i \times N$ rectangular matrices $\mathbf{P}_i(k)$. Any single-particle
quantity, represented as an $N \times N$ matrix $\mathbf{A}^{\mathcal{H}}(k)$,
can be projected from $\mathcal{H}$ into the correlated subspace
$\mathcal{C}_i$ as

$$
\mathbf{A}_i^{\mathcal{C}} = \frac{1}{\mathcal{N}}
\sum_k \mathbf{P}_i(k) \mathbf{A}^{\mathcal{H}}(k) \mathbf{P}^{\dagger}_i(k),
$$

where $\mathcal{N}$ is the number of unit cells on the lattice. The above is
called downfolding in {{DMFT}}[^Maier2005]. Assuming homogeneity such that
$\mathbf{A}_i^{\mathcal{C}}$ is equivalent in all unit cells, the reverse
process from all correlated subspaces $\mathcal{C}_i$ to $\mathcal{H}$ is
given by

$$
\mathbf{A}^{\mathcal{H}}(k) = \sum_i
\sum_k \mathbf{P}^{\dagger}_i(k) \mathbf{A}_i^{\mathcal{C}} \mathbf{P}_i(k).
$$

The above is called upfolding in {{DMFT}}[^Maier2005].

In {{RISB}} the above projectors are used to upfold the renormalization
matrix $\mathbf{\mathcal{R}}_i$ in each correlated subspace $\mathcal{C}_i$ as

$$
\mathbf{\mathcal{R}}(k) = \sum_i \sum_k \mathbf{P}^{\dagger}_i(k)
\mathbf{\mathcal{R}}_i
\mathbf{P}^{}_i(k),
$$

and similarly for the correlation potential matrix
$\mathbf{\lambda}_i$ to obtain $\mathbf{\lambda}(k)$.

The reverse process is used to downfold the quasiparticle density matrix
$\Delta^{\mathrm{qp}}$ of $\hat{H}^{\mathrm{qp}}$ into each correlated
subspace $\mathcal{C}_i$ as

$$
\Delta^{\mathrm{qp}}_i = \frac{1}{\mathcal{N}} \sum_k
\mathbf{P}_i(k)^{} f(\mathbf{H}^{\mathrm{qp}}(k)) \mathbf{P}^{\dagger}_i(k),
$$

where $f(\xi)$ is the Fermi-Dirac function,

$$
\mathbf{H}^{\mathrm{qp}}(k) = \mathbf{\mathcal{R}}(k)
\mathbf{H}_0^{\mathrm{kin}}(k) \mathbf{\mathcal{R}}^{\dagger}(k)
+ \mathbf{\lambda}(k),
$$

is the matrix representation of $\hat{H}^{\mathrm{qp}}$ at each $k$-point,
and $\mathbf{H}_{0}^{\mathrm{kin}}(k)$ is the non-interacting dispersion
matrix on the lattice that does not include the non-interacting quadratic
terms in the subspaces $\mathcal{C}_i$.

Similarly, downfolding is used to obtain the lopsided kinetic energy of the
quasiparticles as

$$
E^{c,\mathrm{qp}}_i = \frac{1}{\mathcal{N}} \sum_k
\mathbf{P}^{}_i(k) \mathbf{H}_0^{\mathrm{kin}}(k)
\mathbf{\mathcal{R}}^{\dagger}(k) f(\mathbf{H}^{\mathrm{qp}}(k))
\mathbf{P}^{\dagger}_i(k).
$$

[^Maier2005]:
    [T. Maier, M. Jarrell, T. Pruschke, and M. H. Hettler,
    _Quantum cluster theories_,
    Rev. Mod. Phys. **77**, 1027 (2005)](https://doi.org/10.1016/j.cpc.2015.04.023)
