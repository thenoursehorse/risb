# About projectors

:::{seealso}
[Using projectors](../how-to/projectors.md).
:::

Sometimes one might want to project onto a correlated subspace
$\mathcal{C}_i = \{ |\chi_{Rm} \rangle_i \}$ from a larger set of states
$\mathcal{H} = \{ |\Psi_{k \nu} \rangle \}$, where $R$ labels a unit cell,
$i$ labels inequivalent subspaces in the reference cell, $m$ labels one of
the $M_i$ correlated orbitals in $\mathcal{C}_i$, $k$ labels a reciprocal
lattice vector $\vec{k}$, and $\nu$ labels one of the $N$ orbitals in
$\mathcal{H}$.

At each $k$, the projection onto $\mathcal{C}_i$ can be encoded into
$M_i \times N$ rectangular matrices $\mathbf{P}_i(k)$. Their rows must be
orthonormal, and different correlated subspaces must not overlap,

$$
\mathbf{P}_i(k) \mathbf{P}_i^\dagger(k) = \mathbf{1}_{M_i},
\qquad
\mathbf{P}_i(k) \mathbf{P}_j^\dagger(k) = 0 \quad (i \ne j).
$$

The matrices $\mathbf{P}_i$ give the coordinates in each subspace. The
square matrix $\mathbf{P}_i^\dagger \mathbf{P}_i$ is the projector onto that
subspace in $\mathcal{H}$. The correlated subspaces need not cover all of
$\mathcal{H}$.

## Downfolding and upfolding

A single-particle quantity represented as an $N \times N$ matrix
$\mathbf{A}^{\mathcal{H}}(k)$ can be projected into each correlated subspace
$\mathcal{C}_i$ as

$$
\mathbf{A}_i^{\mathcal{C}} = \frac{1}{\mathcal{N}}
\sum_k \mathbf{P}_i(k) \mathbf{A}^{\mathcal{H}}(k) \mathbf{P}^{\dagger}_i(k),
$$

where $\mathcal{N}$ is the number of unit cells on the lattice. This is
called downfolding in {{DMFT}}[^Maier2005]. Here the sum is over a uniform
mesh of $\mathcal{N}$ $k$-points. With other integration weights, replace
$\mathcal{N}^{-1}\sum_k$ by $\sum_k w_k$, with $\sum_k w_k = 1$.

A set of local matrices $\mathbf{A}_i^{\mathcal{C}}$, the same in every unit
cell, can be put into the larger space as

$$
\mathbf{A}^{\mathcal{H}}(k) = \sum_i
\mathbf{P}^{\dagger}_i(k) \mathbf{A}_i^{\mathcal{C}} \mathbf{P}_i(k).
$$

This is called upfolding. There is no sum over $k$: the result is a matrix
at each $k$. It is zero on the orbitals in no correlated subspace.

Downfolding this upfolded matrix gives back
$\mathbf{A}_i^{\mathcal{C}}$. The opposite order does not give back a
general $\mathbf{A}^{\mathcal{H}}(k)$, because downfolding retains only the
local matrices in the chosen subspaces. It loses the matrix elements
between subspaces and the remaining $k$ dependence.

## The RISB matrices

In {{RISB}}, the renormalization matrix $\mathbf{\mathcal{R}}_i$ and the
correlation potential matrix $\mathbf{\lambda}_i$ are local matrices of
each correlated subspace. Without ghost quasiparticles they are both
$M_i \times M_i$. Define the projector onto the orbitals in no correlated
subspace as

$$
\mathbf{Q}(k) = \mathbf{1}_N -
\sum_i \mathbf{P}_i^\dagger(k) \mathbf{P}_i(k).
$$

The matrices in the larger space are

$$
\mathbf{\mathcal{R}}(k) = \mathbf{Q}(k) +
\sum_i \mathbf{P}^{\dagger}_i(k) \mathbf{\mathcal{R}}_i \mathbf{P}_i(k),
\qquad
\mathbf{\lambda}(k) = \sum_i
\mathbf{P}^{\dagger}_i(k) \mathbf{\lambda}_i \mathbf{P}_i(k).
$$

The $\mathbf{Q}$ term gives $\mathcal{R} = 1$ on the orbitals in no
correlated subspace, where $\lambda = 0$. Without it their bare dispersion
would be lost. If the correlated subspaces cover $\mathcal{H}$, then
$\mathbf{Q} = 0$.

In the basis of the correlated orbitals and the remaining uncorrelated
orbitals, these are block matrices with blocks $\mathbf{\mathcal{R}}_i$,
$\mathbf{\lambda}_i$, and respectively $1$, $0$ on the uncorrelated
orbitals. Upfolding expresses these matrices in the basis of
$\mathcal{H}$. It gives the usual {{RISB}} hopping
$\mathbf{\mathcal{R}}_i \mathbf{H}_{0,ij}^{\mathrm{kin}}
\mathbf{\mathcal{R}}_j^\dagger$ between subspaces $i$ and $j$.

The quasiparticle Hamiltonian is

$$
\mathbf{H}^{\mathrm{qp}}(k) = \mathbf{\mathcal{R}}(k)
\mathbf{H}_0^{\mathrm{kin}}(k) \mathbf{\mathcal{R}}^{\dagger}(k)
+ \mathbf{\lambda}(k).
$$

Here $\mathbf{H}_{0}^{\mathrm{kin}}(k)$ is the non-interacting dispersion
without the local quadratic terms of the correlated subspaces. Those terms
are included in the impurity Hamiltonians. The local terms of the
uncorrelated orbitals stay in $\mathbf{H}_{0}^{\mathrm{kin}}(k)$; see
[Quadratic terms](../how-to/quadratic_terms.md).

The quasiparticle density matrix is downfolded as

$$
\Delta^{\mathrm{qp}}_i = \left[ \frac{1}{\mathcal{N}} \sum_k
\mathbf{P}_i(k) f(\mathbf{H}^{\mathrm{qp}}(k)) \mathbf{P}^{\dagger}_i(k)
\right]^T,
$$

where $f$ is the Fermi-Dirac function, with the chemical potential included.
The transpose follows the convention
$[\Delta_i^{\mathrm{qp}}]_{ab} =
\langle f_{ia}^\dagger f_{ib} \rangle$.

Similarly, the lopsided kinetic energy is

$$
E^{c,\mathrm{qp}}_i = \frac{1}{\mathcal{N}} \sum_k
\mathbf{P}_i(k) \mathbf{H}_0^{\mathrm{kin}}(k)
\mathbf{\mathcal{R}}^{\dagger}(k) f(\mathbf{H}^{\mathrm{qp}}(k))
\mathbf{P}^{\dagger}_i(k).
$$

These downfolded matrices follow by differentiating the quasiparticle
grand potential with respect to the local $\mathbf{\lambda}_i$ and
$\mathbf{\mathcal{R}}_i$. They are the density and kinetic terms entering
the local {{RISB}} equations. The projectors $\Pi_i$ in Eqs. 112 and 113
of the supplement of Ref. [^Lanata2017] select the same local subspaces
when the lattice is already written in their orbital basis.

Projectors can depend on $k$, for example when local orbitals are
represented in a band basis. The local interaction must be expressed in
the same correlated orbitals. Changing the basis of the larger space,
with both the Hamiltonian and the projectors transformed together, leaves
the problem unchanged. Changing the correlated orbitals themselves
generally changes the interaction matrix elements and can introduce
interactions between clusters.

## Ghost quasiparticles

With ghosts, the number of quasiparticles in a correlated subspace is
larger than the number of physical orbitals, and
$\mathbf{\mathcal{R}}_i$ is rectangular. The physical and quasiparticle
spaces then have different projection matrices. Let
$\mathbf{P}^{\mathrm{qp}}_i(k)$ project onto the quasiparticles of subspace
$i$, and let $\mathbf{P}_u(k)$ and $\mathbf{P}^{\mathrm{qp}}_u(k)$ give
matching coordinates on the uncorrelated orbitals. Then

$$
\mathbf{\mathcal{R}}(k) =
[\mathbf{P}^{\mathrm{qp}}_u(k)]^\dagger \mathbf{P}_u(k) +
\sum_i [\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger
\mathbf{\mathcal{R}}_i \mathbf{P}_i(k),
\qquad
\mathbf{\lambda}(k) =
\sum_i [\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger
\mathbf{\lambda}_i \mathbf{P}^{\mathrm{qp}}_i(k).
$$

The Hamiltonian has the same form as above. Its density matrix is
downfolded with $\mathbf{P}^{\mathrm{qp}}_i$ on both sides. For the
lopsided kinetic energy the left projector is $\mathbf{P}_i$ and the
right projector is $[\mathbf{P}^{\mathrm{qp}}_i]^\dagger$.
{py:func}`risb.helpers.get_ghost_projectors` constructs the quasiparticle
projectors, and {py:func}`risb.helpers.upfold_r_lambda` constructs
$\mathbf{\mathcal{R}}(k)$ and $\mathbf{\lambda}(k)$. This is the
matrix form of the ghost {{RISB}} hopping in Eq. 55 of
Ref. [^Lanata2022].

[^Maier2005]:
    [T. Maier, M. Jarrell, T. Pruschke, and M. H. Hettler,
    _Quantum cluster theories_,
    Rev. Mod. Phys. **77**, 1027 (2005)](https://doi.org/10.1103/RevModPhys.77.1027)

[^Lanata2017]:
    [N. Lanatà, Y.-X. Yao, X. Deng, V. Dobrosavljević, and G. Kotliar,
    _Slave Boson Theory of Orbital Differentiation with Crystal Field Effects:
    Application to UO₂_, Phys. Rev. Lett. **118**, 126401
    (2017)](https://arxiv.org/pdf/1606.09614)

[^Lanata2022]:
    [N. Lanatà, _Operatorial formulation of the ghost rotationally-invariant
    slave-Boson theory_, Phys. Rev. B **105**, 045111
    (2022)](https://arxiv.org/abs/2109.10320)
