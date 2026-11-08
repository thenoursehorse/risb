# Derivation of upfolding and downfolding in RISB

The projectors put the local renormalization matrices and correlation
potentials into the basis of the lattice. This is a change of basis of
the cluster RISB Hamiltonian. Downfolding the quasiparticle density and
lopsided kinetic energy follows from derivatives of its grand potential.

## Correlated subspaces

Suppress the single-particle symmetry block index. There are $N$
physical orbitals at each $k$, and cluster $i$ has $M_i$ orbitals. Its
projection matrix $\mathbf{P}_i(k)$ has shape $M_i \times N$. Assume

$$
\mathbf{P}_i(k)\mathbf{P}_j^\dagger(k)
= \begin{cases}
\mathbf{1}_{M_i}, & i = j,\\
0, & i \ne j.
\end{cases}
$$

The rows are orthonormal within a cluster, and different clusters do
not overlap. The square projector onto the uncorrelated complement is

$$
\mathbf{Q}(k) = \mathbf{1}_N -
\sum_i\mathbf{P}_i^\dagger(k)\mathbf{P}_i(k).
$$

Choose orthonormal rows $\mathbf{P}_u(k)$ spanning this complement, so
$\mathbf{P}_u^\dagger\mathbf{P}_u = \mathbf{Q}$. Stacking all the rows
gives a unitary matrix

$$
\mathbf{W}(k) =
\begin{pmatrix}
\mathbf{P}_1(k)\\
\vdots\\
\mathbf{P}_m(k)\\
\mathbf{P}_u(k)
\end{pmatrix},
\qquad
\mathbf{W}\mathbf{W}^\dagger
= \mathbf{W}^\dagger\mathbf{W} = \mathbf{1}_N.
$$

For a column of lattice annihilation operators $c_k$, the local orbital
coordinates are $d_{ki}=\mathbf{P}_i(k)c_k$. On a uniform mesh,

$$
d_{Ri}=\frac{1}{\sqrt{\mathcal{N}}}
\sum_k e^{ik\cdot R}\mathbf{P}_i(k)c_k.
$$

The row orthonormality gives the canonical anticommutators

$$
\{d_{Ria},d_{R'jb}^\dagger\}
=\delta_{RR'}\delta_{ij}\delta_{ab}.
$$

This allows a separate local Fock space for every translated cluster.
It does not establish that a given physical interaction is local in
these orbitals. That is a separate assumption about the Hamiltonian.

## Upfolding the RISB matrices

First take RISB without ghosts. There is one auxiliary quasiparticle
orbital per physical orbital. The local matrices are

$$
\mathbf{\mathcal{R}}_{\mathrm{loc}}
=\operatorname{diag}(
\mathbf{\mathcal{R}}_1,\ldots,\mathbf{\mathcal{R}}_m,\mathbf{1}_u),
\qquad
\mathbf{\lambda}_{\mathrm{loc}}
=\operatorname{diag}(
\mathbf{\lambda}_1,\ldots,\mathbf{\lambda}_m,0_u).
$$

The renormalization matrix has a quasiparticle row and a physical orbital
column. Choose the same coordinate transformation $\mathbf{W}(k)$ for
the physical and auxiliary lattice spaces. This is a choice of
coordinates for the auxiliary orbitals, not an identification of
physical electrons with quasiparticles.

In the hopping term the mean-field replacement in local coordinates is
$d_k \mapsto \mathbf{\mathcal{R}}_{\mathrm{loc}}^\dagger g_k$, where $g_k$
is the column of auxiliary annihilation operators. With
$g_k=\mathbf{W}(k)f_k$, this gives

$$
c_k \mapsto
\mathbf{W}^\dagger(k)\mathbf{\mathcal{R}}_{\mathrm{loc}}^\dagger
\mathbf{W}(k)f_k
=\mathbf{\mathcal{R}}^\dagger(k)f_k.
$$

Therefore,

$$
\mathbf{\mathcal{R}}(k)
=\mathbf{W}^\dagger\mathbf{\mathcal{R}}_{\mathrm{loc}}\mathbf{W}
=\mathbf{Q}(k)+
\sum_i\mathbf{P}_i^\dagger(k)
\mathbf{\mathcal{R}}_i\mathbf{P}_i(k).
$$

The local auxiliary potential transforms in the same way,

$$
\mathbf{\lambda}(k)
=\mathbf{W}^\dagger\mathbf{\lambda}_{\mathrm{loc}}\mathbf{W}
=\sum_i\mathbf{P}_i^\dagger(k)
\mathbf{\lambda}_i\mathbf{P}_i(k).
$$

The uncorrelated orbitals have $\mathcal{R}=1$ and $\lambda=0$.
Dropping $\mathbf{Q}$ would instead remove their bare dispersion.

Write $\boldsymbol{\varepsilon}(k)=\mathbf{H}_0^{\mathrm{kin}}(k)$.
The quasiparticle Hamiltonian is

$$
\mathbf{H}^{\mathrm{qp}}(k)
=\mathbf{\mathcal{R}}(k)\boldsymbol{\varepsilon}(k)
 \mathbf{\mathcal{R}}^\dagger(k)+\mathbf{\lambda}(k).
$$

Transforming it back to the cluster coordinates gives

$$
\mathbf{W}\mathbf{H}^{\mathrm{qp}}\mathbf{W}^\dagger
=\mathbf{\mathcal{R}}_{\mathrm{loc}}
 \bigl(\mathbf{W}\boldsymbol{\varepsilon}\mathbf{W}^\dagger\bigr)
 \mathbf{\mathcal{R}}_{\mathrm{loc}}^\dagger
+\mathbf{\lambda}_{\mathrm{loc}}.
$$

Its $ij$ hopping block is
$\mathbf{\mathcal{R}}_i\boldsymbol{\varepsilon}_{ij}
\mathbf{\mathcal{R}}_j^\dagger$, the usual cluster RISB expression.
Upfolding introduces no further approximation to that Hamiltonian.
It requires neither invertible nor Hermitian
$\mathbf{\mathcal{R}}_i$, nor commuting
$\mathbf{\mathcal{R}}_i$ and $\mathbf{\lambda}_i$.

The local quadratic terms of the correlated clusters must be in their
impurity Hamiltonians. Accordingly,

$$
\boldsymbol{\varepsilon}(k)
=\mathbf{H}_0(k)-\sum_i\mathbf{P}_i^\dagger(k)
\mathbf{H}_{0,i}^{\mathrm{loc}}\mathbf{P}_i(k).
$$

The local terms of uncorrelated orbitals remain in
$\boldsymbol{\varepsilon}(k)$.

## Downfolding is not an inverse for general lattice matrices

With normalized weights $\sum_k w_k=1$, define

$$
\mathcal{D}_i[\mathbf{A}]
=\sum_k w_k\mathbf{P}_i(k)\mathbf{A}(k)\mathbf{P}_i^\dagger(k),
\qquad
\mathcal{U}[\{ \mathbf{A}_i \}](k)
=\sum_i\mathbf{P}_i^\dagger(k)\mathbf{A}_i\mathbf{P}_i(k).
$$

Orthonormality gives
$\mathcal{D}_i[\mathcal{U}[\{ \mathbf{A}_j \}]]=\mathbf{A}_i$.
In the opposite order, matrix elements between different clusters and
the remaining momentum dependence are lost. For two one-orbital
clusters with identity-row projectors,

$$
\mathbf{A}=\begin{pmatrix}0&t\\t^*&0\end{pmatrix}
\quad\Longrightarrow\quad
\mathcal{U}[\{\mathcal{D}_i[\mathbf{A}]\}]=0.
$$

The useful relation for the variational equations is the trace identity

$$
\sum_k w_k\operatorname{Tr}\!
\left[\mathbf{B}(k)\mathcal{U}[\{\mathbf{A}_i\}](k)\right]
=\sum_i\operatorname{Tr}\!
\left[\mathcal{D}_i[\mathbf{B}]\mathbf{A}_i\right].
$$

It follows by cyclically permuting the matrices inside the trace.

## Density and lopsided kinetic energy from stationarity

Take the projectors and the physical Hamiltonian as fixed. At inverse
temperature $\beta$ and fixed chemical potential $\mu$, the quadratic
quasiparticle grand potential per unit cell is

$$
\Omega_{\mathrm{qp}}
=-\frac{1}{\beta}\sum_k w_k\operatorname{Tr}
\log\!\left[
\mathbf{1}+e^{-\beta(\mathbf{H}^{\mathrm{qp}}(k)-\mu\mathbf{1})}
\right].
$$

Define
$\mathbf{n}(k)=
[\mathbf{1}+e^{\beta(\mathbf{H}^{\mathrm{qp}}(k)-\mu\mathbf{1})}]^{-1}$.
The differential is

$$
\delta\Omega_{\mathrm{qp}}
=\sum_k w_k\operatorname{Tr}
[\mathbf{n}(k)\delta\mathbf{H}^{\mathrm{qp}}(k)].
$$

This also holds at zero temperature away from level crossings, with
the limiting occupations. At fixed quasiparticle filling the Legendre
transform to the free energy gives the same field derivatives.

### Correlation potential

A variation of a local potential gives

$$
\delta\mathbf{H}^{\mathrm{qp}}(k)
=\mathbf{P}_i^\dagger(k)\delta\mathbf{\lambda}_i\mathbf{P}_i(k),
\qquad
\delta_{\lambda_i}\Omega_{\mathrm{qp}}
=\operatorname{Tr}[\mathbf{N}_i\delta\mathbf{\lambda}_i],
$$

where

$$
\mathbf{N}_i
=\sum_k w_k\mathbf{P}_i(k)\mathbf{n}(k)\mathbf{P}_i^\dagger(k).
$$

The density convention in the code is
$[\Delta_i^{\mathrm{qp}}]_{ab}
=\langle f_{ia}^\dagger f_{ib}\rangle$, whereas
$[\mathbf{N}_i]_{ab}=\langle f_{ib}^\dagger f_{ia}\rangle$.
Thus

$$
\Delta_i^{\mathrm{qp}}=\mathbf{N}_i^T,
\qquad
\operatorname{Tr}[\mathbf{N}_i\delta\mathbf{\lambda}_i]
=\sum_{ab}[\Delta_i^{\mathrm{qp}}]_{ab}
             [\delta\mathbf{\lambda}_i]_{ab}.
$$

The local density is exactly the quantity conjugate to the local
correlation potential in the RISB Lagrangian.

### Renormalization matrix

For an arbitrary complex variation of
$\mathbf{\mathcal{R}}_i$,

$$
\delta\mathbf{\mathcal{R}}(k)
=\mathbf{P}_i^\dagger(k)
 \delta\mathbf{\mathcal{R}}_i\mathbf{P}_i(k),
$$

and, with $\mathbf{\lambda}$ fixed,

$$
\delta\mathbf{H}^{\mathrm{qp}}
=\delta\mathbf{\mathcal{R}}\boldsymbol{\varepsilon}
 \mathbf{\mathcal{R}}^\dagger
+\mathbf{\mathcal{R}}\boldsymbol{\varepsilon}
 \delta\mathbf{\mathcal{R}}^\dagger.
$$

Since $\boldsymbol{\varepsilon}$ and $\mathbf{n}$ are Hermitian,
the two trace contributions are complex conjugates. Cyclic permutation
then gives

$$
\delta_{\mathcal{R}_i}\Omega_{\mathrm{qp}}
=2\operatorname{Re}\operatorname{Tr}
[\mathbf{K}_i\delta\mathbf{\mathcal{R}}_i],
\qquad
\mathbf{K}_i
=\sum_k w_k\mathbf{P}_i(k)\boldsymbol{\varepsilon}(k)
 \mathbf{\mathcal{R}}^\dagger(k)\mathbf{n}(k)
 \mathbf{P}_i^\dagger(k).
$$

This is the lopsided kinetic energy stored as
`lopsided_ke_qp[i][bl]`. Together with
`rho_qp[i][bl]`, it is the derivative of the same quasiparticle
functional that contains the upfolded matrices. No separate assumption
about downfolding is needed.

This calculation concerns the quadratic quasiparticle term. The full
finite-temperature ghost functional has further embedding and reference
terms; their temperature dependence is not derived here.

## Momentum dependence and the choice of correlated orbitals

For a unitary change of lattice coordinates
$c'_k=\mathbf{V}(k)c_k$,

$$
\boldsymbol{\varepsilon}'(k)
=\mathbf{V}(k)\boldsymbol{\varepsilon}(k)\mathbf{V}^\dagger(k),
\qquad
\mathbf{P}'_i(k)=\mathbf{P}_i(k)\mathbf{V}^\dagger(k).
$$

The local orbitals $d_{ki}$ are unchanged. Choosing the same change of
auxiliary coordinates gives

$$
\mathbf{\mathcal{R}}'(k)
=\mathbf{V}(k)\mathbf{\mathcal{R}}(k)\mathbf{V}^\dagger(k),
\qquad
\mathbf{\lambda}'(k)
=\mathbf{V}(k)\mathbf{\lambda}(k)\mathbf{V}^\dagger(k).
$$

Consequently $\mathbf{H}^{\mathrm{qp}}$ transforms unitarily and the
downfolded density and kinetic matrices are unchanged. Momentum-dependent
projectors can therefore describe the same local RISB problem in a
different lattice basis.

Changing the correlated orbitals themselves is a different operation.
The interaction must be expressed in the new orbitals; its transformed
matrix elements can couple different clusters or unit cells. Keeping
only local interactions then adds a physical approximation. The
projector algebra alone cannot justify it. Likewise, restricting the
retained band space is a choice of model, not an exact elimination of
the discarded bands.

Nonorthonormal or overlapping projector rows do not satisfy the unitary
construction above. They need a treatment of their metric or a change
to orthonormal orbitals, including consistent interaction matrix
elements. `check_projectors=True` checks the row orthonormality and
non-overlap assumed by the current implementation.

## Ghost quasiparticles

With ghosts, cluster $i$ has $L_i > M_i$ auxiliary quasiparticles and
$\mathbf{\mathcal{R}}_i$ has shape $L_i \times M_i$.
Let $\mathbf{P}^{\mathrm{qp}}_i(k)$ select its quasiparticles from the
larger auxiliary lattice space. These projectors satisfy the same row
orthonormality and non-overlap conditions in that space. Choose matching
coordinates $\mathbf{P}_u$ and $\mathbf{P}^{\mathrm{qp}}_u$ for the
uncorrelated orbitals. Then

$$
\mathbf{\mathcal{R}}(k)
=[\mathbf{P}^{\mathrm{qp}}_u(k)]^\dagger\mathbf{P}_u(k)
+\sum_i[\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger
 \mathbf{\mathcal{R}}_i\mathbf{P}_i(k),
\qquad
\mathbf{\lambda}(k)
=\sum_i[\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger
 \mathbf{\lambda}_i\mathbf{P}^{\mathrm{qp}}_i(k).
$$

Equivalently, with separate unitary matrices for the physical and
auxiliary coordinates,

$$
\mathbf{\mathcal{R}}(k)
=\mathbf{W}_{\mathrm{qp}}^\dagger(k)
 \mathbf{\mathcal{R}}_{\mathrm{loc}}\mathbf{W}(k),
\qquad
\mathbf{\lambda}(k)
=\mathbf{W}_{\mathrm{qp}}^\dagger(k)
 \mathbf{\lambda}_{\mathrm{loc}}\mathbf{W}_{\mathrm{qp}}(k).
$$

Here $\mathbf{\mathcal{R}}_{\mathrm{loc}}$ is rectangular. The
Hamiltonian in auxiliary cluster coordinates still has the usual
$ij$ hopping block
$\mathbf{\mathcal{R}}_i\boldsymbol{\varepsilon}_{ij}
\mathbf{\mathcal{R}}_j^\dagger$.

The same derivative calculation gives

$$
\Delta_i^{\mathrm{qp}}
=\left[
\sum_k w_k\mathbf{P}^{\mathrm{qp}}_i(k)\mathbf{n}(k)
 [\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger
\right]^T,
\qquad
\mathbf{K}_i
=\sum_k w_k\mathbf{P}_i(k)\boldsymbol{\varepsilon}(k)
 \mathbf{\mathcal{R}}^\dagger(k)\mathbf{n}(k)
 [\mathbf{P}^{\mathrm{qp}}_i(k)]^\dagger.
$$

The density is $L_i \times L_i$ and the lopsided kinetic energy is
$M_i \times L_i$, so
$\operatorname{Tr}[\mathbf{K}_i\delta\mathbf{\mathcal{R}}_i]$
is still the required scalar variation.

In the code, `get_ghost_projectors` constructs
$\mathbf{P}^{\mathrm{qp}}_i$ by adding a separate identity block for
each cluster's ghosts. `upfold_r_lambda` pads the columns of
$\mathbf{\mathcal{R}}_i$ with zeros, upfolds with these enlarged
projectors, and drops the extra columns. This realizes the rectangular
formula above. Without ghosts,
$\mathbf{P}^{\mathrm{qp}}_i=\mathbf{P}_i$ and
$[\mathbf{P}^{\mathrm{qp}}_u]^\dagger\mathbf{P}_u=\mathbf{Q}$.

## Relation to the literature and implementation

Lanatà et al. [1] define projectors $\Pi_i$ onto the local single-particle
subspaces in Sec. V of the supplement. Their Eq. 112 gives the local
quasiparticle density with the same transpose convention. Eq. 113 is
the projected kinetic stationarity equation. The rectangular
$\mathbf{P}_i$ used here select coordinates in those subspaces from a
general lattice basis.

Lanatà [2], Eq. 55, writes the ghost quasiparticle hopping as
$\mathbf{\mathcal{R}}_i t_{ij}\mathbf{\mathcal{R}}_j^\dagger$ with a local
$\mathbf{\lambda}_i$. The upfolded Hamiltonian above is its matrix
representation after changing coordinates.

The implementation is in
[`helpers.upfold_r_lambda`](../src/risb/helpers.py) and the
quasiparticle observables of
[`BaseLatticeSolver`](../src/risb/solve_lattice_base.py).

1. N. Lanatà, Y.-X. Yao, X. Deng, V. Dobrosavljević, and G. Kotliar,
   *Slave Boson Theory of Orbital Differentiation with Crystal Field
   Effects: Application to UO₂*, Phys. Rev. Lett. **118**, 126401 (2017).
   [Paper and supplement](https://arxiv.org/pdf/1606.09614).
2. N. Lanatà, *Operatorial formulation of the ghost rotationally-invariant
   slave-Boson theory*, Phys. Rev. B **105**, 045111 (2022).
   [Paper](https://arxiv.org/abs/2109.10320).
