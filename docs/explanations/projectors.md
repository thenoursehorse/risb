# About projectors

Someones one might want to project onto a correlated subspace 
$\mathcal{C}_i = \{ |\chi_{Rm} \rangle_i \}$ from a larger set of states 
$\mathcal{H} = \{ |\Psi_{k \nu} \rangle  \}$, where $R$ labels a reference 
unit cell, $i$ labels inequivalent subspaces in the reference cell, $m$ labels
one of the $M$ correlated orbitals in $\mathcal{C}_i$, $k$ labels a reciprocal 
lattice vector $\vec{k}$, and $\nu$ labels one of the $N$ orbitals in 
$\mathcal{H}$. It is not a requirement that $M$ is the same for each $k$.

At each $k$, the projection onto $\mathcal{C}_i$ can be encoded into 
$M \times N$ rectangular matrices $\mathbf{P}_i(k)$. Any single-particle 
quantity, represented as an $N \times N$ matrix $\mathbf{A}^{\mathcal{H}}(k)$, 
can be projected from $\mathcal{H}$ into the correlated subspace 
$\mathcal{C}_i$ as

$$
\mathbf{A}_i^{\mathcal{C}} = \frac{1}{\mathcal{N}} 
\sum_k \mathbf{P}_i(k) \mathbf{A}^{\mathcal{H}}(k) \mathbf{P}^{\dagger}_i(k),
$$

where $\mathcal{N}$ is the number of unit cells on the lattice. The above is 
called downfolding in {{DMFT}}. Assuming homogeneity such that 
$\mathbf{A}_i^{\mathcal{C}}$ is equivalent in all unit cells, the reverse 
process from all correlated subspaces $\mathcal{C}_i$ to $\mathcal{H}$ is 
given by

$$
\mathbf{A}^{\mathcal{H}}(k) = \sum_i 
\sum_k \mathbf{P}^{\dagger}_i(k) \mathbf{A}_i^{\mathcal{C}} \mathbf{P}_i(k)
$$

The above is called upfolding in {{DMFT}}.