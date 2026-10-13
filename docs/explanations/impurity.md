# About the impurity problem

The main reason the impurity solvers are separated is because the impurity
Hamiltonian is computationally the most expensive part of the {{RISB}}
algorithm, and quickly becomes the computational bottleneck because of the
exponential scaling of the Hilbert space. Other parts of our implementation
are certainly not the most efficient, but solving the impurity Hamiltonian
is the most important part to optimize.

:::{note}
In the {{RISB}} literature this Hamiltonian is called the embedding
Hamiltonian $\hat{H}^{\mathrm{emb}}$ (see [Literature](../about.md#literature-of-original-theory)).
We call it the impurity Hamiltonian $\hat{H}^{\mathrm{imp}}$, and the classes
that solve it impurity solvers, to match the terminology of {{DMFT}}.
:::

## Fast track of theory

:::{admonition} TODO
:class: dropdown
Add diagram of impurity Hamiltonian.
:::

Our goal here is to provide you the minimum ingredients needed to solve the
impurity Hamiltonian for {{RISB}}, possibly without having to read any of the
extended literature. If you have a fast implementation to solve the impurity
problem, it should be trivial to slot it into the rest of our {{RISB}}
implementation.

The impurity Hamiltonian in correlated subspace $\mathcal{C}_i$ is given by

$$
\hat{H}^{\mathrm{imp}}_i = \hat{H}^{\mathrm{loc}}_i
+ \sum^{M_i}_{\alpha} \sum^{M_i}_{a} \left( [\mathcal{D}_i]_{a\alpha}
\hat{c}^{\dagger}_{i\alpha} \hat{f}^{}_{ia} + \mathrm{H.c.} \right)
+ \sum^{M_i}_{a} \sum^{M_i}_{b} [\lambda^c_i]_{ab}
\hat{f}^{}_{ib} \hat{f}^{\dagger}_{ia},
$$

where $\hat{c}_{i\alpha}$ is a physical (electron) orbital and
$\hat{f}_{ia}$ is a bath (quasiparticle) orbital, and $\mathrm{H.c.}$ is the
Hermitian conjugate. The physical orbitals form an impurity that couples to
the bath. The Hilbert space of the bath is a copy of the physical space defined
by the physical orbitals, and so the bath is the same size as the impurity. The local
Hamiltonian $\hat{H}^{\mathrm{loc}}$ is defined by the problem being solved.
The hybridization matrix $\mathbf{D}_i$ and bath coupling matrix
$\mathbf{\lambda}^c_i$ are obtained from the mean-field equations in the
[self-consistent cycle](../tutorials/self-consistent.md).

In the normal phase (non-superconducting) {{RISB}} requires solving the ground
state of the $M_i$ particle sector at each iteration of the self-consistent
loop. Here $M_i$ is the number of physical orbitals
(sites, orbitals, and spin in the correlated subspace $\mathcal{C}_i$). Since
the bath is a copy of the physical space, this particle sector
corresponds to half-filling of the impurity Hamiltonian. Clearly,
compared to {{DMFT}}, this is a much simpler and smaller impurity problem to
solve, and is the biggest advantage of {{RISB}}.

## Exact diagonalization

The simplest way to solve $\hat{H}^{\mathrm{imp}}$ is to use
exact diagonalization in the half-filled particle sector. This is what
`AtomDiagImpuritySolver` does, and orbital sizes up to $M_i = 5$ are possible, but
will take a very long time.
One can also construct a specialized sparse exact diagonalization solver that
takes into account the specific symmetries that the impurity Hamiltonian is
allowed to have. Succinctly, only symmetry allowed $\hat{c}$ and $\hat{f}$
degrees of freedom couple. This symmetry can come from, e.g., point-group
symmetries, spin or orbital symmetries.

## {{DMRG}}

Another method we have employed in the past is to use {{DMRG}}, which we
implemented using [ITensor](https://itensor.org/). Our implementation is
currently very out of date and needs to be updated. Given that
ITensor is even easier to use now with many more helper functions, a {{DMRG}}
solver for the impurity Hamiltonian should be very easy to code.

## Obvious other avenues

There is a very fast sparse exact diagonalization solver
[Pomerol](https://aeantipov.github.io/pomerol/) for finite-temperature
interacting fermions and bosons. If the Hilbert space can be restricted
to a specific particle sector, this could be a very fast solver for big
problems.

The [QuSpin](https://quspin.github.io/QuSpin/) library is another exact
diagonalization solver that offers a lot of flexibility. It has Hilbert
space restriction and parallelization over cores using OpenMP. Because
of the very low-level transparancy in their user_basis class, in principle
it should be able to construct a specialized solver for the impurity
Hamiltonian, all within python, while still being fast.

Other solvers based on Quantum Monte Carlo (QMC), NISQ devices, etc.

:::{note}
If you have any ideas for implementation, contact us and we will be happy
to collaborate. Or edit this page and add it to the list for someone else
to try.
:::

## To interface with our code

:::{seealso}
Source code for `AtomDiagImpuritySolver` for more details. The code is very
minimal and simple.
:::

Any impurity solver can be used with our code provided it implements
:py:class:`risb.types.ImpuritySolver`. Subclass it so that type checkers
verify your implementation. It needs

- `gf_struct`, the block structure of the physical orbitals.
- A method to set the impurity Hamiltonian, called as
  `set_h_imp(Lambda_c, D, h0_loc_matrix)`. `Lambda_c`, `D`, and
  `h0_loc_matrix` are block matrices: dictionaries of `numpy` arrays with the
  blocks of `gf_struct` (see `gf_struct` from {{TRIQS}}). `h0_loc_matrix`
  defines the non-interacting quadratic terms in $\hat{H}^{\mathrm{loc}}$.
- A method `solve()` that finds the ground state in the half-filled particle
  sector and stores the single-particle density matrices `rho_f` (of the
  $\hat{f}$ degrees of freedom), `rho_cf` (between the $\hat{c}$ and
  $\hat{f}$ degrees of freedom), and `rho_c` (of the $\hat{c}$ degrees of
  freedom) for every block.
- Methods `get_rho_f(block)`, `get_rho_cf(block)`, and `get_rho_c(block)` that
  return them.
