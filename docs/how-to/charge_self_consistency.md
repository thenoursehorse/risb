# Charge self-consistency

This guide shows you how to set up a charge self-consistent {{DFT}}+{{RISB}}
calculation. The outer loop updates the DFT charge density using the density
matrix from a {{RISB}} solution, then solves the correlated problem again
with the updated DFT Hamiltonian.

{py:class}`~risb.LatticeSolver` solves the correlated problem for a fixed
`h0_k`. You must supply the DFT calculation, the construction of the
correlated subspaces, and the transfer of the charge correction back to DFT.
The examples below are a skeleton to adapt to your DFT code.

:::{seealso}
:class: dropdown
[Using `LatticeSolver`](lattice_solver.md),
[using projectors](projectors.md), and
[using impurity solvers](impurity_solvers.md).
:::

## Preparing the DFT input

For each DFT update, you need

- `h0_k`: the Kohn-Sham Hamiltonian, as a dictionary of arrays indexed
  as `[k, orbital, orbital]` in each block.
- `density_dft_k`: the DFT one-particle density matrix in the same basis
  and array structure as `h0_k`, using the convention described below.
- `projectors`: a list of dictionaries, one for each correlated cluster,
  with arrays indexed as `[k, local_orbital, orbital]`.
- `gf_struct`: the local block structure of each cluster, and
  `gf_struct_mapping` maps the local block structure to the lattice block
  structure.
- `n_target`: the number of electrons per unit cell in the band window,
  summed over all blocks, including the uncorrelated orbitals in that window.

Use a full, uniform $k$-point mesh for the density and energy helpers below.
Their averages over $k$ assume equal weights. The Hamiltonian, DFT density
matrix, and projectors must use the same orthonormal lattice basis and
$k$-point ordering. Occupations are per spin orbital; if the DFT code
includes a spin degeneracy factor, convert to the block convention used
in `h0_k`.

At every $k$-point the projectors must have orthonormal rows, and the
correlated subspaces must not overlap. They need not cover the whole band
window. Orbitals outside the correlated subspaces remain uncorrelated.

## Interactions and double counting

Construct an interaction `h_int[i]` for each cluster, in the local
basis of `gf_struct[i]`. See [using impurity solvers](impurity_solvers.md)
for examples. With the setup below, `LatticeSolver` extracts the local
quadratic terms from `h0_k` and adds them to the impurity Hamiltonian, so
include only the interaction in `h_int[i]`.

You must also choose a double-counting prescription consistent with your
interaction and correlated subspaces. Supply its potential as a callable

```python
def double_counting(i, rho_c):
    # Return {bl: potential_matrix} for cluster i
    ...
```

`LatticeSolver` calls it each cycle with the cluster index `i` and that
cluster's density matrix from the previous cycle. In the first cycle it
uses the coherent density if no impurity density has been set. The returned
potential is subtracted from the local quadratic terms of the impurity
Hamiltonian. Pass the original DFT Hamiltonian as `h0_k` when using this
hook, so the potential is subtracted once.

The input uses the creation-first correlator convention,
`rho_c[bl][a, b] = <c_dag_a c_b>`. If your double-counting routine uses the
operator-matrix convention, `rho[a, b] = <c_dag_b c_a>`, pass `rho_c[bl].T`
to it. This matters for
complex matrices; [the charge correction section](#constructing-the-charge-correction)
explains the index ordering. The callable returns a potential matrix in
the same convention as the local Hamiltonian.

## Solving at fixed charge density

For one DFT Hamiltonian, construct an impurity solver for each cluster and
then the lattice solver

```python
from risb import LatticeSolver, helpers
from risb.impurity_solvers import FockitImpuritySolver
from risb.kweight import SmearingKWeight

impurity_solvers = [
    FockitImpuritySolver(h_int[i], gf_struct[i]) for i in range(len(gf_struct))
]
kweight = SmearingKWeight(beta=..., n_target=n_target)

S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solvers,
    update_weights=kweight.update_weights,
    projectors=projectors,
    gf_struct_mapping=gf_struct_mapping,
    double_counting=double_counting,
    check_projectors=True,
)
result = S.solve(tol=..., maxiter=..., self_consistency_tol=...)
```

Here `gf_struct`, `h_int` and `projectors` are lists over clusters, even if
there is only one. If the block names agree, omit `gf_struct_mapping` or
set it to `None`.

`result.converged` reports whether the inner stopping criterion was met;
`S.self_consistency_error` measures the RISB equations. Their distinction,
including the atomic limit, is described in
[using `LatticeSolver`](lattice_solver.md#simple-setup).

`LatticeSolver` solves the zero-temperature equations; `beta` here controls
the smearing used for $k$-space integration. For a physical finite-temperature
calculation, see [finite temperature](lattice_solver.md#finite-temperature).

:::{tip}
For ghost {{RISB}}, set the same `bath_size` in the impurity and lattice
solvers and include `S.n_ghost_electrons` in `kweight.n_target`. The DFT
occupations and charge correction still refer to physical electrons. See
[ghost quasiparticles](lattice_solver.md#ghost-quasiparticles).
:::

## Equivalent clusters

`LatticeSolver` iterates over every cluster in `gf_struct`, including
equivalent clusters. It does not identify inequivalent clusters for you.
Keep an entry for every cluster in `gf_struct`, `projectors` and
`impurity_solvers`, but solve only one representative of each equivalence
class. For the others use
{py:class}`~risb.impurity_solvers.DummyImpuritySolver`, which copies the
representative's density matrices.

For example, for two equivalent clusters in the same local basis, use
this list when constructing `LatticeSolver`

```python
from risb.impurity_solvers import DummyImpuritySolver

representative = FockitImpuritySolver(h_int[0], gf_struct[0])
impurity_solvers = [representative, DummyImpuritySolver(representative)]
```

The representative must precede its copies in the list. If the local
bases differ by a symmetry, supply `rotations` to the dummy solver; see
[using dummy impurity solvers](impurity_solvers.md#dummyimpuritysolver).
Use [symmetrizers](lattice_solver.md#enforcing-symmetries) to enforce the
corresponding equivalence of the mean-field matrices.

## Constructing the charge correction

After solving, use {py:func}`~risb.helpers.get_density_matrix_k` to obtain
the physical density matrix in the basis of `h0_k`

```python
density_k = helpers.get_density_matrix_k(
    R_full=S.R_full,
    vectors=S.bloch_vector_qp,
    kweights=S.kweights,
    rho_c=S.rho_c,
    projectors=projectors,
    gf_struct_mapping=gf_struct_mapping,
)
```

Use these attributes from the same completed solver cycle. The helper
constructs the quasiparticle contribution and replaces its local part on
each cluster with the impurity density. This includes the local charge
that is absent from the quasiparticle contribution alone.

`LatticeSolver` stores each impurity's physical density matrix as
`rho_c[bl][a, b] = <c_dag_a c_b>`. This uses the creation-first correlator
convention, as is common in the RISB literature.

`get_density_matrix_k` returns the density in the operator-matrix convention,
as is common in DFT software,
where row $a$ and column $b$ refer to the matrix element
$\rho_{ab} = \langle a | \hat{\rho}_1 | b \rangle$.
The creation and annihilation indices then appear in the opposite order.
To see why, consider one electron with orbital amplitudes $\psi_a$.
Its density operator is $|\psi\rangle\langle\psi|$, so

$$
\rho_{ab} = \psi_a\psi_b^* = \langle c_b^\dagger c_a \rangle.
$$

The same index ordering applies to the one-particle reduced density operator
of a many-electron state. In the local basis, the two arrays are therefore
related by $\rho = \rho_c^T$. `get_density_matrix_k` performs this transpose
internally and returns `density_k[bl][k, a, b] = <c_dag_kb c_ka>`.

With the operator-matrix convention, $E_{\mathrm{band}} = \mathrm{Tr}[h\rho]$
and the real-space density correction is

$$
\Delta n(\mathbf{r}) = \frac{1}{N_k} \sum_{k,ab}
\phi_{ka}(\mathbf{r})\,\Delta N_{ab}(k)\,\phi_{kb}^*(\mathbf{r}),
$$

summed over blocks, where $\phi_{ka}$ are the Bloch basis functions used
for `h0_k`. The returned arrays are indexed as `[k, orbital, orbital]`,
without the $1/N_k$ integration factor, and need no further transpose.
The diagonal occupations agree in both conventions; complex off-diagonal
entries generally differ.

Supply `density_dft_k` in the same basis and operator-matrix convention.
It need not be diagonal. If you use Kohn-Sham eigenstates with independent
band occupations, it reduces to those occupations on the diagonal.
Construct $\Delta N(k) = \rho^{\mathrm{RISB}}(k) - \rho^{\mathrm{DFT}}(k)$
as

```python
delta_n = {bl: density_k[bl] - density_dft_k[bl] for bl in density_k}
```

Pass the full matrix, including off-diagonal entries, to your DFT interface.
That interface must use the Bloch basis functions of this DFT calculation
to convert it into a real-space charge correction and update the Kohn-Sham
potential.
If your interface accepts the full density matrix instead, pass `density_k`.
States outside the chosen band window retain their DFT contribution.

Check that `density_k` and `delta_n` are Hermitian. For a fixed electron
count matching the DFT window, the sum over blocks of the average over $k$
of `Tr[delta_n]` should be zero within the solver and integration tolerances.
Keep the DFT density matrix and basis functions associated with each
correction; a correction in one basis cannot be applied directly in another.

## Calculating the total energy

To get the {{DFT}}+{{RISB}} total energy, use
{py:func}`~risb.helpers.get_band_energy` and
{py:func}`~risb.helpers.get_total_energy`

```python
interaction_energy = ...  # Sum <H_int> over all clusters
double_counting_energy = ...  # Sum over clusters for your prescription

total_energy = helpers.get_total_energy(
    dft_total_energy=...,
    dft_band_energy=helpers.get_band_energy(h0_k, density_dft_k),
    band_energy=helpers.get_band_energy(h0_k, density_k),
    interaction_energy=interaction_energy,
    double_counting_energy=double_counting_energy,
)
```

For a fockit solver, get $\langle H_{\mathrm{int}}\rangle$ with
`solver.overlap(h_int[i])`. A dummy solver has no `overlap` method;
use its representative's interaction energy. Count every equivalent
cluster's contribution to both the interaction and double-counting energies.

This evaluates $E_{\mathrm{DFT}} - E_{\mathrm{band}}^{\mathrm{DFT}}
+ E_{\mathrm{band}}^{\mathrm{RISB}} + \langle H_{\mathrm{int}}\rangle
- E_{\mathrm{dc}}$. Use the original Kohn-Sham `h0_k` and the same band
window for both band energies. The DFT total energy and all other inputs
must refer to the same DFT update and use the same energy units. Supply
$E_{\mathrm{dc}}$ from the same double-counting prescription as the
potential.

## Putting the outer loop together

You can now repeat the fixed-charge calculation and DFT charge update.
For example, the structure of an outer loop could be

```python
dft = run_dft()
for iteration in range(max_charge_iterations):
    # Read h0_k, the DFT density matrix, projectors and the electron count
    S, kweight = prepare_risb(dft)
    result = S.solve(tol=..., maxiter=..., self_consistency_tol=...)

    # Use the density and energy helpers above
    delta_n, total_energy = charge_and_energy(dft, S)
    dft_next = update_dft_charge(dft, delta_n)

    if charge_and_energy_converged(dft, dft_next, total_energy):
        break
    dft = dft_next
```

`run_dft`, `prepare_risb`, `charge_and_energy`, `update_dft_charge` and
`charge_and_energy_converged` are functions you supply. `prepare_risb`
constructs the fixed-charge problem above from the current DFT output.
`charge_and_energy` uses the density and energy examples above.

The inner solve need not converge at every charge update. With a small
`maxiter`, continue from `result.x_next` as `x0` of the next solve; if
rebuilding the solver, also restore `rho_c`. This assumes the same local
basis and parameter layout. Assess charge, energy and RISB self-consistency
together at convergence.

Regenerate `h0_k` and the projectors from each updated DFT calculation.
Constructing a new `LatticeSolver` also rebuilds the local and kinetic terms
that it extracts from `h0_k`.
