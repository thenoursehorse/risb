# Using impurity solvers

This guide shows you how to use the impurity solvers and describes a little
bit about how they do things.

:::{seealso}
:class: dropdown
[About the impurity problem](../explanations/impurity.md).
:::

## `FockitImpuritySolver`

First a `gf_struct` object has to be created that describes the correlated
subspace $\mathcal{C}_i$. It is a list of pairs, each the name of a symmetry
block and its dimension. For example, for the $e_g$ symmetry sector (two
orbitals) in a system where spin is conserved, the structure can be encoded as

```python
gf_struct = [("up_eg", 2), ("dn_eg", 2)]
```

Next a local Hamiltonian has to be constructed. It is built from the
[fockit](https://github.com/thenoursehorse/fockit) operators `c(block, index)`,
`cdag(block, index)` and `nc(block, index)`, with the blocks and indices of
`gf_struct`, and includes the interaction terms. It can include the local
quadratic terms in $\mathcal{C}_i$, but these can also be set later on. It
does not make any difference for the solver. For example, a two-orbital
Hubbard model with hopping between the orbitals can be constructed as

```python
from fockit.algebra import c, cdag, nc, zero

n_orb = 2

U = 1
h_int = zero()
for o in range(n_orb):
    h_int += U * nc("up", o) * nc("dn", o)

# Optional local terms
# V = 0.25
# for s in ["up", "dn"]:
#     h_int += V * (cdag(s, 0) * c(s, 1) + cdag(s, 1) * c(s, 0))
```

{py:mod}`risb.helpers_fockit` has helper functions for some common operators,
e.g., the Kanamori interaction
{py:func}`~risb.helpers_fockit.h_int_kanamori`.

### Constructor

The solver is instantiated as

```python
from risb.impurity_solvers import FockitImpuritySolver

impurity_solver = FockitImpuritySolver(h_int, gf_struct)
```

The interaction Hamiltonian is stored in `impurity_solver.h_int` and the block
matrix structure is stored in `impurity_solver.gf_struct`. The constructor
also builds the basis of the half-filled particle sector of the impurity
problem (physical and bath orbitals), stored in `impurity_solver.basis`.

### Setting `h_imp`

Next the impurity Hamiltonian `h_imp` has to be set with

```python
impurity_solver.set_h_imp(Lambda_c, D)
```

where `Lambda_c` and `D` are block matrices that describe the impurity problem
with the structure of `gf_struct`.
{py:meth}`~risb.impurity_solvers.FockitImpuritySolver.set_h_imp` is internally
called from within the self-consistent loop in `LatticeSolver`. The
hybridization term from `D` is stored as `impurity_solver.h_hybr` and the bath
hopping from `Lambda_c` is stored as `impurity_solver.h_bath`, both as fockit
operators.

If the non-interacting quadratic couplings are not included in `h_int`, then
they must be passed as

```python
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix)
```

where `h0_loc_matrix` is a matrix that describes the couplings, with the same
block matrix structure as `Lambda_c` and `D`. This is stored in
`impurity_solver.h0_loc` as a fockit operator.

### Setting `h_int`

If you want to update the interaction terms of the physical orbitals

```python
# A new h_int
h_int = ...

impurity_solver.set_h_int(h_int)
```

### Solving

It is solved as

```python
impurity_solver.solve()
```

There are no arguments. `solve()` builds the sparse matrix of `h_imp` in
`impurity_solver.basis` and finds its lowest eigenstate with
{py:func}`scipy.sparse.linalg.eigsh`. The ground state energy and vector are
stored in `impurity_solver.gs_energy` and `impurity_solver.gs_vector`. It then
calculates the density matrices described below for every block.

`solve()` raises a `ValueError` if `h_imp` is not Hermitian, and
{py:class}`scipy.sparse.linalg.ArpackNoConvergence` if `eigsh` does not
converge, also after two retries that each double its `ncv`. If the ground state is degenerate, the solver returns one state in
the degenerate subspace, and which one is not specified.

### Choosing the eigensolver

If you want to use the dense eigensolver {py:func}`scipy.linalg.eigh` instead

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, method="eigh")
```

It is exact, but only practical for small impurity problems. Keyword arguments
to the eigensolver are passed as

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, method_options={"ncv": 60})
```

### Symmetries

If the impurity Hamiltonian has symmetries, you can restrict the basis to one
symmetry sector with `diagonal_symmetries`, `permutation_symmetries` and
`onsite_symmetries`. They are passed to `build_basis` from fockit, which
describes each kind. The modes of the bath orbitals are
`("bath_" + block, index)`. For example, to keep only the states with total
$S_z = 0$ that are even under swapping the spins

```python
from fockit.algebra import nc, zero

Sz = zero()
for o in range(n_orb):
    for up, dn in [("up", "dn"), ("bath_up", "bath_dn")]:
        Sz += 0.5 * (nc(up, o) - nc(dn, o))

swap = {"up": "dn", "dn": "up", "bath_up": "bath_dn", "bath_dn": "bath_up"}
spin_swap = {(bl, o): (swap[bl], o) for bl in swap for o in range(n_orb)}

impurity_solver = FockitImpuritySolver(
    h_int,
    gf_struct,
    diagonal_symmetries={Sz: 0},
    permutation_symmetries=[(spin_swap, 1)],
)
```

With permutation or onsite symmetries, the ground state is written in
`impurity_solver.basis`, which only has the diagonal symmetries, to calculate
the density matrices.

The examples show each kind: $S_z$ in `examples/hubbard.py`, swapping the
layers in `examples/bilayer_hubbard.py`, swapping the spins in
`examples/kagome_hubbard.py`, and a $C_3$ rotation as on-site factors in the
basis of the trimer orbitals in `examples/decorated_honeycomb_trimer_basis.py`
and as a permutation of the sites in
`examples/decorated_honeycomb_site_basis.py`.

:::{warning}
The solver finds the lowest state in the chosen sector, so choose the sector
with the state you want, usually the ground state. Choosing another sector
targets a higher state. The eigenvalue of a state also depends on the order of
the fermion modes: the ground state is even under swapping the spins in
`examples/bilayer_hubbard.py` and odd in `examples/kagome_hubbard.py`. The
impurity Hamiltonian must have the symmetries at every step of the
self-consistent loop, e.g., enforced with `symmetrizers` in `LatticeSolver`.
`solve()` raises a `ValueError` if it does not.
:::

### Degenerate ground states

If you want a warning when the ground state is degenerate

```python
impurity_solver = FockitImpuritySolver(
    h_int, gf_struct, degenerate="warn", degenerate_tol=1e-8
)
```

This finds two states instead of one, so it is slower.

### Temperature of the impurity problem

By default the density matrices are from the ground state of the impurity
problem. Giving an inverse temperature averages them over the eigenstates of
the half-filled sector with Boltzmann weights

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, beta=40)
```

This is a convergence aid for the zero-temperature solution. Use the same
inverse temperature as the quasiparticles, e.g., in
{py:class}`~risb.kweight.SmearingKWeight`. It can help in or near a Mott
insulator. There the bath orbitals decouple from the impurity, and in its
ground state they are completely full or empty, while the quasiparticles they
must match are smeared by the temperature, which can make the self-consistent
loop slow to converge or not converge. With `beta` it can converge to the
atomic limit, where the self-consistency equations are still not satisfied,
but the observables do not depend on them (see `self_consistency_tol` in
`LatticeSolver.solve`). When the gap of the impurity problem is much larger
than the temperature, the result is the same as with the ground state. A
degenerate ground state is averaged over.

`overlap` also gives thermal averages. Only the eigenstates with a Boltzmann
weight above $10^{-14}$ relative to the ground state are found, so it is slower
than the ground state when there are many low-lying states. A sector of at most
2000 states is diagonalized as a dense matrix. In a larger one `eigsh` can
miss a state of a degenerate level, so the solver also looks for missed states
below the highest one it found.

### Grand-canonical impurity problem

For thermal averages over every particle number of the impurity problem

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, beta=40, grand_canonical=True)
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix, mu=mu)
```

The chemical potential `mu` is subtracted on the physical and bath orbitals.
The states of each particle number are found from half filling outwards,
until a particle number has a negligible weight. The `diagonal_symmetries`
only split the Hamiltonian of each particle number into blocks, which are all
in the average, and permutation and onsite symmetries cannot be used. The grand
potential $-\log(\sum e^{-\beta E}) / \beta$ over the states is
`impurity_solver.grand_potential`. `ThermalLatticeSolver` needs this mode,
and `LatticeSolver` does not use it.

### Getting the density matrices

The single-particle density matrix of the $f$ electrons in the impurity
problem in one of the blocks specified by `gf_struct` is obtained as

```python
impurity_solver.get_rho_f(block)
```

The off-diagonal density matrix between the $c$ and $f$ electrons is obtained
as

```python
impurity_solver.get_rho_cf(block)
```

The density matrix of the $c$ electrons is obtained as

```python
impurity_solver.get_rho_c(block)
```

The three terms above give the full single-particle density matrix in the
impurity problem. They are also stored for every block as the dictionaries
`impurity_solver.rho_f`, `impurity_solver.rho_cf`, and `impurity_solver.rho_c`.

### Local expectation values

Any local expectation value of an operator `Op` in the impurity problem is
obtained as

```python
impurity_solver.overlap(Op)
```

`Op` must be a fockit operator. For example, the total number of electrons and
the total spin squared on the physical orbitals are

```python
from risb.helpers_fockit import N_op, S2_op

total_number = impurity_solver.overlap(N_op(["up", "dn"], n_orb))
total_spin = impurity_solver.overlap(S2_op(["up", "dn"], n_orb))
```

To get observables in the $f$ electrons, their modes are
`("bath_" + block, index)`, and their structure is obtained as

```python
impurity_solver.gf_struct_bath
```

## `DummyImpuritySolver`

If you want to have some correlated subspaces $\mathcal{C}_i$ as inequivalent,
but they are related by some symmetry, it is not necessary to solve for the
ground state in each subspace separately.

This class is a copy of another
impurity solver. For example, to create a copy of
{py:class}`~risb.impurity_solvers.FockitImpuritySolver` it is instantiated as

```python
from risb.impurity_solvers import DummyImpuritySolver

dummy_solver = DummyImpuritySolver(impurity_solver=impurity_solver)
```

### Rotations

If the copy is related by symmetry, you can pass `rotations` as a list of
functions that operate in sequence on the dictionaries of block matrices that
{py:class}`~risb.impurity_solvers.DummyImpuritySolver` copies. Each function must return a dictionary with
the same blocks as `gf_struct`, so a rotation that mixes orbitals in different
blocks needs a `gf_struct` with those orbitals in the same block. For example, if the
correlated subspace is equivalent except that the spin is rotated,
like in an antiferromagnetic phase, this can be done with

```python
from copy import deepcopy


def rotate_spin(A):
    B = deepcopy(A)
    B["up"] = A["dn"]
    B["dn"] = A["up"]
    return B


dummy_solver = DummyImpuritySolver(impurity_solver=..., rotations=[rotate_spin])
```

:::{note}
`set_h_imp()` does nothing. `solve()` copies the density matrices stored in
the referenced impurity solver and rotates them according to the list of
functions in `rotations`. The referenced impurity solver must therefore be
solved first, i.e., come before the `DummyImpuritySolver` in the list of
impurity solvers passed to `LatticeSolver`. `solve()` also copies the
`entropy` of the referenced solver, which the rotations do not change.
:::

:::{warning}
`DummyImpuritySolver` has no `overlap` method. Rotationally invariant expectation
values can be obtained from the impurity solver that `DummyImpuritySolver` copies.
:::
