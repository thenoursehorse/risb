# Using impurity solvers

This guide shows you how to use the impurity solvers and describes a little
bit about how they do things.

:::{seealso}
:class: dropdown
[About the impurity problem](../explanations/impurity.md).
:::

## `AtomDiagImpuritySolver`

First a `gf_struct` object has to be created that describes the correlated
subspace $\mathcal{C}_i$. This object should be constructed in the same way
as {{TRIQS}} Green's function objects are created: as a list of tuple pairs
describing a symmetry block and its dimension. For example, for the $e_g$
symmetry sector (two orbitals) in a system where spin is conserved, the
structure can be encoded as

```python
gf_struct = [("up_eg", 2), ("dn_eg", 2)]
```

Next a local Hamiltonian has to be constructed. This must
be a {{TRIQS}} operator that includes the interaction terms. It can include
the local quadratic terms in $\mathcal{C}_i$, but these can also be set
later on. It does not make any difference for the solvers. For example,
a two-orbital Hubbard model with hopping between the orbitals can be
constructed as

```python
from triqs.operators import Operator, c_dag, c, n

n_orb = 2

U = 1
h_int = Operator()
for o in range(n_orb):
    h_int += U * n("up", o) * n("dn", o)

# Optional local terms
# V = 0.25
# for s in ['up', 'dn']:
#    h_int += V * ( c_dag(s, 0) * c(s, 1) + c_dag(s, 1) * c(s, 0) )
```

### Constructor

The solver is instantiated as

```python
from risb.impurity_solvers import AtomDiagImpuritySolver

impurity_solver = AtomDiagImpuritySolver(h_int, gf_struct)
```

The interaction Hamiltonian is stored in `impurity_solver.h_int` and the block
matrix structure is stored in `impurity_solver.gf_struct`.

### Setting `h_imp`

Next the impurity Hamiltonian `h_imp` has to be set with

```python
impurity_solver.set_h_imp(Lambda_c, D)
```

where `Lambda_c` and `D` are block matrices that describe the impurity problem
with the structure of `gf_struct`.
:py:meth:`AtomDiagImpuritySolver.set_h_imp` is internally called from within
the self-consistent loop in the {{RISB}} `Solver` classes. The hybridzation
term from `D` is stored as `impurity_solver.h_hybr` and the bath hopping from
`Lambda_c` stored as `impurity_solver.h_bath` as {{TRIQS}} operators.

If the non-interacting quadratic couplings are not included in `h_int`, then
they must be passed as

```python
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix)
```

where `h0_loc_matrix` is a matrix that describes the couplings, with the same
block matrix structure as `Lambda_c` and `D`. This is stored in
`impurity_solver.h0_loc` as a {{TRIQS}} operator.

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

There are no arguments. `solve()` calls
:py:class:`triqs.atom_diag.AtomDiag`, passes it `h_imp`
and restricts the Hilbert space to the half-filled particle sector. The
instance of this class is stored in `impurity_solver.ad`. See the {{TRIQS}}
documentation for its function. It then calculates the density matrices
described below for every block.

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

`Op` must be a {{TRIQS}} operator. See the {{TRIQS}} manual for helper
functions to construct some common observables. To get observables
in the $f$ electrons, the structure is obtained as

```python
impurity_solver.gf_struct_bath
```

The structure of the whole impurity problem (physical and bath orbitals) is
obtained as

```python
impurity_solver.gf_struct_imp
```

## `DummyImpuritySolver`

If you want to have some correlated subspaces $\mathcal{C}_i$ as inequivalent,
but they are related by some symmetry, it is not necessary to solve for the
ground state in each subspace separately.

This class is a copy of another
impurity solver. For example, to create a copy of
:py:class:`AtomDiagImpuritySolver` it is instantiated as

```python
from risb.impurity_solvers import DummyImpuritySolver

impurity_solver = DummyImpuritySolver(impurity_solver=atom_diag_solver)
```

### Rotations

If the copy is related by symmetry, you can pass `rotations` as a list of
functions that operate in sequence on the dictionaries of block matrices that
:py:class:`DummyImpuritySolver` copies. Each function must return a dictionary with
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


impurity_solver = DummyImpuritySolver(impurity_solver=..., rotations=[rotate_spin])
```

:::{note}
`set_h_imp()` does nothing. `solve()` copies the density matrices stored in
the referenced impurity solver and rotates them according to the list of
functions in `rotations`. The referenced impurity solver must therefore be
solved first, i.e., come before the `DummyImpuritySolver` in the list of
impurity solvers passed to `LatticeSolver`.
:::

:::{warning}
`DummyImpuritySolver` has no `overlap` method. Rotationally invariant expectation
values can be obtained from the impurity solver that `DummyImpuritySolver` copies.
:::
