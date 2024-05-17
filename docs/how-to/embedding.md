# Using embedding classes

This guide shows you how to use the embedding classes and describes a little
bit about how they do things.

:::{seealso}
:class: dropdown
[About the embedding Hamiltonian](../explanations/embedding.md).
:::

## `EmbeddingAtomDiag`

First a `gf_struct` object has to be created that describes the correlated
subspace $\mathcal{C}_i$. This object should be constructed in the same way
as {{TRIQS}} Green's function objects are created: as a list of tuple pairs
describing a symmetry block and its dimension. For example, for the $e_g$
symmetry sector (two orbitals) in a system where spin is conserved, the
structure can be encoded as

```python
gf_struct = [ ('up_eg', 2),  ('dn_eg', 2)]
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
    h_int += U * n('up', o) * n('dn', o)

# Optional local terms
#V = 0.25
#for s in ['up', 'dn']:
#    h_int += V * ( c_dag(s, 0) * c(s, 1) + c_dag(s, 1) * c(s, 0) )
```

### Constructor

The solver is instantiated as

```python
from risb.embedding import EmbeddingAtomDiag
embedding = EmbeddingAtomDiag(h_int, gf_struct)
```

The interaction Hamiltonian is stored in `embedding.h_int` and the block
matrix structure is stored in `embedding.gf_struct`.

### Setting `h_emb`

Next the embedding Hamiltonian `h_emb` has to be set with

```python
embedding.set_h_emb(Lambda_c, D)
```

where `Lambda_c` and `D` are block matrices that describe the impurity problem
with the structure of `gf_struct`.
:py:meth:`EmbeddingAtomDiag.set_h_emb` is internally called from within
the self-consistent loop in the {{RISB}} `Solver` classes. The hybridzation
term from `D` is stored as `embedding.h_hybr` and the bath hopping from
`Lambda_c` stored as `embedding.h_bath` as {{TRIQS}} operators.

If the non-interacting quadratic couplings are not included in `h_int`, then
they must be passed as

```python
embedding.set_h_emb(Lambda_c, D, h0_loc_matrix)
```

where `h0_loc_matrix` is a matrix that describes the couplings, with the same
block matrix structure as `Lambda_c` and `D`. This is stored in
`embedding.h0_loc` as a {{TRIQS}} operator.

### Setting `h_int`

If you want to update the interaction terms on the impurity

```python
# A new h_int
h_int = ...

embedding.set_h_int(h_int)
```

### Solving

It is solved as

```python
embedding.solve()
```

There are no arguments. All `solve()` does is call
:py:class:`triqs.atom_diag.AtomDiag`, passes it `h_emb`
and restricts the Hilbert space to the half-filled particle sector. The
instance of this class is stored in `embedding.ad`. See the {{TRIQS}}
documentation for its function.

### Getting the density matrices

The single-particle density matrix of the $f$ electrons in the embedding
space in one of the blocks specified by `gf_struct` is obtained as

```python
embedding.get_rho_f(block)
```

The off-diagonal density matrix between the $c$ and $f$ electrons is obtained
as

```python
embedding.get_rho_cf(block)
```

The density matrix of the $c$ electrons is obtained as

```python
embedding.get_rho_c(block)
```

The three terms above give the full single-particle density matrix in the
embedding space.

### Local expectation values

Any local expectation value of an operator `Op` in the embedding space is
obtained as

```python
embedding.overlap(Op)
```

`Op` must be a {{TRIQS}} operator. See the {{TRIQS}} manual for helper
functions to construct some common observables. To get observables
in the $f$ electrons, the structure is obtained as

```python
embedding.gf_struct_bath
```

The whole embedding space structure is obtained as

```python
embedding.gf_struct_emb
```

## `EmbeddingDummy`

If you want to have some correlated subspaces $\mathcal{C}_i$ as inequivalent,
but they are related by some symmetry, it is not necessary to solve for the
ground state in each subspace separately.

This class is a copy of another
`Embedding` class. For example, to create a copy of
:py:class:`EmbeddingAtomDiag` it is instantiated as

```python
from risb.embedding import EmbeddingDummy
embedding = EmbeddingDummy(embedding = embedding_atom_diag_instance)
```

### Rotations

If the copy is related by symmetry, you can pass `rotations` as a list of
functions that operate in sequence on the block matrices that
:py:class:`EmbeddingDummy` returns. For example, if the
correlated subspace is equivalent except that the spin is rotated,
like in an antiferromagnetic phase, this can be done with

```python
from itertools import deepcopy
def rotate_spin(A):
    B = deepcopy(A)
    B['up'] = A['dn']
    B['dn'] = A['up']
    return B

embedding = EmbeddingDummy(embedding = ...,
                           rotations = [rotate_spin])
```

:::{note}
`set_h_emb()` and `solve()` are methods that just `pass` and do nothing.
The density matrix functions `get_rho_f`, `get_rho_cf`, and `get_rho_c` grab
what is stored in `embedding_atom_diag` and rotates according to the list of
functions in `rotations`.
:::

:::{warning}
The operator passed to `overlap` is not rotated. Likely only rotationally
invariant quantities make sense to calculate, which are can be obtained from
the embedding class that `EmbeddingDummy` copies.
:::
