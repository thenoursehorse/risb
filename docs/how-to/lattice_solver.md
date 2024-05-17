# Using`LatticeSolver`

If you want to solve a strongly correlated model on a lattice you can use
:py:class:`LatticeSolver`. This guide shows you how to use it for most kinds
of problems you might want to solve.

## Simple setup

First define a non-interacting coupling matrix that describes electron
hopping between sites and orbitals and orbital energies, with some defined
block structure. E.g., if spin is a good quantum number and there are
$N$ orbitals per unit cell

```python
h0_k = {'up' : k by N by N ndarray, 'dn' : k by N by N ndarray}
```

:::{tip}
:class: dropdown
`h0_k` can contain any local hopping terms `h0_loc` that are in the subspace
$\mathcal{C}_i$. In {{RISB}} these terms are included in the embedding
Hamiltonian and are typically separated out from `h0_k`. `LatticeSolver` works
these terms out automatically. But they can be included in
the interactions `h_int` by hand if you want. It makes no difference to
the solver.
:::

Next define a class that determines the integration weight at each $k$ point
on the lattice, e.g., `class:SmearingKWeight`.

```python
kweight = KWeightClass(...)
```

It must have a member function `update_weights()` that is called as
`update_weights(energies, **kwargs)` where `energies` are a `dict[ndarray]`
that are the eigenenergies of `h0_k` in each block (e.g., `'up'` and `'dn'`
spin blocks).

Next define the local block matrix structure of the correlated subspace
as a list of pairs. Each pair is the name of a block `bl` and the number of
orbitals in that block as `n_orb`.

```python
gf_struct = [(bl, n_orb), ...]
```

Following the block structure of `gf_struct`, construct the interacting
Hamiltonian in the correlated subspace $\mathcal{C}_i$. This can include
the quadratic terms in a cluster if you want.

```python
h_int = ...
```

:::{tip}
:class: dropdown
The {{TRIQS}} library makes it very simple to define
`h_int` as second-quantized operators. All of the `Embedding` classes we
provide assume that `h_int` is a {{TRIQS}} operator.
:::

Next define the class that solves in the correlated subspace $\mathcal{C}_i$
the impurity problem for {{RISB}}.

```python
embedding = EmbeddingClass(h_int, gf_struct)
```

Finally, set up the solver class and solve

```python
S = LatticeSolver(h0_k = h0_k,
                  gf_struct = gf_struct,
                  embedding = embedding,
                  update_weights = kweight.update_weights,
)
S.solve(tol = ...)
```

## Multiple clusters

:::{seealso}
:class: dropdown
[Using projectors](../how-to/projectors.md) for more details and examples.
:::

If you want a correlated subspace $\mathcal{C}_i$ for each cluster $i$ you
must construct a `gf_struct` for each subspace as a list

```python
gf_struct = [gf_struct_1, gf_struct_2, ...]
```

You must construct an embedding solver for each subspace $\mathcal{C}_i$ as
a list

```python
embedding = [embedding_1, embedding_2, ...]
```

You must construct a projector from the larger space of `h0_k` to each
subspace $\mathcal{C}_i$ as a list

```python
projectors = [projector_1, projector_2, ...]
```

(Optional) Depending on how the problem is set up, you might also need to
construct a mapping from the block structure defined in each `gf_struct[i]`
to the larger block structure space of `h0_k`. For each correlated subspace
$\mathcal{C}_i$ `gf_struct_mapping_i` is a dictionary of
`str` $\rightarrow$ `str`

```python
gf_struct_mapping_1 = {'bl_in_gf_struct_1' : 'bl_in_h0_k', ...}
...
gf_struct_mapping = [gf_struct_mapping_1, gf_struct_mapping_2, ...]
```

These lists are passed to the solver as

```python
S = LatticeSolver(...
                  gf_struct = gf_struct,
                  embedding = embedding,
                  projectors = projectors,
                  gf_struct_mapping = gf_struct_mapping,
                  ...
)
```

## Enforcing symmetries

:::{admonition} Thanks <3
This way to do symmetries is unashamedly taken from
[TRIQS/hartree_fock](https://triqs.github.io/hartree_fock). There are other
ways to enforce symmetries on the matrices that we also implement at the
same time, but this is very easy and quick for a user to cater to their
specific needs.
:::

If you want to enforce symmetries this is done through functions that are
called at each self-consistent loop. For example, if the block structure in
each cluster is just the spin, paramagnetism can be enforced as

```python
def paramagnetism(A):
    n_clusters = len(A)
    for i in range(n_clusters):
        A[i]['up'] = 0.5 * (A[i]['up'] + A[i]['dn'])
        A[i]['dn'] = A[i]['up']
    return A
```

If you want to enforce another symmetry then you can define another
function in a similar way

```python
def symmetry1(A):
    do something here, maybe only to a single cluster i
    return A
```

The functions are passed as a list to the solver

```python
S = LatticeSolver(...
                  symmetries = [paramagnetism, symmetry1, ...],
                  ...
)
```

and are called in the sequence they are given in the list.

## Using other functions to find a root

If you want to change the function that finds the roots of the
self-consistent procedure you can specify it with

```python
S = LatticeSolver(...
                  root = root,
                  ...
)
```

`root` is called exactly the same as what :py:func:`scipy.optimize.root`
requires. It must be a callable function that takes as input
`root(S._target_function, x0, args=, **kwargs)`. Here `S._target_function`
by default returns a `tuple[ndarray,ndarray]` of a flattened ndarray `x_new`
of a new `Lambda` and `R` matrix and a flattened ndarray `x_err` of the error
from the `f1` and `f2` root functions.

If you do not want `S._target_function` to return `x_new` and only return
`x_err` then you can specify

```python
S = LatticeSolver(...
                  root = root,
                  return_x_new = False,
                  ...
)
```

You can change whether `x_err` is from the root functions `f1` and
`f2` or as a recursion from successive changes to `x` as
`x_err = x_new - x` with

```python
S = LatticeSolver(...
                  root = root,
                  error_fun = 'root' or 'recursion',
                  ...
)
```

For example, to use :py:func:`scipy.optimize.root`

```python
from scipy.optimize import root
S = LatticeSolver(...
                  root = root,
                  return_x_new = False,
                  ...
)
S.solve(tol = , method = 'broyden1, hybr, krylov, etc', other kwargs = )
```

:::{warning}
In our experience scipy's `root` functions do not work as well as either
standard linear mixing or {{DIIS}} (which seems to work very well for
{{RISB}}, and is the default method).
:::

## Real or complex?

In general the matrices in {{RISB}} should be complex. If you choose a basis
where you know they all will be real and you want to force the matrices to
be real you can do that with

```python
S = LatticeSolver(...
                  force_real = True,
                  ...
)
```

This will make most `EmbeddingClass` solvers faster, and the other matrix
operations faster. For many systems forcing the matrices to be real is fine.
