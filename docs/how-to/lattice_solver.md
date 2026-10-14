# Using`LatticeSolver`

If you want to solve a strongly correlated model on a lattice you can use
{py:class}`~risb.LatticeSolver`. This guide shows you how to use it for most kinds
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
$\mathcal{C}_i$. In {{RISB}} these terms are included in the impurity
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
`update_weights(energies)` where `energies` are a `dict[ndarray]`
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
Build `h_int` with the second-quantized operators `c`, `cdag` and `nc` from
`fockit.algebra`, whose modes are `(block, index)` following `gf_struct`. See
[Using impurity solvers](impurity_solvers.md).
:::

Next define the class that solves in the correlated subspace $\mathcal{C}_i$
the impurity problem for {{RISB}}.

```python
impurity_solver = ImpuritySolverClass(h_int, gf_struct)
```

Finally, set up the solver class and solve

```python
S = LatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solver,
    update_weights=kweight.update_weights,
)
S.solve(tol=...)
```

By default `tol` is on how much $\mathcal{R}$ and $\lambda$ change in one
iteration, which can stop changing before the {{RISB}} self-consistency
equations are satisfied, e.g., at $\mathcal{R} = 0$. So after solving, the
largest entry of the two self-consistency equations is
`S.self_consistency_error`, and `solve` warns if it is above
`self_consistency_tol` (default $10^{-6}$)

```python
S.solve(tol=..., self_consistency_tol=1e-6)
```

In the atomic limit ($\mathcal{R} = 0$, e.g., a Mott insulator) the equations
are not satisfied, but there the bath of the impurity problem decouples and
the observables do not depend on it.

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

You must construct an impurity solver for each subspace $\mathcal{C}_i$ as
a list

```python
impurity_solvers = [impurity_solver_1, impurity_solver_2, ...]
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
                  impurity_solvers = impurity_solvers,
                  projectors = projectors,
                  gf_struct_mapping = gf_struct_mapping,
                  ...
)
```

A projector that depends on $k$ is an array indexed as `[k, row, column]`. At
every $k$-point the projectors must have orthonormal rows and not overlap
between clusters. Orbitals of `h0_k` that are in no cluster have
$\mathcal{R} = 1$ and $\lambda = 0$, and their local terms are in
`S.h0_kin_k`. To check the projectors, so that `LatticeSolver` raises a
`ValueError` if they are not orthonormal or overlap

```python
S = LatticeSolver(..., projectors=projectors, check_projectors=True)
```

## Ghost quasiparticles

In ghost {{RISB}}[^LanataGhost2017] [^Lanata2022] each cluster has more
quasiparticles than orbitals. The extra (ghost) quasiparticles describe the
Hubbard bands, e.g., a Mott insulator that still has a finite double
occupancy. Give the number of quasiparticles of each block of each cluster as
`bath_size`, which is also the number of bath orbitals of its impurity
problem. E.g., for three quasiparticles for the one orbital of each spin

```python
bath_size = {"up": 3, "dn": 3}
impurity_solver = FockitImpuritySolver(h_int, gf_struct, bath_size=bath_size)
S = LatticeSolver(...
                  impurity_solvers = impurity_solver,
                  bath_size = [bath_size],
                  ...
)
```

The `bath_size` of a block must be a multiple of the size of the block, and
without `bath_size` there are no ghosts. The quasiparticle bands hold
`S.n_ghost_electrons` more electrons than the physical ones, half the number
of ghosts, so a filling `n_target` of the $k$-space integration weights must
include them

```python
kweight = SmearingKWeight(beta=..., n_target=n_target)
S = LatticeSolver(..., update_weights=kweight.update_weights, bath_size=...)
kweight.n_target = n_target + S.n_ghost_electrons
```

In a Mott insulator the ground state of the impurity problem can break a
symmetry of the solution, e.g., it can be spin polarized. Enforce the
symmetry with the symmetrizers below. For a paramagnet it is not enough to
restrict the impurity problem to $S_z = 0$, because the bath can make up the
spin of the physical orbitals. If you set the starting `S.R` yourself, give
the ghosts a non-zero entry, because a ghost with $\mathcal{R} = 0$ stays
decoupled.

With ghosts `S.Z` $= \mathcal{R}^{\dagger} \mathcal{R}$ is not the
quasiparticle weight, but the spectral weight summed over all frequencies. The
quasiparticle weight and the self-energy of block `bl` of cluster `i` are

```python
from risb import helpers

Z = helpers.get_z(S.R[i][bl], S.Lambda[i][bl], kweight.mu)
Sigma = helpers.get_sigma_w(
    omega, S.R[i][bl], S.Lambda[i][bl], kweight.mu, S.h0_loc_matrix[i][bl]
)
```

See [the self-energy and the quasiparticle weight](../explanations/self_energy.md)
for how they are obtained, and `examples/ghost_hubbard.py` for the Hubbard
model on the Bethe lattice with and without ghosts.

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
        A[i]["up"] = 0.5 * (A[i]["up"] + A[i]["dn"])
        A[i]["dn"] = A[i]["up"]
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
                  symmetrizers = [paramagnetism, symmetry1, ...],
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

`root` is called exactly the same as what {py:func}`scipy.optimize.root`
requires. It must be a callable function that takes as input
`root(S._target_function, x0, **kwargs)`. Here `S._target_function`
by default returns a `tuple[ndarray,ndarray]` of a flattened ndarray `x_new`
of a new `Lambda` and `R` matrix and a flattened ndarray `x_err` of the error,
which by default is the change `x_new - x` (see `error_fun` below).

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

For example, to use {py:func}`scipy.optimize.root`

```python
from scipy.optimize import root
S = LatticeSolver(...
                  root = root,
                  return_x_new = False,
                  ...
)
S.solve(tol = , method = 'broyden1, hybr, krylov, etc', other kwargs = )
```

For `krylov`, `broyden1`, `broyden2`, `anderson`, `linearmixing`,
`diagbroyden` and `excitingmixing`, scipy turns `tol` into a stop on the step
size. The {{RISB}} equations have directions in which the error does not
change: rotating the basis of the quasiparticles maps a solution to another
solution, so the solutions are not isolated, and then the steps need not
shrink. Stop on the size of the error instead

```python
S.solve(method="krylov", options={"fatol": 1e-10})
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

This will make most impurity solvers faster, and the other matrix
operations faster. For many systems forcing the matrices to be real is fine.

[^LanataGhost2017]:
    [N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević,
    _Emergent Bloch excitations in Mott matter_,
    Phys. Rev. B **96**, 195126 (2017)](https://doi.org/10.1103/PhysRevB.96.195126)

[^Lanata2022]:
    [N. Lanatà,
    _Operatorial formulation of the ghost rotationally invariant slave-boson
    theory_,
    Phys. Rev. B **105**, 045111 (2022)](https://doi.org/10.1103/PhysRevB.105.045111)
