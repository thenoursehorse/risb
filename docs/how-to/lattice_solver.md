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

In ghost {{RISB}}[^LanataGhost2017] [^Lanata2022] the quasiparticle
Hamiltonian has $B$ orbitals for each orbital of a cluster instead of one, so
$\mathcal{R}$ is $B \nu \times \nu$ for $\nu$ orbitals. The $B - 1$ extra
quasiparticle orbitals per orbital, the ghosts, give the self-energy of the
cluster up to $(B - 1) \nu$ poles, while without ghosts it is linear in frequency
(see [the self-energy and the quasiparticle weight](../explanations/self_energy.md)).
The poles can describe, e.g., the Hubbard bands and the gap of a Mott
insulator. $B = 1$ is {{RISB}} without ghosts.

Set $B$ with `bath_size`, the number of quasiparticle orbitals of each block,
which is $B$ times the size of the block. It is also the number of bath
orbitals of the impurity problem, so both solvers take it. E.g., for $B = 3$

```python
B = 3
bath_size = {bl: B * bl_size for bl, bl_size in gf_struct}
impurity_solver = FockitImpuritySolver(h_int, gf_struct, bath_size=bath_size)
S = LatticeSolver(...
                  impurity_solvers = impurity_solver,
                  bath_size = [bath_size],
                  ...
)
```

The impurity problem is half filled, so the ghosts hold $(B - 1) / 2$
electrons per orbital. The quasiparticle bands then hold
`S.n_ghost_electrons` more electrons than the physical ones, and the filling
`n_target` of the $k$-space integration weights must include them

```python
kweight = SmearingKWeight(beta=..., n_target=n_target)
S = LatticeSolver(..., update_weights=kweight.update_weights, bath_size=...)
kweight.n_target = n_target + S.n_ghost_electrons
```

Half filling needs an even number of ghosts in a cluster, and for a paramagnet
an even number for each spin. Otherwise the impurity problem has an odd number
of electrons, so it has no $S_z = 0$ state and its ground state is degenerate.
With the same $B$ for every orbital, an odd $B$ always works, and an even $B$
only if each spin has an even number of orbitals.

Blocks can have different $B$, e.g., a larger $B$ for the more correlated
orbitals. They still share one impurity problem, with all the interactions of
the cluster. E.g., for a $d$ shell with blocks of the $t_{2g}$ and $e_g$
orbitals of each spin, $B = 3$ for $t_{2g}$ and $B = 2$ for $e_g$ give
$6 + 2 = 8$ ghosts per spin

```python
bath_size = {"up_t2g": 9, "dn_t2g": 9, "up_eg": 4, "dn_eg": 4}
```

while $B = 2$ for $t_{2g}$ and $B = 3$ for $e_g$ give $3 + 4 = 7$.

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
for how they are obtained.

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

## Choosing the optimizer

`S.solve()` uses {py:class}`~risb.optimize.DIIS` by default. If you want
another optimizer, or other options for it

```python
from risb.optimize import DIIS, LinearMixing

S.solve(optimizer=DIIS(history_size=10, alpha=0.5), tol=1e-10, maxiter=500)
S.solve(optimizer=LinearMixing(alpha=0.3))
```

The optimizer decides when the solve has converged: when `Lambda` and `R`
stop changing (`error_fun="recursion"`, the default), or when the root
functions `f1` and `f2` are zero (`error_fun="root"`, the default of
`ThermalLatticeSolver`)

```python
S.solve(optimizer=DIIS(error_fun="root"))
```

If you want to use {py:func}`scipy.optimize.root`, give its keyword arguments
to {py:class}`~risb.optimize.ScipyRoot`

```python
from risb.optimize import ScipyRoot

S.solve(optimizer=ScipyRoot(method="krylov", options={"fatol": 1e-10}))
```

For `krylov`, `broyden1`, `broyden2`, `anderson`, `linearmixing`,
`diagbroyden` and `excitingmixing`, give `fatol` as above, so that scipy
stops on the size of the error and not of the step.

:::{warning}
In our experience scipy's `root` functions do not work as well as either
standard linear mixing or {{DIIS}} (which seems to work very well for
{{RISB}}, and is the default method).
:::

`S.solve()` returns a {py:class}`~risb.optimize.SolveResult`

```python
result = S.solve(tol=1e-10)
result.converged, result.cycles, result.error
```

Afterwards `S.R`, `S.Lambda` and every other attribute of `S`, and the
impurity solvers, are those of `result.x`, the last guess of the solve.

## Running code during the solve

If you want to run your own code after every cycle, e.g., to print the
progress or change the optimizer, give a function that takes the
{py:class}`~risb.optimize.Driver` and the lattice solver

```python
def progress(driver, S):
    print(driver.cycle, driver.error)


S.solve(callbacks=[progress])
```

A callback can end the solve with `driver.stop()`, or replace
`driver.optimizer`.

For example, {py:class}`~risb.optimize.OnPlateau` restarts the optimizer when
the error stops falling

```python
from risb.optimize import OnPlateau

S.solve(callbacks=[OnPlateau(patience=20)])
```

## Finite temperature

`LatticeSolver` is at zero temperature. If you want ghost {{RISB}} at a finite
temperature[^Giuli2026], use {py:class}`~risb.ThermalLatticeSolver` with an
impurity solver of the thermal state over every particle number at the same
inverse temperature

```python
from risb import ThermalLatticeSolver
from risb.impurity_solvers import FockitThermalImpuritySolver

impurity_solver = FockitThermalImpuritySolver(
    h_int, gf_struct, beta=beta, bath_size=bath_size
)
S = ThermalLatticeSolver(
    h0_k=h0_k,
    gf_struct=gf_struct,
    impurity_solvers=impurity_solver,
    beta=beta,
    mu=mu,
    bath_size=[bath_size],
)
S.solve(tol=...)
```

The chemical potential `mu` is fixed. If you want a fixed number of electrons
per unit cell instead

```python
S = ThermalLatticeSolver(...
                         n_target = n_target,
                         ...
)
```

and the chemical potential is `S.mu` after solving. `n_target` does not
include `S.n_ghost_electrons`.

By default the solve uses {py:class}`~risb.optimize.OnPlateau`. When the
error stops falling it restarts {{DIIS}}, or, near a solution, anchors the
fits with `S.anchor()`. At high temperature the equations barely fix some
combinations of `Lambda`, `R`, `D` and `Lambda_c`, and the anchor pins them.
If you want to turn this off

```python
S.solve(callbacks=[])
```

The grand potential and the entropy per unit cell are

```python
from risb import helpers

omega = helpers.get_thermal_grand_potential(
    energies=S.energies_qp,
    R=S.R,
    Lambda=S.Lambda,
    D=S.D,
    Lambda_c=S.Lambda_c,
    beta=beta,
    mu=S.mu,
    impurity_grand_potential=[impurity_solver.grand_potential],
)
entropy = helpers.get_thermal_entropy(
    ...
    impurity_entropy=[impurity_solver.entropy],
)
```

If you find more than one solution, the physical one has the lowest grand
potential.

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

[^Giuli2026]:
    [S. Giuli, T.-H. Lee, Y.-X. Yao, G. Kotliar, A. E. Ruckenstein, O. Gingras,
    and N. Lanatà,
    _Unifying variational and dynamical quantum embedding: from ghost
    Gutzwiller approximation to dynamical mean-field theory_,
    arXiv:2603.20559 (2026)](https://arxiv.org/abs/2603.20559)
