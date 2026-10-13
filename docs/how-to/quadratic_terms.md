# Constructing tight-binding models

This guide shows various methods for constructing the non-interacting
terms of a model required for {{RISB}}.

## Constructing `h0_k`

:::{warning}
The ordering of `k` matters for some $k$-space integration methods, e.g.,
linear tetrahedron. If that is the case you must permute `h0_k` to the
required ordering.
:::

There are many ways to construct the hopping matrices in $k$-space. Below
are a few of them that have been useful. Any method can be used as long
as it is possible to return an `numpy.ndarray` indexed as
`k, orb_i, orb_j` where `k` is a $k$-point on a meshgrid and `orb_i` is one of
the `n_orb` orbitals that corresponds to a site, orbital, or spin in a unit
cell. These can possibly be partitioned into symmetry blocks, e.g., by spin.

### Explicit method

If you have an equation for the matrix representation of $\hat{H}_0$
at each $k$-point, then this is a simple way to code it to have the
correct structure for {{RISB}}.

First construct a Monkhorst-Pack $k$-point mesh in fractional coordinates

```python
import numpy as np
from itertools import product

# The spatial dimensions of the problem
d = ...

# How many points in each spatial dimension
n_k_1 = ...
...
n_k_d = ...
...
n_k_list = [n_k1, ..., n_k_d]

# Total number of k-points
n_k = np.product(n_k_list)

# Shift to apply to each spatial dimension if want to move it off
# high symmetry points. It should be a value between 0 and 1 (usually 0.5)
shift_1 = ...
...
shift_d = ...
shift_list = [shift_1, ..., shift_d]

# Method 1: Using np.meshgrid

# Create linear array in each spatial dimension
k_linear_list = [
    np.linspace(0, 1, n_k_list[i], endpoint=False) + shift_list[i] / n_k_list[i]
    for i in range(d)
]

# Create a meshgrid
k_mesh = np.meshgrid(*k_linear_list, indexing="ij")

# Make it a list indexed as idx, k1, k2, ...
k_mesh = np.reshape(k_mesh, (d, -1)).T

# Method 2: Using an explicit for loop

# Create empty mesh and populate it
k_mesh = np.empty(shape=(n_k, d))
for idx, coords in enumerate(
    product(*[range(n_k_list[i]) for i in range(len(n_k_list))])
):
    for dim in range(d):
        k_mesh[idx, dim] = (coords[dim] + shift_list[dim]) / n_k_list[dim]
```

If your function for $\hat{H}_0$ is in a different basis then you will
have to rotate `k_mesh` into this basis. For example, to Cartesian
coordinates can be done as

```python
# The unit cell lattice vectors
a_1 = [float_1, ..., float_d]
...
a_d = [float_1, ..., float_d]
...

# A matrix of unit cell lattice vectors where each column is a vector
A = np.array([a_1, a_2, ...]).T

# The reciprocol lattice vectors as a matrix where each column is a vector
G = 2 * np.pi * np.linalg.inv(A).T

# Rotate mesh into this basis as k_cartesian = G @ k_fractional

# Method 1: Using numpy broadcasting
k_mesh = (G @ k_mesh.T).T

# Method 2: Using an explicit for loop
for k in range(k_mesh.shape[0]):
    k_mesh[k, :] = G @ k_mesh[k, :]
```

Next, construct `h0_k` as

```python
h0_k = dict()

# For each symmetry block:

# Name of block
bl_name = 'some string'

# Number of orbitals (including sites, orbitals, spin, etc) at each k-point
n_orb = ...

h0_k[bl_name] = np.zeros([n_k, n_orb, n_orb], dtype=complex)
for k in range(n_k):
    h0_k[bl_name][k][...] = n_orb by n_orb array that is the function for h0_k in symmetry block bl_name
```

:::{tip}
The [TRIQS](https://triqs.github.io/) library can also construct tight-binding
models.
:::

### Using [TBmodels](https://github.com/Z2PackDev/TBmodels)

Work in progress.

## Constructing `h0_loc` as a matrix

:::{seealso}
[Using projectors](projectors.md).
:::

If you want to get $\hat{H}_i^{\mathrm{loc}}$ for each correlated
space $\mathcal{C}_i$ calculated from `h0_k` then there are some helper
functions.

If there are no projectors and the correlated space is the entire unit cell

```python
from risb.helpers import get_h0_loc_matrix

h0_loc_matrix = dict()
for bl in h0_k.keys():
    h0_loc_matrix[bl] = get_h0_loc_matrix(h0_k[bl])
```

If there are multiple correlated spaces, indexed as `i`

```python
# Total number of clusters on the lattice
n_clusters = ...

# A list of gf_struct objects in each correlated space
gf_struct = ...

# A list of mappings of the block structure from each correlated
# subspace to the larger space of h0_k
gf_struct_mapping = ...

# A list of projectors into each correlated subspace
projectors =

h0_loc_matrix = [dict() for i in range(n_clusters)]
for i in range(n_clusters):
    for bl, bl_size in gf_struct:
        bl_full = gf_struct_mapping[bl]
        h0_loc_matrix[i][bl] = get_h0_loc_matrix(h0_k[bl_full], projectors[i][bl] )
```

### As a fockit operator

If you have a block matrix representation of a single-particle operator and
you want it as a fockit operator

```python
from risb.helpers_fockit import matrix_to_Op

# The block matrix
A = ...

# A dictionary of the operator in each block
Op = matrix_to_Op(A)
```

The operator in each block `bl` is
`sum_ij A[bl][i, j] * cdag(bl, i) * c(bl, j)`.

## Constructing `h0_kin_k`

:::{seealso}
[Using projectors](projectors.md).
:::

If you want only the kinetic terms in `h0_k` with all of the local terms
`h0_loc` from $\hat{H}_i^{\mathrm{loc}}$ removed then you can use

```python
from risb.helpers import get_h0_kin_k

# A list of gf_struct objects in each correlated space
gf_struct = ...

# A list of mappings of the block structure from each correlated
# subspace to the larger space of h0_k
gf_struct_mapping = ...

# A list of projectors into each correlated subspace
projectors =

h0_kin_k = get_h0_kin_k(h0_k, projectors, gf_struct_mapping)
```
