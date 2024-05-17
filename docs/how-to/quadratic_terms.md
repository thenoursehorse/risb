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
k_linear_list = [np.linspace(0, 1, n_k_list[i], endpoint = False) + shift_list[i] / n_k_list[i] for i in range(d)]

# Create a meshgrid
k_mesh = np.meshgrid(*k_linear_list, indexing='ij')

# Make it a list indexed as idx, k1, k2, ...
k_mesh = np.reshape(k_mesh, (d,-1)).T

# Method 2: Using an explicit for loop

# Create empty mesh and populate it
k_mesh = np.empty(shape=(n_k, d))
for idx, coords in enumerate(product( *[range(n_k_list[i]) for i in range(len(n_k_list))] )):
    for dim in range(d):
        k_mesh[idx,dim] = (coords[dim] + shift_list[dim]) / n_k_list[dim]
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
    k_mesh[k,:] = G @ k_mesh[k,:]
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

### Using {{TRIQS}} lattice tools

If you know the positions of the orbitals in real-space then there is a
(currently poorly documented) way to do this in {{TRIQS}}. Here is some
complimentary information on top of what is provided by {{TRIQS}}.

First specify the lattice vectors as

```python
units = [a_1, ..., a_d]
```

The allowed number of spatial dimensions is one of $d = 1, 2, 3$.

Next give a list of tuples that give the positions of each atom and orbital in
the unit cell. The units are in fractional coordinates of the lattice vectors
that are specified in `units`, and there are `n_orb` total sites, orbitals,
and maybe spin.

```python
orbital_positions = [
    (frac of a_1, ... , frac of a_d),     # Position of orbital 1
    ...
    (frac of a_1, ... , frac of a_d),     # Position of orbital n_orb
]
```

Next create a $k$-space mesh from this unit cell

```python
from triqs.lattice.tight_binding import BravaisLattice, BrillouinZone, MeshBrZone

# How many points in each spatial dimension
n_k_1 = ...
...
n_k_d = ...
...
n_k_list = [n_k1, ..., n_k_d]

bl = BravaisLattice(units=units)
bz = BrillouinZone(bl)
mk = MeshBrZone(bz, n_k_list)
```

Next set up a dictionary of hoppings between orbitals, within a unit cell
and between unit cells.

Each key in the dictionary is a tuple of the displacement from a reference unit
cell at $\mathbf{R} = (0, \ldots, 0_d)$.
The coordinates for this displacement is in the basis of lattice
vectors $\vec{a}_1, \ldots, \vec{a}_d$. For example, in two-dimensions, a
displacement by $\vec{a}_1$ is the key `(1,0)`. A displacement by
$2\vec{a}_1 + \vec{a}_2$ is the key `(2,1)`. Generally, the displacements are
`(n,m,p)` where `n`, `m`, and `p` have to be integers.

The value of each key is a matrix of hopping amplitudes from the orbitals
in the reference unit cell to the unit cells defined by the displacement key.
Hopping between orbitals within a unit cell is encoded as a matrix and indexed
with the key `(0,...,0_d)`.

Here is an example of how this structure looks like in code

```python
# Number of sites and orbitals (and maybe spin) in each unit cell
n_orb = ...

hoppings = {
    (0,...,0_d) : n_orb by n_orb ndarray,
    (1,...,0_d) : n_orb by n_orb ndarray,
    ...
    (0,...,1_d) : n_orb by n_orb ndarray,
    ...
    (-1,...,0_d) : n_orb by n_orb ndarray
    ...
}
```

Next construct the tight-binding model as

```
from triqs.lattice.tight_binding import TBLattice
tbl = TBLattice(units=units, hoppings=hoppings, orbital_positions=orbital_positions)
```

:::{note}
If `h0_k` has some block structure you want to encode, the above process
can be done for each block (by definition of being a block matrix structure
there cannot be coupling between orbitals of different blocks).
Each `TBLattice` instance must use the same `units` and $k$-space
meshgrid `mk`.
:::

The last and important part is to Fourier transform the tight-binding model
onto the $k$-space meshgrid `mk` and have it indexed as `k, orb_i, orb_j` for
the {{RISB}} `Solver` class. This is very easily done as

```python
h0_k = dict()

# For each block (a string bl_name) in h0_k
h0_k[bl_name] = tbl.fourier(mk).data
```

:::{warning}
I have only used this class on hypercubic lattices. It may work poorly for
something like kagome, or if the unit cell is very complicated. If this is not
an issue then let me know and I will remove this warning.
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

### As a {{TRIQS}} operator

If you have a block matrix representation of a single-particle operator and
you want it as a {{TRIQS}} operator

```python
from risb.helpers_triqs get_C_Op

# The block matrix
A = ...

# A gf_struct object of the structure of the space
gf_struct = ...

# A list/vector of operators
C_Op = get_C_Op(gf_struct, dagger=False)
C_dag_Op = get_C_Op(gf_struct, dagger=True)

Op = dict()
for bl, bl_size in gf_struct:
    Op[bl] = C_dag_Op[bl] @ A[bl] @ C_Op[bl]
```

Or you can simply use

```python
from risb.helpers_triqs matrix_to_Op

Op = matrix_to_Op(A, gf_struct)
```

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
