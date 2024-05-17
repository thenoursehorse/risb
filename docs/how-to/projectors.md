# Using projectors

This guide shows you how to project onto ineqvuialent correlated subspaces
$\mathcal{C}_i$.
There are two technical choices for projectors.
Projecting onto subspaces with the same block matrix structure as the larger set
of orbitals. Projecting onto subspaces with a different block matrix structure
than the larger space, e.g., to enforce some local symmetry.

:::{seealso}
:class: dropdown
[About projectors](../explanations/projectors.md) for more details and the
theory of projectors.
:::

## Simple projectors

If you want a correlated model where each inequivalent correlated subspace
has the same block structure as the non-interacting model, (e.g., the
dispersion matrix `h0_k`) you can construct very simple projection matrices
by hand.

In this example each subspace $i$ is a site
$\alpha \in \{A, B, C\}$ with spin $\sigma$ within the three-site unit cell
of the kagome lattice. Hence, you can define six
$1 \times 3$ rectangular matrices that project onto these spaces. First
define a `gf_struct` for each correlated space $i$

```python
import numpy as np

# The block structure of the non-interacting space
spin_names = ['up', 'dn']

# Define number of clusters/correlated subspaces
n_clusters = 3

# Define a local structure for each correlated subspace
gf_struct = [ [(bl, 1) for bl in spin_names] for i in range(n_clusters) ]
```

Next define the projectors as rectangular matrices

```python
import numpy as np

# Dictionaries of projectors onto each block in gf_struct[i]
P_A = { bl : np.array( [ [1, 0, 0] ] ) for bl in spin_names }
P_B = { bl : np.array( [ [0, 1, 0] ] ) for bl in spin_names }
P_C = { bl : np.array( [ [0, 0, 1] ] ) for bl in spin_names }

# A list holding each projector, 1 for each i in gf_struct
projectors = [P_A, P_B, P_C]
```

You do not need to define the projector for all $k$ because at each $k$ they
are equivalent. Next define an embedding solver for each inequivalent cluster

```python
from ... import EmbeddingClass

# Define h_int for each cluster as a list
...

embedding = [EmbeddingClass(h_int[i], gf_struct[i]) for i in range(n_clusters)]
```

The solvers are initialized as

```python
from risb import SomeSolver

S = SomeSolver(...
               gf_struct = gf_struct,
               embedding = embedding,
               projectors = projectors,
               ...
)
```

:::{note}
:class: dropdown
In this case a correlated subspace for each orbital within a unit cell was
defined. This is not a requirement. It is fine if there were multiple orbitals
per site and only some of them were treated as correlated, or if only some
sites in a unit cell were treated as correlated.
:::

## Complicated projectors with `gf_struct_mapping`

If the block matrix structure of the non-interacting Hamiltonian $\hat{H}_0$
(`h0_k` in code) is not the same as the block matrix structure of the
correlated subspaces $\mathcal{C}_i$ then you can use a mapping dictionary
to go between them.

This is used if there is some well defined local
symmetry in the subspace $\mathcal{C}_i$ that is not valid in the larger
space of $\hat{H}_0$. An example is if projecting onto a $d$-orbital subspace
of a metal and because of crystal field symmetries the $t_{2g}$ and $e_g$
orbitals make sense as block matrices in the subspace of $\mathcal{C}_i$,
but only spin up and spin down block matrices are possible in the larger
space of $\hat{H}_0$.

The mapping is used as

```python
# h0_k is a dict[ndarray] with, e.g., blocks 'up', 'dn'.
h0_k =

# The structure of each correlated subspace
gf_struct_subspace_1 = [('block_1', n_orb_1), ...]
...

# The mapping from a correlated subspace to the space of h0_k
gf_struct_mapping_1 = {'block_1': 'block_in_h0_k', ...}
...

# Make a list of all subspaces
gf_struct = [gf_struct_subspace_1, ...]
gf_struct_mapping = [gf_struct_mapping_1, ...]
```

An example of a $d$ orbital with octohedral/tetrahedral symmetry

```python
h0_k = {'up' : ..., 'dn' : ...}

gf_struct_d = [('up_eg', 2), ('dn_eg', 2), ('up_t2g', 3), ('dn_t2g', 3)]
gf_struct_mapping_d  {'up_eg' : 'up', 'up_t2g' : 'up', 'dn_eg' : 'dn', 'dn_t2g' : 'dn'}
```

There are several helper functions that might need a `gf_struct_mapping`,
either as a list of all correlated subspaces or for a single correlated
subspace. The documentation will say what is required.

If you need to use the mapping for the `Solver` classes

```python
from risb import LatticeSolver

S = LatticeSolver(...,
                  gf_struct = list of gf_struct in each subspace,
                  gf_struct_mapping = list of mappings in each subspace,
)
```
