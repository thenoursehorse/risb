# Using projectors

This guide shows you how to project onto ineqvuialent correlated subspaces
$\mathcal{C}_i$. 
There are two technical choices for projectors.
Projecting onto subspaces in the same basis as the larger set of orbitals. 
Projecting onto subspaces with a different block matrix structure than 
the larger space, e.g., to enforce some local symmetry.

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

Work in progress.