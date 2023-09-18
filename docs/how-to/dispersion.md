---
title: Construct dispersion
parent: How-to guides
---

## Method 2: Using [TRIQS](https://triqs.github.io/) functions

We can also use the methods within [TRIQS](https://triqs.github.io/) to 
constrcut Bravai lattices and get dispersion relations. In this case we 
don't use the mesh we constructed ourselves. This method is not straight 
forward to use with linear tetrahedron, so it's often easier to just 
construct your own dispersion relations.

```python
from triqs.lattice.tight_binding import *

def cubic_kin(t = 1, nkx = 6, spatial_dim = 3, orb_dim = 2):
    # Cubic lattice
    units = np.eye(spatial_dim)
    bl = BravaisLattice(units = units, orbital_positions= [ (0,0) ] ) # only do one orbital because all will be the same
    
    hop = {}
    for i in range(spatial_dim):
        hop[ tuple((units[:,i]).astype(int)) ] = [[t]]
        hop[ tuple((-units[:,i]).astype(int)) ] = [[t]]
    
    tb = TightBinding(bl, hop)
    
    energies = energies_on_bz_grid(tb, nkx)
    mesh_num = energies.shape[1]

    di = np.diag_indices(orb_dim)
    
    # want dispersion indexed as k, a, b so transpose
    h_kin = dict()
    for s in ["up","dn"]:
        h_kin[s] = np.zeros([mesh_num, orb_dim, orb_dim])
        h_kin[s][:, di[0],di[1]] = np.transpose(energies, (1,0))
    
    return h_kin
```