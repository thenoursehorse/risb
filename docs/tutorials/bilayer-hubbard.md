---
title: Bilayer Hubbard model
parent: Tutorials
mathjax: true
---

# Introduction
We will try to reproduce the results of Sec. IIIB of 
PRB **76**, 155102 (2007).

The bilayer Hubbard model on the hypercubic lattice is given by

$$
\hat{H} = \hat{H}^{\mathrm{kin}} + \sum_i \hat{H}_i^{\mathrm{loc}},
$$

where $$i$$ indexes a site.

The kinetic part of the Hamiltonian is given by

$$
\hat{H}^{\mathrm{kin}} = - \frac{1}{d} \sum_{\sigma} \sum_{\alpha=1,2} 
t_{\alpha} \hat{d}_{i \alpha \sigma}^{\dagger} \hat{d}_{j \alpha \sigma},
$$

where $$d$$ is the number of spatial dimensions ($$d=3$$ on the cubic lattice), 
$$\sigma$$ is a spin label, $$\alpha$$ is an orbital label, and $$t_{\alpha}$$
is the hopping parameter.

For the bilayer Hubbard model the two orbitals have the dispersion relation

$$
\varepsilon_{k\alpha} = -\frac{2}{d} t_{\alpha} \sum_{\mu}^d \cos(k_\mu a),
$$

where $$k$$ labels a reciprocol lattice vector. Hence, 
$$\hat{H}_{k \alpha \beta}^{\mathrm{kin}} = \varepsilon_{k\alpha}$$ 
for $$\alpha = \beta$$ and zero otherwise.

The local part of the Hamiltonian is given by

$$
\hat{H}_i^{\mathrm{loc}} = 
V \sum_{\sigma} (\hat{d}_{i 1 \sigma}^{\dagger} \hat{d}_{i 2 \sigma} 
+ \mathrm{H.c.})
+ U \sum_{\alpha} \hat{n}_{i \alpha \uparrow} \hat{n}_{i \alpha \downarrow},
$$

where $$V$$ is the interlayer hopping between the orbitals, and $$U$$ is 
the local Coulomb repulsion.

# Construct the matrix representation of $$\hat{H}^{\mathrm{kin}}$$

The Hamiltonian is block diagonal in spin. Hence, we only need to create
the matrix for each spin. For each spin $$\sigma$$, the hopping terms are 
given by the kinetic Hamiltonian 
$$\hat{H}^{\mathrm{kin}}_{k \sigma \alpha\beta}[\sigma]$$. This is 
represented as an $$N \times M \times M$$ matrix, where $$N$$ is the number of 
sites on the lattice and $$M$$ is the number of orbitals. Below, ``s`` labels 
each spin.

## Method 1: Explicit construction

The easiest way to create this is as an array using `numpy`. First create the 
$$k$$-grid mesh.

```python
import numpy as np

spatial_dim = 3 # for cubic lattice
nkx = 6 # linear dimension of k-mesh

# Build shifted equally spaced hypercubic mesh
mesh = np.empty(shape=(nkx**spatial_dim, spatial_dim))
coords = [range(nkx) for _ in range(spatial_dim)]
for idx,coord in enumerate(zip(*coords)):
    for i in range(len(coords)):
        mesh[idx,i] = coord[i]/nkx + 0.5/nkx
```

Use the mesh to construct $$\hat{H}^{\mathrm{kin}}$$

```python
from itertools import product

def cubic_kin(mesh, t = 1, a = 1, orb_dim = 2):
    mesh_num = mesh.shape[0]
    di = np.diag_indices(orb_dim)

    h_kin = dict()
    for s in ["up","dn"]:
        h_kin[s] = np.zeros([mesh_num, orb_dim, orb_dim])
        h_kin[s][:,di[0],di[1]] = -2.0 * t * np.sum(np.cos(a * 2.0 * np.pi * mesh), axis=1)[:, None]

    return h_kin
```

## Method 2: Using `TRIQS` functions

We can also use the methods within `TRIQS` to constrcut Bravai lattices
and get dispersion relations. In this case we don't use the mesh we 
constructed ourselves. This method is not straight forward to use with 
linear tetrahedron, so I always just construct my own dispersion relations.

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
    
    h_kin = dict()
    for s in ["up","dn"]:
        h_kin[s] = np.zeros([mesh_num, orb_dim, orb_dim])
        h_kin[s][:, di[0],di[1]] = energies[:, None]
    
    return h_kin
```

# Construct local Hamiltonian $$\hat{H}_i^{\mathrm{loc}}$$

The local Hamiltonian has to include all of the many-body interactions on each
site $$i$$, as well as the quadratic terms that describe orbital energies and
hopping between orbitals on site $$i$$. For more details refer to the 
`TRIQS` documentation on constructing second-quantized operators.

We label the spin as the first index of the operators, while the second index
labels the orbital.

```python
from triqs.operators import *

def bilayer_loc(V = 0.25, U = 0):
    
    h_loc = Operator()
    for o in [1,2]:
        h_loc += U * n("up",o) * n("dn",o)
    for s in spin_names:
        h_loc += V * ( c_dag(s,1)*c(s,2) + c_dag(s,2)*c(s,1) )

    return h_loc
```

# Setup the mean-field matrices

The simplest way to enforce the block structure of $$\hat{H}$$ is to construct 
matrices in the same way that `TRIQS` constructs block Green's functions. Each 
block will represent the spin, and each matrix within that block is the 
two-orbital space on each site of the bilayer Hubbard model. In RISB the 
homogenous assumption is taken, so that the matrices are the same on every 
site.

The mean-field matrices that characterize the quasiparticle Hamiltonian 
$$\hat{H}^{\mathrm{qp}}$$ are the renormalization matrix $$\mathcal{R}$$ and
the correlation matrix $$\lambda$$. The quasiparticle density matrix 
of $$\hat{H}^{\mathrm{qp}}$$ is $$\Delta^p$$ and the lopsided kinetic energy 
is $$\mathcal{K}$$.

The impurity is described by the hybridization matrix $$\mathcal{D}$$, and the 
matrix that describes the bath $$\lambda^c$$.

The single-particle density matrices of the impurity are the density matrix of 
the quasiparticles (the bath) $$N^f$$, the off-diagonal density matrix terms 
between the bath and the impurity $$M^{cf}$$, and the density matrix of the 
physical electron (the impurity) $$N^c$$.

```python       
gf_struct = [ ["up", [1, 2]], ["dn", [1, 2]] ]

R = dict()
Lambda = dict()

pdensity = dict()
ke = dict()

D = dict()
Lambda_c = dict()

Nf = dict()
Mcf = dict()
Nc = dict()
    
for s,ind in gf_struct:
    # Mean-field matrices 
    R[s] = np.zeros((len(ind),len(ind)))
    Lambda[s] = np.zeros((len(ind),len(ind)))

    # H^qp density matrices
    pdensity[s] = np.empty((len(ind),len(ind)))
    ke[s] = np.empty((len(ind),len(ind)))
    
    # Hybridization and bath energies
    D[s] = np.empty((len(ind),len(ind)))
    Lambda_c[s] = np.empty((len(ind),len(ind)))
    
    # Density matrices of impurity
    Nf[s] = np.empty((len(ind),len(ind)))
    Mcf[s] = np.empty((len(ind),len(ind)))
    Nc[s] = np.empty((len(ind),len(ind)))    
```
# Calculating the mean-field density matrices

At each iteration, the embedding problem requires a few different quantities. 
This requires calculating the quasiparticle density matrix 
$$\Delta^p$$ from the mean-field, the single-particle (lopsided) kinetic 
energy $$\mathcal{K}$$, and $$\mathcal{D}$$ and $$\lambda^c$$. There are 
helper functions to this that are imported as

```python
import risb.sc_cycle as sc
```

The helper functions are used as below.

```python
# Quasiparticle density
pdensity[s] = sc.get_pdensity(vec, wks)

# Lopsided kinetic energy
disp_R = sc.get_disp_R(R[s], dispersion, vec)                
ke[s] = sc.get_ke(disp_R, vec, wks)

# Hybridization matrix
D[s] = sc.get_d(pdensity[s], ke[s])

# Bath orbital energies
Lambda_c[s] = sc.get_lambda_c(pdensity[s], R[s], Lambda[s], D[s])
```

After the embedding Hamiltonian is solved and the single-particle density 
matrices are obtained from it, there are two ways to do the self-consistency 
loop. The first is to calculate a new guess for $$\mathcal{R}$$ and 
$$\lambda$$. The helper functions to do this are

```python
Lambda[s] = sc.get_lambda(R[s], D[s], Lambda_c[s], Nf[s])
R[s] = sc.get_r(Mcf[s], Nf[s])
```

The second is as a root problem ensuring that the density matrices from
$$\hat{H}^{\mathrm{qp}}$$ match the density matrices from $$\hat{H}^{\mathrm{emb}}$$.

```python
f1 = sc.get_f1(Mcf[s], pdensity[s], R[s])
f2 = sc.get_f2(Nf[s], pdensity[s])
```

# The $$k$$-space integrator

RISB requires taking an integral of many mean-field matrices. The way to do 
this that generalizes to many kinds of $$k$$-space integration methods is to 
find the weight to that integral at each $$k$$-point. This is, e.g., how 
linear tetrahedron works. 

The weights are with respect to the quasiparticle Hamiltonian 

$$
\hat{H}^{\mathrm{qp}} = 
\mathcal{R} \hat{H}^{\mathrm{kin}} \mathcal{R}^{\dagger}
+ \lambda
$$

## Method 1: A simple Fermi weight
The simplest method is to just calculate the weight using the Fermi-Dirac 
distribution function. The weight at each $$k$$-point has to be updated at 
each iteration of the self-consistency method, but in practice this can be very fast.

Then the weight to each point in the integral is given by the Fermi weight 
of the eigenenergies of $$\hat{H}^{\mathrm{qp}}$$. There is a helper 
function that does this diagonalization at each $$k$$-point.

```python
import risb.sc_cycle as sc

def get_wks(R, Lambda, h_kin, mu=0, beta=10):
    eig, vec = sc.get_h_qp(R, Lambda, h_kin)
    
    fermi_fnc = lambda eks : 1.0 / (np.exp(beta * (eks - mu)) + 1.0)
    wks = fermi_fnc(eig, beta) / h_kin.shape[0]
    
    return eig, vec, wks
```

## Method 2: Linear tetrahedron

The linear tetrahedron method is far more accurate and faster than using 
simple smearing methods. It assumes zero temperature. First set it up and 
use its mesh to construct $$\hat{H}^{\mathrm{kin}}$$ instead of the mesh 
we made by hand.

```python
from kint import Tetras

kintegrator = Tetras(nkx,nkx,nkx)
mesh = kintegrator.getMesh
```

The $$k$$-space integration weights can now be calculated as

```python
def get_wks(R, Lambda, h_kin, mu=0)
    eig, vec = sc.get_h_qp(R, Lambda, h_kin)
    
    kintegrator.setEks(np.transpose(eig))
    kintegrator.setEF(mu)
    #kintegrator.setEF_fromFilling(N_elec / 2.0)
    #mu_calculated = kintegrator.getEF
    wks = np.transpose(kintegrator.getWs)
    
    return eig, vec, wks
```

The commented line sets the chemical potential $$\mu$$ for a desired electron 
filling $$N_{\mathrm{elec}}$$ of the lattice. The line assumes paramagnetic 
solutions.

# Setup the embedding Hamiltonian $$\hat{H}^{\mathrm{emB}}$$ solver

Solving the embedding Hamiltonian $$\hat{H}^{\mathrm{emb}$$ is 
done in a way that is intended to bevery modular. It is kept this way 
because this is the most computationally expensive part. The simplest thing 
to do is blindly use exact diagonalization. There is also a specialized 
exact diagonalization implementation for the specific structure of the 
embedding Hamiltonian, and an outdated solver using `DMRG`. There are so 
many avenues to go down for this: `Pomerol`, `QuSpin`, `QuTiP`, QMC, etc.

## Method 1: Using `AtomDiag` from `TRIQS`

There is a simple testing implementation that only uses `TRIQS` functions.

```python
from risb.embedding_atom_diag import *

emb_solver = EmbeddingAtomDiag(gf_struct)
```

There are a few methods that are kept consistent across the different 
implementations. The first sets the embedding Hamiltonian from the local 
Hamiltonian and the mean-field matrices $$\mathcal{D}$$ and $$\lambda^c$$.

```python
emb_solver.set_h_emb(h_loc, Lambda_c, D)
```

The second solves for the ground state in the $$2M$$ particle sector, which 
corresponds to the embedding problem being half-filled.

```python
emb_solver.solve()
```
The rest are methods to get the single-particle density matrices of the 
impurity

```python
Nf[s] = emb_solver.get_nf(s)
Mcf[s] = emb_solver.get_mcf(s)
Nc[s] = emb_solver.get_nc(s)
```

where `s` specifies the block as defined in `gf_struct`. 

## Method 2: Using our implementation of exact diagonalization

This embedding solver is specialized to only solve for the embedding state
that RISB requires. A big difference is that it needs the local Hamiltonian 
to initialize because it automatically finds and enforces many symmetries.

```python
from embedding_ed import *

emb_solver = EmbeddingEd(h_loc, gf_struct)
```

Compared to `AtomDiag` the only difference in using it is the local 
Hamiltonian is not passed to the method that constructs the embedding 
Hamiltonian.

```python
emb_solver.set_h_emb(Lambda_c, D)
emb_solver.solve()
```

There are some additional parameters that one can use in solve to do with 
the Lanczos algorithm.

```python
psiS_size = emb_solver.get_psiS_size()
emb_solver.solve(ncv = min(30,psiS_size), max_iter = 1000, tolerance = 0)
```

Here `ncv` is to do with how many vectors it keeps in its iterative solver, 
and the rest should be self explanatory. You usually want to keep as many 
vectors as possible, but for large problems it is not possible. I find that 
30 often works well, but this is all problem dependent.

## Method 3: Using `DMRG`

# Putting it all together for the self-consistent loop

Below is a simple self-consistent loop using everything we have set up.

```python
from copy import deepcopy

U = 1
V = 0.25
mu = U / 2.0
num_cycles = 25
beta = 10

# Initialize Lambda and R to the non-interacting values
for s in block_names:
    Lambda[s] = np.eye(Lambda[s].shape[0]) * mu
    np.fill_diagonal(R[s], 1)

for cycle in range(num_cycles):    
    # For convergence checking
    norm = 0
    R_old = deepcopy(R)
    Lambda_old = deepcopy(Lambda)

    for s in ["up","dn"]:
        eig, vec, wks = get_wks(R[s], Lambda[s], h_kin[s], mu, beta)
        
        # H^qp density matrices
        pdensity[s] = sc.get_pdensity(vec, wks)
        disp_R = sc.get_disp_R(R[s], dispersion, vec)
        ke[s] = sc.get_ke(disp_R, vec, wks)

        # H^emb parameters
        D[s] = sc.get_d(pdensity[s], ke[s])
        Lambda_c[s] = sc.get_lambda_c(pdensity[s], R[s], Lambda[s], D[s])

    # Solve H^emb
    emb_solver.set_h_emb(h_loc, Lambda_c, D, mu)
    emb_solver.solve()

    for s in ["up","dn"]:
        # H^emb density matrices
        Nf[s] = emb_solver.get_nf(s)
        Mcf[s] = emb_solver.get_mcf(s)
        Nc[s] = emb_solver.get_nc(s)

        # New guess for mean-field matrices
        Lambda[s] = sc.get_lambda(R[s], D[s], Lambda_c[s], Nf[s])
        R[s] = sc.get_r(Mcf[s], Nf[s])

        # Check how close the guess was
        norm += np.linalg.norm(R[s] - R_old[s])
        norm += np.linalg.norm(Lambda[s] - Lambda_old[s])

        if norm < 1e-6:
            break
    
# Quasiparticle weight
Z = dict()
for s in ['up','dn']:
    Z[s] = np.dot(R[s], R[s].conj().T)
```

# Exercises

1. Put the simplest version together.
1. Put the complicated version with `EmbeddingED` and `Tetras` together.
1. Solve for a range of $$U$$ values.
1. What is the evolution of the quasiparticle weight at half-filling (Fig. 7)?
1. What is the filling of the electrons in the natural $$\pm$$ basis?
1. What is the evolution of the quasiparticle weight at $$n = 1.88$$ (Fig. 10)?