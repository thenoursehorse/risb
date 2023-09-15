# RISB loop

## Introduction

In this tutorial, you will construct the self-consisten loop in rotationally 
invariant slave-bosons, and use it to solve the bilayer Hubbard model.

We will try to reproduce the results of Sec. IIIB of 
PRB **76**, 155102 (2007).

The bilayer Hubbard model on the hypercubic lattice is given by

$$
\hat{H} = \hat{H}_0^{\mathrm{kin}} + \sum_i \hat{H}_i^{\mathrm{loc}},
$$

where $i$ indexes a site.

The non-interacting kinetic part of the Hamiltonian that describes hopping 
between clusters (sites) on the lattice is given by

$$
\hat{H}_0^{\mathrm{kin}} = - \frac{1}{d} 
\sum_{\langle i j \rangle} \sum_{\sigma} \sum_{\alpha=1,2} 
( t_{\alpha} \hat{d}_{i \alpha \sigma}^{\dagger} \hat{d}_{j \alpha \sigma}
+ \mathrm{H.c.} ),
$$

where $d$ is the number of spatial dimensions ($d=3$ on the cubic 
lattice), $\sigma$ is a spin label, $\alpha$ is an orbital label, 
$\langle i j \rangle$ indicates nearest-neighbor bonds, 
${\mathrm{H.c.}}$ is Hermitian conjugate, and $t_{\alpha}$ is the 
probability amplitude to move an electron between nearest neighbor sites.

For degenerate orbitals on the cubic lattice, the dispersion for each 
orbital is given by

$$
\varepsilon_{k\alpha} = -\frac{2}{d} t_{\alpha} \sum_{\mu}^d \cos(k_\mu a),
$$

where $k$ labels a reciprocol lattice vector. Hence, 
$\hat{H}_{0, k \alpha \beta}^{\mathrm{kin}} = \varepsilon_{k\alpha}$ 
for $\alpha = \beta$ and zero otherwise.

The local part of the Hamiltonian is given by

$$
\hat{H}_i^{\mathrm{loc}} = 
V \sum_{\sigma} (\hat{d}_{i 1 \sigma}^{\dagger} \hat{d}_{i 2 \sigma} 
+ \mathrm{H.c.})
+ U \sum_{\alpha} \hat{n}_{i \alpha \uparrow} \hat{n}_{i \alpha \downarrow},
$$

where $V$ is the interlayer hopping between the orbitals, and $U$ is 
the local Coulomb repulsion.

## A: Construct $\hat{H}_0^{\mathrm{kin}}$

First, construct the dispersion between clusters on the lattice. It should 
not include the non-interacting quadratic terms within a cluster. 

Notice that the Hamiltonian is block diagonal in spin. Hence, the hopping terms 
are given by the kinetic Hamiltonian $\hat{H}^{\mathrm{kin}}_{0,k \alpha\beta}[\sigma]$, 
and we can construct an array for each spin $\sigma$ separately. Each array 
will be $\mathcal{N} \times n_{\mathrm{orb}} \times n_{\mathrm{orb}}$, where 
$\mathcal{N}$ is the number of unit cells (sites in this case) on the 
lattice and $n_{\mathrm{orb}}$ is the number of orbitals in each unit cell. 
If the Hamiltonian was not block diagonal in spin, we would instead construct a 
single $\mathcal{N} \times 2M \times 2M$ array.

The easiest way to create this is as an array using `numpy`. We will 
intentionally do this in a computationally slow way. The intention here is to 
be explicit for clarity. First create the 
$k$-grid mesh of the Brillouin zone in crystal coordinates

```python
import numpy as np
from itertools import product

def make_kmesh(n_kx = 6, d = 3):
    """
    Return a shifted Monkhorst-Pack k-space mesh on a hypercubic Bravais 
    lattice.

    Parameters
    ----------

    n_kx : optional, int
        Linear dimension of k-mesh in each dimension. Default 6.

    
    d : optional, int
        Number of dimensions, e.g., d = 2 is the square lattice. Default 3.
    
    """
    
    # Total number of k-points on the mesh
    n_k = n_kx**d

    mesh = np.empty(shape=(n_k, d))
    coords = [range(n_kx) for _ in range(d)]
    for idx,coord in enumerate(product(*coords)):
        for i in range(len(coord)):
            mesh[idx,i] = coord[i]/float(n_kx) + 0.5/float(n_kx)
    
    return mesh
```

Now construct $\hat{H}_0^{\mathrm{kin}}$ on this mesh.

```python
def make_h0_k(mesh, gf_struct, t = 1, a = 1):
    """
    Return dispersion of degenerate orbitals on a hypercubic lattice as a 
    dictionary, where each key is the spin.

    Parameters
    ----------

    mesh : ndarray
        The k-space mesh of the Bravais lattice.

    gf_struct : list of pairs [ (str,int), ... ]
        Structure of matrices. 

    t : optional, float
        Hopping amplitude. Default 1.

    a : optional, float
        Lattice spacing. Default 1.

    """
    n_k = mesh.shape[0]

    h0_k = dict()
    for bl, n_orb in gf_struct:
        di = np.diag_indices(n_orb)
        h0_k[bl] = np.zeros([n_k, n_orb, n_orb])
        h0_k[bl][:,di[0],di[1]] = -2.0 * t * np.sum(np.cos(2.0 * a * np.pi * mesh), axis=1)[:, None]

    return h0_k
```

Note that the structure of the matrices is given in the same way as TRIQS 
Green's functions as

```python
gf_struct = [ ("up", 2), ("dn", 2) ],
```

## B: Construct $\hat{H}_i^{\mathrm{loc}}$

Next, construct the local Hamiltonian on each cluster (site) $i$. It has to 
include all of the many-body interactions on each cluster as well as the 
non-interacting quadratic terms that describe orbital energies and
hopping between orbitals on site $i$. For more details refer to the 
[TRIQS](https://triqs.github.io/) documentation on constructing 
second-quantized operators.

```python
from triqs.operators import *

def make_h_loc(V = 0.25, U = 5):
    """
    Return the local terms of the bilayer Hubbad model as a TRIQS operator.
    
    Parameters
    ----------
    
    V : optional, float
        Hopping between orbitals on each cluster.

    U : optional, float
        Local Coulomb repulsion on each site.

    """
    
    h_loc = Operator()
    for o in range(2):
        h_loc += U * n("up", o) * n("dn", o)
    for bl in ["up", "dn"]:
        h_loc += V * ( c_dag(bl, 0) * c(bl, 1) + c_dag(bl, 1) * c(bl, 0) )

    return h_loc
```

## C: Setup mean-field matrices

We now need to initialize the mean-field matrices used in RISB. In RISB the
homogenous assumption is taken, so that the matrices are the same on every site.
Below we describe what each mean-field matrix physically relates to within 
the algorithm.

The mean-field matrices that characterize the quasiparticle Hamiltonian 
$\hat{H}^{\mathrm{qp}}$ are the renormalization matrix $\mathcal{R}$ and
the correlation matrix $\lambda$. The single-particle quasiparticle density 
matrix of $\hat{H}^{\mathrm{qp}}$ is $\Delta^p$ and the lopsided 
quasiparticle kinetic energy is $\mathcal{K}$.

The non-interacting quadratic parts of the embedding Hamiltonian 
$\hat{H}^{\mathrm{emb}}$ are described by the hybridization matrix 
$\mathcal{D}$, and the matrix that describes the couplings in the bath 
$\lambda^c$.

The single-particle density matrices of $\hat{H}^{\mathrm{emb}}$ are the 
density matrix of the f-electrons (the bath, these are quasiparticles) $N^f$, 
the density matrix of the c-electrons (the impurity, these are physical 
electrons), and the off-diagonal density matrix between the c- and f- 
electrons (the impurity and the bath) $M^{cf}$.

```python 
# H^qp parameters R and Lambda
R = dict()
Lambda = dict()
for bl, bl_size in gf_struct:
    R[bl] = np.zeros((bl_size, bl_size))
    Lambda[bl] = np.zeros((bl_size, bl_size))

# H^qp single-particle density matrix and (lopsided) kinetic energy
rho_qp = dict()
kinetic_energy_qp = dict()
for bl, bl_size in gf_struct:
    rho_qp[bl] = np.zeros((bl_size, bl_size))
    kinetic_energy_qp[bl] = np.zeros((bl_size, bl_size))
    
# H^emb hybridization and bath terms
D = dict()
Lambda_c = dict()
for bl, bl_size in gf_struct:
    D[bl] = np.zeros((bl_size, bl_size))
    Lambda_c[bl] = np.zeros((bl_size, bl_size))
    
# H^emb single-particle density matrices
Nf = dict()
Mcf = dict()
Nc = dict()
for bl, bl_size in gf_struct:
    Nf[bl] = np.zeros((bl_size, bl_size))
    Mcf[bl] = np.zeros((bl_size, bl_size))
    Nc[bl] = np.zeros((bl_size, bl_size))   
```

### Helper functions

As an aside, let me describe how to obtain the above mean-field matrices, which 
has to be done at eacheiteration of the self-consistent process. There are 
helper functions that do this for you. They simply take in numpy arrays 
and either use `numpy.einsum` or multiply arrays together. Below, the 
definition of `bloch_qp` and `kweights` will be described later.

Remember that you can check the docstring of a helper function with 
`help(helpers.function)`.

```python
from risb import helpers

# H^qp single-particle density
for bl, bl_size in gf_struct:
    rho_qp[bl] = helpers.get_pdensity(bloch_qp[bl], kweights[bl])

# H^qp (lopsided) quasiparticle kinetic energy
for bl, bl_size in gf_struct:
    h0_R = helpers.get_h0_R(R[bl], h0_k[bl], bloch_qp[bl])
    kinetic_energy_qp[bl] = helpers.get_ke(h0_R, bloch_qp[bl], kweights[bl])

# H^emb hybridization
for bl, bl_size in gf_struct:
    D[bl] = helpers.get_d(rho_qp[bl], kinetic_energy_qp[bl])

# H^emb bath
for bl, bl_size in gf_struct:
    Lambda_c[bl] = helpers.get_lambda_c(rho_qp[bl], R[bl], Lambda[bl], D[bl])
```

After the embedding Hamiltonian is solved and the single-particle density 
matrices of the impurity model are obtained, there are two ways to do the 
self-consistency loop. The first is to calculate a new guess for 
$\mathcal{R}$ and $\lambda$ which re-parameterizes $H^{\mathrm{qp}}$. 
The helper functions to do this are

```python
for bl, bl_size in gf_struct:
    Lambda[bl] = helpers.get_lambda(R[bl], D[bl], Lambda_c[bl], Nf[bl])
    R[bl] = helpers.get_r(Mcf[bl], Nf[bl])
```

The second is as a root problem, adjusting $\mathcal{R}$ and $\lambda$ to
ensure that the density matrices from $\hat{H}^{\mathrm{qp}}$ match the 
density matrices from $\hat{H}^{\mathrm{emb}}$.

```python
for bl, bl_size in gf_struct:
    f1 = helpers.get_f1(Mcf[bl], pdensity[bl], R[bl])
    f2 = helpers.get_f2(Nf[bl], rho_qp[bl])
```

## D: The $k$-space integrator

Next you will specify how k-space integrals are performed. RISB requires 
integrating many mean-field matrices. The way to do this that generalizes to 
many kinds of $k$-space integration methods is to 
find the weight of the integral at each $k$-point. This is, e.g., how 
linear tetrahedron works and smearing methods work. As you will see below, 
because the reference energy for the integration are the eigenenergies of 
$\hat{H}^{\mathrm{qp}}$, the weight at each $k$-point has to be updated at 
each iteration of the self-consistency method. Not to worry though, 
in practice this can be very fast. Below we will describe the general theory 
for taking these integrals in multi-orbital cases. 

The weights are with respect to the quasiparticle Hamiltonian 

$$
\hat{H}^{\mathrm{qp}} = 
\mathcal{R} \hat{H}^{\mathrm{kin}} \mathcal{R}^{\dagger}
+ \lambda,
$$

which, in this case, are block diagonal in $k$. All of the integrals take 
the form

$$
\lim_{\mathcal{N} \rightarrow \infty} 
\frac{1}{\mathcal{N}} \sum_k A_k f(\hat{H}_k^{\mathrm{qp}}),
$$

where $\mathcal{N}$ is the number of unit cells, $A_k$ is a generic 
function, and $f(\xi_n)$ is the Fermi-Dirac distribution. The meaning 
of $f(\hat{H}^{\mathrm{qp}}$ is specifically the operation

$$
U_k U_k^{\dagger} f(\hat{H}_k^{\mathrm{qp}}) U_k U_k^{\dagger} 
= U_k f(\xi_{kn}) U_k^{\dagger},
$$

where $U_k$ is the matrix representation of the unitary that diagonalizes 
$\hat{H}_k^{\mathrm{qp}}$, $\xi_{kn}$ are its eigenenergies, and 
$f(\xi_{kn})$ is a diagonal matrix of the Fermi-Dirac distribution for 
each eigenvalue.

The integral can be converted to a series of finite $k$-points, with an 
appropriate integration weight such that the integral now takes the form 

$$
\sum_k A_k w(\varepsilon_{kn}).
$$

There is a helper function that constructs $\hat{H}^{\mathrm{qp}}$ and 
returns its eigenvalues and eigenvectors at each $k$-point on the finite 
grid.

```python
from risb import helpers

energies_qp = dict()
bloch_qp = dict()
for bl, bl_size in gf_struct
energies_qp[bl], bloch_qp[bl] = helpers.get_h_qp(R[bl], Lambda[bl], h0_k[bl])
```

The simplest definition for the integration weight is to just calculate the 
integration weight using the Fermi-Dirac distribution function on a finite grid 
at the inverse temperature $\beta$. That is,

$$
w(\varepsilon_{kn}) = \frac{1}{\mathcal{N}} f(\varepsilon_{kn}).
$$

The code to perform this is

```python
def update_weights(energies, mu=0, beta=10):
    """
    Return the integration weights at each k-point on the lattice.

    Parameters
    ----------

    energies : ndarray
        Energies at each k-point for each band n, indexed as k,n.

    mu : optional, float
        Chemical potential. Default 0.

    beta : optional, float
        Inverse temperature. Default 0.

    """
    n_k = energies.shape[0]
    return 1.0 / (np.exp(beta * (energies - mu)) + 1.0) / n_k
```

## E: Solving $\hat{H}^{\mathrm{emb}}$

Now we have to solve the impurity problem defined by the embedding Hamiltonian 
$\hat{H}^{\mathrm{emb}}$. There is a simple, but relatively slow, 
implementation that only uses [TRIQS](https://triqs.github.io/).

```python
from risb.embedding import EmbeddingAtomDiag

embedding = EmbeddingAtomDiag(h_loc, gf_struct)
```

Setting the embedding Hamiltonian $\hat{H}^{\mathrm{emb}}$ is done with 

```python
embedding.set_h_emb(Lambda_c, D)
```

Solving for the ground state in the $2M$ particle sector, which 
corresponds to the embedding problem being half-filled, is done with 

```python
embedding.solve()
```

The methods to calculate the single-particle density matrices are

```python
for bl, bl_size in gf_struct:
    Nf[bl] = embedding.get_nf(bl)
    Mcf[bl] = embedding.get_mcf(bl)
    Nc[bl] = embedding.get_nc(bl)
```

## Exercises

1. Can you match each part above with the self-consistent loop defined in the 
literature?
1. Piece together everything above and write your own code.
1. Solve for a range of $U$ values at half-filling ($\mu = U / 2$).
1. How does $\beta$ and the size of the k-space mesh affect the results?
1. What is the evolution of the quasiparticle weight $Z$ at 
half-filling (Fig. 7)?
1. What is the evolution of the electron filling in the 
bonding/anti-bonding ($\pm$) basis?
1. Implement a method to solve for the chemical potential $\mu$ at a fixed 
electron density $n$ (you may find `brentq` or `bisect` from `scipy.optimize` 
useful).
1. What is the evolution of the quasiparticle weight at electron filling 
$n = 1.88$ (Fig. 10)?

## Conclusion

You have built the self-consistent loop for RISB and solved a (not so simple) 
interacting fermion problem. The code in `LatticeSolver` is not much more 
complicated than what you have done. You should now easily be able to mofify, 
implement, and contribute to any parts in the library. You also now understand 
the basic ingredients needed for most self-consistent procedures in much 
more sophisticated codes for DFT and DMFT.

Below is some code that should be very easy to fill in. But you will understand 
much more about RISB if you try to piece everything together from the 
self-consistent equations found in the literature found in 
[About]({% link about.md %})

### Skeleton code cheat sheet

Below is a simple self-consistent loop that relies on everything we have set up.

```python
from copy import deepcopy
from risb import helpers

U = 4
V = 0.25
mu = U / 2.0
n_cycles = 25
beta = 40
n_orb = 2
block_names = ['up','dn']
gf_struct = [(bl, n_orb) for bl in block_names]

# Implement A
h0_k = 

# Implement B
h_loc = 

# Implement C

# Implement D

# Implement E
embedding = 

# H^qp parameters R and Lambda initialized to the non-interacting values
for bl, bl_size in gf_struct:
    np.fill_diagonal(Lambda[bl], mu)
    np.fill_diagonal(R[bl], 1)

for cycle in range(n_cycles):    
    # For convergence checking
    norm = 0
    R_old = deepcopy(R)
    Lambda_old = deepcopy(Lambda)

    # H^qp and integration weights
    for bl, bl_size in gf_struct:
        eig_qp[bl], vec_qp[bl] = helpers.get_h_qp(R[bl], Lambda[bl], h0_k[bl])

    # k-space integration weights
    for bl, bl_size in gf_struct:
        kweights[bl] = 

    # H^qp density matrices
    for bl, bl_size in gf_struct:
        pdensity[bl] = helpers.get_pdensity(vec_qp[bl], wks[bl])
        h0_R[bl] = helpers.get_h0_R(R[bl], h0_k[bl], vec_qp[bl])
        kinetic_energy_bl[bl] = helpers.get_ke(h0_R[bl], vec_qp[bl], kweights[bl])

    # H^emb parameters
    for bl, bl_size in gf_struct:
        D[bl] = helpers.get_d(pdensity[bl], kinetic_energy_bl[bl])
        Lambda_c[bl] = helpers.get_lambda_c(pdensity[bl], R[bl], Lambda[bl], D[bl])

    # Solve H^emb
    embedding.set_h_emb(h_loc, Lambda_c, D)
    embedding.solve()

    # H^emb density matrices
    for bl, bl_size in gf_struct:
        Nf[bl] = embedding.get_nf(bl)
        Mcf[bl] = embedding.get_mcf(bl)
        Nc[bl] = embedding.get_nc(bl)

    # New guess for H^qp parameters
    for bl, bl_size in gf_struct:
        Lambda[bl] = sc.get_lambda(R[bl], D[bl], Lambda_c[bl], Nf[bl])
        R[bl] = sc.get_r(Mcf[bl], Nf[bl])

    # Check how close the guess was
    for bl, bl_size in gf_struct:
        norm += np.linalg.norm(R[bl] - R_old[bl])
        norm += np.linalg.norm(Lambda[bl] - Lambda_old[bl])

        if norm < 1e-6:
            break
    
# Quasiparticle weight
Z = dict()
for bl, bl_size in gf_struct:
    Z[bl] = R[bl] @ R[bl].conj().T
```