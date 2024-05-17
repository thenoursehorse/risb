# Constructing the {{RISB}} self-consistent loop

In this tutorial, you will construct the self-consistent loop in rotationally
invariant slave-bosons, and use it to solve the bilayer Hubbard model. This
will allow you to easily expand upon the algorithms that we provide so that
you can tailor-make {{RISB}} for your own research problems.

We will try to reproduce the results of Sec. IIIB of Ref. [^Lechermann2007].

## Example model: bilayer Hubbard

:::{tip}
In `examples/bilayer_hubbard.py` we provide an example using the
`LatticeSolver` class that you can compare your answers to.
:::

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
( t_{\alpha} \hat{c}_{i \alpha \sigma}^{\dagger} \hat{c}_{j \alpha \sigma}
+ \mathrm{H.c.} ),
$$

where $d$ is the number of spatial dimensions ($d=3$ on the cubic
lattice), $\sigma$ is a spin label, $\alpha$ is an orbital label,
$\langle i j \rangle$ indicates nearest-neighbor bonds,
${\mathrm{H.c.}}$ is Hermitian conjugate, $t_{\alpha}$ is the
probability amplitude to move an electron between nearest neighbor sites, and
$\hat{c}^{(\dagger)}_{i\alpha\sigma}$ is an annihilation (creation) operator.

For degenerate orbitals on the cubic lattice, the dispersion for each
orbital is given by

$$
\varepsilon^{\mathrm{kin}}_{k\alpha} = -\frac{2}{d} t_{\alpha} \sum_{\mu}^d \cos(k_\mu a),
$$

where $k$ labels a reciprocal lattice vector. Hence,
$\hat{H}_{0, k \alpha \beta}^{\mathrm{kin}} = \varepsilon^{\mathrm{kin}}_{k\alpha}$
for $\alpha = \beta$ and zero otherwise.

The local part of the Hamiltonian is given by

$$
\hat{H}_i^{\mathrm{loc}} =
V \sum_{\sigma} (\hat{c}_{i 1 \sigma}^{\dagger} \hat{c}_{i 2 \sigma}
+ \mathrm{H.c.})
+ U \sum_{\alpha} \hat{n}_{i \alpha \uparrow} \hat{n}_{i \alpha \downarrow},
$$

where $V$ is the interlayer hopping between the orbitals, $U$ is
the local Coulomb repulsion, and
$\hat{n}_{i\alpha\sigma} \equiv \hat{c}^{\dagger}_{i\alpha\sigma} \hat{c}_{i\alpha\sigma}$.

:::{important}
In {{RISB}} the non-interacting quadratic terms within a cluster $i$ have
to be separated from the non-interacting quadratic terms coupling between
clusters. That is why the above is split into $\hat{H}^{\mathrm{kin}}_0$
and $\hat{H}^{\mathrm{loc}}_i$. Note that this is done automatically when
using `LatticeSolver`.
:::

## A: Construct $\hat{H}_0^{\mathrm{kin}}$

:::{seealso}
[Constructing tight-binding models](../how-to/quadratic_terms.md) for
some methods.
:::

First, construct the dispersion between clusters on the lattice. It should
not include the non-interacting quadratic terms within a cluster.

Notice that the Hamiltonian is block diagonal in spin. Hence, the hopping terms
are given by the kinetic Hamiltonian $\hat{H}^{\mathrm{kin}}_{0,k \alpha\beta}[\sigma]$,
and we can construct an array for each spin $\sigma$ separately. Each array
will be $\mathcal{N} \times n_{\mathrm{orb}} \times n_{\mathrm{orb}}$, where
$\mathcal{N}$ is the number of unit cells (sites in this case) on the
lattice and $n_{\mathrm{orb}}$ is the number of orbitals in each unit cell.
If the Hamiltonian was not block diagonal in spin, we would instead construct a
single $\mathcal{N} \times 2 n_{\mathrm{orb}} \times 2 n_{\mathrm{orb}}$
array. You can block diagonalize however is appropriate for your problem.

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
    n_kx : int, optional
        Linear dimension of k-mesh in each dimension.
    d : int, optional
        Number of dimensions, e.g., d = 2 is the square lattice.

    Returns
    -------
    mesh : numpy.ndarray
        The mesh in fractional coordinates, indexed as k, dim_1, dim_2, ..., dim_d
    """

    # Total number of k-points on the mesh
    n_k = n_kx**d

    mesh = np.empty(shape=(n_k, d))
    coords = [range(n_kx) for _ in range(d)]
    for idx, coord in enumerate(product(*coords)):
        for i in range(len(coord)):
            mesh[idx,i] = coord[i] / float(n_kx) + 0.5 / float(n_kx)

    return mesh
```

Now construct $\hat{H}_0^{\mathrm{kin}}$ on this mesh.

```python
def make_h0_kin_k(mesh, gf_struct, t = 1, a = 1):
    """
    Return dispersion of degenerate orbitals on a hypercubic lattice as a
    dictionary, where each key is the spin.

    Parameters
    ----------
    mesh : numpy.ndarray
        The k-space mesh of the Bravais lattice.
    gf_struct : list of pairs [ (str,int), ... ]
        Structure of matrices.
    t : float, optional
        Hopping amplitude.
    a : float, optional
        Lattice spacing.

    Returns
    -------
    h0_kin_k : numpy.ndarray
        Hopping matrix indexed as k, orbital_i, orbital_j.
    """
    n_k = mesh.shape[0]

    h0_kin_k = dict()
    for bl, n_orb in gf_struct:
        di = np.diag_indices(n_orb)
        h0_kin_k[bl] = np.zeros([n_k, n_orb, n_orb])
        h0_kin_k[bl][:,di[0],di[1]] = -2.0 * t * np.sum(np.cos(2.0 * a * np.pi * mesh), axis=1)[:, None]

    return h0_kin_k
```

Note that the structure of the matrices is given in the same way as {{TRIQS}}
Green's functions as

```python
gf_struct = [ ("up", 2), ("dn", 2) ]
```

## B: Construct $\hat{H}_i^{\mathrm{loc}}$

Next, construct the local Hamiltonian on each cluster (site) $i$. It has to
include all of the many-body interactions on each cluster as well as the
non-interacting quadratic terms that describe orbital energies and
hopping between orbitals on site $i$. In this case, we use the
second-quantized operators that the {{TRIQS}} library provides.

```python
from triqs.operators import *

def make_h_loc(V = 0.25, U = 4):
    """
    Return the local terms of the bilayer Hubbad model as a TRIQS operator.

    Parameters
    ----------
    V : float, optional
        Hopping between orbitals on each cluster.
    U : float, optional
        Local Coulomb repulsion on each site.

    Returns
    -------
    h_loc : triqs.operators.Operator
    """

    h_loc = Operator()
    for o in range(2):
        h_loc += U * n("up", o) * n("dn", o)
    for bl in ["up", "dn"]:
        h_loc += V * ( c_dag(bl, 0) * c(bl, 1) + c_dag(bl, 1) * c(bl, 0) )

    return h_loc
```

## C: Construct $\hat{H}^{\mathrm{qp}}$

The first step in the self-consistent loop is to obtain the mean-field
quasiparticle Hamiltonian

$$
\hat{H}^{\mathrm{qp}} =
\mathcal{R} \hat{H}^{\mathrm{kin}} \mathcal{R}^{\dagger}
+ \lambda,
$$

where $\mathcal{R}$ is the renormalization matrix and $\lambda$ is the
correlation potential matrix. They are assumed to be the same on every site,
so that they are square matrices whose dimensions are $2 n_{\mathrm{orb}}$.
It is useful to construct them with a block matrix structure, similarly to how
$\hat{H}^{\mathrm{kin}}_0$ was done.

$\mathcal{R}$ and $\mathcal{\lambda}$ are mean-field matrices and are the
input initial guesses to the solution to the self-consistent loop. Often a
reasonable initial guess is to choose $\mathcal{R}$ as the identity and
$\lambda$ as the zero matrix.

There is a helper function that constructs $\hat{H}^{\mathrm{qp}}$, and
it is simple to get its eigenvalues and eigenvectors at each $k$-point on the
finite grid.

```python
from risb import helpers

energies_qp = dict()
bloch_vector_qp = dict()
for bl, bl_size in gf_struct
    h_qp = helpers.get_h_qp(R[bl], Lambda[bl], h0_kin_k[bl])
    energies_qp[bl], bloch_vector_qp[bl] = np.linalg.eigh(h_qp)
```

## D: Setup k-integrator function

:::{seealso}
[About k-integration](../explanations/kweight.md) for the theory of how
$k$-space integration is numerically done in most condensed matter codes.

[Using `SmearingKWeight`](../how-to/kweight.md) for a class that implements
the below weighting factor.
:::

Next you will need to construct how $k$-space integrals are performed. All
integrals are with respect to the eigenenergies $\xi^{\mathrm{qp}}_{kn}$ ($n$
is a band index) of $\hat{H}^{\mathrm{qp}}$.

The simplest approximation for the integration weight is to just use the
Fermi-Dirac distribution function $f(\xi)$ on a finite grid
at the inverse temperature $\beta$. That is,

$$
w(\xi^{\mathrm{qp}}_{kn}) = \frac{1}{\mathcal{N}} f(\xi^{\mathrm{qp}}_{kn}).
$$

The code to perform this is

```python
def update_weights(energies, mu=0, beta=10):
    """
    Return the integration weights for each band at each
    k-point on the lattice.

    Parameters
    ----------
    energies : dict[numpy.ndarray]
        Energies at each k-point. Each key is a symmetry block, and its value
        is the energy in each band n, indexed as k, n.
    mu : float, optional
        Chemical potential.
    beta : float, optional
        Inverse temperature.

    Returns
    -------
    weights : dict[numpy.ndarray]
    """
    n_k = energies.shape[0]
    return {bl : 1.0 / (np.exp(beta * (energies[bl] - mu)) + 1.0) / n_k for bl in energies.keys()}
```

Then the integration weight at each $k$-point can be calculated

```python
kweights = update_weights(energies = energies_qp, mu = ..., beta = ...)
```

## E: Setup mean-field matrices

We now need to initialize the mean-field matrices used in
{{RISB}}. In {{RISB}} the homogenous assumption is taken,
so that the matrices are the same on every site.
Below we describe what each mean-field matrix physically relates to within
the algorithm.

The single-particle quasiparticle density
matrix of $\hat{H}^{\mathrm{qp}}$ is $\Delta^{\mathrm{qp}}$ and the lopsided
quasiparticle kinetic energy is $E^{c,\mathrm{qp}}$.

The non-interacting quadratic parts of the embedding Hamiltonian
$\hat{H}^{\mathrm{emb}}$ are described by the hybridization matrix
$\mathcal{D}$, and the matrix that describes the couplings in the bath
$\lambda^c$.

The single-particle density matrices of $\hat{H}^{\mathrm{emb}}$ are the
density matrix of the f-electrons (the bath, these are quasiparticles)
$\Delta^{f}$,
the density matrix of the c-electrons (the impurity, these are physical
electrons) $\Delta^{c}$,
and the off-diagonal density matrix between the c- and f- electrons
(the impurity and the bath) $\Delta^{cf}$.

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
rho_f = dict()
rho_cf = dict()
rho_c = dict()
for bl, bl_size in gf_struct:
    rho_f[bl] = np.zeros((bl_size, bl_size))
    rho_cf[bl] = np.zeros((bl_size, bl_size))
    rho_c[bl] = np.zeros((bl_size, bl_size))
```

### Helper functions

As an aside, let me describe how to obtain the above mean-field matrices, which
has to be done at each iteration of the self-consistent process. There are
helper functions that do this for you. They simply take in numpy arrays
and either use `numpy.einsum` or `numpy.dot` arrays together.

Remember that you can check the docstring of a helper function with
`help(helpers.function)`, which will even tell you which equation it relates
to within the [literature](../about.md#literature-of-original-theory).

```python
from risb import helpers

# H^qp single-particle density
for bl, bl_size in gf_struct:
    rho_qp[bl] = helpers.get_rho_qp(bloch_vector_qp[bl], kweights[bl])

# H^qp (lopsided) quasiparticle kinetic energy
for bl, bl_size in gf_struct:
    h0_kin_k_R[bl] = helpers.get_h0_kin_k_R(R[bl], h0_kin_k[bl], bloch_vector_qp[bl])
    lopsided_kinetic_energy_qp[bl] = helpers.get_ke(h0_kin_k_R, bloch_vector_qp[bl], kweights[bl])

# H^emb hybridization
for bl, bl_size in gf_struct:
    D[bl] = helpers.get_d(rho_qp[bl], kinetic_energy_qp[bl])

# H^emb bath
for bl, bl_size in gf_struct:
    Lambda_c[bl] = helpers.get_lambda_c(rho_qp[bl], R[bl], Lambda[bl], D[bl])
```

See [Solving $\hat{H}^{\mathrm{emb}}$](#f-solving-hat-h-mathrm-emb) for the
single-particle density matrices of $\hat{H}^{\mathrm{emb}}$.

## F: Solving $\hat{H}^{\mathrm{emb}}$

:::{seealso}
[About the embedding Hamiltonian](../explanations/embedding.md).

[Using embedding classes](../how-to/embedding.md).
:::

Now we have to solve the impurity problem defined by the embedding Hamiltonian
$\hat{H}^{\mathrm{emb}}$. There is a simple (but poorly scaled to large
problems) implementation that only uses {{TRIQS}}.

```python
from risb.embedding import EmbeddingAtomDiag

embedding = EmbeddingAtomDiag(h_loc, gf_struct)
```

Setting the embedding Hamiltonian $\hat{H}^{\mathrm{emb}}$ is done with

```python
embedding.set_h_emb(Lambda_c, D)
```

Solving for the ground state in the $n_{\mathrm{orb}}$ particle sector, which
corresponds to the embedding problem being half-filled, is done with

```python
embedding.solve()
```

The methods to calculate the single-particle density matrices are

```python
for bl, bl_size in gf_struct:
    rho_f[bl] = embedding.get_rho_f(bl)
    rho_cf[bl] = embedding.get_rho_cf(bl)
    rho_c[bl] = embedding.get_rho_c(bl)
```

## G: Closing the loop

After the embedding Hamiltonian is solved and the single-particle density
matrices of the embedding Hamiltonian are obtained, there are two ways to do
the self-consistency loop. The first is to calculate a new guess for
$\mathcal{R}$ and $\lambda$ which re-parameterizes $H^{\mathrm{qp}}$.
The helper functions to do this are

```python
from risb import helpers

for bl, bl_size in gf_struct:
    Lambda[bl] = helpers.get_lambda(R[bl], D[bl], Lambda_c[bl], rho_f[bl])
    R[bl] = helpers.get_r(rho_cf[bl], rho_f[bl])
```

The second is as a root problem, adjusting $\mathcal{R}$ and $\lambda$ to
ensure that the density matrices from $\hat{H}^{\mathrm{qp}}$ match the
density matrices from $\hat{H}^{\mathrm{emb}}$.

```python
for bl, bl_size in gf_struct:
    f1 = helpers.get_f1(rho_cf[bl], rho_qp[bl], R[bl])
    f2 = helpers.get_f2(rho_f[bl], rho_qp[bl])
```

## Exercises

1. Can you match each part above with the self-consistent loop defined in the
   [literature](../about)?
1. Piece together everything above and write your own code.
1. Solve for a range of $U$ values at half-filling ($\mu = U / 2$).
1. How does $\beta$ and the size of the k-space mesh affect the results?
1. What is the evolution of the quasiparticle weight $Z$ at
   half-filling (Fig. 7)?
1. What is the evolution of the electron filling in the
   bonding/anti-bonding ($\pm$) basis?
1. Implement a method to solve for the chemical potential $\mu$ at a fixed
   electron density $n$ (you may find :py:func:`scipy.optimize.brentq`
   or :py:func:`scipy.optimize.bisect` from :py:mod:`scipy.optimize` useful).
1. What is the evolution of the quasiparticle weight at electron filling
   $n = 1.88$ (Fig. 10)?

## Conclusion

You have built the self-consistent loop for
{{RISB}} and solved a (not so simple)
interacting fermion problem. The code in :py:class:`risb.solve_lattice.LatticeSolver`
is not much more complicated than what you have done. You should now easily be
able to mofify, implement, and contribute to any parts in the library.
You also now understand the basic ingredients needed for most self-consistent
procedures in much more sophisticated codes for {{DFT}} and {{DMFT}}.
Hopefully, you can easily build upon this simple example to do much more
complicated things.

Below is some code that should be very easy to fill in. But you will understand
much more about {{RISB}} if you try to piece everything together from the
self-consistent equations in the [literature](../about).

## Skeleton code cheat sheet

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

# Implement step A
h0_kin_k =

# Implement step B
h_loc =

# Implement step D
update_weights =

# Implement step E
initialized_mean_field_matrices =
...

# Implement step F
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

    # Step C: construct H^qp
    for bl, bl_size in gf_struct:
        energies_qp[bl], bloch_vector_qp[bl] = helpers.get_h_qp(R[bl], Lambda[bl], h0_kin_k[bl])

    # k-space integration weights
    kweights = update_weights(energies_qp, beta = , mu = )

    # Step E is below

    # H^qp density matrices
    for bl, bl_size in gf_struct:
        rho_qp[bl] = helpers.get_rho_qp(bloch_vector_qp[bl], kweights[bl])
        h0_kin_k_R[bl] = helpers.get_h0_kin_k_R(R[bl], h0_kin_k[bl], bloch_vector_qp[bl])
        lopsided_kinetic_energy_qp[bl] = helpers.get_ke(h0_kin_k_R, bloch_vector_qp[bl], kweights[bl])

    # H^emb parameters
    for bl, bl_size in gf_struct:
        D[bl] = helpers.get_d(rho_qp[bl], kinetic_energy_bl[bl])
        Lambda_c[bl] = helpers.get_lambda_c(rho_qp[bl], R[bl], Lambda[bl], D[bl])

    # Solve H^emb
    embedding.set_h_emb(Lambda_c, D)
    embedding.solve()

    # H^emb density matrices
    for bl, bl_size in gf_struct:
        rho_f[bl] = embedding.get_rho_f(bl)
        rho_cf[bl] = embedding.get_rho_cf(bl)
        rho_c[bl] = embedding.get_rho_c(bl)

    # Step G: New guess for H^qp parameters
    for bl, bl_size in gf_struct:
        Lambda[bl] = sc.get_lambda(R[bl], D[bl], Lambda_c[bl], rho_f[bl])
        R[bl] = sc.get_r(rho_cf[bl], rho_f[bl])

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

[^Lechermann2007]:
    [F. Lechermann, A. Georges, G. Kotliar, and O. Parcollet,
    _Rotationally invariant slave-boson formalism and momentum dependence of the
    quasiparticle weight_,
    Phys. Rev.B **76**, 155102 (2007)](https://doi.org/10.1103/PhysRevB.76.155102)
