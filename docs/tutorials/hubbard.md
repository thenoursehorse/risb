# Brinkman-Rice metal-insulator transition

In this tutorial you will use `LatticeSolver` and `EmbeddingAtomDiag` to solve
the half-filled Hubbard model. This is one of the simplest strongly correlated
electron models, yet in general does not have an exact solution. Using this
model, you will learn about the Brinkman-Rice [^Brinkman1970] description of
a Mott insulator, the quintessential strongly correlated phase of matter.
At the end of this tutorial you will have an idea of some of the kinds
ground states {{RISB}} can capture, and importantly some of its limitations.

:::{seealso}
[Using `LatticeSolver`](../how-to/lattice_solver.md) for more details on the
{{RISB}} solver class.
:::

:::{tip}
In examples/hubbard.py we provide an example if you are stuck. But you will
learn a lot more if you try to write everything yourself.
:::

## The model

The Hubbard model is given by

$$
\hat{H} = t \sum_{ij\sigma}
\hat{c}^{\dagger}_{i\sigma} \hat{c}^{}_{j\sigma}
+ U \sum_i \hat{n}^{}_{i\uparrow} \hat{n}^{}_{i\downarrow},
$$

where $t$ is the hopping amplitude between sites,
$\hat{c}^{(\dagger)}_{i\sigma}$ is a (creation) annihilation
operator on site $i$ with spin $\sigma \in \{\uparrow, \downarrow\}$, and
$\hat{n}_{i\sigma} \equiv \hat{c}^{\dagger}_{i\sigma} \hat{c}^{}_{j\sigma}$
is the number operator.

The non-interacting part is given by the first term

$$
\hat{H}^0 = t \sum_{ij\sigma}
\hat{c}^{\dagger}_{i\sigma} \hat{c}^{}_{j\sigma},
$$

which wants to move electrons around and create plane waves on the lattice.
The interacting part on every site is the summand in the second term

$$
\hat{H}_i^{\mathrm{int}} =
U \hat{n}^{}_{i\uparrow} \hat{n}^{}_{i\downarrow},
$$

which wants to keep electrons at arms length (apart). By themselves, each term
can be solved analytically. The headache (and all of the interesting resulting
physics) happens when the two terms compete.

In the paramagnetic state (where spin rotational symmetry does not break) all
single-site {{RISB}} calculations will qualitatively give the same solutions.
The differences will occur in larger {{RISB}} clusters, where more spatial
correlations are captured, and in magnetic states captured at the single-site
level (similar to how magnetic states are captured in Hartree-Fock).

## Construct the non-interacting terms $\hat{H}^{0}$

:::{seealso}
[Constructing tight-binding models](../how-to/quadratic_terms.md) for some
other ways to construct $\hat{H}^{0}$.
:::

In this tutorial, you can pick any single-atom unit cell. But, for simplicity,
lets consider the cubic lattice.

First create a cubic $k$-space mesh in crystal coordinates with lattice
spacing $a = 1$

```python
import numpy as np

# Spatial dimensions
d = 3

# Number of points on grid in each dimension
n_k_x = 10

# Total number of points on the grid
n_k = n_k_x**d

# Linear array in each spatial dimension
# endpoint = False otherwise you will count the boundary twice
k_linear = np.linspace(0, 2 * np.pi, n_k_x, endpoint=False)

# Make the mesh
# Can you figure out the outcome of the pythonic syntax for the first param?
k_mesh = np.meshgrid(*[k_linear]*3, indexing='ij')

# Reshape to list
k_mesh = np.reshape(k_mesh, (n_k, d))
```

Next create the hopping marix for each spin $\sigma$ in the structure that
`LatticeSolver` expects. In general, this is a [numpy.ndarray](numpy.ndarray)
indexed as $(k, \mathrm{orb}_i, \mathrm{orb}_j)$, where $k$ is a point on the
grid, and $\mathrm{orb}_i$ is an atom, orbital, and/or spin in the unit cell.
In this case there is only a single atom, and the two spin orbitals can be
treated separately because there is no spin-orbit coupling.

```python
# Hopping amplitude
t = 1

# Number of orbitals on a site
n_orb = 1

# Hopping matrix
h0_k = dict()
for bl in ['up', 'dn']:
    h0_k[bl] = np.zeros([n_k, n_orb, n_orb])
    for orb in range(n_orb):
        h0_k[bl][:, orb, orb] = -2.0 * t * ( np.cos(k_mesh[:,0]) \
                                           + np.cos(k_mesh[:,1]) \
                                           + np.cos(k_mesh[:,2]) )
```

## Local structure on a site

Now set up how the local structure looks like. This follows the same
conventions as Green's functions in {{TRIQS}}

```python
gf_struct = [('up', n_orb), ('dn', n_orb)]
```

All single-particle matrices defined on a cluster (a site) will follow this
block structure. It basically says that the `'up'` and '`dn'` spins do not
have off-diagonal elements between them, and each matrix in each block is a
$n_{orb} \times n_{orb}$ matrix.

The structure of `gf_structure` also determines what local operators are
on a cluster (a site) that you will use in the next section.

## Construct the interactions $\hat{H}_i^{\mathrm{int}}$

Since you are using `EmbeddingAtomDiag`, you might as well use all of the great
second-quantized operator tools provided by {{TRIQS}}.

Because {{RISB}} will make all equivalent clusters homogenous, you only need to
construct the interactions for a single site.

Now construct the interaction terms. The Hubbard interaction is simply given
as

```python
from triqs.operators import n

# Hubbard U
U = 5

h_int = U * n('up', 0) * n('dn', 0)
```

There are also some great helper tools for constructing interactions
in the `triqs.operators.util.hamiltonians` module provided by {{TRIQS}}. It
is completely overkill for this interaction, but let's do it anyway.

```python
from triqs.operators.util.hamiltonians import h_int_density

# 2D matrix of U_ij^{sigma sigma} (same spins)
U = np.zeros(shape = [n_orb, n_orb])

# 2D matrix of U_ij^{sigma nu} (different spins)
Uprime = np.full(shape = [n_orb, n_orb], fill_value = 5)

h_int = h_int_density(spin_names = ['up', 'dn'],
                      n_orb = n_orb,
                      U = U,
                      Uprime = Uprime,
                      off_diag=True
)
```

## Enforce paramagnetism

There will be some numerical noise in the solutions that can make
the `'up'` and `'dn'` spin parts of the solution slightly different. It helps
the self-consistent process to ensure that this symmetry is strictly enforced.
You can do this by passing a symmetrization function to the solver

```python
# The object LatticeSolver expects is list[dict[ndarray]], where each
# item in the list is a cluster. In this case there is only one
# cluster.
def force_paramagnetic(A):
    A[0]['up'] = 0.5 * (A[0]['up'] + A[0]['dn'])
    A[0]['dn'] = A[0]['up']
    return A
```

## Setup the k-space integral calculator

:::{seealso}
[About k-integration](../explanations/kweight.md).

[Using `SmearKWeight`](../how-to/kweight.md).
:::

You need to specify a function to `LatticeSolver` that gives the integration
weights at each $k$-point on the lattice. The easiest thing to do is to use
the `SmearKWeight` class that assumes the weights are given by
Fermi-Dirac functions

```python
# The smearing, in units of inverse temperature. It is an approximation
# because RISB is at zero temperature
beta = 40

# 1 electron on average per unit cell, half-filling
n_target = 1 # half-filling

kweight = SmearingKWeight(beta=beta, n_target=n_target)
```

## Setup the solvers and find a self-consistent solution

:::{seealso}
[Using embedding classes](../how-to/embedding.md).

[Using `LatticeSolver`](../how-to/lattice_solver.md).

[Constructing the {{RISB}} self-consistent loop](../tutorials/self-consistent.md).
:::

You need a class that solves the
[embedding Hamiltonian](../explanations/embedding.md). A simple one is
:py:class:`risb.embedding.EmbeddingAtomDiag`

```python
embedding = EmbeddingClass(h_int, gf_struct)
```

Next setup the {{RISB}} solver for the self-consistent loop. This will make a
guess for some parameters (renormalization matrix $\mathcal{R}$ and
correlation potential matrixx $\lambda$) and go through a self-consistent
loop until $\mathcal{R}$ and $\lambda$ reach a fixed point. On a lattice
the solver is setup as

```python
S = LatticeSolver(h0_k = h0_k,
                  gf_struct = gf_struct,
                  embedding = embedding,
                  update_weights = kweight.update_weights,
                  symmetries = [force_paramagnetic]
)
```

Now you can solve the model

```python
# Tolerance for when self-consistent procedure is stopped
tol = 1e-6

S.solve(tol = tol)
```

## Observables on a site

In {{RISB}} it is very simple to calculate local quantities on a cluster
(a site). Using {{TRIQS}} there are some simple helper functions to construct
some common observables. The main one we are interested in is the total spin
on a site.

The average total spin per cluster $S$ is
the solution to

$$
S(S+1) = \frac{1}{\mathcal{N}} \sum_i \langle \hat{S}_i^2 \rangle
$$

where $\hat{S}_i$ is the total spin operator on a cluster (a site). Because
the homogenous assumption is used in {{RISB}}, the spin per site is the same
on every site, so you only have to (and only can) calculate it on one site.

In {{TRIQS}} the total spin $\hat{S}^2$ is given by

```python
from triqs.operators.util.observables import S2_op

total_spin_Op = S2_op(spin_names, n_orb, off_diag=True)
```

To calculate the overlap with the solution {{RISB}} finds

```python
total_spin = embedding.overlap(total_spin_Op)
```

## Exercises

1.  Solve for a range of $U$ values from $U = 0$ to $U = 12$. You may find
    `embedding.set_h_int()` useful.

1.  Plot the average spin $S$ on a site versus $U$. Plot the quasiparticle
    weight $Z$ (accessed by `S.Z`) versus $U$. What happens to $S$ and
    $Z$ for large $U$?

1.  Plot the hybridization coupling $D$ (accessed by `embedding.D`) of the
    [embedding Hamiltonian](../explanations/embedding.md). What happens to $D$
    for large $U$? What does this imply about the impurity and the bath coupling
    in the embedding Hamiltonian $\hat{H}^{\mathrm{emb}}$?

1.  You will notice that $Z$, $S$, and $D$ both have an abrupt change, and then
    plateau, at some critical interaction strength $U_c$. What is the phase of
    matter to the left of $U_c$? What is the phase of matter to the right of
    $U_c$? What type of phase transition occurs at $U_c$?

1.  Construct an operator that calculates the number variance

    $$
    \mathrm{Var}(N) = \langle (\hat{N} - \langle N \rangle)^2 \rangle,
    $$

    and total spin variance

    $$
    \mathrm{Var}(N) = \langle (\hat{N} - \langle N \rangle)^2 \rangle,
    $$

    and plot these as a function of $U$. How do they look like to the left
    and right of the critical interaction $\hat{U}_c$?

1.  If you know anything about the Mott metal-insulator phase transition,
    you might find the phase of matter to the right of $U_c$ a bit strange. In the
    limit $t/U$ is small, a perturbative expansion gives the low-energy effective
    theory of the half-filled Hubbard model as the Heisenberg model, given by

        $$
        \hat{H} = J \sum_{i \neq j} \vec{S}_i \cdot \vec{S}_j,
        $$

        where $\vec{S}_i = (\hat{S}_i^x, \hat{S}_i^y, \hat{S}_i^z)$, and
        $J = 4 t^2 / U$.

        The above implies that the localized spin-$1/2$s on each site
        have a spin-spin exchange interaction between them arising from charge
        fluctuations (high-energy virtual processes). In the atomic limit
        $t / U \rightarrow 0$ the spin exchange $J \rightarrow 0$, with an
        isolated spin-$1/2$ on each site because there is no coupling between
        them.

        Recalling the observables you calculated to the right of the critical
        interaction $U_c$ (which happened at a finite $U$) what does this imply
        about the ground state that {{RISB}} predicts in the Mott phase?

        What physics is {{RISB}} not capturing that, e.g., {{DMFT}} captures,
        and why?

        :::{hint}
        In {{RISB}} the frequency dependence of the self-energy is linear and
        goes like $\Sigma(\omega) \sim (I - Z^{-1}) \omega$, where $I$ is the
        identity.
        :::

## Conclusion

You have solved the Hubbard model using the {{RISB}} approximation and found
a Brinkman-Rice metal-insulator transition to the Mott insulating phase.
In cluster extensions to {{RISB}} this type of transition can also occur. You
should now know one of the ways that {{RISB}} captures insulators, and
understand the limitations [^GHOST] for the kinds of ground states that {{RISB}}
can describe.

## Bonus

In this tutorial you only considered solutions where every site was the same.
Construct a two-site super cell of the cubic lattice, and treat the
two sites as separate embedding spaces $\mathcal{C}_i$ (see
[Using projectors](../how-to/projectors.md)). You will also find a
metal-insulator transition (of the Slater type).

1. How does this differ to the Brinkman-Rice metal-insulator transition?

1. How does this differ to the mean-field solution (Hartree-Fock) to the
   Hubbard model on the cubic lattice?

1. Knowing that the Mott insulating state that {{RISB}} captures in the
   Brinkman-Rice transition should have spin-spin exchange, what is the expected
   ground state of the Hubbard model on the cubic lattice at half-filling?

[^Brinkman1970]:
    [W. F. Brinkman and T. M. Rice,
    _Application of Gutzwiller's Variational Method to the Metal-Insulator Transition_,
    Phys. Rev. B **2**, 4302 (1970)](https://doi.org/10.1103/PhysRevB.2.4302).

[^GHOST]:
    [N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević,
    _Emergent Bloch excitations in Mott matter_,
    Phys. Rev. B **96**, 195126 (2017)](https://doi.org/10.1103/PhysRevB.96.195126),
    and the papers that cite this paper for extensions that improve upon
    normal {{RISB}}.
