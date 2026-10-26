# Using impurity solvers

This guide shows you how to use the impurity solvers and describes a little
bit about how they do things.

:::{seealso}
:class: dropdown
[About the impurity problem](../explanations/impurity.md).
:::

## `FockitImpuritySolver`

First a `gf_struct` object has to be created that describes the correlated
subspace $\mathcal{C}_i$. It is a list of pairs, each the name of a symmetry
block and its dimension. For example, for the $e_g$ symmetry sector (two
orbitals) in a system where spin is conserved, the structure can be encoded as

```python
gf_struct = [("up_eg", 2), ("dn_eg", 2)]
```

Next a local Hamiltonian has to be constructed. It is built from the
[fockit](https://github.com/thenoursehorse/fockit) operators `c(block, index)`,
`cdag(block, index)` and `nc(block, index)`, with the blocks and indices of
`gf_struct`, and includes the interaction terms. It can include the local
quadratic terms in $\mathcal{C}_i$, but these can also be set later on. It
does not make any difference for the solver. For example, a two-orbital
Hubbard model with hopping between the orbitals can be constructed as

```python
from fockit.algebra import c, cdag, nc, zero

n_orb = 2

U = 1
h_int = zero()
for o in range(n_orb):
    h_int += U * nc("up", o) * nc("dn", o)

# Optional local terms
# V = 0.25
# for s in ["up", "dn"]:
#     h_int += V * (cdag(s, 0) * c(s, 1) + cdag(s, 1) * c(s, 0))
```

{py:mod}`risb.helpers_fockit` has helper functions for some common operators,
e.g., the Kanamori interaction
{py:func}`~risb.helpers_fockit.h_int_kanamori`.

### Constructor

The solver is instantiated as

```python
from risb.impurity_solvers import FockitImpuritySolver

impurity_solver = FockitImpuritySolver(h_int, gf_struct)
```

The interaction Hamiltonian is stored in `impurity_solver.h_int` and the block
matrix structure is stored in `impurity_solver.gf_struct`. The constructor
also builds the basis of the half-filled particle sector of the impurity
problem (physical and bath orbitals), stored in `impurity_solver.basis`.

### Setting `h_imp`

Next the impurity Hamiltonian `h_imp` has to be set with

```python
impurity_solver.set_h_imp(Lambda_c, D)
```

where `Lambda_c` and `D` are block matrices that describe the impurity problem
with the structure of `gf_struct`.
{py:meth}`~risb.impurity_solvers.FockitImpuritySolver.set_h_imp` is internally
called from within the self-consistent loop in `LatticeSolver`. The
hybridization term from `D` is stored as `impurity_solver.h_hybr` and the bath
hopping from `Lambda_c` is stored as `impurity_solver.h_bath`, both as fockit
operators.

If the non-interacting quadratic couplings are not included in `h_int`, then
they must be passed as

```python
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix)
```

where `h0_loc_matrix` is a matrix that describes the couplings, with the same
block matrix structure as `Lambda_c` and `D`. This is stored in
`impurity_solver.h0_loc` as a fockit operator.

### Setting `h_int`

If you want to update the interaction terms of the physical orbitals

```python
# A new h_int
h_int = ...

impurity_solver.set_h_int(h_int)
```

### Solving

It is solved as

```python
impurity_solver.solve()
```

There are no arguments. `solve()` builds the sparse matrix of `h_imp` in
`impurity_solver.basis` and finds its lowest eigenstate with
{py:func}`scipy.sparse.linalg.eigsh`. The ground state energy and vector are
stored in `impurity_solver.gs_energy` and `impurity_solver.gs_vector`. It then
calculates the density matrices described below for every block.

`solve()` raises a `ValueError` if `h_imp` is not Hermitian, and
{py:class}`scipy.sparse.linalg.ArpackNoConvergence` if `eigsh` does not
converge. If the ground state is degenerate, the solver returns one state in
the degenerate subspace, and which one is not specified.

### Choosing the eigensolver

If you want to use the dense eigensolver {py:func}`scipy.linalg.eigh` instead

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, method="eigh")
```

It is exact, but only practical for small impurity problems. Keyword arguments
to the eigensolver are passed as

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, method_options={"ncv": 60})
```

### Symmetries

If the impurity Hamiltonian has symmetries, you can restrict the basis to one
symmetry sector with `diagonal_symmetries`, `permutation_symmetries` and
`onsite_symmetries`. They are passed to `build_basis` from fockit, which
describes each kind. The modes of the bath orbitals are
`("bath_" + block, index)`. For example, to keep only the states with total
$S_z = 0$ that are even under swapping the spins

```python
from fockit.algebra import nc, zero

Sz = zero()
for o in range(n_orb):
    for up, dn in [("up", "dn"), ("bath_up", "bath_dn")]:
        Sz += 0.5 * (nc(up, o) - nc(dn, o))

swap = {"up": "dn", "dn": "up", "bath_up": "bath_dn", "bath_dn": "bath_up"}
spin_swap = {(bl, o): (swap[bl], o) for bl in swap for o in range(n_orb)}

impurity_solver = FockitImpuritySolver(
    h_int,
    gf_struct,
    diagonal_symmetries={Sz: 0},
    permutation_symmetries=[(spin_swap, 1)],
)
```

A permutation or onsite symmetry that changes a diagonal symmetry, like the
spin swap with $S_z$, makes the solve use more memory. If a permutation or
onsite symmetry changes the particle number, or the symmetries do not commute
with each other, the ground state is written in a basis without the
permutation and onsite symmetries, which is slower.

:::{warning}
The solver finds the lowest state in the chosen sector, so choose the sector
with the state you want, usually the ground state. Choosing another sector
targets a higher state. The eigenvalue of a state also depends on the order of
the fermion modes. The impurity Hamiltonian must have the symmetries at every
step of the self-consistent loop, e.g., enforced with `symmetrizers` in `LatticeSolver`.
`solve()` raises a `ValueError` if it does not.
:::

### Degenerate ground states

If you want a warning when the ground state is degenerate

```python
impurity_solver = FockitImpuritySolver(
    h_int, gf_struct, degenerate="warn", degenerate_tol=1e-8
)
```

This finds two states instead of one, so it is slower.

### Temperature of the impurity problem

By default the density matrices are from the ground state of the impurity
problem. If you want them averaged over the eigenstates of the half-filled
sector with Boltzmann weights

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, beta=40)
```

This can help `LatticeSolver` converge in or near a Mott insulator, and is
not a finite-temperature solution. Use the same inverse temperature as in
{py:class}`~risb.kweight.SmearingKWeight`. `overlap` then also gives thermal
averages.

### Grand-canonical impurity problem

If you want thermal averages over every particle number of the impurity
problem, as `ThermalLatticeSolver` needs

```python
from risb.impurity_solvers import FockitThermalImpuritySolver

impurity_solver = FockitThermalImpuritySolver(h_int, gf_struct, beta=40)
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix, mu=mu)
```

The chemical potential `mu` is subtracted on the physical and bath orbitals.
The grand potential is `impurity_solver.grand_potential`. The symmetries are
given without eigenvalues, because every eigenvalue is in the average

```python
impurity_solver = FockitThermalImpuritySolver(
    h_int,
    gf_struct,
    beta=40,
    diagonal_symmetries=[Sz],
    permutation_symmetries=[layer_swap],
)
```

The diagonal symmetries must be sums of number operators, like $S_z$, and the
permutation and onsite symmetries must not change them, so use either $S_z$
or the spin swap.

### Getting the density matrices

The single-particle density matrix of the $f$ electrons in the impurity
problem in one of the blocks specified by `gf_struct` is obtained as

```python
impurity_solver.get_rho_f(block)
```

The off-diagonal density matrix between the $c$ and $f$ electrons is obtained
as

```python
impurity_solver.get_rho_cf(block)
```

The density matrix of the $c$ electrons is obtained as

```python
impurity_solver.get_rho_c(block)
```

The three terms above give the full single-particle density matrix in the
impurity problem. They are also stored for every block as the dictionaries
`impurity_solver.rho_f`, `impurity_solver.rho_cf`, and `impurity_solver.rho_c`.

### Local expectation values

Any local expectation value of an operator `Op` in the impurity problem is
obtained as

```python
impurity_solver.overlap(Op)
```

`Op` must be a fockit operator. For example, the total number of electrons and
the total spin squared on the physical orbitals are

```python
from risb.helpers_fockit import N_op, S2_op

total_number = impurity_solver.overlap(N_op(["up", "dn"], n_orb))
total_spin = impurity_solver.overlap(S2_op(["up", "dn"], n_orb))
```

To get observables in the $f$ electrons, their modes are
`("bath_" + block, index)`, and their structure is obtained as

```python
impurity_solver.gf_struct_bath
```

## `Block2ImpuritySolver`

This solver finds the ground state of the impurity problem with the density
matrix renormalization group (DMRG) from
[block2](https://block2.readthedocs.io), for impurity problems too large for
exact diagonalization. It needs the optional dependency block2, installed with

```shell
pip install "risb[block2]"
```

or `uv add "risb[block2]"` in a [uv](https://docs.astral.sh/uv) project.

The impurity problem must be collinear and real: each block of `gf_struct` is
one spin, and `h_int` conserves the total spin projection.

### Constructor

The solver is instantiated as

```python
from risb.impurity_solvers import Block2ImpuritySolver

impurity_solver = Block2ImpuritySolver(h_int, gf_struct, bond_dim=500)
```

`bond_dim` is the bond dimension of the matrix product state. The larger it
is, the closer the state is to the exact ground state, and the longer a solve
takes.

The spin-up and spin-down block of each pair of blocks with the same orbitals
is given as

```python
gf_struct = [("up_eg", 2), ("dn_eg", 2), ("up_t2g", 3), ("dn_t2g", 3)]
impurity_solver = Block2ImpuritySolver(
    h_int, gf_struct, spin_blocks=[("up_eg", "dn_eg"), ("up_t2g", "dn_t2g")]
)
```

The default is `[("up", "dn")]`.

### Symmetries

By default only the particle number is conserved, and the solver finds the
lowest state at half filling. The total spin projection `"sz"` and the total
spin `"s"` can be conserved as well, with the sector of each

```python
# The lowest state with Sz = 0
impurity_solver = Block2ImpuritySolver(h_int, gf_struct, symmetries={"sz": 0})

# The lowest singlet
impurity_solver = Block2ImpuritySolver(h_int, gf_struct, symmetries={"s": 0})
```

With `None` as the sector, every sector is solved and the lowest is kept

```python
impurity_solver = Block2ImpuritySolver(h_int, gf_struct, symmetries={"s": None})
```

With `"s"`, the density matrices average over the states of the spin
multiplet. Adding `"sz"` picks one of them, e.g., `{"s": 1, "sz": 1}`. The
sector of the ground state is stored in `impurity_solver.sector`.

With `"s"`, `solve()` raises a `ValueError` if the spin-up and spin-down terms
of `h_imp` differ.

### Mixing sweeps

DMRG keeps a state in the sectors of every quantity that the Hamiltonian
conserves, also the ones the solver does not, such as the parity of each
orbital with its bath orbitals in a Kanamori impurity. It can then end in an
excited state. Each solve therefore starts with the sweeps of Appendix A1 of
Wolf, et al.[^Wolf2015], with extra random hoppings between the orbitals that
connect these sectors. They keep the symmetries in `symmetries`. The hoppings
fall in stages, each a strength relative to the largest one-body coupling, a
number of sweeps at the full bond dimension, and a noise

```python
impurity_solver = Block2ImpuritySolver(
    h_int,
    gf_struct,
    mix_schedule=[(0.1, 6, 1e-5), (0.01, 6, 1e-5), (0.001, 6, 1e-5), (1e-4, 6, 1e-5)],
)
```

which is the default. They are turned off with

```python
impurity_solver = Block2ImpuritySolver(h_int, gf_struct, mix_schedule=())
```

`Block2ThermalImpuritySolver` takes the same `mix_schedule` for
`method="lowest"`.

### Setting `h_imp` and solving

As for `FockitImpuritySolver`,

```python
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix)
impurity_solver.solve()
```

`solve()` stores the ground state energy in `impurity_solver.gs_energy`, and
the largest weight discarded in the last sweep in
`impurity_solver.discarded_weight`. The density matrices, `set_h_int` and
`overlap` are as described for `FockitImpuritySolver`. The operators of
`overlap` have at most two-body terms. With `"s"` and no `"sz"`, expectation
values average over the states of the spin multiplet, as the density
matrices do.

### Grand-canonical impurity problem

If you want thermal averages over every particle number of the impurity
problem, as `ThermalLatticeSolver` needs

```python
from risb.impurity_solvers import Block2ThermalImpuritySolver

impurity_solver = Block2ThermalImpuritySolver(
    h_int, gf_struct, beta=40, bond_dim=500, time_step=0.1
)
impurity_solver.set_h_imp(Lambda_c, D, h0_loc_matrix, mu=mu)
impurity_solver.solve()
```

The thermal state is a matrix product state of the orbitals and an ancilla
next to each, evolved in imaginary time from infinite
temperature[^Feiguin2005] in steps of at most `time_step`. `symmetries` is
`["sz"]` (the default), `["s"]`, or `[]` for the particle number only, and
every sector is in the average.
`spin_blocks` and `bath_size` are as for `Block2ImpuritySolver`.

If you want the thermal average over the lowest states instead, for low
temperatures

```python
impurity_solver = Block2ThermalImpuritySolver(
    h_int, gf_struct, beta=40, bond_dim=500, method="lowest", eta=8, delta=2
)
```

The lowest states of every sector within `delta` particles of half filling
are found together by DMRG, and the states with
$\beta (E - E_0) \le$ `eta` are in the average[^Giordano2026].
`solve()` warns if the sectors `delta` particles from half filling have
states in the average, or if the states it found outside the average have a
weight above $10^{-4}$.

`solve()` stores the thermal average of `h_imp` in `impurity_solver.energy`,
the grand potential in `impurity_solver.grand_potential`, and the entropy in
`impurity_solver.entropy`. `overlap` gives thermal averages.

### MPI ranks

The sectors can be solved in parallel on MPI ranks, with the optional
dependency [mpi4py](https://mpi4py.readthedocs.io) built against the MPI of
the machine, installed with

```bash
pip install --no-binary mpi4py "risb[mpi]"
```

or with [uv](https://docs.astral.sh/uv)
`uv pip install --no-binary mpi4py "risb[mpi]"`.

Pass the communicator to the solver

```python
from mpi4py import MPI

impurity_solver = Block2ThermalImpuritySolver(
    h_int, gf_struct, beta=40, method="lowest", comm=MPI.COMM_WORLD
)
```

and run the script on every rank, e.g., `mpirun -n 32 python script.py`. Every
rank must call `solve()` and `overlap`, and every rank gets the same results.
The ranks of `Block2ImpuritySolver` split the sectors of `symmetries`, and the
ranks of `Block2ThermalImpuritySolver` split the sectors of `"lowest"`.
`"purification"` runs on rank 0. Each rank uses `OMP_NUM_THREADS` threads.

[^Wolf2015]:
    [F. A. Wolf, A. Go, I. P. McCulloch, A. J. Millis, and U. Schollwöck,
    _Imaginary-Time Matrix Product State Impurity Solver for Dynamical
    Mean-Field Theory_,
    Phys. Rev. X **5**, 041032 (2015)](https://doi.org/10.1103/PhysRevX.5.041032)

[^Feiguin2005]:
    [A. E. Feiguin and S. R. White,
    _Finite-temperature density matrix renormalization using an enlarged
    Hilbert space_,
    Phys. Rev. B **72**, 220401(R) (2005)](https://doi.org/10.1103/PhysRevB.72.220401)

[^Giordano2026]:
    [L. Giordano, Y. Tan, Z.-H. Cui, and C. Sun,
    _Ab initio quantum embedding at finite temperature with density matrix
    embedding theory_,
    arXiv:2601.01641](https://arxiv.org/abs/2601.01641)

## `DummyImpuritySolver`

If you want to have some correlated subspaces $\mathcal{C}_i$ as inequivalent,
but they are related by some symmetry, it is not necessary to solve for the
ground state in each subspace separately.

This class is a copy of another
impurity solver. For example, to create a copy of
{py:class}`~risb.impurity_solvers.FockitImpuritySolver` it is instantiated as

```python
from risb.impurity_solvers import DummyImpuritySolver

dummy_solver = DummyImpuritySolver(impurity_solver=impurity_solver)
```

### Rotations

If the copy is related by symmetry, you can pass `rotations` as a list of
functions that operate in sequence on the dictionaries of block matrices that
{py:class}`~risb.impurity_solvers.DummyImpuritySolver` copies. Each function must return a dictionary with
the same blocks as `gf_struct`, so a rotation that mixes orbitals in different
blocks needs a `gf_struct` with those orbitals in the same block. For example, if the
correlated subspace is equivalent except that the spin is rotated,
like in an antiferromagnetic phase, this can be done with

```python
from copy import deepcopy


def rotate_spin(A):
    B = deepcopy(A)
    B["up"] = A["dn"]
    B["dn"] = A["up"]
    return B


dummy_solver = DummyImpuritySolver(impurity_solver=..., rotations=[rotate_spin])
```

:::{note}
`set_h_imp()` does nothing. `solve()` copies the density matrices stored in
the referenced impurity solver and rotates them according to the list of
functions in `rotations`. The referenced impurity solver must therefore be
solved first, i.e., come before the `DummyImpuritySolver` in the list of
impurity solvers passed to `LatticeSolver`. `solve()` also copies the
`entropy` of the referenced solver, which the rotations do not change.
:::

:::{warning}
`DummyImpuritySolver` has no `overlap` method. Rotationally invariant expectation
values can be obtained from the impurity solver that `DummyImpuritySolver` copies.
:::
