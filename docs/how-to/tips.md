# Tips and tricks

Practical advice for when things do not go as planned.

## If the solve does not converge

Start from a nearby solution, e.g., the one at the previous `U` or
temperature in a scan

```python
S.solve(tol=...)
R, Lambda = S.R, S.Lambda
...
S_next.R, S_next.Lambda = R, Lambda
S_next.solve(tol=...)
```

Try another history size for {{DIIS}}, or plain linear mixing

```python
from risb.optimize import DIIS, LinearMixing

S.solve(optimizer=DIIS(history_size=10))
S.solve(optimizer=LinearMixing(alpha=0.5))
```

Both lattice solvers and {{DIIS}} stop on the root functions by default.

For `LatticeSolver`, recursion can converge faster near a metal-insulator
transition. Without ghosts, it can also help in the atomic Mott limit
(`R = 0`), where the decoupled bath can leave nonzero density residuals

```python
S.solve(optimizer=DIIS(error_fun="recursion"))
```

Check `S.self_consistency_error` and the physical filling: unchanged parameters
can still be a non-solution. With ghosts, a Mott insulator can retain finite
hybridization, so being insulating alone does not justify ignoring residuals.

`ThermalLatticeSolver` restarts or anchors the solve when it stalls (see
{py:class}`~risb.optimize.OnPlateau`). If you want it to act sooner or later

```python
from risb.optimize import OnPlateau

S.solve(callbacks=[OnPlateau(patience=10, near=1e-5)])
```

The root finders of scipy usually only converge when started close to a
solution.

## Several solutions

Near a phase transition, or at a fixed chemical potential, there can be more
than one solution, and which one you find depends on where you start.
Continue from a nearby solution to stay on the same one. With
`ThermalLatticeSolver` the physical solution is the one with the lowest grand
potential (see [Finite temperature](lattice_solver.md#finite-temperature)).

## A temperature for the impurity problem of `LatticeSolver`

Decoupled bath orbitals can be completely full or empty in the embedding
ground state, while the quasiparticles they must match are smeared by the
temperature of
{py:class}`~risb.kweight.SmearingKWeight`. This can make the solve slow or
stop it from converging. Giving the impurity solver the same inverse
temperature can help

```python
impurity_solver = FockitImpuritySolver(h_int, gf_struct, beta=beta)
kweight = SmearingKWeight(beta=beta, n_target=n_target)
```

In the atomic limit (`R = 0`) the self-consistency equations are then still
not satisfied, but the observables do not depend on them (see
`self_consistency_tol` in `LatticeSolver.solve`).

## How the thermal impurity solve works

With `beta`, `FockitImpuritySolver` keeps the eigenstates with a Boltzmann
weight above $10^{-14}$ relative to the ground state, so at high temperature
it needs many states and is slower. A sector of at most 2000 states is
diagonalized as a dense matrix. In a larger one it uses `eigsh`, retries
twice with a larger `ncv` if `eigsh` does not converge, and looks for states
of a degenerate level that `eigsh` missed. `FockitThermalImpuritySolver`
goes through the particle numbers outwards from half filling, until a
particle number has a negligible weight. Symmetries make the sectors smaller
and the solve faster.

## Examples of impurity symmetries

- $S_z$ as a diagonal symmetry: `examples/hubbard.py`.
- Swapping the layers as a permutation symmetry: `examples/bilayer_hubbard.py`.
  The ground state is even under the swap.
- Swapping the spins as a permutation symmetry: `examples/kagome_hubbard.py`.
  The ground state is odd under the swap.
- A $C_3$ rotation as on-site factors in the basis of the trimer orbitals:
  `examples/decorated_honeycomb_trimer_basis.py`, and as a permutation of the
  sites: `examples/decorated_honeycomb_site_basis.py`.

## More examples

- Ghost {{RISB}} for the Hubbard model on the Bethe lattice, with and without
  ghosts: `examples/ghost_hubbard.py`.
- Ghost {{RISB}} at finite temperature, from low to high temperature:
  `examples/thermal_hubbard.py`.
