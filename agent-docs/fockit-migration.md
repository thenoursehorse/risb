# fockit migration

Make [fockit](https://github.com/thenoursehorse/fockit) (PyPI `fockit`) the default
impurity solver and TRIQS optional. One step at a time; stop after each step for the
user to review. Update this file as steps finish. Delete it when done.

## Steps

1. [x] Check fockit works for risb (see Findings).
2. [x] Implement a fockit impurity solver, modelled after `AtomDiagImpuritySolver`.
3. [x] Test it on the same things AtomDiag is tested on (`tests/test_atomdiag.py`,
   `tests/test_latticesolver.py`, `tests/test_dummy_impurity_solver.py`).
4. [ ] Benchmark fockit vs AtomDiag. **Stop**: user decides the defaults.
5. [ ] Swap all examples to fockit, with fockit's own syntax for building operators
   and solving.
6. [ ] Make TRIQS an optional dependency. Keep `AtomDiagImpuritySolver`; fockit is
   the default.
7. [ ] Swap docs (tutorials, how-to, explanations, README) to fockit.
8. [ ] Handle degenerate ground states (see Findings and Open questions). Until
   then `solve` uses eigsh with `k=1` and a fixed random start vector, which,
   like AtomDiag, picks an arbitrary state when the ground state is degenerate.
9. [ ] Explore pinning RISB's gauge redundancy. At the bilayer solution the root
   function's Jacobian has 3 zero singular values out of 24 (condition number
   1e16, null directions in the R parameters), so roots are not isolated and
   step-size stopping rules fail (see Step 3 notes). Ideas: fix the gauge (e.g.
   a parametrization of R or the f-orbital basis that removes the redundant
   directions), or a root finder that projects out the null space. Check first
   that the null directions really are the gauge.

## Commits

One commit per step (or smaller), not one at the end. At the end of each step,
list its files and suggest a commit message; the user commits.

- Step 1: committed (`9f62a29`).
- Step 2: committed (`aa53926`).
- Step 3: `src/risb/helpers_fockit.py`, `docs/api/risb.helpers_fockit.rst`,
  `docs/index.md`, `src/risb/impurity_solvers/fockit.py`,
  `src/risb/impurity_solvers/__init__.py`, `tests/test_common.py`,
  `tests/test_atomdiag.py` -> `tests/test_impurity_solvers.py`,
  `tests/test_latticesolver.py`, `tests/test_dummy_impurity_solver.py`, this file.

## Step 2 notes

- `FockitImpuritySolver` in `src/risb/impurity_solvers/fockit.py`. Same public API
  as `AtomDiagImpuritySolver` (`set_h0_loc/h_int/h_bath/h_hybr/h_imp(mu=)`, `solve`,
  `get_rho_*`, `overlap`, `gs_energy`, `gf_struct_bath`). `basis` (built once in
  `__init__`) replaces `ad`. Operators are fockit `OpSum`s; bath modes are
  `("bath_" + block, i)`.
- Checked by hand against AtomDiag on the step-1 models: max diff about 1e-14.
- `docs/conf.py`: fockit types in `nitpick_ignore` (fockit has no hosted docs or
  intersphinx inventory).

## Step 3 notes

- Public `src/risb/helpers_fockit.py`: `N_op`, `S2_op`, `h_int_kanamori`. Each
  docstring says "Adapted from TRIQS's ...". No TRIQS code or text is copied, so
  there is no TRIQS copyright header. Other helpers in this file need not be
  from TRIQS. Kanamori matches TRIQS (lattice Kanamori test passes to 1e-10).
- `solve` checks `h_imp` is Hermitian (`ValueError` otherwise; tested). It calls
  scipy's `eigsh` directly with a fixed random start vector, and
  dense `eigh` when the sector dimension is < 3. fockit's `groundstate` is a thin
  wrapper and not used.
- `tests/test_atomdiag.py` -> `tests/test_impurity_solvers.py`. Models are written
  once in fockit syntax; AtomDiag gets them through `fockit.algebra.to_triqs`. The
  impurity, lattice and dummy tests are parametrized over `["fockit", "atom_diag"]`
  (`tests/test_common.py`: `make_impurity_solver`, `overlap`). The AtomDiag cases
  `importorskip("triqs")`.
- Pulled forward from step 6: `risb/impurity_solvers/__init__.py` imports
  `AtomDiagImpuritySolver` lazily (module `__getattr__`). In a venv without TRIQS
  these three test files give 13 passed and 13 skipped. `test_examples.py` and
  `test_docs.py` still need TRIQS (steps 5/6).
- **Problem: eigsh breaks scipy's `root(method="krylov")`.**
  `test_scipy_root[fockit-bilayer-krylov]` took anywhere from 0.3 s to 172 s.
  Krylov uses finite-difference Jacobians; eigsh's answer is accurate (about 5e-15
  from dense) but not smooth in the parameters. Bilayer, 3 runs each:

  | ground state | time | impurity solves | converged |
  | --- | --- | --- | --- |
  | eigsh, random `v0` | 4 / 0.3 / 172 s | 798 / 54 / 32833 | yes |
  | eigsh, fixed `v0` | about 170 s | 33412 | no |
  | dense `eigh` | 0.5 s | 85 | yes |
  | eigsh, `v0` = previous ground state | 0.3-0.5 s | 66-112 | yes |

  Cause found: the problem is the test's stopping rule, not eigsh. The RISB
  Jacobian at the solution has 3 zero singular values out of 24 (condition number
  1e16, null directions in the R parameters; most likely the gauge freedom). So the
  roots are not isolated. `root(method="krylov", tol=1e-10)` stops on the step
  size (`xtol`), and along the null directions any noise makes the steps
  arbitrary. Untested guess: dense and AtomDiag get away with it because their
  rounding noise is the same on every call. With a residual stop (`options={"fatol": 1e-10}`), Krylov took
  0.2 s and 33-36 calls for every eigsh start (random, or warm with weight 0 to
  1e-1). `hybr` also stops on step size and failed once in 5.
  Done: `test_scipy_root` uses `options={"fatol": 1e-10}` (0.7 s, 5 runs out of 5).
- Warm start (tried, removed): a pure warm start (`v0` = previous ground state)
  gets stuck in the wrong spin sector at a level crossing (dh_trimer + Zeeman
  field, energy off by 0.026); mixing in >= 1e-8 of a fixed random vector fixes
  that. Its only possible benefit now is speed (fewer Lanczos iterations): measure
  in step 4 and add back if it pays.

## Findings (step 1)

fockit 0.4.0: pure Python, numpy + scipy only, Python >= 3.14, MIT (fine with GPL-3).

API mapping from `AtomDiagImpuritySolver`:

- Operators: `fockit.algebra.c/cdag/nc(bl, i)` give mode `(bl, i)`, the same labels
  as the AtomDiag `fops`. `hc(op)` is `dagger`.
- `AtomDiag(h, fops, n_min=M, n_max=M)` ->
  `build_basis({FermionSpace(): modes}, diagonal_symmetries={N_total: M})`. The
  basis only depends on `gf_struct`, so build it once in `__init__`.
- `act`/`overlap` -> `expectation(to_matrix(op, basis), psi)`.
- `to_matrix` returns a real matrix unless an element is complex, so no
  `h_imp.real` workaround is needed.
- Ground state: `fockit.solve.groundstate` (`eigsh`, dense `eigh` if dim tiny) or
  `eigh_full`.

Checked with a scratch script (fockit from source, the repo `.venv` for TRIQS): the
`test_atomdiag.py` models (one_band, bilayer, dh_trimer) and a complex trimer
(random complex `h0_loc` and `D`, like decorated_honeycomb).

| model | sector dim | max diff vs AtomDiag | fockit eigsh | fockit eigh | AtomDiag |
| --- | --- | --- | --- | --- | --- |
| one_band | 6 | 3e-16 | 0.9 ms | 0.8 ms | 0.1 ms |
| bilayer | 70 | 5e-15 | 3.3 ms | 3.6 ms | 1.4 ms |
| dh_trimer | 924 | 1e-14 | 15 ms | 196 ms | 153 ms |
| complex trimer | 924 | 7e-15 | 19 ms | 249 ms | 1042 ms |

Times are one `set_h_imp` + `solve` + all density matrices (basis built once, ~0.5
ms). Building the fockit Hamiltonian as an `OpSum` is < 1 ms. All fixture values
match to < 1e-14 (tests use 1e-12).

Problems:

- **Degenerate ground states.** `eigsh` starts from a random vector. With `D = 0` in
  one_band (4-fold degenerate) `rho_c['up']` came out 0.35, 0.37, 0.97, 0.84 on four
  runs. AtomDiag is deterministic but arbitrary (always 1.0 up, 0.0 down). Non-
  degenerate cases agree run to run to 1e-14.
- eigsh cannot reliably find a whole degenerate manifold (Krylov space from one
  start vector). With `D = 0`: dh_trimer is 18-fold degenerate and eigsh `k=18`
  found 14-17 copies, `k=36` found all 18. The complex trimer is 8-fold degenerate
  and `k=16` found 6-8. So "eigsh + average the ground states it finds" is not
  robust on its own.
- Physics: the main real case is the RISB Mott phase (paramagnetic `R = 0`, so
  `D = 0`): the impurity decouples and its atomic multiplet is degenerate. The
  symmetric answer is the equal-weight mixture over the multiplet (`T -> 0` of the
  thermal state). To get a magnetic or orbitally ordered answer, break the symmetry
  on purpose (a field in `h0_loc`).
- fockit has no `N_op`, `S2_op`, `h_int_kanamori`, `set_operator_structure`; risb's
  `helpers_triqs.get_h0_loc` returns TRIQS operators.
- `examples/hubbard.py` and `examples/bilayer_hubbard.py` use TRIQS Green's
  functions and `TBLattice`; fockit has nothing like them.
- `risb/impurity_solvers/__init__.py` imports `atom_diag` eagerly, so
  `import risb.impurity_solvers` needs TRIQS. TRIQS is not in `pyproject` deps; CI
  installs it with `environment.yml`.
- fockit is young; pin a lower bound.

## Open questions

- Step 8: degenerate ground states. Idea (not yet agreed): `solve(n_states=1,
  dense=False, degeneracy_tol=1e-10)`. Default eigsh with 2 states, logging a
  warning if `E1 - E0 < tol`. `n_states=k` asks eigsh for k states; `dense=True`
  uses `eigh` on the whole sector. In both cases, average the density matrices
  over the states within tol of E0 (equal-weight mixture).
- Step 7: docs should say to stop scipy root finders on the residual (`fatol`),
  not the step size, because RISB roots are not isolated.
- Step 8: eigsh can fail by not converging (small gap, `ArpackNoConvergence`), and
  silently gives garbage if `h_imp` is not Hermitian. Catch and fall back to dense?
- Resolved (step 3): the `N`/`S2`/Kanamori helpers live in `risb.helpers_fockit`.
- Step 5: hubbard and bilayer examples need TRIQS for Green's functions. Keep TRIQS
  for that part only, or drop it?
- Step 5/6: should `AtomDiagImpuritySolver` accept fockit operators (fockit has
  `to_triqs`) so there is one operator syntax?
