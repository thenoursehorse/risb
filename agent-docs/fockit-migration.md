# fockit migration

Make [fockit](https://github.com/thenoursehorse/fockit) (PyPI `fockit`) the default
impurity solver and TRIQS optional. One step at a time; stop after each step for the
user to review. Update this file as steps finish. Delete it when done.

## Steps

1. [x] Check fockit works for risb (see Findings).
2. [x] Implement a fockit impurity solver, modelled after `AtomDiagImpuritySolver`.
3. [x] Test it on the same things AtomDiag is tested on (`tests/test_atomdiag.py`,
   `tests/test_latticesolver.py`, `tests/test_dummy_impurity_solver.py`).
4. [x] Benchmark fockit vs AtomDiag (see Step 4 notes). Decided: **fockit is the
   default**. eigsh gets a larger Krylov space (`ncv=40`); **no automatic dense
   fallback** (impossible at large sizes; the user chooses, step 8).
5. [x] Swap all examples to fockit, with fockit's own syntax for building operators
   and solving (see Step 5 notes).
6. [x] Make TRIQS an optional dependency. Keep `AtomDiagImpuritySolver`; fockit is
   the default (see Step 6 notes).
7. [x] Swap docs (tutorials, how-to, explanations, README) to fockit (see Step 7
   notes).
8. [ ] Solver options and degenerate ground states. The user picks eigh or eigsh
   and can pass settings to eigsh (or other solvers); never an automatic
   fallback. Also degenerate ground states (see Findings and Open questions).
   Until then `solve` uses eigsh with `k=1`, `ncv=40` and a fixed random start
   vector, which, like AtomDiag, picks an arbitrary state when the ground state
   is degenerate.
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
- Step 3: committed (`e811380`).
- Step 4: committed (`ccdb042`, `5de4190`, `ff2bb5e`).
- Step 5: (a) `examples/kagome_hubbard.py`; (b) `src/risb/helpers_fockit.py`,
  `examples/decorated_honeycomb.py`; (c) `examples/hubbard.py`,
  `examples/bilayer_hubbard.py`, `examples/green_functions.py`,
  `tests/test_examples.py`, this file. Committed (`2f1c55c`, `56219dd`,
  `0def080`), plus the `green_functions.py` TRIQS argument fix.
- Step 6: committed.
- Step 7: (a) `README.md`, `docs/about.md`; (b) `docs/how-to/impurity_solvers.md`,
  `docs/how-to/lattice_solver.md`, `docs/how-to/quadratic_terms.md`;
  (c) `docs/tutorials/hubbard.md`, `docs/tutorials/self-consistent.md`;
  (d) `docs/explanations/impurity.md`; (e) roles only: `docs/how-to/diis.md`,
  `docs/how-to/kweight.md`, `docs/tutorials/kagome.md`; this file.

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

## Step 4 notes

Script: `benchmarks/impurity_solvers.py` (`uv run python benchmarks/impurity_solvers.py`;
not in the wheel or sdist; in mypy `files`). Logs with `logging`.

Impurity solve (`set_h_imp` + `solve`, including density matrices), seconds. Run
with U=4, J=0.8 (the script now uses U=3, J=0.6; timings hardly depend on U).
Generic random Lambda_c, D, h0 (seed 0). |dE| between solvers <= 2.5e-14.

| model | orbitals | dim | fockit | AtomDiag | AtomDiag largest block |
| --- | --- | --- | --- | --- | --- |
| hubbard / kanamori / complex | 1 | 6 | 0.0007 | 0.0001 | 4 |
| hubbard / kanamori / complex | 2 | 70 | 0.003-0.004 | 0.001-0.002 | 36 |
| hubbard | 3 | 924 | 0.012 | 0.27 | 400 |
| kanamori / complex | 3 | 924 | 0.018 / 0.020 | 0.30 / 0.72 | 400 |
| hubbard | 4 | 12870 | 0.26 | 477 | 4900 |
| kanamori / complex | 4 | 12870 | 0.39 / 0.41 | 593 / 1742 | 4900 |
| hubbard | 5 | 184756 | 7.5 | not run | 63504 (~32 GB dense) |
| kanamori / complex | 5 | 184756 | 13.2 / 12.6 | not run | 63504 |

fockit breakdown at 5 orbitals (kanamori real): building `h_imp` 0.001 s, basis
(once) 0.02 s, `to_matrix` 5.4 s, eigsh 5.4 s, density matrices 2.5 s. At 4
orbitals: matrix 0.21 s, eigsh 0.13 s, density matrices 0.05 s. Operator algebra
in Python is not a bottleneck.

LatticeSolver, cubic lattice, DIIS, `tol=1e-8`, paramagnetic, U=3 J=0.6 (Hubbard
U=3):

| model | fockit | AtomDiag | iterations (f / AD) | Z (both) |
| --- | --- | --- | --- | --- |
| hubbard 1 orbital | 0.02 s | 0.01 s | 16 / 16 | 0.68698322 |
| bilayer 2 | 0.10 s | 0.07 s | 17 / 17 | 0.69566633 |
| kanamori 2 | 0.09 s | 0.05 s | 17 / 17 | 0.57494032 |
| kanamori 3 | 0.91 s (`ncv=40`) | 2.84 s | 42 / 39 | 0 (Mott) |

- **Robustness problem:** kanamori 3 with the current solver raises
  `ArpackNoConvergence` (9241 iterations) at solve 37: a DIIS iterate near the
  Mott point (D ~ 0) has an 8-fold degenerate ground state plus more levels
  within 5e-4. On that matrix: default eigsh (`ncv=20`) fails in 3.4 s;
  `ncv=40` works (0.005 s); `ncv=60` works; k=2 works; k=10 fails;
  `maxiter=1e6` works in 79 s; dense `eigh` 0.34 s. AtomDiag is unaffected
  (dense). U=4 J=0.8 is Mott on this lattice too; neither solver converged in 999
  DIIS iterations there (degenerate impurity, step 8).
- Warm start (scratchpad, `ncv=40`): no real gain (kanamori 3: 0.78 s vs 0.91 s;
  others identical). Not worth adding.
- Done: `ncv=min(dim, 40)` in `solve`. No measurable cost at 3-5 orbitals (dim
  184756: 5.0 s vs 5.4 s default). New test
  `test_three_orbital_kanamori_is_mott_insulator` (cubic, U=3, J=0.6, Z -> 0,
  both solvers) failed with `ArpackNoConvergence` before the fix. Simple
  degenerate inputs (D=0, 135-fold) did not break default eigsh; it needs the
  near-degenerate cluster.
- Bug from step 3 (in `e811380`): the lazy import hid `AtomDiagImpuritySolver`
  from autodoc, so the API page lost it and the strict docs build failed
  (`LatticeSolver` docstring link). Fixed with a module `__dir__` returning
  `__all__`. My earlier docs checks piped through `tail`, which hid Sphinx's exit
  code.

## Step 5 notes

Decisions:
- D1: `hubbard.py` and `bilayer_hubbard.py` become TRIQS-free (numpy cubic h0_k,
  fockit impurity). The Green's-function construction moves to a new TRIQS-only
  `examples/green_functions.py` (based on the bilayer); its test skips without
  TRIQS.
- D2: port `get_h0_loc` to `risb.helpers_fockit` (with a matrix-to-operator
  helper), and keep the decorated honeycomb's structure.
- D3: examples switch from `print` to `logging` (results at INFO, plain format;
  `logging.getLogger("risb")` set to WARNING to hide per-iteration DIIS logs).
- One example per commit; `tests/test_examples.py` reference values must not
  change.

Done:
- `kagome_hubbard.py`, `decorated_honeycomb.py`, `hubbard.py`,
  `bilayer_hubbard.py` use fockit operators, `FockitImpuritySolver`,
  `risb.helpers_fockit`, and `logging`; none needs TRIQS. New
  `examples/green_functions.py` (bilayer + TRIQS `TBLattice`/Green's functions,
  fockit impurity); its test does `importorskip("triqs")`.
- `risb.helpers_fockit`: new `matrix_to_Op(A)` and `get_h0_loc(h0_k, P,
  gf_struct_mapping, force_real)` (fockit versions of the `helpers_triqs` ones,
  without the `gf_struct` argument, which fockit does not need).
- Hubbard/bilayer h0_k is now numpy on a Gamma-centred mesh (same k-points as
  TRIQS `MeshBrZone`).
- Reference values unchanged. Decorated honeycomb differs from its references by
  2e-7 to 1e-6 (the example solves to `tol=1e-4`; the test checks 1e-3).
- `test_hubbard_example`: the monotonic-Z and 0 <= Z <= 1 checks now allow 1e-12.
  fockit gives Z ~ 1e-33 noise in the Mott region and Z(U=0) = 1 + 4e-16; all Z
  still match their references to 1e-5.
- Decorated honeycomb: removed the unused `NOp = N_op(...)` line and the
  commented-out N/S2 code (its modes `("up", o)` never matched the example's
  blocks `"up_A"`, ...). `get_hubb_trimer` no longer calls `.real` (fockit has no
  equivalent; with `tk=0` all coefficients are real).
- Without TRIQS: `test_examples.py` gives 4 passed, 1 skipped, but needs
  `matplotlib` (hubbard plots), which is not in the `test` dependency group (CI
  gets it from conda). Step 6.
- Step 7: `docs/tutorials/self-consistent.md` points to
  `examples/bilayer_hubbard.py` for Green's functions; now
  `examples/green_functions.py`.

## Step 6 notes

Without TRIQS (before step 6): pytest failed only in `test_docs.py` (imports
`risb.helpers_triqs`) and `test_examples.py` (needs matplotlib); mypy needed
matplotlib; the strict docs build failed (autodoc imports `helpers_triqs` and
`atom_diag`).

Decisions:
- matplotlib goes in the `test` dependency group; `test_docs.py` skips modules
  that need the missing TRIQS.
- D1: CI has two jobs: "Test (no TRIQS)" (uv only: lint, mypy, pytest) and
  "Test (TRIQS)" (conda, as before: pytest and the strict docs build).
- D2: docs keep building with TRIQS installed (CI and CD); `docs/conf.py`
  unchanged.
- D3: no custom error message; without TRIQS users get Python's
  `No module named 'triqs'`.

Done:
- `pyproject.toml`: `matplotlib>=3.11.2` in the `test` group (`uv add`; it also
  swapped a few conda packages in the local `.venv`, e.g. six, for PyPI builds;
  TRIQS 4.0.2 still works there).
- `tests/test_docs.py`: skips modules and `__all__` names whose import fails
  because `triqs` is missing.
- `.github/workflows/ci.yml`: jobs `test` "Test (no TRIQS)" (`uv sync --locked`;
  lint, mypy, pytest) and `test-triqs` "Test (TRIQS)" (conda; pytest, strict docs
  build). `cd.yml` unchanged (calls `ci.yml`; builds docs with TRIQS). actionlint
  clean.
- Checked: fresh `uv sync --locked` env without TRIQS: lint, mypy, pytest
  (42 passed, 15 skipped). TRIQS env: pytest 57 passed, strict docs build exit 0.
- Step 7: README "Dependencies"/"Tests" sections still say the tests need TRIQS.

## Step 7 notes

Decisions (user): the docs use fockit only. TRIQS and `AtomDiagImpuritySolver` are
mentioned once, in `docs/how-to/impurity_solvers.md`, saying it exists. "No one
should be using the AtomDiag solver." No doubling up of fockit and TRIQS
anywhere else (tutorials, how-tos, explanations, about). The hubbard tutorial's
density-density example is written out by hand (no new helper). The docs do
not mention the benchmarks.

Done:
- `how-to/impurity_solvers.md`: rewritten for `FockitImpuritySolver` (what
  `solve` does and raises, `overlap` with `helpers_fockit`, bath modes); a short
  `AtomDiagImpuritySolver` note (needs TRIQS, slower, use fockit, cite TRIQS).
- Tutorials `hubbard.md`, `self-consistent.md`: fockit operators and solver.
  The hand-written density-density interaction equals TRIQS `h_int_density`
  exactly (checked with `to_triqs`, random U, U', 3 orbitals); with Uprime=5 it
  gives the same Z and S2 as `5 * nc("up", 0) * nc("dn", 0)`.
- `how-to/lattice_solver.md`: fockit tip; stop scipy's quasi-Newton methods on
  the residual (`options={"fatol": ...}`).
- `how-to/quadratic_terms.md`: removed "Using TRIQS lattice tools"; "As a fockit
  operator" with `helpers_fockit.matrix_to_Op`.
- `explanations/impurity.md`, `about.md`, `README.md`: fockit; no TRIQS. README
  tests and docs use plain `uv sync` (docs build without TRIQS warns that it
  cannot import `risb.helpers_triqs`; CI builds with TRIQS).
- Found: MyST does not parse `:py:class:`-style roles (they rendered as text, or
  as a broken field list at the start of a line). All 19 converted to
  `{py:class}` etc. with full targets.
- Checked: code blocks of the hubbard tutorial run end to end (it never imports
  `LatticeSolver`/`SmearingKWeight`, pre-existing); the how-to, self-consistent
  and quadratic-terms snippets run; strict docs build exit 0; pytest 57 passed;
  pre-commit clean.
- Open: `how-to/lattice_solver.md` still credits TRIQS/hartree_fock for the
  symmetrizer idea.

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
- `ArpackNoConvergence` with k=1 has no result to return, so "warn and continue"
  is not possible (stale or NaN density matrices would corrupt the iteration).
  Proposed: keep raising, with a message saying what the user can change; warn
  only where a partial result exists (k > 1, step 8). Not yet agreed.
- Resolved (step 3): the `N`/`S2`/Kanamori helpers live in `risb.helpers_fockit`.
- Step 5/6: should `AtomDiagImpuritySolver` accept fockit operators (fockit has
  `to_triqs`) so there is one operator syntax?
