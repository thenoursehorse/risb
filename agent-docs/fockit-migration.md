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
8. [x] Solver options (see Step 8 notes): `method` ("eigsh" or "eigh"),
   `method_options` (passed to that method), `diagonal_symmetries`,
   `permutation_symmetries`, `onsite_symmetries` (passed to fockit's
   `build_basis`; the solver adds half filling; diagonal symmetries checked
   against `h_imp`; density matrices via `expand_state` into the
   diagonal-restricted basis), `degenerate` ("ignore" default, "warn") and
   `degenerate_tol`. Never an automatic fallback. "average" is item 13.
9. [x] Explore pinning RISB's gauge redundancy (see Step 9 notes; user: no
   code, docs wording in the lattice_solver how-to). At the bilayer solution the root
   function's Jacobian has 3 zero singular values out of 24 (condition number
   1e16, null directions in the R parameters), so roots are not isolated and
   step-size stopping rules fail (see Step 3 notes). Ideas: fix the gauge (e.g.
   a parametrization of R or the f-orbital basis that removes the redundant
   directions), or a root finder that projects out the null space. Check first
   that the null directions really are the gauge.
10. [x] (done, see Item 10 notes; docs sentence fixed) Investigate the claim in `docs/how-to/green_functions.md`: "The
   interacting Green's function from RISB only has the quasiparticle part, so its
   filling is reduced by Z." It implies RISB's G(k, w) = R^dag G_qp R is missing
   a (local/incoherent) part. Checked so far (bilayer example only): G filling =
   n Z to 1e-8 and G_qp filling = n. Find out what the correct statement is (and
   whether the filling of G should be n), then fix the docs sentence.
11. [x] (done, see Item 11 notes: fockit `_conjugate` keeps float zeros) Investigate why fockit's `build_basis` is slow (> 10 min, stopped) for the
   decorated honeycomb impurity (924-state N sector) with C3 as an onsite
   symmetry, factors `c c^dag + w^q c^dag c` (q = 0, 1, 2 for A, E1, E2). Is it
   fockit, or the setup? See Step 8 notes.
12. [x] (done, see Item 12 notes) Add the impurity symmetries to the examples to show users how to use
   them: Sz and spin swap (hubbard, kagome), plus layer swap (bilayer), plus C3
   (decorated honeycomb, once item 11 is resolved).
   Decisions (user): hard-code sectors after verifying they hold the ground
   state at every iteration; comments say the solver finds the lowest state in
   the chosen sector (a sector can also target a higher state, not only speed);
   one kind each: hubbard Sz (diagonal), bilayer layer swap (permutation),
   kagome spin swap (permutation), decorated honeycomb C3 (onsite).
13. [x] (deferred into item 15 by the user, see Item 13 notes) `degenerate="average"`: the equal-weight mixture over the degenerate
   level (eigsh doubling + `scipy.linalg.orth`; tested exact, see Step 8
   notes). First check the finite-temperature RISB/gGA literature on whether a
   mixed embedding state is the right prescription, and find a case that
   `symmetrizers` cannot handle.
14. [x] Make risb's RISB cycle gauge covariant (see Step 14 notes). Done: fixes 1-3
   (hybridization index order, Lambda_c / lambda from the gauge-invariant
   term with the exact matrix derivative, f2), Z = R^dag R, get_sigma_w with
   (R^dag R)^-1, covariance tests. Before, for complex non-commuting models the
   result depended on the starting gauge and Sigma was wrong when R is not
   normal; all current examples were unaffected (R normal). Independent
   check: stationarity of `helpers.get_lagrangian` at risb's solutions (test
   added).
15. [x] (done: thermal impurity `beta`, see Item 15 notes; includes `degenerate="average"` from item 13: the only exact
   impurity degeneracy found is the Mott state, D = 0) Robustness at rho_qp eigenvalues 0 or 1 (insulating regime). Every K^(-1/2)
   formula (get_d, get_r, get_lambda_c, get_lambda, get_f1) is singular when a
   quasiparticle band is completely full or empty; iterates reach that in the
   Mott/transition regime of the complex 2-orbital model (U >= 4, J = 0.2 U):
   the original code crashed at U = 5, 6 and reached Z = 0 at U = 7, 8 only via
   a blow-up; after the covariance fixes it crashes at all U >= 4; with linear
   mixing both crash (see Step 14 notes). Derive the R -> 0 / rho-edge limit,
   treat it consistently in all those helpers, re-run the U scan (DIIS and
   linear mixing), add a complex Mott test for both solvers. Step 9 may explain
   the large DIIS steps (lambda eigenvalues -51, 194) that reach these states.
16. [ ] `helpers.get_lagrangian` for several clusters. It handles one cluster equal
   to the unit cell only, but `LatticeSolver` allows several clusters with
   `projectors` and `gf_struct_mapping`; the helper should take the same
   arguments (upfolded R(k), lambda(k) in the quasiparticle term; embedding
   and Delta_p terms summed over clusters). Extend the stationarity test to a
   multi-cluster model.
17. [ ] Decorated honeycomb in the site basis. Rename
   `examples/decorated_honeycomb.py` to `decorated_honeycomb_trimer_basis.py`
   (update tests, docs, references) and add
   `examples/decorated_honeycomb_site_basis.py`: the same model without the
   trimer transformation (3x3 spin blocks), with C3 as a permutation symmetry
   (cyclic relabelling of the sites, physical and bath, eigenvalue w^k; fockit
   accepts it: site-basis trimer impurity, sectors 312/306/306, w^0 holds the
   ground state). Checks to go with it: a symmetrizer that averages R, Lambda
   over the rotation (the site basis has a U(3) bath gauge per block, so h_imp
   is only C3 symmetric if D, Lambda_c commute with the permutation; the
   trimer basis has 1x1 blocks so it is automatic); the k-mesh respects C3;
   the ground state is in the chosen sector at every iteration; both examples
   give the same physics (Z eigenvalues, densities, energies); add to
   `test_examples.py`.

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
- Step 8: `src/risb/impurity_solvers/fockit.py`,
  `tests/test_impurity_solvers.py`, `docs/how-to/impurity_solvers.md`, this file.
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
  fockit impurity). The Green's function construction moves to a new TRIQS-only
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
- `explanations/impurity.md`, `about.md`, `README.md`: fockit. README lists TRIQS
  4.x as optional for Green's functions (`risb.helpers_triqs`; user request). README
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

Follow-ups (user):
- `about.md` lists both solvers again (AtomDiag with the TRIQS reference).
- `explanations/impurity.md`: "we use the same `gf_struct` convention as TRIQS"
  (checked: TRIQS 4.0.2 `BlockGf` takes risb's `gf_struct` as is;
  `set_operator_structure` gives the same pairs).
- `how-to/quadratic_terms.md`: a tip that TRIQS can construct tight-binding
  models, linking only to TRIQS (no walkthrough, no API links).
- New `how-to/green_functions.md` ("Constructing Green's functions",
  `risb.helpers_triqs`, says TRIQS must be installed). Formulas checked against
  the code; its code run on the `examples/green_functions.py` setup reproduces
  the example (G_qp filling n, G filling n Z, both G constructions agree).
- Write "Green's function(s)", never "green functions".

## Step 8 notes

User's idea: compute 2 states; if E1 - E0 < tol, warn. Then find ways to resolve
the degeneracy. Solver options must live on the instance (constructor), because
`LatticeSolver` calls `solve()` with no arguments.

Detection (eigsh with k=2, fixed and 29 random v0, tol 1e-8): every degenerate
case was flagged 30/30 with E0 right (hubbard1 D=0 4-fold, kanamori3 generic
3-fold, kanamori3 D=0 3-fold, complex kanamori3 D=0 3-fold, step-4 matrix 12
within 1e-8); non-degenerate controls never flagged; no convergence failures.
Note: a generic 3-orbital Kanamori impurity with D != 0 has a 3-fold (triplet)
ground state, so degeneracy is not only a D = 0 (Mott) issue.

Cost of k=2 vs k=1 (hubbard, random couplings, `ncv=40`):

| orbitals | whole solve (k=1) | eigsh k=1 | eigsh k=2 | extra per solve |
| --- | --- | --- | --- | --- |
| 3 | 12 ms | 3 ms | 4 ms | ~+10% |
| 4 | 0.24 s | 0.074 s | 0.15 s | ~+30% |
| 5 | 7.1 s | 2.6 s | 5.1 s (4.2 s with ncv=60) | ~+25-35% |

Dense eigh on the sector: 0.17 s (3 orbitals), 173 s (4 orbitals).

Resolving (equal-weight mixture over the degenerate level vs exact from dense
eigh, 5 random starts each, 3 orbitals unless stated):
- eigsh with doubling k until the top returned eigenvalue is above E0 + tol:
  finds the right count; exact for real H; for complex H the returned
  degenerate eigenvectors are not orthogonal (Gram error up to 5e-2, ARPACK's
  complex path), so rho was off by 5e-3. With `scipy.linalg.orth` on the
  returned vectors: exact (2e-14).
- LOBPCG with doubling block size: right count, but ~10x slower than eigsh and
  rho only good to 1e-8 to 1e-11 (convergence warnings). Not worth it.
- Splitting by total Sz (fockit `diagonal_symmetries`), ground states of every
  sector at E0: exact (<3e-15) in every case, ~0.03-0.04 s. Only valid when
  `h_imp` conserves Sz, and the solver must be told which blocks are up/down.
- Dense eigh: exact, impractical from 4 orbitals.

Real case: the 3-orbital Kanamori Mott lattice run (U=3, J=0.6) had 9 impurity
solves flagged by k=2, each an 80-fold exactly degenerate level (gap 1.6 to the
next). eigsh doubling + orth: all 80, exact (1e-15), 1.2-2.8 s. Sz sectors: all
80 in 7 sectors (1, 10, 19, 20, 19, 10, 1), exact, 0.23 s.
Hard case: the step-4 matrix (a near-degenerate cluster, 72 levels within 1e-8)
makes eigsh doubling raise `ArpackNoConvergence` at k=32 (28/32 converged).
4 orbitals, generic Kanamori (dim 12870): 3-fold; eigsh doubling + orth 0.47 s,
Sz sectors 0.35 s, dense 168 s; both exact (1.6e-15).
Note for tests: risb warnings are errors under pytest (`error:::risb`), so a
degeneracy warning would fail `test_three_orbital_kanamori_is_mott_insulator`.

Decisions (user): default "ignore" (k=1, no check); "average" uses eigsh doubling
+ orthonormalization; options are `method` ("eigsh" or "eigh") and
`method_options` (not `eigsh_options`). Also pass fockit symmetry constraints
(diagonal, permutation, onsite) to the impurity basis.

Symmetries (bilayer, spin-symmetric couplings; one band for particle-hole):
- Diagonal (Sz = 0): dim 70 -> 36, same E0, density matrices built in that
  basis exact (8e-15).
- Permutation (spin flip incl. bath): fockit refuses to build the single
  density-matrix operators in the symmetric basis (`ValueError`, not
  symmetric); `expand_state` to the plain basis then measuring is exact (1e-15)
  in the sector holding the ground state. The other sector is a higher state.
- Onsite (particle-hole, physical c + cdag, bath c - cdag, one band with
  h0_loc = -U/2, Lambda_c = 0): exact via `expand_state` in the right sector;
  fockit raises `ValueError` if `h_imp` is not symmetric (wrong signs).

Correction: only density-matrix elements that a symmetry moves to a different
operator (rho_up <-> rho_dn under spin swap; c^dag c <-> c c^dag under
particle-hole) cannot be built in the symmetric basis; invariant operators can.
Since the ground state of a sector is an eigenstate of every symmetry operation
g (|character| = 1), <O> = <(1/|G|) sum_g g^dag O g>, and that symmetrized
operator can be built in the symmetric basis. Checked: spin swap (+1 sector)
1.6e-15; particle-hole (-1 sector, O -> P O P^dag, P the product of the
factors) 2.8e-16. So two exact routes: expand the state into the plain basis
(generic, little code, measurement in the full-size basis), or symmetrize the
operators (stays in the small basis, but risb must build the whole group's
action on operators).

Diagonal symmetries are not checked by fockit: restricting an `h_imp` that does
not conserve Sz to Sz = 0 silently gave E0 -2.2806 instead of -2.4573. The
solver should check `[h_imp, Q] = 0` for each diagonal symmetry and raise.

Symmetries of the converged `h_imp` in the examples (fockit's own checks for
permutation/onsite; commutator for Sz):
- hubbard, kagome: Sz, spin swap.
- bilayer: Sz, spin swap, layer swap (0 <-> 1).
- decorated honeycomb: Sz, spin swap, and C3 (sum of q n with q = 0, 1, 2 for
  A, E1, E2 is conserved mod 3: every term changes it by -3, 0 or 3). C3 is
  not a fockit diagonal symmetry (not additive). As an onsite symmetry with
  factors `c c^dag + w^q c^dag c`, `build_basis` ran > 10 min on the 924-state
  sector without finishing (stopped; not investigated: fockit performance or
  my setup?). Factors written as `1 + (w^q - 1) n` are rejected by fockit.
- No example had particle-hole symmetry (all sign patterns tried).

Decisions (user): three explicit constructor arguments mirroring fockit's names
(`diagonal_symmetries`, `permutation_symmetries`, `onsite_symmetries`);
degeneracy option `degenerate="ignore" | "warn" | "average"` with
`degenerate_tol=1e-8`.

Questions answered (user asked):
- Permutation/onsite symmetries: expand the state into the basis with all
  diagonal symmetries (N and e.g. Sz), i.e. the same `build_basis` call without
  permutation/onsite symmetries (what `expand_state` expects); not just the N
  sector.
- Does degeneracy matter? Expectation values differ between degenerate states
  for operators the symmetry does not leave alone (step 1: rho_c up 0.35, 0.37,
  0.97, 0.84 across eigsh runs, one band, D = 0). The spin-summed rho_up + rho_dn
  was 1.00 every time (for an SU(2) multiplet it is the same for every state and
  superposition). risb's `symmetrizers` act on rho_f, rho_cf, rho_c, so a
  paramagnetic symmetrizer already averages spin-multiplet degeneracy away
  (why the Mott test converges with "ignore"). It matters only when no
  symmetrizer covers it (magnetism allowed, orbital or accidental degeneracy).
- "average" is the mixed state P0 / g = (1/g) sum_i |i><i| (average of
  expectation values), not a sum of state vectors. It is the T -> 0 limit of
  exp(-beta H) / Z and basis independent; for spin multiplets it equals a
  paramagnetic symmetrizer. Not verified as the correct RISB prescription (T = 0
  RISB is a pure-state variational theory): check the literature (finite-T
  RISB/gGA) before implementing "average". Open: is "average" needed at all,
  given symmetrizers?

Done (step 8):
- `FockitImpuritySolver(..., method, method_options, diagonal_symmetries,
  permutation_symmetries, onsite_symmetries, degenerate, degenerate_tol)`.
  Unknown `method`/`degenerate` raise `ValueError`. Diagonal symmetries are
  checked with `normal_order(commutator(h_imp, Q))` (coefficients > 1e-10
  raise; ~7-82 ms per solve at 2-5 orbitals). With permutation/onsite
  symmetries the ground state is found in a symmetric basis and
  `expand_state`d into `basis` (N + diagonal symmetries). "warn" uses k=2 and
  `warnings.warn(RuntimeWarning)`. User `method_options` override the eigsh
  defaults (`which="SA"`, fixed `v0`, `ncv=min(dim, 40)`).
- Tests: eigh, Sz, and Sz + spin swap reproduce the three impurity fixtures
  (spin-swap eigenvalue of the ground state: one_band -1, bilayer +1,
  dh_trimer -1); particle-hole sector reproduces the plain solve; missing
  symmetries raise; "warn" warns at D = 0 and not with hybridization.
- How-to: "Choosing the eigensolver", "Symmetries", "Degenerate ground states".

## Item 15 notes (Mott regime)

- Trace (scratchpad `item15_trace.py`, complex 2-orbital model 6^3,
  Kanamori J = 0.2 U, beta 40, method eigh, start R = 1, lambda = mu):
  - DIIS, U = 4 and 5 (complex and real hopping alike): raises after 141 / 15
    (real 294 / 21) cycles. Before the crash DIIS extrapolates to absurd points
    (lambda - mu eig -172 .. +163, Z eig 1e-6) where Delta eig is 1e-8 or
    1 - 1e-7; that is where the K^(-1/2) helpers fail.
  - Linear mixing 0.1, U = 4 and 5: no crash; goes to Z ~ 0.01-0.08,
    lambda - mu ~ 0, Delta eig ~ 1/2 (no edges), then a period-2 oscillation
    for 3000 cycles (residual ~0.8): lambda - mu eig (-0.000, +0.009) <->
    (-0.009, +0.000), Delta (0.454, 0.502) <-> (0.498, 0.546).
  - So the edge states are DIIS artifacts; the real problem is the
    oscillation near the Mott state.
- Hypothesis: near D = 0 the bath orbitals decouple and the T = 0 embedding
  gives a step function of the bath level, while rho_qp = n_F(lambda - mu)
  is smooth (beta = 40): f2 = rho_f - Delta cannot vanish near lambda = mu, so
  the map is effectively discontinuous. At D = 0 exactly, the zero bath level
  is degenerate (item 13): only a mixture gives rho_f = 1/2 = Delta. Test:
  embedding at the quasiparticles' beta (Gibbs average in the N = M sector),
  as in the finite-T functionals (Giuli et al. 2026; Ayral et al. 2017).
- Test (scratchpad `item15_thermal.py`, dense prototype: Gibbs weights at
  beta = 40 over the eigenstates of the N = M sector, weights < 1e-14
  dropped; same model, start, tol 1e-9, maxiter 3000):

  | U | ground, DIIS | ground, linear 0.1 | thermal, DIIS | thermal, linear 0.1 |
  | --- | --- | --- | --- | --- |
  | 3 | 11 it | 439 it | 11 it (identical) | 437 it |
  | 4 | raises (141) | oscillates | 33 it, Z = 0 | 297 it, Z = 0 |
  | 5 | raises (15) | oscillates | 12 it, Z = 0 | 236 it, Z = 0 |
  | 6 | raises (72) | raises (68) | 9 it, Z = 0 | 222 it, Z = 0 |
  | 8 | raises (15) | raises (68) | 9 it, Z = 0 | 213 it, Z = 0 |

  Thermal Mott solutions: Z ~ 1e-33, Delta eig near 1/2 but lambda - mu not
  unique (DIIS +-0.004 to +-0.028, linear +-1e-5 to +-1e-4): at D = 0 the
  decoupled bath and the flat quasiparticle band are Fermi functions of the
  same level at the same beta, so f2 holds for a family of lambda (like mu in
  the gap); the impurity, hence the physics, does not depend on it.
  U = 3 metal: thermal = ground (gap >> T).
- Side finding: without a symmetrizer, linear mixing at U = 3 converges to a
  spin-polarized solution (Tr Delta_up = 1.19, n_target 2 over both spins),
  DIIS to the paramagnetic one; the thermal run found the opposite spin.
- Ensemble (user asked to check first): canonical, the half-filled sector.
  Giuli et al. say the T = 0 problems "are to be understood in the half-filled
  sector" within the "particle-number conserving (normal-state) variational
  setup" (their Refs. 45, 46): the sector is a property of the variational
  space, so every pure state of a thermal ensemble lies in it. Inference; the
  finite-T trace's space is not stated explicitly where I read.
- Gibbs-average methods (user asked to check fockit's finite-T solvers;
  scratchpad `item15_methods.py`, Kanamori impurities, beta 40, D = 0.5 metal
  and 0.05 near Mott): states with weight > 1e-14: 5 (metal) to 13 / 33 / 61
  (2 / 3 / 4 orbitals near Mott). Dense: exact, 0.04 s (924), 164 s (12870).
  fockit `eigh_full` + `thermal_expectation`: exact, 0.2 s (924, per operator,
  full spectrum). fockit TPQ: noisy at this beta (100 samples: 5e-3 to 3e-1
  error near Mott), 55 s for 2 operators at 12870. Truncated eigsh (double k
  until the top weight < 1e-14, Rayleigh-Ritz): exact (<= 1e-14), 1.1 s at
  12870. Chosen: dense for `method="eigh"`, truncated eigsh otherwise. Possible
  fockit addition (user's call): a low-T thermal_expectation from the lowest k
  states.
- Implemented: `FockitImpuritySolver(..., beta=None)`; attributes `states`,
  `weights`; `overlap` is the weighted average; `gs_vector` is the lowest
  state. Tests: degenerate impurity averaged at beta 1000 (rho 1/2), eigsh =
  eigh thermal densities near Mott (1e-10), complex 2-orbital U = 5 lattice
  reaches Z = 0 with beta 40 (raises without it). How-to section
  "Temperature of the impurity problem". AtomDiagImpuritySolver unchanged
  (users should use fockit).
- U scan with the implementation (complex 2-orbital, beta 40, eigsh): U = 3
  as before; U = 4-8 converge to Z = 0, DIIS 9-33 it, linear mixing 0.1
  213-297 it.
- Not done: guards for Delta eigenvalues at 0 or 1 (no longer reached here;
  a DIIS overshoot could still reach them in other models); stationarity of
  the thermal solutions (get_lagrangian is the T = 0 embedding functional).

## Item 13 notes (degenerate="average")

- Literature (definitions checked, derivations where marked mine):
  - Lanata, Deng, Kotliar PRB 92, 081108 (2015), finite-T GA: thermal
    quasiparticle ensemble (their Eq. 35) but a single slave-boson amplitude
    phi (one pure embedding state, Eqs. 28, 30); T -> 0 flows to a ground
    state of H_emb, not a mixture.
  - Sandri, Capone, Fabrizio PRB 87, 205108 (2013): one variational matrix
    Phi_i per site (pure embedding state), entropy term in Phi^dag Phi.
  - Ayral, Lee, Kotliar PRB 96, 235139 (2017), Appendix C (finite-T RISB,
    final expression only): embedding term T sum Tr log[1 - e^{-beta (H_emb -
    E_c)}] + E_c. Mine: its derivatives weight embedding eigenstates by
    Bose-like factors normalized to 1 by E_c; T -> 0 puts the weight on the
    ground level, split equally over an exactly degenerate level.
  - Giuli, Lee, Yao, Kotliar, Ruckenstein, Gingras, Lanata arXiv:2603.20559
    (2026), finite-T ghost-GA (Eqs. 196-206): embedding enters as
    -(1/beta) ln Tr e^{-beta H_emb}, expectation values are Gibbs averages;
    beta -> infinity with an exact degeneracy gives the equal mixture. They
    state it does not reduce to Ayral's finite-T RISB functional even at
    B = 1.
  - Also mine: the T = 0 embedding energy E0(D, Lambda_c) is not
    differentiable where a degenerate level splits (degenerate perturbation
    theory: the derivative depends on the direction); the equal mixture is
    the T -> 0 limit of the smooth finite-T derivative.
  So the finite-T RISB/ghost-GA functionals support "average" as the T -> 0
  limit; the pure-state finite-T GA papers do not use mixtures.
- Search for a case symmetrizers cannot handle (scratchpad `item13.py`,
  `item13_gaps.py`):
  - 3-orbital Kanamori, cubic 10^3, no symmetrizers (magnetism allowed),
    "ignore" (eigsh default v0 and 2 seeds, eigh) vs a dense "average"
    prototype: U = 2: non-degenerate, all identical (Z 0.7917, spin and
    orbital polarization <= 4e-15). U = 3: none converge in 1000 it, average
    included (Z 0.003-0.007, heading to the Mott state, item 15); final level
    non-degenerate.
  - Impurity ground level at converged solutions (symmetrize_blocks, 8^3,
    eigh): 2 orbitals n = 1-3 and 3 orbitals n = 1-5, U = 1, 2, 3: every
    metallic solution non-degenerate with gap E1 - E0 0.37-1.30; the only
    degenerate one is the Mott state (3 orbitals n = 3 U = 3, Z = 0, 24-fold,
    D = 0); 3 orbitals n = 4, 5 at U = 3 raise (item 15).
  - Pattern (not a proof): with D != 0 the bath screens the impurity into a
    non-degenerate ground state; exact degeneracy appears where D = 0 (Mott),
    which is item 15's regime, and there the spin symmetrizer already copes
    (the Mott test converges with "ignore").

## Item 12 notes (symmetries in the examples)

- Ground-state sector at each example's converged solution (dense, all
  sectors): hubbard U = 10 triplet (N_up - N_dn = -2, 0, 2 all hold it);
  bilayer layer swap +1 (-1 is 0.33 higher); kagome spin swap -1 (+1 is 0.96
  higher; the sign is from reordering fermions); decorated honeycomb C3 w^0
  (w^1, w^2 0.35 higher). Spin swap with Sz = 0: bilayer +1, kagome -1.
- Added: hubbard `diagonal_symmetries={N_up - N_dn: 0}`, bilayer layer swap
  (+1), kagome spin swap (-1), decorated honeycomb C3 onsite factors (1).
- Verified (scratchpad `item12_verify.py`): every impurity solve of every
  example compared with the unrestricted ground energy: |difference| <=
  1.1e-13 (hubbard 180 solves over the U scan, bilayer 21, kagome 20,
  decorated honeycomb 17). Results vs the previous examples: bilayer 2.8e-11,
  kagome 1.5e-12, decorated honeycomb 3.6e-10; hubbard Z, S2, N identical at
  every U, but at U = 10 (Mott, R = 0) mu = lambda differ (2.826 vs 2.807):
  mu is not fixed inside the Mott gap (item 15 regime), lambda - mu = 0 both.
- `docs/how-to/impurity_solvers.md`: points to the examples; the warning now
  says the solver finds the lowest state in the chosen sector (another sector
  targets a higher state), with the bilayer/kagome spin-swap signs.

## Item 11 notes (C3 slowness)

- Cause is fockit, not the setup. `build_basis` with the C3 onsite factors
  takes 0.0 s (dim 312 of 924). The time is in `to_matrix` ->
  `check_symmetric` -> `_conjugate` (symmetries.py), which computes
  U op U^dag symbolically: for each ladder operator, `normal_order(factor *
  image * hc(factor))` over the 8 factors. Each factor commutes with c_k on
  other modes (factor factor^dag = 1), but the exact cancellation (w w^* - 1)
  leaves float coefficients down to 1e-80 that are never dropped: the image of
  c(up_E1, 0) has 2, 2, 4, 6, 12, 18, 36, 50 terms after each factor instead of
  1. Then `term * images[single]` multiplies up to 4 such images per monomial
  of h_imp (~50^4 terms), unordered and undropped. cProfile: 11M monomials
  built in 90 s, not finished.
- Fix (tested by monkeypatching): `drop_zeros(..., tol=1e-12)` after each
  conjugation step. Images stay 1 term; C3 sectors w^0, w^1, w^2: dims 312,
  306, 306, build_basis + to_matrix 0.01 s; ground state in w^0, E0
  -15.022185199237 = unsymmetrized. Upstream fockit main (ef4596d) has the
  same `_conjugate`. A relative tolerance would be safer than an absolute one.
- From reading `tabulate_onsite`: the factor form `1 + (w^q - 1) n` is
  rejected because each monomial is tabulated as its own level map and a
  level hit by two monomials (identity and n both act on level 1) fails the
  "one level with a phase" check; summing the monomials per target level
  would accept it.
- Fixed upstream in fockit 0.4.1 (`0f1a17f`, drops terms below 1e-12 x
  max|coeff| in `_conjugate`); risb requires `fockit>=0.4.1`. Reproducer:
  0.01 s (10 s on 0.4.0).
- For risb: no speed reason to use C3 here: the whole unsymmetrized
  to_matrix + dense eigh on 924 states takes 0.07 s.

## Item 10 notes (physical Green's function)

- User's question (theory exploration): R only dresses the itinerant
  (hopping) part; the local part is exact in the slave bosons (embedding).
  G = R^dag G_qp R is not an isometry (weight Tr R^dag R = Tr Z, not M), so
  its filling drops with Z. Should G be an itinerant part with R plus a local
  part from the bosons, G = R^dag G_qp R + G_inc(w), and does that fix the
  filling?
- Sum rules fix G_inc's weight (1 - R^dag R) and equal-time density
  (rho_c - R^dag n_qp R), not its w dependence (no boson dynamics at the
  saddle point). Validity needs 0 <= rho_c - R^dag n_qp R <= 1 - R^dag R.
- Scope (user): A literature (definitions only, checked), B derivation in
  risb's conventions, C numbers (validity condition; embedding c Green's
  function as a G_inc candidate) on bilayer half filled, bilayer n = 1.6, M2,
  one large-U point, D report. No code.
- Literature (definitions, checked against our numbers):
  Lechermann et al. PRB 76, 155102 (2007): H_loc incl. its one-body part is
  represented exactly in the bosons (their Eq. 47), only the intersite
  hopping through R (Eq. 49); G from Dyson with the linear Sigma (Eqs. 53,
  55); intro: incoherent (high-energy) excitations come from fluctuations
  around the saddle point. Lanata, Lee, Yao, Dobrosavljevic PRB 96, 195126
  (2017), supplement Eqs. 45-49: A(k, w) = R^dag delta(w - H_qp) R from
  inserting projectors on the Gutzwiller quasiparticle excitations, "not a
  complete basis", so the weight is R^dag R = Z; with ghost orbitals the
  weight is [R~^dag R~]_11 ~ 1 for all U and the Hubbard bands appear (main
  text). Raimondi & Castellani PRB 48, 11453 (1993): Hubbard bands from
  Gaussian boson fluctuations (abstract only, not read).
- Derivation (sketch): the embedding state's one-body density matrix Gamma
  over (c, f) has 0 <= Gamma <= 1. Its Schur complements with the
  stationarity condition rho_cf = R^T sqrt(Delta (1 - Delta)) give, in G's
  index order ([a, b] = <c^dag_b c_a>), n_inc = rho_c^T - R^dag rho_qp^T R
  >= 0 and (1 - R^dag R) - n_inc >= 0. So G_inc with the sum-rule weight and
  density is always a valid local spectral function at a RISB solution, and
  Z = R^dag R <= 1.
- Numbers (scratchpad `item10.py`, per spin block, method eigh):

  | model | Tr Z | filling G | n Z_mean | n_inc eig | (1-Z) - n_inc eig |
  | --- | --- | --- | --- | --- | --- |
  | bilayer U=4 n=2 | 0.906 | 0.4529 | 0.4529 | 0.257, 0.290 | 0.257, 0.290 |
  | bilayer U=4.5 n=2 | 0.587 | 0.2933 | 0.2933 | 0.331, 0.376 | 0.331, 0.376 |
  | bilayer U=4 n=1.6 | 1.321 | 0.5284 | 0.5286 | 0.125, 0.147 | 0.193, 0.214 |
  | M2 n=2 | 1.270 | 0.6349 | 0.6349 | 0.150, 0.215 | 0.150, 0.215 |
  | M2 n=1.6 | 1.462 | 0.5814 | 0.5846 | 0.079, 0.140 | 0.135, 0.185 |

  Filling of G = Tr(R^dag rho_qp^T R); equals n Z_mean only with
  particle-hole symmetry or equal Z. G_qp and rho_c fillings = n. Validity
  condition holds strictly everywhere. Bilayer U >= 5 does not converge
  (item 15).
- Embedding c Green's function (Lehmann, full Fock space): weight 1 and
  density rho_c (to 1e-12, by construction); poles at |w - mu| < 0.6 and
  Hubbard-like ones at |w - mu| ~ 2.3-3 (bilayer U = 4). But its low-energy
  weight (bilayer U=4: 0.61 vs Tr Z 0.91; M2: 0.89 vs 1.27) and high-energy
  occupied weight (0.70 vs Tr n_inc 0.55) do not match the RISB split, so it
  is not R^dag G_qp R + G_inc and cannot be used as G_inc directly.

## Step 9 notes

Scope (user): 1) derive how R, lambda, D, Lambda_c, rho_qp, rho_f, rho_cf
transform under a quasiparticle rotation u in risb's conventions; 2) test
numerically whether one RISB cycle commutes with the gauge (U(n), O(n), U(1)
per block; with/without symmetrizers; force_real); 3) compare with the measured
flat directions of the Jacobian (one_band, bilayer, Kanamori 2 orbitals) and
explain the 3 of 24; 4) try gauge fixing (R parametrization, lambda or rho_qp
diagonal = natural-basis gauge, null-space projection) and measure conditioning,
Krylov with a step-size stop, DIIS iterations. Scratchpad only; stop before
changing LatticeSolver.
Reference (user): the supplement of arXiv:1606.09614 (Lanata et al., PRL 118,
126401 (2017)) discusses the gauge; `get_rho_qp` cites its Eq. 112
(natural-basis gauge) and says risb does not use that gauge. risb's equations
follow PRX 5, 011008 (2015), Eqs. 32-39.
Naive count: U(2) per block, two blocks, so 8; measured 3.

Results so far (bilayer, U=4, V=0.25, `method="eigh"`, DIIS tol 1e-13):
- Flat directions of the Jacobian: 3 of 24 (symmetrized, complex), 6 of 24
  (not symmetrized, complex), 0 of 12 with `force_real` (R restricted to real
  symmetric matrices removes the gauge here; smallest singular value 0.07).
- Of the U(2) rotations, only those commuting with lambda at the solution
  (e.g. exp(i t sigma_x); the solution is layer symmetric, R = 0.673 * 1) map
  the root to a root. A real rotation exp(i t sigma_y) gives |F| = 0.59 for
  every convention tried (R -> u^dag R, R u, u^T R, R u*).
- Stage by stage under that rotation: rho_qp -> u^dag rho_qp u and
  D -> u^dag D are covariant; Lambda_c is not, so rho_f, rho_cf, R, Lambda
  after it are not. Cause: item 14. Lanata's gauge group (supplement Eqs.
  41-43) is the full U(n), so the 3 (or 6) flat directions are what is left
  of the gauge after risb's non-covariant Lambda_c.
- Lanata et al. (supplement Sec. VI) argue the gauge freedom speeds up
  quasi-Newton convergence, and that natural-basis gauge fixing is costly
  (an extra equation, Eq. 126) and breaks covariance under changes of basis.

## Step 14 notes

- The exact derivative of Tr[Y f(rho)] + c.c., f(x) = sqrt(x (1 - x)), is
  -(W^T + W^*) with W = V [F o (V^dag Y V)] V^dag, rho = V diag(delta) V^dag,
  F_ij = (1 - delta_i - delta_j) / (f(delta_i) + f(delta_j)) (stable divided
  difference). Matches finite differences to 1e-9 (real 2x2, complex 3x3,
  4x4). risb's `get_lambda_c` and `get_lambda` (same form) differ by O(1) for
  random non-commuting inputs, and agree to 1e-12 when R D commutes with rho.
- But in a non-commuting model (2 orbitals, cubic, crystal field 0.4,
  inter-orbital hopping 0.3 cos kx, Kanamori U=3 J=0.6, |[rho_qp, (RD)^T]| =
  6.9e-3), risb's formula and the exact one give the same physics to 8 digits
  (eigenvalues of Z and of rho_c), and the same in three orbital bases (the
  Kanamori term is invariant under the rotation). The exact one converged in
  42-45 DIIS iterations vs 147 (another run: 156 vs 261). At the solution the
  raw derivative terms differ by 1.4e-2, but Lambda_c is projected onto
  Hermitian matrices right after, and lambda_new = lambda + T(rho_qp) - T(rho_f)
  makes T cancel at the fixed point (rho_f = rho_qp). So earlier "possible
  correctness bug" was an overclaim: no wrong result has been shown.
- A change of orbital basis cannot test this: it rotates the physical and
  quasiparticle indices together, and risb's formulas are covariant under that.
  Only a gauge rotation (quasiparticle index alone) exposes it.
- Gauge covariance of the risb cycle is broken in more than one place:
  `get_d` returns D with the quasiparticle index first (transforms as
  u^dag D, like Lanata's D_a,alpha), but the impurity solvers use D[alpha, a]
  with the physical index first, i.e. they get D^T; and risb's Lambda formula
  contracts R's physical index with D's quasiparticle index (R @ D), where
  Lanata contracts the physical indices (R D^T). Best combination found
  (Y = R D^T, impurity gets D^T, R -> u^dag R): |F(u x*)| = 9e-3, still not
  covariant, so something else is too (not yet found; candidates: get_r,
  get_f1, get_rho_qp's transpose, the lopsided ke).
- Option (a) (user): are risb's results ever physically wrong? Models (8^3
  cubic mesh, `method="eigh"`, no symmetrizers): M1 2 orbitals, crystal field,
  real inter-orbital hopping; M2 same with complex hopping; M3 3 orbitals,
  three crystal fields, inter-orbital hoppings, Kanamori U=2.5 J=0.5; M4 one
  4x4 block (2 orbitals x 2 spins), complex spin-orbit-like coupling, Hubbard.
  risb vs the "exact" Lambda formula: M1, M2 same physics (1e-12) and at
  risb's solution the Hermitian parts of the two derivative terms agree to
  ~1e-14 (so risb's formula is exact at those solutions). M4 same agreement
  (2e-14) at risb's solution, but the exact formula did not converge (2999 it)
  where risb's did (31). M3: both converge to |F| ~ 3e-14 at tol 1e-13 but to
  different solutions (Z eigenvalues differ by 4.3e-6; Hermitian parts differ
  by 3.2e-5 at both), so at least one is wrong. The "exact" formula is not a
  reliable judge: it keeps risb's R @ D contraction.
- Gauge-rotated starts with risb's unmodified code (Lambda = mu0 * 1,
  R = u^dag for random U(n) u; DIIS, tol 1e-12, cap 3000): every unrotated
  start (R = 1) converged (195-960 it); every rotated start failed (12/12,
  M1-M4). With a 20000 cap, some M1/M2 rotated starts converged after
  5313-16803 it. In a gauge-covariant loop a rotated start is an equivalent
  problem. So risb's broken covariance makes convergence depend on starting
  from a gauge-special point like R = 1.
- Those that converge (cap 20000): M1 rotated start 2 (5477 it) has the same
  physics as the unrotated start (Z 2.7e-15, rho_c 8.8e-14). M2 (complex):
  rotated starts 2 (16836 it) and 3 (9582 it) converge to the same solution as
  each other, which differs from the unrotated start's solution (Z eigenvalues
  by 4.4e-4, rho_c eigenvalues by 1.6e-3). Either M2 has two RISB solutions,
  or risb's answer depends on the starting gauge; cannot tell which is
  physical with the current, non-covariant cycle.
- Conclusion of (a): for every current example and test (commuting/symmetric
  cases) risb is unaffected. For complex, non-commuting models risb's result
  can depend on the starting point, convergence from anything but R = 1 is
  very slow, and two Lambda formulas give different solutions in M3. A
  gauge-covariant cycle (option (b)) is needed to know the right answer.
- Option (b) (user): make the whole cycle gauge covariant, explaining every
  step. Reference for "correct" (user): (iii) checks that must hold whatever
  the formalism (U = 0 limit for any starting gauge, covariance of one cycle,
  derivative formulas vs finite differences), then a direct variational
  minimization on M2 as the final judge. Lanata's papers may be used for
  definitions, but every definition is checked, not trusted.
- Lattice-side transformation laws, derived from risb's own definitions (f =
  u f'): R -> u^dag R, lambda -> u^dag lambda u, rho_qp[a, b] = <f^dag_a f_b>
  -> u^T rho_qp u*, ke[alpha, b] = (h0 R^dag rho_qp^T)[alpha, b] -> ke u,
  D = K^(-1/2) ke^T -> u^T D (agree with Lanata Eqs. 41, 43). The embedding is
  then equivalent under the rotation only if the hybridization is
  sum D[a, alpha] c^dag_alpha f_a (D's qp index first); risb's solvers use
  D[alpha, a].
- U = 0 limit, M2 (complex), risb's current code: correct physics from every
  start (Z = 1, rho_c = exact non-interacting density matrix to 4e-13), but
  the unrotated start converges in 1 iteration and rotated starts (R = u^dag,
  a gauge copy of the exact solution) in 765-1464. A covariant cycle would take
  1 from any rotated start: a formalism-independent test for each fix.
- Fix 1 (applied): both impurity solvers' `set_h_hybr` use D transposed, i.e.
  sum D[a, alpha] c^dag_alpha f_a (docstrings state the index order). Tests:
  69 pass. (Its first measurement was wrong: `measure.py` imported scratch
  code that replaced risb's Lambda functions. Clean numbers below.)
- Impurity-side laws (derived, with sum D[a, alpha] c^dag_alpha f_a): rho_f ->
  u^T rho_f u*, rho_cf -> rho_cf u*, R_new -> u^dag R_new, lambda_new ->
  u^dag lambda_new u; hole-form bath sum f_a Lambda_c[a, b] f^dag_b needs
  Lambda_c -> u^T Lambda_c u*. So f2 = rho_f - rho_qp^T is not covariant
  (fix 3: rho_f - rho_qp).
- Fix 2 (proposed): from the gauge-invariant term Tr(D R^T f(rho_qp)) + c.c.,
  lambda + lambda_c = -(W^T + W^*), W = V [F o (V^dag D R^T V)] V^dag; risb's
  Lambda_c is the transpose (hole-form bath): get_lambda_c = -(W + W^dag) -
  Lambda^T, get_lambda = -(W_f^T + W_f^*) - Lambda_c^T (rho_f in place of
  rho_qp).
- Fix 2 (applied): `helpers.get_lambda_c` = -G^T - Lambda^T and
  `get_lambda` = -G(rho_f) - Lambda_c^T, with G = W^T + W^* the gradient of
  Tr(D R^T f(rho)) + c.c. (private `_sqrt_term_gradient`; checked against
  finite differences to 1e-8 for complex 2x2-4x4).
- Clean measurements (`measure.py`, M2 interacting, one generic U(2) u, risb's
  own functions asserted):
  original: |F(u x*)| 0.83, 3/8 stages covariant, U = 0 rotated starts
  [1176, 972, no], M2 from R = 1 195 it;
  fix 1 only: 4.5e-3, 3/8, [1035, no, no], 1125 it;
  fix 1 + 2: 3.8e-13 (= |F(x*)|), 8/8, [1, 1, 1], 108 it.
  The default cycle (error_fun="recursion") is now gauge covariant. f2 is
  only used with error_fun="root" and is still not covariant (fix 3).
- After fix 2: 2 tests fail, `test_scipy_root[{fockit,atom_diag}-bilayer-
  krylov]`: Lambda differs from the expected matrix by 3e-9 (tolerance 1e-10)
  but is a small gauge rotation of it (same eigenvalues; mu passes). Krylov
  stops on the residual, which is now zero along the whole gauge orbit.
- Found: `LatticeSolver.Z` is R R^dag (quasiparticle basis, gauge
  dependent); the physical weight is R^dag R (same eigenvalues).
  `helpers_triqs.get_sigma_w` uses (R R^dag)^-1 where G = R^dag G_qp R gives
  (R^dag R)^-1 (not yet verified).
- Proposed order: (1) lattice tests compare gauge-invariant quantities (mu,
  eigenvalues of Lambda and Z); (2) fix 3 (f2 = rho_f - rho_qp) and measure
  error_fun="root"; (3) Z = R^dag R; (4) check get_sigma_w and the other
  Green's function helpers for gauge invariance; (5) covariance tests.
- Done after fix 2 (user approved the order):
  (1) `test_latticesolver.do_assert` compares mu and the eigenvalues of
  Lambda and Z (gauge invariant); the 2 Krylov tests pass again.
  (2) Fix 3: `get_f2` = rho_f - rho_qp. Before: error_fun="root" residual at
  the rotated solution 7.8e-3; after 2.3e-13 (= at x*). At the R = 1 solution
  rho_qp happened to be real symmetric, so both forms were 0 there.
  (3) `LatticeSolver.Z` = R^dag R (physical orbitals, gauge invariant; same
  eigenvalues as the old R R^dag; equal when R is normal, as in all examples).
  (4) `helpers_triqs.get_sigma_w`: Z^-1 = (R^dag R)^-1 (from G = R^dag G_qp R:
  Sigma = (1 - (R^dag R)^-1)(w + mu) + R^-1 lambda R^-dag - h0_loc). M2 (R not
  normal): Dyson vs R^dag G_qp R disagreed by 1.2e-2 at the solution and 0.45
  at the rotated one, Sigma changed by 0.73 under rotation; after: 1e-15,
  2e-15. `docs/how-to/green_functions.md` says Z = R^dag R.
  (5) Tests (behaviour, public interface), each failing on the original code
  and passing now: `test_quasiparticle_basis_of_the_starting_guess_does_not_
  change_the_physics` and `test_rotated_copy_of_the_non_interacting_solution_
  is_converged` (test_latticesolver, both solvers; complex 2-orbital model
  `build_two_orbital_complex_h0_k` and `random_unitary` in test_common), and
  `test_self_energy_does_not_depend_on_the_quasiparticle_basis` (new
  tests/test_helpers_triqs.py, skips without TRIQS).
- Interacting M2 now: rotated starts converge in 25-28 it like the unrotated
  start, same physics (Z 6e-12, mu 3e-10), both solvers. 74 tests pass (56 +
  18 skipped without TRIQS); lint, mypy, strict docs pass.
- Derivative method: Daleckii-Krein (eigensystem, divided differences) kept
  over the equivalent Sylvester form (sqrt(K) Z + Z sqrt(K) = D R^T,
  W = (1 - rho) Z - Z rho; agree to 2e-14): 3-4.5x faster for block sizes
  1-10 (5-19 us vs 22-84 us). Docstrings of `get_lambda_c` and
  `_sqrt_term_gradient` name the method. The scalar f'(rho) form with the
  corrected D R^T pairing is also gauge covariant but differs from the exact
  derivative (1.4 on a random non-commuting 3x3), so covariance alone does not
  fix the formula.
- Insulating regime check (user asked), complex 2-orbital model
  (`build_two_orbital_complex_h0_k`, Kanamori J = 0.2 U), DIIS from R = 1:
  original code: U = 3 converges (Z 0.55, 0.69); U = 4 Z = 0 after 530 it;
  U = 5, 6 crash (singular matrix); U = 7, 8 reach Z = 0, but by a blow-up
  (error 1.2e8 at call 6, then lambda = 0, R = 0, mu = 0, the trivial Mott
  fixed point). New code: U = 3 identical; U >= 4 all crash (ARPACK error,
  NaN, singular matrix). With linear mixing both codes crash at U >= 4.
  Cause: iterates reach rho_qp eigenvalues at 0 or 1 (e.g. 1 + 3e-15), where
  every K^(-1/2) formula (get_d, get_r, get_lambda_c, get_lambda, get_f1) is
  singular; the new gradient gives NaN there (real sqrt of a tiny negative).
  Complex-branch sqrt or clipping eigenvalues in the gradient alone does not
  help (other helpers then hit a singular matrix). Starting exactly at R = 0
  is accepted as converged at once by both; the existing Mott tests (real,
  symmetric) pass.
- Docs matched to the corrected code: `explanations/impurity.md` (bath term
  sum lambda^c_ab f_a f^dag_b, the code's order; interface section states
  D[bl][a, alpha], Lambda_c[bl][a, b], rho_f[a, b] = <f_b f^dag_a>,
  rho_cf[alpha, a] = <c^dag_alpha f_a>, rho_c[alpha, beta] =
  <c^dag_alpha c_beta>), `explanations/projectors.md` (Delta^qp transposed, as
  get_rho_qp), `tutorials/self-consistent.md` (Z = R^dag R; code slips fixed:
  missing colon, update_weights on a dict, h0_kin_k_R / kinetic_energy names,
  sc. -> helpers., break outside the block loop). The tutorial's functions and
  skeleton, with its blanks filled from the tutorial, run: bilayer converges
  in 9 cycles, Z = 0.7526.
- Independent check (user chose stationarity of the Lagrangian over a full
  minimization): new public `helpers.get_lagrangian` (single cluster = unit
  cell, no projectors; Fermi-Dirac smearing; embedding terms through the
  impurity's expectation values at fixed ground state; Eq. 40 of the PRL
  126401 supplement in risb's conventions). Its partial derivatives vanish
  exactly where risb's equations hold (lambda: rho_qp = Delta_p; Lambda_c:
  f2; D: f1; Delta_p: fix 2's formula; R: the D equation).
  Finite-difference derivatives (step 1e-6, floor ~1e-9) along every
  direction of R, D (real and imaginary parts) and lambda, Lambda_c, Delta_p
  (Hermitian basis), complex 2-orbital model, Kanamori U = 3, J = 0.6:
  fixed code's solution: all <= 9e-10 (stationary). Controls: R perturbed by
  0.01: 1e-3 to 1e-2 (the check is sensitive); U = 0: <= 2e-10. Original
  code (git worktree at e47aa26^, verified loaded): from R = 1 it stalls at
  error 4e-9 but that point is stationary to 5e-9 (same physics); from a
  rotated start (M2, 8^3 mesh) it does not converge in 20000 it and ends at a
  point with its own f1/f2-type derivatives 1e-6 / 1e-7 but the Delta_p
  derivative 3.5e-2, Z 0.5712 vs 0.5742: not a RISB solution.
- Item 14 independent check: done (above). Test added:
  `tests/test_helpers.py::test_converged_solution_is_a_stationary_point_of_
  the_lagrangian` (fockit only; complex 2-orbital model, rotated starting R,
  tol 1e-8; finite-difference derivatives of `get_lagrangian` in every
  direction < 1e-7; largest measured 7e-9). 0.14 s. The original code fails
  it (no convergence in 999 it; derivatives up to 5e-2). A 4^3 k-mesh does
  not converge (999 it), so the test keeps 6^3.
- `get_lagrangian` handles one cluster equal to the unit cell only: item 16.
- Unblocked: step 9 (gauge freedom and convergence speed; Lanata's claim).
- Tests to add once the cycle is covariant (user, done above): through `S.R`, `S.Lambda`,
  `S.solve()`: U = 0, a gauge-rotated copy of the exact solution converges in
  1-2 iterations; complex non-commuting interacting model, rotated and
  unrotated starts give the same physics without many more iterations.

## Step 9 notes (gauge redundancy, after the covariance fixes)

- Flat directions at the bilayer solution (`method="eigh"`): no symmetrizer,
  complex: 8 = the full U(2) x U(2) gauge (every rotation maps the root to a
  root, |F| 1.6e-14); `symmetrize_blocks`: 4 (only rotations equal on both
  blocks survive the up = dn symmetrizer); `force_real`: 0 (R is restricted
  to real symmetric matrices, which pins the gauge). The null directions are
  exactly the gauge. Step 3's "3 of 24" came from the old non-covariant code.
- Any R can be made Hermitian by a gauge rotation (polar decomposition
  R = U P, rotate by U), so "R Hermitian" is a complete gauge choice.
- Decisions (user): compare free gauge, Hermitian-R gauge and Lanata's natural
  basis (rho_qp diagonal); include the Mott scan (U = 4-8, complex model).
  Natural basis is done as a gauge-fixing rotation after each cycle (sorted
  eigenvalues of rho_qp, phases making diag(R) real positive), root finder on
  the full variables; not Lanata's eta formulation. Hermitian-R: polar
  rotation after each cycle, root finder also on the full variables (the
  slice R = R^dag is linear, so DIIS stays in it after the first step; same
  as restricting R's parameters).
- Why fixing after the cycle works: the cycle F is covariant, so g o F (g =
  rotation into the slice) is gauge invariant; along a gauge direction its
  derivative is 0, so the Jacobian of g(F(x)) - x is -1 there, not 0. Natural
  basis uses rho_f of the same cycle (rho_f = Delta_p at convergence), ill
  defined when rho_f is degenerate. A fixed point of g o F only gives
  F(x) = u x, so the free residual |F(x) - x| is reported too.
- Jacobian condition number of the recursion error at the solution (finite
  differences, DIIS to 1e-12, scratchpad `step9.py`): bilayer free 2.0e14,
  Hermitian R 17, natural 28; M2 complex free 7.0e12, Hermitian R 11,
  natural 17. Either gauge choice makes the roots isolated.
- Convergence (scratchpad `step9.py converge`, recursion error, tol 1e-10,
  `method="eigh"`; 3 starts: R = 1 and 2 random unitary rotations; impurity
  solves per run; every run reported converged has free |F(x) - x| <= 1e-10,
  so no spurious fixed points):

  | model | root finder | free | Hermitian R | natural |
  | --- | --- | --- | --- | --- |
  | bilayer | DIIS | 14 | 14-16 | 16 |
  | bilayer | Krylov, residual stop | 54-60 | 56-650 | 43-93 |
  | bilayer | Krylov, step stop | fails (8000+) | 88-679 | 74-145 |
  | bilayer | hybr | 43-44 | 44-81 | 55-105, 1 of 3 fails |
  | Kanamori degenerate | DIIS | 20 | 20 | raises |
  | Kanamori degenerate | Krylov, residual stop | 60-68 | 55-76 | fails |
  | Kanamori degenerate | Krylov, step stop | fails (8000+) | 84-108 | fails |
  | Kanamori degenerate | hybr | 41 | 41-50 | fails |
  | M2 complex | DIIS | 26-29 | 85-103 | 79-124 |
  | M2 complex | Krylov, residual stop | 49-51 | 37-74 | 40-64 |
  | M2 complex | Krylov, step stop | fails (8000+) | 70-104 | 72-93 |
  | M2 complex | hybr | 43-45 | 42-44 | 42-47 |

  Free-gauge Krylov with a step stop sits at the root (residual 1e-15) but
  never stops: the steps along the gauge directions do not shrink. The
  natural basis fails whenever rho_f is degenerate (degenerate orbitals), as
  expected. Neither pinned gauge speeds up DIIS (M2: 3-4x slower), so
  Lanata's claim does not carry over to risb's default root finder.
- Mott scan (complex 2-orbital model, U = 4-8, DIIS): every gauge raises
  (ValueError / LinAlgError from K^(-1/2)) at every U, same as free. Pinning
  does not help item 15.

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
