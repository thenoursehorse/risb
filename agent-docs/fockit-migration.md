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
15. [x] (done; the Mott states have a traceless f2 residual, harmless in the atomic limit, see Item 15 notes; includes `degenerate="average"` from item 13: the only exact
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
16. [x] (done: several clusters, and finite T via embedding_entropy; see Item 16 notes) `helpers.get_lagrangian` for several clusters. It handles one cluster equal
   to the unit cell only, but `LatticeSolver` allows several clusters with
   `projectors` and `gf_struct_mapping`; the helper should take the same
   arguments (upfolded R(k), lambda(k) in the quasiparticle term; embedding
   and Delta_p terms summed over clusters). Extend the stationarity test to a
   multi-cluster model. Also a finite-T version (user): with a thermal
   embedding (`FockitImpuritySolver(beta=...)`) the embedding terms become
   its free energy -(1/beta) ln Tr_{N=M} e^{-beta H_emb} (Gibbs, Giuli et al.
   2026; Ayral et al. 2017 Eq. C3 has a Bose form T Tr log[1 - e^{-beta (H_emb
   - E_c)}] instead: decide which), and add the stationarity check of the
   thermal Mott solutions (complex 2-orbital, U = 5).
17. [x] (done, see Item 17 notes) Decorated honeycomb in the site basis. Rename
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
18. [x] (done: DIIS fix, mu bracket, core treatment of full / empty orbitals; see Item 18 notes) Frozen quasiparticle orbitals (user): the analytic treatment of Delta
   eigenvalues at 0 or 1 in the condensed-boson (pure embedding) functional,
   e.g. with exact T = 0 k-integration. In the Mott state (R = 0, flat bands at
   lambda) a band away from mu is exactly full or empty (Delta at an edge,
   every K^(-1/2) helper singular), and a flat band at mu is fractionally
   filled with the fraction fixed by n_target, so the bath needs a mixture
   with those weights. Derive the limit (frozen rows of R and D, frozen bath,
   the 0 x infinity terms in the gradient), check it against the literature
   (ghost-GA has non-invertible R) and numerically (approach T -> 0), then
   decide on an implementation.
19. [x] (done, see Item 19 notes; scope (user): k-dependence and coverage; non-orthonormal / overlapping
   is item 21. lambda gets no Q term: uncorrelated states keep their local
   part in h0_kin, have no constraint, so lambda = 0 there; R = 1 there gives
   their bare band) Relax the projector assumptions (user: "a VERY big restriction",
   "100% sure that can be relaxed"). Now required: projectors the same at
   every k-point, orthonormal rows, no overlap between clusters, covering
   every orbital of h0_k (an uncovered orbital gets R = 0, lambda = 0, a
   flat band at 0). Relax at least k-independence (e.g. Wannier-like
   projectors that vary with k) and full coverage (uncorrelated orbitals
   keeping their bare band: R = 1, lambda = 0 there, h0_kin unchanged).
   Check the up/down-folding maths for each, extend `check_projectors`, and
   test (the supercell test is a starting point; a partially covered cell
   with an uncorrelated orbital should reproduce the covered result).
20. [x] (closed, not worth it: underdetermined by the overall lambda shift; the hybrid works for metals but costs more and fails at edges / Mott; see Item 20 notes) Investigate (R, Delta^p) as the input of the self-consistent cycle
   instead of (R, Lambda) (user). Initialising with R = 1 and Delta^p = the
   non-interacting density matrix is more natural than guessing Lambda;
   Lambda would then be found from Delta^p (e.g. a fit / inversion of
   Delta^p = n_F(R eps R^dag + lambda - mu) for lambda). Find out whether it is
   worth it: how lambda is obtained and whether that is well posed, cost per
   cycle, effect on convergence and on the R = 0 / edge cases, gauge
   covariance, and what the literature does (Lanata's codes, ghost-GA).
21. [x] (dropped (user): it would only be Lowdin orthonormalization at each k, (P P^dag)^(-1/2) P over all clusters' projectors, which users can do themselves; check_projectors=True flags when needed) Investigate allowing non-orthonormal or overlapping projectors (user;
   after item 19), e.g. orthonormalizing internally, as DFT+DMFT codes do
   with raw projections onto local orbitals.
22. [x] (dropped by the user) Rename `h0_kin_k` (user: later; the proposed h0_qp_k / h0_lat_k "aren't
   great"; old-name aliases also decided later). It is h0_k minus the local
   terms of the correlated clusters, so it also holds the local terms outside
   all projectors (after item 19); until renamed, document that.
23. [x] (closed for now (user): helpers and hook done; checking DFT+RISB properly needs much more than the NiO run; see Item 23 notes) DFT+RISB helpers (user; later): double-counting helpers, a helper for
   the DFT+RISB total energy, a helper for the charge self-consistent (CSC)
   update, and a test on real input (the user has NiO projectors and h_k, and
   another workflow to point to). Not needed: k-point symmetrization and
   projector conditions.
24. [x] (done: steps 1-4 at T = 0; finite T is item 30) Investigate adding ghost RISB (g-RISB / ghost-GA) to risb (user): from
   Lanata's theory papers only, not other codes. What changes (bath larger
   than the impurity, rectangular R, the ghost orbitals' Lambda, the
   embedding sector), what it costs with fockit, and how it fits risb's
   interfaces.
25. [ ] Investigate overlapping fragments, as in bootstrap embedding (user):
   clusters that share orbitals, how their overlapping parts are matched,
   and what that would need beyond risb's non-overlapping projectors.
26. [x] (done: mixed B per block works, odd ghosts per cluster rejected, docs on choosing B; see Item 26 notes) Investigate different numbers of ghost bath orbitals per block of an
   impurity (user), e.g. B = 2 for t2g and B = 3 for eg. Questions: an even
   B, or a mix, has no obvious half-filled sector ((B + 1) nu / 2 per block
   need not be an integer); what Lanata's papers say about the embedding's
   particle number then.
27. [x] (done: removed the solver, the tests' solver parametrization (101 tests, no skips), the benchmark's AtomDiag columns, the docs section and about entry; the TRIQS citation moved to the Green's function how-to; get_C_Op stays for item 28 because helpers_triqs.matrix_to_Op uses it) Remove AtomDiag (user): AtomDiagImpuritySolver and what only it
   needs.
28. [x] (done: helpers_triqs removed, numpy Green's function helpers in helpers, TRIQS out of CI/CD; see Item 28 notes) Investigate the benefit of having helpers_triqs and its Green's
   function helpers at all (user). Should the Green's function helpers be in
   helpers without TRIQS, like helpers.get_sigma_w?
29. [x] Review comments and docstrings (user): they have started to narrate
   the conversation (e.g., "ARPACK can fail (error 3, no shifts could be
   applied)") and to use phrases that do not make sense (e.g., "Renormalization
   matrix in the blocks of the lattice, quasiparticles by orbitals").
30. [ ] Ghost RISB at finite temperature (user; from item 24 step 5). In the
   4c model (two-site cubic cell, U = 2, fixed mu = 1, B = 3) the cycle does
   not converge above T ~ 0.03, with a thermal or a ground-state impurity.
   risb's finite T (Gibbs weights in the half-filled sector) is not Lanata's
   finite-T ghost functional (arXiv:2603.20559: grand-canonical traces and an
   extra quadratic embedding H_0emb), which GEM implements by fitting thermal
   density matrices.
31. [ ] Choose the electron offset M_i of a cluster's embedding (user; from
   item 26): g-RISB allows any integer 0 <= M_i <= ghosts, each a variational
   ansatz, the physical one the lowest energy. Needed for even B in a
   paramagnet. Would need one value per cluster shared by
   FockitImpuritySolver (sector) and LatticeSolver (n_ghost_electrons).
32. [x] (superseded (user): the half-filled thermal mode stays as a convergence aid for T = 0, documented so; thermal averages over every particle number and symmetry value are the grand-canonical mode of item 30 step 2) Thermal average over one symmetry sector (user; found in item 26):
   with `beta` and `diagonal_symmetries` (e.g. {Sz: 0}) FockitImpuritySolver
   averages only over the eigenstates of that sector, not a Gibbs average
   over all of the half-filled sector. E.g. one band, B = 5: the first
   excited level is a triplet (S = 1); with {Sz: 0} its Sz = +-1 states are
   left out. Fix idea: sum the Gibbs weights over all sectors of the
   symmetry, solving each.
33. [ ] Redundant ghosts (from item 26): one band, U = 2.4, B = 5 converges
   to the B = 3 solution plus two ghosts per spin with R = 0 near mu, and
   the self-consistency error stays 0.05-0.6 with every impurity mode. The
   decoupled orbitals of item 18 and the U = 0 ghosts of item 24. Options to
   investigate: the core treatment for ghosts, or detecting and removing
   ghosts that decouple.
34. [ ] Permutation and onsite symmetries in the grand-canonical impurity
   solve (user; from item 30 step 2): fockit builds one eigenvalue of each.
   Idea: enumerate each symmetry's eigenvalues (2m-th roots of unity for
   order m, since fermion signs can make g^m = -1), build every sector,
   expand to the particle-number basis, and raise unless the sector
   dimensions add up to the particle-number dimension (non-commuting
   symmetries leave states in no sector). An optimization: smaller blocks,
   and symmetry-degenerate states in different blocks.
35. [ ] Revisit a separate grand-canonical impurity solver class (user; from
   item 30 step 2): the two modes treat symmetries differently, e.g.
   diagonal_symmetries take a value (fockit's convention) that has no
   meaning in the grand-canonical mode, where they only split blocks.
36. [ ] Non-unique finite-T ghost solutions at high T (user; from item 30
   step 3): above T ~ 0.1 (one band, B = 3, U = 2) the Jacobian of Eqs.
   200-203 has more zero directions than the 6 of the gauge (about 4 more
   at T = 1). Two T = 1 solutions have the same d, E and L_beta to 6 digits
   but Tr Delta 1.62 vs 1.50 and Sigma(i w_1) 2% apart. Investigate what the
   extra directions are, whether Sigma(i w_n) is then ill-defined, and how to
   pick a solution (e.g., continuation from low T).

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

## Item 30 notes (ghost RISB at finite temperature)

Step 1, the paper (Giuli et al., arXiv:2603.20559, Secs. III A, III F, IV,
VI, VII B):
- T = 0 reformulation (Eq. 21): L = L_qp + L_emb - L_0emb. The standard
  mixing term F[Delta] = Tr((Lambda + Lambda_c) Delta) + 2 Re Tr(D R^T
  sqrt(Delta (1 - Delta))) (Eq. 93, risb's get_lagrangian) is replaced by
  the ground-state energy of a quadratic H_0emb (Eq. 27) on f (B nu ghost-
  extended modes) and b (bath): Lambda on f, D R^T coupling f-b, Lambda_c on
  b. Sec. IV: at T = 0 its ground state is the purification of Delta
  (Eqs. 95-97: <f+f> = Delta, <f+b> = sqrt(Delta (1 - Delta)), <b b+> =
  Delta), so the two Lagrangians are equivalent (same stationary points,
  larger gauge group).
- Finite T (Eq. 196): L_beta = Omega_qp + sum Omega_emb - sum Omega_0emb,
  each -T ln Tr e^{-beta H}, grand canonical. Stationarity (Eqs. 200-203):
  thermal <f+f>, <f+ D b> of H_0emb match the qp; thermal <b b+>, <R f+ b>
  of H_0emb match H_emb. beta -> inf gives Eq. 21.
- Sec. VI B-C (checked the algebra: trace-log, Eqs. 212-227): L_beta equals
  the DMFT functional (Kotliar RMP 2006) restricted to Sigma(i w_n) and
  Delta(i w_n) with the ghost pole parametrizations (Eqs. 207-208). Claimed
  -> DMFT as B -> inf at any T ("carries over", not shown in detail).
- Entropy from the stationary value: S = beta (E - mu N - L_beta*).
- Does not reduce to the finite-T RISB of Ayral, Lee, Kotliar (PRB 96,
  235139) even at B = 1.
- Algorithm (Sec. III F): (R, Lambda) -> qp -> fit (D, Lambda_c) to Eqs.
  43-44 -> H_emb -> fit (R, Lambda) to Eqs. 45-46. At T = 0 the first fit
  is analytic (Eqs. 88-89 = risb's get_d, get_lambda_c); the second they do
  numerically. At T > 0 both are fits (a thermal H_0emb is mixed, no
  Schmidt purification).
- Validation: only the half-filled paramagnetic one band on the Bethe
  lattice, B = 1 and 3, against NRG (Fig. 7: d, (E - mu N)/N, S vs T at
  U = 1, 2, 3.2).
- risb now: L = Omega_qp (Fermi smearing) + <H_emb> - T S_imp (Gibbs in the
  half-filled sector) - F[Delta] with the T = 0 purification form. A mixed
  qp state is matched with the pure-state relation sqrt(Delta (1 - Delta)),
  which is the likely cause of the item 24 finite-T stalls (Tr Delta 2.08 vs
  Tr rho_f 1.49) and of the decoupled-orbital residuals (items 18, 33): a
  decoupled f level and its bath are both Fermi functions in thermal H_0emb.

Plan (user): a separate ThermalLatticeSolver (solve_lattice_thermal.py)
with new helpers; LatticeSolver stays (T = 0, beta=None); if this works it
becomes the finite-T path. Steps: 1 H_0emb helpers; 2 grand-canonical
impurity solve (absorbs item 32); 3 prototype then ThermalLatticeSolver
(root over R, Lambda, D, Lambda_c); 4 grand potential and entropy; 5 retest
item 33 and the item 24 stalls; 6 example, docs, decision.

Step 1 (done): helpers.get_h0_emb (2M x 2M one-body matrix in risb's
conventions, psi = (f, b): [[Lambda - mu, R D^T], [(R D^T)^dag, -Lambda_c^T
- mu]] plus the constant Tr Lambda_c from the b b^dag ordering),
get_h0_emb_densities (thermal <f+f> as rho_qp, <f+b>, <b b+> as rho_f),
get_h0_emb_grand_potential. risb's Lambda and Lambda_c exclude mu, the
paper's include it, so mu is a uniform shift of all f and b modes. Checked
at a converged B = 3 solution (U = 2.4, Bethe, mu ~ 0): at beta 1e3 to 1e8
the thermal state reproduces Delta, sqrt(Delta (1 - Delta)), Delta, the
impurity rho_f and R^T <f+b> = rho_cf to 2e-13, with N = M (half filling)
and a +- symmetric spectrum. Tests: exact diagonalization of the operator
written with fockit over the whole Fock space (M = 3, nu = 2, complex,
beta 3, mu 0.3; densities and grand potential to 1e-12; fails if W is
D R^T, if Lambda_c is not transposed, or without Tr Lambda_c), and the
purification at a converged solution (1e-10).

Step 2 (done): FockitImpuritySolver(..., grand_canonical=True) needs beta;
permutation / onsite symmetries rejected (they keep one irrep). Particle
numbers built lazily (cached fockit bases) from half filling outwards until
the lowest energy of a particle number has a negligible weight (assumes no
second minimum in N); each particle number's states split by the values of
the diagonal symmetries (diagonal in the Fock basis), all kept; states of
every kept block found with the thermal path relative to the global E0.
New attributes: ensemble [(basis, states, weights)] per particle number
(ground state's first; overlap sums over it in both modes), grand_potential.
set_h_imp(mu) now subtracts mu on physical and bath orbitals (was physical
only, undocumented, unused). The half-filled thermal mode is documented as a
convergence aid for T = 0 (user). Tests vs exact thermal averages over the
whole Fock space: one spin-mixing complex block, doped (mu 0.4, beta 5,
reference H built without mu minus mu times every mode); Sz split (beta 3,
mu 0.3); both 1e-10. They fail with half filling only, one Sz value, or mu on
the physical orbitals only.

Step 3 prototype (scratchpad thermal/, one band, Bethe D = 1, 400 k, mu = 0
fixed, paramagnetic, real): Eqs. 200-203 in risb conventions per block:
E1 <f+f>_0emb = rho_qp; E2 <f+b>_0emb D = ke, ke[a, alpha] = mean_k eps_k
(n_F(H_k)^T R*)[a, alpha] (= risb's lopsided ke transposed, checked 3e-14);
E3 <b b+>_0emb = rho_f(imp); E4 R^T <f+b>_0emb = rho_cf(imp); impurity
grand canonical with set_h_imp(mu). At LatticeSolver's T = 0 solution all
four hold to 1e-14 with H_0emb at beta 1e8 and a ground-state impurity.
- (A) least_squares over all 18 reals: converges at every T (residual
  1e-10 to 1e-15) but 39-1176 residual calls up to T = 0.3 and 10629 at
  T = 0.5 (each a grand-canonical impurity solve).
- (B) outer DIIS on (R, Lambda); inner least_squares fit of (D, Lambda_c) to
  E1-E2 (H_0emb only), one impurity solve, fit of (R, Lambda) to E3-E4:
  4-67 impurity solves per T, same d and E to 4 digits. Chosen.
- U = 2, B = 3 vs Fig. 7 (read off): d 0.0762 / 0.077 (T 0.03), 0.0526 /
  0.054 (0.1), 0.0534 / 0.052 (0.2, minimum), 0.0875 / 0.08 (0.5), 0.142 /
  0.13 (1), 0.190 / 0.19 (2); (E - mu N)/N -1.083 / -1.08 low T, -0.680 /
  -0.68 at T = 2. U = 1: d 0.167 low T, min 0.142 near T 0.2, 0.193 (1),
  0.220 (2); E -0.716. U = 3.2: d 0.0133, E -1.640 low T; d 0.092 (1), 0.157
  (2); T = 0.5 stops at 300 iterations with residual 3.5e-9.
- T -> 0: d 0.0856 at T = 0.005 vs LatticeSolver 0.08559.
- <N_emb> = 4 (half filling) to 1e-6 up to T 0.3 on the continuation path,
  physical n_c = 0.5 always.
- Non-uniqueness at high T: Jacobian (18 x 18, central differences) has 6
  zero singular values (the u, v gauge) for T <= 0.1; the smallest others
  fall with T (2.9e-4 at 0.1, 7.5e-6 at 0.2) and about four more are zero at
  T = 1. Two T = 1 solutions (jump from T = 0 vs continuation) have the same
  d 0.141837, E -0.820781, L_beta -2.059247 but Tr Delta 1.616 vs 1.499 and
  Sigma(i w_1) -0.10239i vs -0.10456i, no decoupled mode: thermodynamics
  unique, the pole parametrization not.
- Item 33: B = 5, U = 2.4 converges (1e-15 at T = 0.005, 0.01, 0.1; 3.5e-13
  at 0.03), E -1.2619 vs B = 3 -1.2615 (T = 0.005), same d. Not yet
  checked: the item 24 two-site-cell stall (needs projectors).

Step 3b-i (refactor only): `BaseLatticeSolver` (ABC, `solve_lattice_base.py`)
holds setup, flatten/project, `_target_function`, `solve`, `Z`, and
`one_cycle`: bands, `update_weights` (abstract), rho_qp and lopsided ke, then
`update_bath` (abstract, sets D and Lambda_c), `solve_impurities` (shared;
passes `mu` to `set_h_imp` only when `self.mu` is not None), `update_qp`
(abstract, sets f1, f2, Lambda, R). `LatticeSolver` keeps the analytic T = 0
steps and the core-orbital placement. `ImpuritySolver.set_h_imp` takes
`mu=None`. All 110 tests unchanged. Next (3b-ii): `ThermalLatticeSolver`
with the approach-B fits, Fermi weights at fixed beta, mu.

## Item 26 notes (different ghost counts per block)

Step 1, literature (papers only):
- Giuli et al., arXiv:2603.20559, Eq. 16, and Lanata's ghost DMET (PRB
  2023), Eq. 63 of its supplement: the selection rule N(n) - N(Gamma) = m_i
  for each fragment i, on the total numbers of the fragment (all blocks), so
  the embedding holds B nu_i - m_i electrons (M_i - m_i with M_i quasiparticle
  orbitals). m_i is any integer, each a different variational ansatz.
- Ghost DMET, main text: different assignments of fermions give different
  variational states, "among which the physical solution is the one
  resulting in the lowest variational energy"; previous work found it is
  generally the half-filled one, nu_i (B + 1) / 2, "integer for odd B".
- Every paper read uses one odd B for all orbitals. The molecules paper
  (arXiv:2403.05157) uses 2 and 4 ghosts per orbital, i.e. B = 3 and 5. None
  uses an even B or a different B per block.
- Lanata, g-RISB operatorial formulation (PRB 105, 045111 (2022),
  arXiv:2109.10320): B_i nu_i auxiliary f modes per fragment, "B_i >= 1 is a
  given integer" (Eq. 6), then B_i = B assumed for simplicity. Bosons
  Phi_{Gamma n} only for N(n) - N(Gamma) = M_i (Eq. 8), M_i = nu_i (B - 1) / 2
  (Eq. 11) chosen so the mean field is g-GA; "different choices B and M_i
  would be also possible, in principle". The reformulation is exact for any
  B (Eqs. 23, 29, 39). Embedding state (Eq. 63) = |Gamma> x U_PH |n>, so it
  holds B nu_i - M_i electrons; Eq. 11 is half filling. Only the number of f
  modes of the fragment and M_i enter, not the block structure (mine).
  Mine, from Eqs. 13, 15, 28: every physical state needs a non-empty S_i^Gamma
  and nonzero D, C, i.e. 0 <= M_i <= (number of f modes) - nu_i.
- "Generally" / "usually" (ghost DMET; arXiv:2603.20559) is empirical: the
  physical choice of M_i is the lowest-energy one, half filling is what
  previous work found, with no proof that it always is.
- risb's code: bath_size is per block (a multiple of the block size), so a
  mix like t2g B = 2, eg B = 3 is accepted. Gaps when nu_i + M_i is odd:
  FockitImpuritySolver takes len(modes) // 2 (floor) and
  LatticeSolver.n_ghost_electrons sums (M - nu) / 2 (a half-integer), which
  disagree (the floor sector has m = ceil((M - nu) / 2)).
- Step 2a: test (test_latticesolver) two half-filled Hubbard orbitals (U =
  2.4, Bethe, no coupling) in one cluster with B = 3 and B = 1: each matches
  its own cluster, Z and docc to 1e-10 in the prototype (Z 0.16424 / 0.49980).
  Only particle-hole symmetric: away from it the total-N constraint of the
  shared embedding is weaker than one per orbital.
- Step 2c: how-to/lattice_solver.md ghost section: mixed B, even ghosts per
  cluster and per spin (t2g 3 + eg 2 per orbital: 8 per spin; 2 + 3: 7).
- Even B (mine): M_i in [0, (B - 1) nu_i]; one band with spin, B = 2: M = 0,
  1, 2 give N_emb = 4, 3, 2; half filling (3) has no Sz = 0 state, M = 0 or 2
  do. So even B with spin symmetry needs a non-half-filled M_i.
- Odd ghosts per spin, checked (one band, Bethe, U = 2.4, beta 200 smearing,
  thermal impurity beta 2000): the parity is right (2 degenerate states for
  B = 2 and 4), but "no paramagnet" was too strong. B = 2: converged (1e-12),
  ghost levels +-0.918, none at mu, Z = 0, docc 0.0258, energy -0.0549, an
  insulator where B = 1 (Z 0.4998, E -0.0364) and B = 3 (Z 0.1642,
  E -0.0615) are metals; at U = 1.5 B = 2 does not converge (error 2e-2).
  B = 4: the B = 3 solution plus a level at mu with |R| 0.02 and occupation
  0.5 that carries the odd spin (Z, docc, E the same as B = 3).
- B = 5 with the thermal solver (beta 2000) fails at cycle 11 (sector 924):
  E - E0 = 0, then 4.64057e-3 three times and 4.64059e-3 (3-fold exact plus
  one 2e-8 away), 9.28e-3. On that H (seed 0 v0, ncv 40): eigsh k = 2 does
  not converge (maxiter, 7.5 s); ncv 80 converges. k = 4 "converges" but
  returns 0, 4.64e-3 x 2, 9.28e-3: it misses a degenerate copy, silently
  (seeds 1-5: 4 of 5 right). Single-vector Lanczos sees one vector per
  degenerate eigenspace in exact arithmetic, so the thermal solver's
  eigsh path can drop degenerate states without an error.
- B = 5 levels: ground state, then an S = 1 triplet (exact 3-fold) and a
  singlet 2e-8 above it.
- Options measured (thermal, dim 924 at beta 2000; 2-orbital Kanamori B = 3,
  dim 12870, beta 200 / 40 / 20 with 3 / 13 / 68 states): current doubling
  fails / 0.33 / 0.76 / 4.8 s; deflation one state at a time 0.08 / 0.31 /
  1.77 / 13.8 s (beta 10 still running after 25 min: too slow at high T);
  doubling + ncv retries + deflated check 7.2 / 0.45 / 0.90 / 5.3 s; dense
  0.02 s at 924, 108 s at 12870; lobpcg 0.48 s at 924 (tolerance warnings).
- Done (user: dense below a size, current doubling above, with bounded
  retries): thermal solves diagonalize sectors <= 2000 densely; _eigsh
  retries twice with doubled ncv (capped at dim) then raises; after the
  doubling (and for degenerate="warn", k = 2) _add_missed_states shifts the
  found states up and adds the lowest state of the shifted H while it is
  below the highest found, switching to dense once the states are a quarter
  of the sector (bounded, no arbitrary cap: a highly degenerate level can
  have more missed copies than eigsh found). Test: one band B = 6 (3432),
  rng 34 inputs, eigsh = eigh to 1e-10; fails without the change (11 vs 12
  states). A miss depends on rounding: the same inputs printed to 17 digits
  (1 ulp off) are not missed, and an exactly degenerate model (two decoupled
  zero-energy baths, 20 states) is found by plain eigsh.
- B = 5 after the fix: no crash, but no solution with any impurity mode
  (ground state 999 it error 0.58; beta 200 error 0.05; beta 2000 error
  0.33). It is the B = 3 solution (levels +-1.0665, 0, same |R|, energy
  -0.06152) plus two ghosts per spin with R = 0 at +-0.0014, occupations
  0.57 / 0.43: decoupled ghosts, item 33.
- Step 2b (user: option 1): LatticeSolver raises if a cluster has an odd
  number of ghosts, FockitImpuritySolver if its number of modes is odd;
  n_ghost_electrons is an int. Choosing M_i (non-half-filled embeddings,
  even B in a paramagnet) is not done.
- Spin (mine): with equal spin blocks, half filling per spin needs
  nu_s + M_s even, i.e. an even number of ghosts per spin. t2g B = 2 (3
  ghosts) + eg B = 3 (4 ghosts) gives 7 per spin: the half-filled embedding
  has an odd number of electrons, so no Sz = 0 state. t2g B = 3 (6) + eg
  B = 2 (2) gives 8, fine. One band with even B has B - 1 ghosts per spin,
  odd, the same problem.

## Item 24 notes (ghost RISB: literature and how hard)

Sources (papers only, no code read):
- Lee, Lanata, Kotliar, PRB 107, L121104 (2023), arXiv:2212.07515: T = 0
  g-RISB equations (their Eqs. 2-10) are RISB's with R (B nu x nu),
  lambda, lambda_c (B nu x B nu), D (B nu x nu), bath of B nu orbitals.
  B = 3 already gives energy and double occupancy within 1% of DMFT.
- Giuli, Lee, Yao, Kotliar, Ruckenstein, Gingras, Lanata, arXiv:2603.20559
  (2026): ghost-GA -> DMFT as B -> infinity. Embedding in the sector with
  (B + 1) nu / 2 electrons (their Eq. 48; B odd). Finite T: L_beta = Omega_qp
  + sum Omega_emb - sum Omega_0emb, grand-canonical traces, with a new
  quadratic embedding H_0emb (ghost f coupled to bath b). Reduces to the
  T = 0 functional as beta -> infinity.
- GEM paper, arXiv:2609.32633 (2026): finite T done by fitting thermal
  density matrices (with a movement penalty); at T = 0 no fitting, the
  usual saddle-point equations. Linear mixing 0.05 in its examples. ED only.
- Giuli et al., arXiv:2607.27156 (2026): the iterative (DMFT-like) scheme
  is fragile in the paramagnetic Mott phase (ad-hoc recipes, fail in a
  field); direct minimization of the energy functional is robust there.
- arXiv:2608.00254 (2026): g-RISB for Sr2RuO4 t2g with SOC.

risb against the T = 0 equations:
- Helpers already rectangular-safe in the cycle: get_d uses the lopsided
  h0 R^dagger (no inv(R)); get_r, get_f1 use _sqrt_k pseudo-inverses.
  inv(R) only in get_d2 (unused) and helpers_triqs.get_sigma_w.
- solve_lattice: qp space (sum B nu_i + uncovered) differs from the band
  space, so R_full is rectangular and upfold (same space both sides) needs
  a qp-side layout. Initial R = 1, Lambda = h0_loc must change: R = [1; 0]
  with degenerate ghost levels is a fixed point equal to plain RISB, so
  the start must split the ghost levels. Core treatment and edge orbitals
  need revisiting.
- Filling: from Eq. 10 (bath density = 1 - Delta) and Eq. 48,
  <n_c> = Tr Delta - (B - 1) nu / 2 at self-consistency, so the qp
  n_target shifts by (B - 1) nu_i / 2 per cluster (derived, not checked).
- fockit solver: bath size = block size today; the half-filled sector
  (len(modes) // 2) already gives (B + 1) nu / 2 once the bath is B nu.
- Cost (B = 3, Sz = 0 half-filled): 1 orbital 36 states; 2 orbitals 4900;
  3 orbitals 853,776; Ni d shell 3.4e10 (ED impossible; NiO plain RISB was
  63,504).
- Finite T: item 15's Gibbs sum in the half-filled sector is not Lanata's
  finite-T functional (grand-canonical, with Omega_0emb).

## Item 24 step notes (g-RISB, slowly, step by step)

Steps: 1 prototype in the scratchpad; 2 bath size in the fockit solver; 3
LatticeSolver quasiparticle space (rectangular R_full, ghost count, filling
shift, ghost-splitting start); 4 tests, Sigma(w), core treatment, Lagrangian,
docs, one at a time; 5 finite T decided later.

Step 1 (scratchpad item24_ghost.py, nothing in risb changed): one band, Bethe
D = 1 (energies at equal-probability points of the semicircle, 4000, smearing
beta 200), half filling, risb helpers unchanged with R (B x 1), own 8-mode ED
embedding in N = B + 1, Sz = 0 (36 states). DIIS from R = [0.9, 0.3, 0.3],
ghost levels -1, 0, 1. Lee/Lanata/Kotliar Figs. 3-4 read off by eye:
- U = 2.4, B = 1: d 0.0732, E -0.0364, Ekin -0.2120, Z 0.500 (Brinkman-Rice
  0.500, d 0.0733).
- U = 2.4, B = 3: d 0.0534, E -0.0615, Ekin -0.1896, Z 0.164 (paper ~0.054,
  -0.061, -0.190, 0.17). 11 it.
- U = 3.5, B = 3: d 0.0109, E -0.0365, Ekin -0.0748, Z 0, R != 0 (paper
  ~0.0108, -0.037, -0.075). 18 it.
- U = 0.1, 0.5, 1.5, B = 3 converge in ~11 it; f1, f2 ~1e-11 everywhere.
- <n_c> = Tr Delta - (B - 1) / 2 holds to 4 decimals; Sigma(w) from Dyson is
  k-independent (spread ~1e-11); B = 3 energy below B = 1.
- U = 0 exactly, B = 3: no convergence (499 it, f2 0.1). The ghosts carry no
  weight there: a family of solutions with ghost bands fully full or empty
  (the core case, not handled for ghosts).

Step 2: FockitImpuritySolver(..., *, bath_size={block: n}) (default the
block size); gf_struct_bath holds the sizes and set_h_bath, set_h_hybr and
the rho_f, rho_cf loops use them. Half-filled sector unchanged in code
(len(modes) // 2). Test: two decoupled extra bath orbitals (levels +-10)
leave rho_c, rho_cf, rho_f of the one-bath problem unchanged (1e-12). Against
the step 1 prototype at U = 2.4, B = 3 (fed its converged D, Lambda_c):
rho_f, rho_cf agree to 4e-15; sector 70 states without Sz. Suite 106 passed.

Step 3a (layout "C, padding only R"): the quasiparticle space of a block of
h0_k is its orbitals then the ghosts of the clusters (order of clusters and
their blocks). helpers.get_ghost_projectors(h0_k, bath_size, projectors,
gf_struct_mapping) gives P_qp,i = [P_i, 0; 0, 1] (M_i x n_qp, k-dependent if
P_i is). upfold now sizes a block by the projectors (or A) mapped into it,
else by h0_k. R padded to M x M with zero columns, upfolded with outside=1,
then R_full[..., :n_bands] is the rectangular R of ghost RISB, used with the
unpadded h0_kin_k (no padded eps stored: about 1.8x the lattice memory
otherwise; padded matmuls ~15% slower on the NiO-like 8 bands, B = 3).
Tests: no ghosts gives the projectors back (k-independent and k-dependent);
two-site cell, site 0 with B = 3, site 1 uncovered: eigenvalues of H_qp equal
the ghost form, also in a basis rotating with k. Moving the ghosts one slot
fails it. Suite 109 passed.

Step 3b: LatticeSolver(..., *, bath_size=[{block: M}]) (M a multiple of the
block size, else ValueError; default no ghosts); S.n_ghost_electrons = sum
(M - n) / 2, for the user's n_target. _projectors_qp from
get_ghost_projectors when there are ghosts (else the projectors). R padded,
upfolded with _projectors_qp, sliced to the bands; Lambda_full with
_projectors_qp; get_rho_qp with P_qp; get_ke(h0R, vec, w, P, projector_qp)
(new optional argument, right side). _flatten/_unflatten: square blocks keep
the Hermitian basis (now cached by size), rectangular blocks by entries;
_unflatten takes like= for shapes; round trips are _project. Start (own
guess, not from a paper): R = [sqrt(1 - 0.18) 1; sqrt(0.18 / (B - 1)) 1;
...] (B = 3: 0.906, 0.3, 0.3 as the prototype), Lambda = 0 on the orbitals
(as without ghosts) and the B - 1 copies at levels linspace(-W, W) with W the
kinetic half bandwidth. Not yet ghost-aware: Z (= R^dag R), Sigma(w),
get_density_matrix_k, get_lagrangian, docs. Suite 109 passed (B = 1
unchanged); one ghost cycle runs with the right shapes, Tr Delta 1.5.

Step 3c (scratchpad item24_3c.py): LatticeSolver with bath_size, Bethe
D = 1, 4000 k, smearing beta 200, Sz = 0 impurity sector, paramagnetic
symmetrizer, impurity beta 200. Same E, d, Ekin as the step 1 prototype to
5 decimals at U = 0.5 (-0.30977), 2.4 (-0.06153), 3.5 (-0.03651); R differs
by a ghost rotation, Lambda eigenvalues equal. Start scan (ghost weight
0.05, 0.18, 0.5 x level spread W/2, W, 2W x U = 0.5, 2.4, 3.5): all 27
converge in 7-16 it to the same E, d (err <= 7e-11). 200 k changes E by
1e-5.
- Without Sz = 0 and the paramagnetic symmetrizer, U = 3.5 broke spin
  symmetry (impurity n_c 0.989) and ended at R = 0 with the core treatment
  pushing the ghost levels to +-10.96 (core moves started when two Delta
  eigenvalues hit 1). Turning the core treatment off did not help (R = 0,
  f2 = 1). With the symmetries it converges.
- In the Mott phase the impurity ground state is doubly degenerate even in
  Sz = 0 (a quasiparticle band sits at mu, the zero eigenvalue of Lambda):
  the ground-state solver returns one of them (n_c 0.506 / 0.494, E and d
  unchanged); beta = 200 averages them to 0.5000.
Test test_ghost_risb_gives_the_published_energy_of_the_hubbard_model (fockit
only, 200 k, U = 2.4 and 3.5): self-consistency < 1e-8, n_c = 0.5, d and E
within 1e-3 of Lee/Lanata/Kotliar Figs. 3-4 (read by eye). tests/test_common
bethe_h0_k(n_k). Suite 111 passed, 2 skipped.

Answers after 3c (U = 3.5, B = 3, 1000 k): the paramagnetic symmetrizer is
what makes it converge; Sz = 0 alone still ends at R = 0 (impurity polarized
with the bath compensating). Symmetrizer alone: right solution (d 0.01093).
The symmetrizer does not reach rho_c: one_cycle symmetrizes R, Lambda, D,
Lambda_c, rho_qp, rho_f, rho_cf, f1, f2 but not rho_c, so S.rho_c is the one
degenerate ground state the solver returned (n_c up/dn 0.013/0.987 without
Sz = 0, 0.463/0.537 with). Not ghost specific; rho_c feeds double_counting
and get_density_matrix_k. Fix proposed: symmetrize rho_c too (not done).

Step 4a: S.Z kept as R^dag R, documented as the spectral weight of R^dag G_qp
R over all frequencies (Lanata et al. PRB 96, 195126 (2017), supplemental
Eq. 49): the quasiparticle weight without ghosts, the sum rule with ghosts (1
if R is an isometry; "almost equal to 1 for all U" there). Here 0.969 at
U = 2.4 and 0.982 at U = 3.5 against Z 0.164 and 0. The quasiparticle weight
with ghosts goes with Sigma(w) in 4b. Docs build with -W passes.

rho_c fix (user, before 4b): one_cycle now projects and symmetrizes rho_c
with rho_f and rho_cf. Plain RISB showed it too: the three-orbital Kanamori
Mott test with symmetrize_blocks had S.rho_c spin traces 3.0/0.0 (AtomDiag)
and 1.15/1.85 (fockit). That test is now
test_three_orbital_kanamori_is_a_paramagnetic_mott_insulator and asserts
rho_c up = dn; it fails for both solvers without the fix.

Step 4b: arXiv:2601.16401 (Makaresz, Gingras, Lee, Lanata, Powell, Nourse)
Eq. 11 = Sigma(z) = z + mu - eps0 - [R^dag (z + mu - Lambda)^-1 R]^-1 (App. A,
Woodbury). App. B pole form Eq. B7 has a sign slip in B3-B7: the R^dag
Lambda R term and the pole term are + (numerically: printed - off by 0.6, +
agrees with A7 to 2e-16; + also gives B10 and Lanata 2017 Eq. 51). Neither
appendix gives Z; from B7, 1 - Sigma'(0) = S^-1 (S + B D B^dag) S^-1, D =
diag(1/(mu - e_j)^2), so Z = S - B [D^-1 + B^dag S^-1 B]^-1 B^dag (local, no
upfolding: Sigma is k-independent). Matches a finite difference to ~1e-10;
Bethe B = 3: Z 0.1643 (U 2.4), 0 (U 3.5, both ghost poles at mu).
helpers.get_sigma_w (was get_self_energy)(omega, R, Lambda, mu, h0_loc) in the pole form (Eq.
11 directly is singular at every eigenvalue of Lambda, e.g., a real mesh
point at w + mu = lambda gave 0.11 error without ghosts); helpers.
get_z (was get_quasiparticle_weight)(R, Lambda, mu); private _self_energy_poles (ghost
subspace from the SVD, as scipy null_space, whose stubs reject complex).
helpers_triqs.get_sigma_w uses get_self_energy (old formula to 3e-14 on
real and Matsubara meshes, both code paths). Tests: Green's function at
every k with ghosts; no ghosts linear incl. w + mu at eigenvalues of Lambda;
Z = slope of Sigma (with and without ghosts); the published ghost test also
checks Z (0.17, 0). Flipping the pole sign fails 2 tests. S.Z docstring
points to get_quasiparticle_weight. Suite 115 passed, 2 skipped; docs -W ok.

After 4b (user): renamed to helpers.get_sigma_w and helpers.get_z (user's
names; helpers_triqs imports the module to avoid the clash with its own
get_sigma_w). Derivation recorded in docs/explanations/self_energy.md: G^-1
= g^-1 - eps (Woodbury), the pole form by block inversion (Schur
complement) in the basis of R's columns and the ghosts (gives B10, then B7
gauge-free), Z by differentiating and Woodbury, R^dag R as the sum rule, and
several clusters. Paper errors located (rendered page 11): step B2 -> B3
keeps Woodbury's minus where Sigma = z - [..]^-1 flips it to plus (B3 as
printed off by 0.61, with + 9e-16); B3 also lacks a dagger (F R S^-1 R F ->
F R S^-1 R^dag F); B4-B6 carry the minus (off 0.61); B7 has - on S^-1 R^dag
Lambda R S^-1 (should be +) but + on the pole term (off 1.1 as printed);
A1 has H_qp = R eps R^dag - Lambda, but A3, A6, A7, Eq. 11 use + Lambda.
Correct: A2-A7, B1, B2, B10 (3e-16). Lattice self-energy with several
clusters is the upfold of each cluster's Sigma_i with the physical
projectors: checked with two clusters (2 + 1 orbitals, B = 3), k-dependent
projectors, an orbital in no cluster: Dyson vs R_full^dag G_qp R_full 8e-15.
Pass S.h0_loc_matrix for Sigma_i - V_dc (lattice Dyson) or h0_loc - V_dc for
Sigma_i. Eq. 11 directly vs the pole form only differs on real frequencies
without broadening at or near eigenvalues of Lambda.

Step 4c: helpers.upfold_r_lambda(R, Lambda, h0_kin_k, projectors_qp,
gf_struct_mapping) -> (R_full, Lambda_full) (pad R, upfold, slice), used by
LatticeSolver and get_lagrangian. get_lagrangian reads the bath sizes from
R's shapes (no new argument) and adds mu * n_ghost_electrons: without it L =
Omega - mu n_g (checked: mu 1.2, n_g 2, L - Omega = -2.400000).
get_density_matrix_k and the entropy helpers needed no change (docstring:
R_full may be quasiparticles by orbitals). Tests with ghosts (two-site cubic
cell, site 0 B = 3, site 1 in no cluster): L stationary, density matrix
local block = rho_c^T and trace, band + interaction = L + mu N + TS,
entropy = -dL/dT. Removing the mu n_g term fails the energy test only.
- Stationarity finite difference: a ghost Delta eigenvalue near 0 or 1
  (0.0011, 0.9961) makes sqrt(Delta (1 - Delta)) steep; the derivative
  scales with step^2 (-2.3e-3, -2.3e-5, -2.3e-7 for 1e-4, 1e-5, 1e-6; 6e-9 at
  1e-7), so the helper's step is now 1e-7 (was 1e-6).
- Thermal impurity solver bug (not ghost specific): the dense switch was
  dim <= 2k + 1, so eigsh ran with ncv 65 of a 70-state sector and ARPACK
  error 3 appeared; now dense once dim <= 2 (2k + 1) (Krylov space more
  than half the sector), as the comment says.
- Finite T with ghosts (for step 5): two-site cubic cell, U = 2, fixed mu =
  1, B = 3 on site 0. beta 40 converges (err 8e-15, S_imp 0.007); from it,
  beta 30, 20, 15, 12, 10 all stall (err 0.2 to 0.06) with the ghost levels
  collapsed onto mu (0.98, 1.00, 1.03) and S_imp ~1.7 to 1.9; from the
  default start beta 10 stalls too, also with a ground-state impurity
  (Tr Delta 2.08 vs Tr rho_f 1.49). The (R, Lambda) recursion stops where the
  sqrt-gradients at Delta and rho_f agree, which does not force Delta =
  rho_f. So the ghost entropy test runs at T = 0.025. Our finite T (Gibbs
  sum in the half-filled sector) is not Lanata's finite-T ghost functional.

Step 4d: examples/ghost_hubbard.py: one band, Bethe D = 1,
B = 1 and 3, U 0.25 to 4 in steps of 0.25, plots Z (helpers.get_z) and the
double occupancy; Sz = 0 sector, paramagnetic symmetrizer,
kweight.n_target = 1 + S.n_ghost_electrons after building S. With smearing
beta 200 and 1000 k the B = 3 metal ended at U ~ 2.7 and B = 1 at 3.25
(Brinkman-Rice Uc 3.39): the quasiparticle bandwidth ~ Z is then comparable
to T = 0.005. With beta 2000 and 4000 k: B = 3 Z -> 0 at U ~ 2.86 to 2.88
(Lanata 2017 ghost-GA Uc2 ~ 2.88, Uc1 ~ 2); B = 1 Z(3.25) = 0.0834
(Brinkman-Rice 0.0836). Runs in 3 s.
- DIIS bug (not ghost specific): NewtonSolver.solve did not clear the
  history (x, g_x, error) of the previous call, so in a scan with one
  LatticeSolver each solve extrapolated with iterates of the previous U. B =
  3 at U = 3.0 took 999 iterations (residual 6e-4) after U = 2.75, against
  25 from a fresh solver; the example took 18.6 s, now 3.0 s. Fixed: solve
  clears it. Test: a second solve with the same solver does the same
  iterations as a fresh one (failed for DIIS before).
- After the fix tests/test_examples.py::test_hubbard_example fails at U =
  4.5: the scan (steps of 0.5) now lands on Z = 0.1020 instead of the stored
  0.2681. Both are solutions (residual ~1e-12, mu 2.25, N 1); Omega
  -2.269989 (Z 0.2681, fresh start) < -2.267644 (Z 0.1020). With steps of
  0.25 or 0.1 the scan stays on Z = 0.2681. User: steps of 0.25. Done:
  examples/hubbard.py U steps 0.25; test reference regenerated (41 points,
  Z = 0 from U = 4.75; old points agree to 2e-6).
- tests/test_examples.py::test_ghost_hubbard_example: B = 3 Z and double
  occupancy at U = 0.25 to 4 against the run (d 0.01093 at U = 3.5, Lee et
  al. Fig. 4 ~0.0108).
- How-to: "Ghost quasiparticles" section in docs/how-to/lattice_solver.md:
  bath_size for the solver and LatticeSolver, the n_target shift with
  S.n_ghost_electrons, symmetrizers in the Mott insulator (Sz = 0 alone is
  not enough), a ghost with R = 0 stays decoupled (checked through
  LatticeSolver: Z 0.5003, d 0.0733 at U = 2.4, the B = 1 values), S.Z vs
  helpers.get_z, helpers.get_sigma_w, links to the explanation and the
  example. No finite T, no cost (user). 4f dropped (item 27 removes
  AtomDiag).

Citations (user: these two only): Lanata, Lee, Yao, Dobrosavljevic, PRB 96,
195126 (2017) (ghost-GA) and Lanata, PRB 105, 045111 (2022) (g-RISB), in
the README citation block (included by docs/about.md), the about page's
literature section, the how-to ghost section (footnotes), and LatticeSolver's
bath_size docstring (inline links; named link targets after Parameters are
not found by docutils). Makaresz et al. stays where it is used: the
self-energy page and get_sigma_w/get_z, whose docstring now cites Eq. B7
(B10) for the pole form it evaluates, and that B7's second term is printed
with a minus.

Step 4e: the step 1 non-convergence at U = 0 was the prototype (no core
treatment). Through LatticeSolver (example setup, Bethe, beta 2000, 4000 k)
B = 3 at U = 0 converges in 10 it (err 3e-14): the ghost bands are fully
empty and full (Delta eigenvalues 0, 1), the core treatment puts their
levels at +-10.96, and the result is exact (Z 1, d 0.25, E -0.4244125, as B =
1). At U = 1e-4 and 1e-3 the default start also decouples the ghosts (the B
= 1 solution, E -0.42416253 at 1e-3); warm-started from U = 0.01 the
ordinary ghost solution (levels +-0.675) exists there too, E -0.42416254
(lower by 1e-8, d 0.249917 vs 0.249926). From U = 0.01 the default start
gives the ghost solution. Test (user):
test_ghost_risb_without_interactions_is_the_non_interacting_solution (Bethe,
200 k, U = 0, B = 3: Z = 1, d = 0.25, Ekin = -4 / (3 pi)). Step 4 done.

## Item 29 notes (review comments and docstrings)

Reviewed every comment and docstring in the uncommitted diff (ghost work).
Rewritten: the ARPACK comment (debugging history) in fockit.py; _GHOST_WEIGHT
and the ghost start comment (narrated a test) in solve_lattice.py; the
"X by Y" shape shorthand in get_lagrangian, get_density_matrix_k, upfold,
upfold_r_lambda, get_ghost_projectors, get_sigma_w, get_z ("with a row for
each quasiparticle and a column for each orbital"); get_z's Notes and
_self_energy_poles's comment; two test comments (one overclaimed: "no
solution the cycle reaches" -> "does not converge"); a ragged paragraph in
the how-to. Restored the commented-out lopsided_ke_qp line that the rename
had rewritten. Then the committed migration work (9f62a29..26e0868):
- Source: "The ground state is found in this basis, then written in basis"
  (broken); "Bounds the bands of R h0_kin_k R^dag, because Z = R^dag R <= 1"
  (unproven, not Z with ghosts); the mu bracket comment in
  _from_triqs_hartree; a stray ")" in get_h_qp's R; a ragged paragraph in
  get_lagrangian.
- Tests: "(textbook Anderson acceleration with 3 vectors needs 84 iterations
  here)" (a measurement); the thermal Mott test comment claimed the
  decoupled bath orbitals are half filled: they are not (rho_f eigenvalues
  0.44, 0.56; rho_qp 0.45, 0.55; residual 0.11). Its ground-state version
  converges too, in 171 iterations instead of 12.
- Docs: impurity_solvers "This is can help", "does not converge" without
  beta (it can be slow: 171 vs 12 iterations in that test), "Away from a Mott
  insulator ... the same as the ground state" (too broad);
  green_functions: the no-ghost self-energy and "only describes the
  quasiparticles" now say "without ghosts" and point to the explanation (the
  TRIQS quasiparticle Green's functions with ghosts are untested, item 28);
  "Hubbad" typo in the self-consistent tutorial.
- Left: commented-out old code in tests/test_impurity_solvers.py (hubb_N)
  and examples/decorated_honeycomb_trimer_basis.py (an impurity_solvers
  line). Item 29 done.

## Item 28 notes (helpers_triqs)

Audit: group 1, TRIQS-operator helpers (get_C_Op, matrix_to_Op,
get_h0_loc_blocks, get_h0_loc, get_gf_struct_from_g): unused, with fockit
twins in helpers_fockit; deleted (user). Group 2, Green's function helpers:
only one cluster covering the whole cell (R and Lambda of the cluster used
directly, no projectors, clusters or ghosts), slow duplicate code paths;
replaced by numpy helpers (user): helpers.get_g0_k_w(omega, h0_k, mu),
get_g_qp_k_w and get_g_k_w(omega, R_full, Lambda_full, h0_kin_k, mu) (from the
eigenvectors of H_qp, private _resolvent), get_g_w_loc(g_k_w, projectors,
gf_struct_mapping) (a list per cluster, the k average of P g P^dag).
Tests: the old TRIQS test ported (Sigma independent of the quasiparticle
basis, Dyson = R^dag G_qp R locally); G with ghosts, two clusters, k-dependent
projectors and an orbital in no cluster = Dyson with the upfolded Sigma; the
local G does not depend on a basis that rotates with k (a missing conjugate
in get_g_w_loc fails it). Example green_functions.py in numpy: Matsubara G
both ways (agree to 2e-15), spectral function on w + i 0.05, weight 0.9028
vs Tr R^dag R 0.9057 (test 1e-2). Found: S.R_full / S.Lambda_full are from
the start of the last cycle, ~1e-8 behind S.R after solve (docs now say so;
the example uses upfold_r_lambda(S.R, S.Lambda, ...)). How-to rewritten,
with a checked snippet that fills a TRIQS BlockGf. Removed: helpers_triqs,
its test and API page, environment.yml, CI's TRIQS job (docs build moved to
the main job), micromamba in CD's docs build, the triqs mypy override,
intersphinx and type mappings, README's optional TRIQS, test_docs's TRIQS
skips, TRIQS links in helpers_fockit (now literals). The TRIQS citation
too (user): nothing in risb uses TRIQS, so it is gone from the how-to,
CITATION.bib and CITATION.cff; the how-to only says the arrays are easily
converted to TRIQS Green's functions. 103 passed.

## Item 23 notes (DFT+RISB helpers)

- User: risb does not provide double-counting formulas or a CSC loop; it
  provides hooks/helpers for another workflow (~/GitHub/spincrossover,
  examples/nio/vasp, run_csc.py). That workflow: one-body.h5 per DFT cycle
  (hamiltonian (k, nb, nb) per spin in the band basis, projectors (k, n, nb),
  occupations (k, nb), weights, total_energy, num_electrons); the DMFT step
  returns ChargeCorrection = Delta N(k) = rho(k) - diag(f_DFT(k))
  (`embedding.density_correction`, rho from TRIQS Gf.density()) and energy
  components total = E_DFT - E_band^DFT + E_band^DMFT + E_int - E_dc
  (`triqs/_dmft.py::_energy_components`). NiO dp PLO file: 4096 k (uniform
  weights, full 16^3), 8 bands per spin (Ni 3d rows 0-4, O 2p rows 5-7),
  projectors unitary at every k (6e-15), 14 electrons in the window.
- Decisions (user): double counting is a LatticeSolver callable
  `double_counting(i, rho_c_i)` returning a potential per block, subtracted
  from cluster i's local terms in the impurity problem each cycle; the
  k-resolved density matrix is returned in the f(h) convention
  rho[a, b] = <c^dag_b c_a> (TRIQS / Lowdin, what Delta N(k) takes); NiO
  check in the scratchpad only.
- Convention (user asked why): Lanata defines [Delta^p]_ab = <f^dag_a f_b>
  (PRL 118, 126401 Eq. 11); ghost DMET (2023) suppl. Eq. 53: Delta_ij =
  <a^dag_i a_j> = [f(h)]_ji; Giuli et al. 2026: "the transpose of n_F(h)".
  risb follows it (get_rho_qp takes .T of f(h); rho_c[alpha, beta] =
  <c^dag_alpha c_beta>; the Lagrangian's Tr(lambda^T Delta^p)). TRIQS /
  Lowdin / Green's functions use the matrix of the density operator,
  rho_ab = <c^dag_b c_a> = f(h)_ab = G_ab(0^-). Both standard; transposes;
  equal for real problems.
- Implemented: `LatticeSolver(..., double_counting=callable)`: called each
  cycle as double_counting(i, rho_c) (rho_c of cluster i from the previous
  cycle, rho_qp in the first), the result subtracted from the cluster's
  local terms in its impurity problem; `LatticeSolver.rho_c` is now filled
  each cycle (it was declared but never set); `R_full`, `Lambda_full`
  documented. Helpers: `get_density_matrix_k(*, R_full, vectors, kweights,
  rho_c, projectors, gf_struct_mapping)` = R^dag n_qp R plus, on each cluster,
  P^dag (rho_c^T - its coherent local part) P, in the f(h) convention;
  `get_band_energy(h0_k, density_k)` = sum_k Tr[h0_k rho(k)] / n_k;
  `get_total_energy(*, dft_total_energy, dft_band_energy, band_energy,
  interaction_energy, double_counting_energy)`.
- Tests: the density matrix's local block on each cluster is rho_c^T and its
  trace the filling (complex 2-orbital; 2-site cell with a site in no
  cluster); band energy + <H_int> = L + mu N + T S (same two models; fails
  without the local correction, and the complex one with the transposed
  convention; the local blocks are real so the density test alone cannot see
  the convention); a constant double counting equals shifting the local
  energies in h0_k (both solvers), and is called every cycle with the cluster
  index and each block. Suite 105 passed.
- NiO check (scratchpad `item23_nio.py`): VASP dp PLO one-body.h5, Ni d
  rows 0-4 as one cluster (k-dependent projectors, O p in no cluster,
  check_projectors passes), 14 electrons, Kanamori U = 6.2, J = 1.1, nominal
  FLL (N0 = 8) through the double_counting hook, paramagnetic, ground-state
  impurity in N_up - N_dn = 0 (63504 states; the full 184756-state thermal
  run took minutes per cycle). DIIS 7 it, 24 s, self-consistency 1e-9.
  Delta N(k): (4096, 8, 8) per spin, Hermitian, sum of traces -2e-9.
  Energies (eV): DFT total -12.379, DFT band 62.101, RISB band 66.851,
  interaction 161.778, double counting 160.400, DFT+RISB total -6.252.
  Physics not meaningful: Z ~ 0.993 and 9.78 Ni d electrons, because the
  nominal FLL (average U = Slater F0) was paired with Kanamori (U' = 4.0,
  average ~4.5): the potential 42.65 eV over-subtracts by ~13 eV. A real run
  needs the workflow's Slater interaction and its double counting.
- Problem of that run (user asked): the workflow's U = 6.2, J = 1.1 are
  Slater (F0, J = (F2 + F4) / 14); I built a Kanamori interaction with them.
  FLL (made for Slater, average interaction F0 = 6.2) then over-subtracts
  ~14.5 eV against Kanamori's mean ~3.76 (Held's U), sinking the Ni d shell.
- Rerun with the workflow's own Slater tensor (spincrossover.coulomb
  .slater_tensor, chemist [ij|kl], Wannier90 cubic order, F4/F2 = 0.63;
  spherical average back to F0 6.2, J 1.1), H_int = 1/2 sum [ij|kl]
  c+_is c+_ks' c_ls' c_js, and its spin-averaged FLL at N0 = 8 (potential
  42.65, energy 160.40): DIIS 11 it, 67 s, self-consistency 1e-8, mu 9.138;
  Z eig 0.567 (x2), 0.668 (x3) (paramagnetic metal in RISB); Ni d occupation
  8.248; Delta N(k) (4096, 8, 8), Hermitian, traces sum to -7e-9; energies
  (eV): DFT band 62.101, RISB band 63.523, interaction 171.062, double
  counting 160.400, DFT+RISB total -0.296 (DFT -12.379). No DMFT result on
  disk to compare (data/dp has csc, crpa, plo only).
- Open (fockit?): the first Slater run raised "h_imp does not conserve a
  diagonal symmetry" with np.int64 mode labels (indices from np.nonzero);
  casting them to int fixed it. A small np.int64 term alone commutes fine, so
  the mechanism is not pinned down.

## Item 20 notes ((R, Delta^p) as the input of the cycle)

- Theory (mine): cycle (R, Delta^p) -> lambda from rho_qp(R, lambda) =
  Delta^p (inversion), D, Lambda_c as now, impurity -> rho_f, rho_cf, new R
  = get_r(rho_cf, rho_f), new Delta^p = rho_f. Fixed points are the RISB
  solutions (lambda condition exact by the inversion, Lambda_c condition is
  the update). Inversion well posed at T > 0 for Delta^p strictly in (0, 1)
  (Delta^p = dOmega/dlambda^T, Omega concave in lambda: a Legendre
  transform); not at an edge (lambda -> -+inf) or at T = 0 with a gap. Only
  lambda - mu enters; with mu fixed by the filling an inconsistent Tr
  Delta^p has no exact inverse, so the inner problem is a least-squares fit
  (as in DMET). Covariant. At R = 0 lambda is set by Delta^p, removing the
  lambda -> lambda degeneracy of the (R, lambda) cycle (untested).
- Literature: Lanata et al. PRB 85, 035133 (2012), arXiv:1108.0180:
  "minimization only in the variational density matrix, in the spirit of the
  Levy and Lieb formulation of DFT", R "a fixpoint" at fixed density; their
  2015-2017 papers use the (R, lambda) root problem. DMET fits its
  correlation potential to density matrices by least squares; Wu et al. PRB
  102, 085123 (2020), arXiv:2003.00873: the fit "is often hindered by the
  numerical stability", worst "when the effective single-particle system is
  gapless or nearly gapless".
- Prototype (scratchpad `item20.py`, inner least-squares inversion for lambda
  with mu fixed by n_target, DIIS on (R, Delta^p), start R = 1, Delta^p =
  non-interacting, lambda = h0_loc): converges to non-solutions. Bilayer:
  Z = 0, |rho_f - rho_qp| 1; complex U = 3: Z = 1, 0.5. (R, lambda) from
  the same start: 20 / 21 cycles, correct.
- Cause (trace `item20_trace.py`, bilayer): Tr rho_f is the impurity's
  charge (N_c + N_f = M), and Tr Delta^p went 2.00 -> 1.16 -> ... -> 0.05,
  fit error 0.2-0.5. On the lattice only lambda - mu enters and mu is fixed
  by the filling, so a constant shift of lambda shifts mu and leaves
  Delta^p unchanged: the inversion cannot fix lambda's overall shift. That
  shift sets the bath levels relative to the impurity, i.e. the impurity's
  charge; in (R, lambda) it comes from the stationarity update of lambda
  (uses the impurity), in (R, Delta^p) nothing fixes it. So (R, Delta^p) is
  underdetermined by one number (an embedding chemical potential) when the
  clusters cover the lattice. Cost: 6730-19528 lattice evaluations (finite-
  difference least squares) for ~110 impurity solves vs ~20 / ~20.
- Fix options: carry the shift as an extra variable updated from the
  stationarity condition (a hybrid close to (R, lambda)), or an inner root
  on the impurity's charge (impurity solves inside the inversion).
- Hybrid tested (user; scratchpad `item20_hybrid.py`): x = (R, Delta^p, s),
  lambda = lambda'(R, Delta^p) + s with lambda' the exact inversion at
  mu = 0 (hybr), Delta^p <- rho_f, s updated by the trace of the
  stationarity lambda. Same start (R = 1, lambda = h0_loc). Impurity /
  lattice evaluations, (R, lambda) vs hybrid: bilayer 20 / 20 vs 26 / 950;
  complex U = 3 21 / 21 vs 23 / 994; 3-orbital Kanamori U = 2 19 / 19 vs
  31 / 5255; all correct (|rho_f - rho_qp| <= 4e-10). Kanamori U = 3 Mott
  and the frozen shell: inversion fails (fit error 1; Delta^p at an edge /
  flat band at mu); complex Mott U = 5 with beta: no convergence in 500.
- Side finding for item 18: the frozen-shell model from lambda = h0_loc =
  diag(-2, 2) still ends at R = 0 (self-consistency 0.6) with the core
  treatment; default lambda = 0 and diag(0, 4), (-2, 4), (0, 2), (2, 7)
  converge. Different trigger: the half-filled orbital's bath level (~2) is
  far below its impurity level (~6 = crystal field 2 + Hartree 4), so
  cycle 0 empties it. Open.

## Item 19 notes (projectors that depend on k or leave orbitals uncovered)

- Literature: PRL 118, 126401 supplement (arXiv:1606.09614): epsilon_k spans
  the whole single-particle space ("contains also the uncorrelated
  orbitals"), energy sum_kij R_i eps_k,ij R_j^dag; R = 1, lambda = 0 on
  uncorrelated orbitals is not stated explicitly there. Lanata et al. PRX
  2015 (arXiv:1407.4862) main text has no formalism.
- Maths: R(k) = sum_i P_i(k)^dag R_i P_i(k) + (1 - sum_i P_i(k)^dag P_i(k)),
  lambda(k) = sum_i P_i(k)^dag lambda_i P_i(k) (no outside term: uncorrelated
  states have no constraint; their local terms stay in h0_kin_k). Downfolding
  (rho_qp, ke, h0_loc) already broadcast over a k-dependent P.
- Implemented: `helpers.upfold(A, h0_k, projectors, gf_struct_mapping,
  outside)` used by LatticeSolver.one_cycle (R with outside = 1, Lambda 0)
  and get_lagrangian; `_dagger` (conjugate transpose of the last two axes) in
  get_h0_loc_matrix, get_h0_kin_k, get_h_qp, get_h0_kin_k_R so P and R can
  depend on k; check_projectors checks orthonormal rows and no overlap at
  every k (coverage no longer required); docstrings (projectors,
  check_projectors, h0_kin_k holds the local terms of orbitals in no cluster,
  get_h_qp, get_lagrangian) and the how-to.
- Tests: check_projectors rejects overlapping / not orthonormal / not
  orthonormal at one k, accepts partial coverage and k-dependent projectors;
  an orbital in no cluster keeps its band (2-site cell: site 1 in no cluster
  = site 1 a U = 0 cluster; fails without the R = 1 term); k-dependent
  projectors give the same solution (random unitary at every k on h0 and P;
  Z, Lambda to 1e-8); the Lagrangian is stationary with an orbital in no
  cluster (fails with a Lagrangian missing the R = 1 term). Suite 96 passed
  (74 passed, 27 skipped without TRIQS).

## Item 17 notes (decorated honeycomb in the site basis)

- `examples/decorated_honeycomb.py` renamed (git mv) to
  `decorated_honeycomb_trimer_basis.py`; new
  `decorated_honeycomb_site_basis.py`: same mesh and parameters, h0_k in the
  sites (-tk between the sites of a triangle, -tg e^{ik d_a} from site a of
  one triangle to site a of the other), U sum_a n_a,up n_a,dn, blocks
  [("up", 3), ("dn", 3)], projectors rows of the identity, no
  gf_struct_mapping, C3 as `permutation_symmetries=[(c3_rotation, 1)]`
  (a -> a + 1 on physical and bath blocks), symmetrizer paramagnetic + C3
  average (P A P^T over the three rotations), `force_real=True` (time reversal:
  E2 = E1^* in the trimer basis is real circulant matrices in the sites).
- Verified (scratchpad `item17_verify.py`, both at tol 1e-10): 81 it each,
  self-consistency error 2e-11 / 3e-11; Z, lambda, rho_c eigenvalues, mu and
  the impurity E0 agree to 1e-12 (Z 0.5912 x2, 0.8080; lambda 1.7913, 4.0374
  x2; mu 4.0866; E0 -15.0221753). C3 eigenvalue 1 holds the ground state at
  all 82 solves (<= 9e-14); w^1, w^2 0.349 higher. The Gamma-centred 18 x 18
  mesh is C3 symmetric (C3 is the integer matrix [[-1, 1], [-1, 0]] in reduced
  coordinates).
- Tests: the trimer test renamed; new site-basis test asserting the
  eigenvalues of Z and Lambda equal the trimer example's values (shared
  constants). How-to impurity_solvers points to both examples.

## Item 16 notes (multi-cluster and finite-T get_lagrangian)

- Projector up/down-folding (user asked to check first): R_full = sum_i
  P_i^dag R_i P_i (and lambda) up; rho_qp, ke, h0_loc down with P (...) P^dag;
  h0_kin = h0 - sum_i P_i^dag h0_loc_i P_i. Nothing else is upfolded.
  Assumptions (were unchecked and undocumented): P k-independent, orthonormal
  rows, no overlap between clusters, full coverage of h0_k (an uncovered
  orbital gets R = 0, lambda = 0: a flat band at 0). Supercell check: the
  one-site cubic Hubbard model (U = 4) as a 2-site cell, one cluster per
  site: Z identical to 12 digits (plain and renamed blocks), lambda to 2e-7.
  Done (user): assumptions documented (projectors docstring, how-to
  "Multiple clusters"), `check_projectors=False` option on LatticeSolver
  (ValueError: k-dependent, not covering, not orthonormal / overlapping;
  kagome and decorated honeycomb pass), tests: the supercell reproduces the
  one-site Z (both solvers, plain and renamed blocks), check_projectors
  rejects the three bad cases and is off by default. Suite 85 passed.
- Decisions (user): keyword-only per-cluster lists; finite-T via an entropy
  argument (fockit only), but "be careful".
- Finite T: the Gibbs-averaged impurity (`beta`) is not a consistent
  finite-T RISB. U = 0, complex 2-orbital: ground-state impurity gives
  Z = 1 and L = Omega_0 (exact) at every beta (2, 5, 10, 20, 40); thermal
  impurity gives Z = 0 (trap) at beta 2, Z = 0.937 / 0.975 at beta 5 (a
  genuine self-consistent solution, error 1e-13), 0.99989 at 10, 1 at 20;
  L = -1.5094 and L - S/beta = -1.5630 vs Omega_0 = -1.5555 at beta 5. So
  `beta` renormalizes the quasiparticles without interactions when T is
  comparable to the impurity's gaps; only safe for T far below them. The
  proposed -S/beta term would not fix it (the solution itself is off).
  Giuli et al.'s finite-T functional subtracts the free energy of an
  auxiliary quadratic embedding (Omega_0,emb), presumably to cancel this
  (not verified); Ayral et al. use a Bose form. A correct finite-T embedding
  changes the self-consistency equations, not only the averaging.
- User: RISB / slave bosons are known to have a problem with the entropy in
  the free energy; the criterion is that the definitions / maths are right
  (the finite-T Lagrangian's stationary points are the thermal cycle's
  solutions), not that it gives the physical entropy (check the literature on
  the known issue). Multi-cluster first; finite T afterwards.
- Known entropy issue (literature): Lanata, Deng, Kotliar PRB 92, 081108
  (2015), arXiv:1505.01951: earlier finite-T GA (their Refs. 18, 19, Wang et
  al. 2010; Sandri et al. 2013) is "underestimating the entropy using an
  entropy inequality", which "generates a few pathologies of the theory, such
  as giving a negative entropy at low temperatures"; they evaluate S not from
  the variational parameters but from dS = dE/T with S(T = 0) = 0. A search
  for a statement about Kotliar-Ruckenstein slave-boson mean field found only
  leads (slave-fermion mean field overestimating the entropy; free-energy
  fluctuation corrections that did not preserve the noninteracting limit).
- Bose vs Gibbs (user asked): Ayral's Eq. C3 treats the embedding
  eigenstates as bosonic modes with the one-boson constraint imposed on
  average by E_c (grand canonical): weights 1 / (e^{beta (E_n - E_c)} - 1),
  sum 1. Gibbs (what `beta` does) has exactly one boson: e^{-beta E_n} / Z.
  Equal at high T (small fugacity); at low T Bose needs e^{beta (E0 - E_c)}
  ~ 2 and gives excited states ~ (1/2) e^{-beta Delta} vs Gibbs e^{-beta
  Delta}; both the ground level (equal split if degenerate) at T -> 0.
- Implemented: `FockitImpuritySolver.entropy` (-sum w ln w of its weights;
  0 for the ground state); `get_lagrangian(..., embedding_entropy=...)`
  subtracts sum_i S_i / beta (Gibbs only; docstring Notes: the stationary
  points are the solutions with that ensemble; S_i defined as -sum w ln w;
  the summary says L's value at a solution is RISB's grand potential, -dL/dT
  at fixed mu RISB's entropy). Bug found in item 15's code: the k doubling for the thermal sum
  reached k = 32 on a 70-state sector and ARPACK failed (error 3); now dense
  once 2k + 1 >= dim. Tests: a 4-fold degenerate impurity at beta 1000 has
  entropy ln 4; the Lagrangian is stationary at a thermal metallic solution
  (complex 2-orbital U = 3, beta 20: Z 0.5367 / 0.6801 vs ground 0.5402 /
  0.6810, entropy 0.016, |f2| 2e-9). The entropy term is constant at fixed
  ensemble, so stationarity does not test it (passes without it too).
- Entropy helpers (user): `get_entropy_qp(energies, beta, mu)` (Fermi-Dirac,
  per unit cell, as the quasiparticle term of get_lagrangian),
  `get_entropy_impurity(weights)` (-sum w log w; FockitImpuritySolver.entropy
  uses it), `get_entropy(*, energies, beta, mu, impurity_entropy)` (their
  sum). get_lagrangian's argument renamed embedding_entropy ->
  impurity_entropy (user: "impurity"; a flat list, one float per cluster's
  impurity problem). Docstring summary: L at a solution is RISB's grand
  potential, minus its T-derivative at fixed mu RISB's entropy.
  DummyImpuritySolver has no entropy (use the referenced solver's).
- Thermodynamic test (user: small, fast, with partial coverage): one band,
  2-site cell, 4^3, U = 2, fixed mu = 1, thermal impurities at T = 0.1, a
  cluster on every site and on site 0 only: get_entropy = -dL/dT (central
  difference, dT = 1e-3 T) to 3e-7 (abs 1e-5 in the test); 0.05-0.11 s
  each; fails if the Lagrangian drops the impurity entropy. (Check of the
  check: complex 2-orbital U = 3 beta 20, F = L + mu N: -dF/dT 0.574883 vs
  0.574816.) U = 4 and U = 2 at T >= 0.2 collapse to Z = 0 or do not
  converge on this cell.
- Thermal solutions at beta 10 and 5 collapse to Z = 0 (|f2| 0.4 / 0.02,
  entropy 2.6 / 3.0); the ground-state impurity at beta 5 does not converge
  (999 it).
- Thermal Mott (complex U = 5, beta 40): max |dL| R 0, lambda 0, D 0,
  Delta_p 9e-10, Lambda_c 0.25 (= f2, the traceless atomic-limit residual;
  self-consistency 0.13). Reported, not asserted (user).
- Multi-cluster `get_lagrangian` done: keyword-only, per-cluster lists (as
  LatticeSolver stores them), `projectors`, `gf_struct_mapping`; R and lambda
  upfolded as in LatticeSolver. Stationarity test made a helper; new test on
  a 2-site cell with a cluster per site. The sites must differ in their
  hopping (site 0's y, z hopping x 1.3): a staggered on-site energy is local,
  goes into h0_loc, leaves h0_kin swap-symmetric, and then swapped projectors
  still pass. With the hopping difference, swapped projectors fail the test
  (R derivative of cluster 0). `one_band_cubic_h0_k` moved to test_common.
  Suite 86 passed (65 passed, 21 skipped without TRIQS).

## Item 18 notes (frozen quasiparticle orbitals)

- Derivation (mine, risb conventions, natural gauge Delta = diag(delta)):
  D equation rho_cf[alpha, a] = R[a, alpha] f_a and R equation
  Q[b, alpha] = f_b D*[b, alpha] with Q = avg_k n_k R eps_k. For a frozen b
  (delta_b = 0 or 1): n_k[b, :] = delta_b e_b at every k (a projector with a
  0 / 1 diagonal entry), so Q[b, :] = delta_b R[b, :] avg_k eps_k = 0 (the
  local part is removed from h0_kin_k), and a sharp bath occupation gives
  rho_cf[:, b] = 0. Both equations hold for any R[b, :], D[b, :]: they are
  not fixed by stationarity. A frozen full band can even disperse at no
  energy cost (sum over a full band of eps = 0). lambda_b is fixed only to a
  half-line (full / empty). The gradient's frozen-frozen divided differences
  (1 - d_i - d_j) / (f_i + f_j) diverge but multiply (D R^T) entries that are
  0 when R_b = D_b = 0; active-frozen entries are finite times 0.
  Natural choice: R_b = D_b = 0 (what a sharp impurity orbital forces).
- Exact edges need an exactly decoupled orbital. Model: 2 orbitals, crystal
  field -+2, cubic 10^3, Kanamori U = 2, J = 0.4, n = 3 (orbital 0 full).
  With inter-orbital hopping 0.2: 1 - delta saturates at 9.7e-4 as beta ->
  1280 (k-dependent orbital character of the full band), everything finite.
  With hopping 0: current code raises at beta >= 4 (ground-state impurity:
  rho_f eigenvalue -6e-18 in cycle 0, NaN in the gradient's sqrt).
- At beta = 3 the deep orbital has R ~ 1e-10, D ~ 3e-11 and delta = n_F(lambda
  - mu) = 0.76 of a flat level: same structure as the Mott state. In the
  condensed functional at T > 0 a decoupled pure bath is 0 / 1 while the flat
  level is a Fermi function, so f2 = 0 only as lambda_b -> -+infinity (a
  solution at infinity; matches DIIS's runaway lambda steps).
- Wrong reference check: Kanamori pair hopping (J) moves a pair from the full
  orbital into the other: R_0 = 0 exactly (parity of n_0 conserved) but
  delta_0 = 0.999, not frozen, and Z_1 0.92 vs one band 0.86. With J = 0 the
  mapping to one band (U = 2, Z 0.8519 / 0.8612 / 0.8615 at beta 10 / 40 /
  160) is exact.
- Prototype (scratchpad `item18_projection.py`): pseudo-inverse of K^(1/2) in
  get_d / get_r (frozen rows 0), sqrt K in get_f1, frozen-frozen divided
  differences 0, frozen lambda pushed to mu -+ 40/beta. Needed a wider mu
  bracket (brentq fails when mu sits on a flat level: SmearingKWeight brackets
  with [e_min, e_max], no margin). J = 0, beta 40:
  - linear mixing 0.2: a genuine solution (|f1| 5e-12, |f2| 3e-11) but not
    the expected one: the deep orbital stays a dispersive nearly-full band
    (Z 0.92, delta 0.999), the half-filled orbital collapses to Z = 0 (flat
    at mu, delta 0.501) with its fractional rho_f from an arbitrary state of
    a degenerate level (item 13).
  - DIIS: recursion error 0 but |f2| = 0.25: not a solution. For frozen
    orbitals the cycle maps lambda_b to itself, so the default recursion
    error can report convergence when the RISB equations fail.
  - The expected solution (orbital 0 frozen, orbital 1 = one band, Z 0.86)
    was not found by either.
- Literature: ghost DMET (Lanata 2023 supplement): natural orbitals with
  occupation 0 / 1 are "core", "decoupled from the fragment A and its bath
  B", so they have no bath partner. Giuli et al. 2026 App. B "assume that
  rho(1 - rho) is invertible" (the edge is excluded, not treated). Ghost GA
  (Lanata et al. 2017) solves the root problem in F1, F2 by quasi-Newton.
- The 0 x infinity gradient entry is not 0 in the limit: it is finite and
  sets the frozen bath level on the correct side (hybridized run, 1 - delta
  = 1e-3: G = -(Lambda_c + lambda) = 2.70). With it set to 0, Lambda_c_00 =
  -lambda_00 = -6 and the frozen orbital's bath came out half filled
  (rho_f00 = 0.5), pulling orbital 1 off the one-band answer. For an exactly
  decoupled orbital the condition is again an inequality (bath empty if
  Delta = 1, full if 0), so the core treatment fixes it explicitly:
  prototype sets Lambda_c = -+50 in frozen Delta directions (get_lambda_c).
- With the core bath: J = 0 two-orbital model, the one-band-equivalent point
  (R = diag(0, sqrt Z_one), lambda = diag(7 - 40/beta, 7)) is an exact fixed
  point (|f1| 3e-14, |f2| 9e-16, Z = 0.86117659 = one band). From the generic
  start DIIS finds it at beta 10 / 40 / 160 in 13-15 it (Z = one band to all
  digits, |f1|, |f2| <= 1e-11); at 640 it stalls just above tol at the right
  Z (k-mesh mu). Linear mixing 0.2 finds another solution.
- The other solution is regular and lower in free energy (F = L + mu N via
  get_lagrangian, beta 40): one-band-equivalent F 3.73510 (Z_1 0.8612,
  Delta eig 1.0, 0.5); other F 3.73021 (Z_1 0.9199, Delta eig 0.999, 0.501,
  rho_f not at edges, D != 0). In both orbital 0 has Z = 0, rho_c = 1. In the
  lower one physical orbital 1 couples to both quasiparticle orbitals (R
  column components 0.33, 0.90): the full orbital's spare quasiparticle
  orbital acts as a ghost orbital for orbital 1, lowering F by 0.005 and
  raising Z_1. (Earlier reading of this solution as "orbital 1 collapsed /
  degenerate-state artifact" was a mislabeling of eigenvalues.) The exact
  one-band mapping holds for the model, not for RISB's variational space.
- The two-orbital J = 0 thermal run raised (not investigated).
- Robustness (user chose; uncommitted in `src/risb/helpers.py`): `_sqrt_k`
  (sqrt of rho (1 - rho) from clipped eigenvalues, or its pseudo-inverse with
  0 for eigenvalues below `_EDGE_TOL` = 1e-12) in get_d, get_r, get_f1; the
  gradient's divided differences are 0 between two edge orbitals. No core
  bath, no lambda push. Suite: 78 passed.
  Decoupled full-shell model (scratchpad `item18_robust.py`, start lambda =
  diag(0, 4)): no more crashes. Linear mixing 0.2 reaches the regular
  solution at beta 10 / 40 / 160, J = 0 and 0.4 (Z_1 0.91-0.92, |f1|, |f2|
  ~ 1e-11). DIIS "converges" in 3 it to R = 0 with |f2| = 0.9 (with
  error_fun="root" it reports failure instead).
- Trap mechanism (`item18_trap.py`): the start puts orbital 1's bath (cost 4)
  below orbital 1 on the impurity (~6), so cycle 0 empties impurity orbital
  1 and returns R ~ 0; flat bands then give ke = 0, D = 0, rho_cf = 0, R = 0.
  R = 0, D = 0, any lambda is a fixed point of the recursion in every RISB
  problem (violating f2 unless the impurity is thermal / degenerate), and it
  absorbs. From lambda = diag(0, 6) both DIIS and linear mixing fall into it
  (|f2| = 1). Before the robustness change these paths crashed instead.
- Self-consistency check (user: name `self_consistency_tol`, uncommitted in
  `src/risb/solve_lattice.py`): after a self-consistent `solve`,
  `S.self_consistency_error` = max |f1|, |f2| from the last cycle, and a
  RuntimeWarning above `self_consistency_tol` (default 1e-6). User chose to
  warn on the full residual and explain the atomic limit in the message,
  docstring and how-tos (lattice_solver: after "Simple setup";
  impurity_solvers: temperature section). Tests: completely full orbital
  in the helpers (finite, decoupled rows, the rest unchanged), the warning on
  the decoupled full-shell trap (both solvers), the thermal Mott test expects
  the warning, the Kanamori Mott test ignores it (only fockit warns). Suite
  81 passed (62 passed, 19 skipped without TRIQS).
- DIIS check (user asked; scratchpad `item18_diis_check.py`): coefficients
  (bordered Gram system, sum c = 1, pinv), history alignment and restarts
  are right; the step is x = sum c_i g(x_i) (Anderson with beta = 1; `alpha`
  only in the plain step every n_period = 3 iterations, undamped by
  default; the first step is the plain recursion). With error_fun="root" the
  coefficients come from (f2, f1), not g - x: a Pulay-style heuristic. On a
  linear fixed point (n = 8) with full history it matches textbook Anderson
  exactly (9 it = n + 1, iterates to 1e-13). BUG: with history 3 it needed
  147 it vs 84: the iterates separate at |e| ~ 4e-8, where pinv's relative
  cutoff drops the Gram matrix (entries ~1e-16) next to the -1 border.
  Fixed: B scaled to max |B| = 1 before bordering (same c; guarded for B =
  0); 84 it. Test: DIIS is Anderson acceleration on a linear problem (n + 1
  with full history; history 3 within 100 it; fails without the fix). Suite
  88 passed, 9.7 s (was 14.5 s).
- mu bracket (user: first): `update_mu` brackets with e_min - 50/beta, e_max
  + 50/beta (was [e_min, e_max]: failed when mu sits at a flat level that
  is not half filled). Test: a flat band filled to 3/4 (all three smearing
  methods; Fermi mu = ln 3 / beta); fails without the fix. Suite 91 passed.
- Where the trap happens (user asked): only the decoupled full-shell model
  (2 orbitals, crystal field -+2, no inter-orbital hopping, orbital 0 a
  completely full band, U = 2, J = 0 / 0.4, n = 3, beta 40): every R = 1
  start (lambda diag(0, 4), (0, 6), Hartree (2, 7)) and every root finder
  (DIIS, DIIS root, hybr, Krylov, linear warm-up, restart on collapse) end at
  R = 0 with |f2| 0.3-1 and 2 instead of 3 impurity electrons. Not with 0.2
  inter-orbital hopping, not in the bilayer / complex controls, the 22
  Kanamori metals or the examples. The Mott runs end at R = 0 too but there
  it is the atomic limit (traceless residual). Cause: Delta of the decoupled
  full orbital is exactly 1 from cycle 0, the edge-safe gradient zeroes its
  0 x infinity term, so its bath level -lambda is misplaced, the impurity
  fills that bath and empties orbital 1. Not the start: DIIS started at the
  regular solution (or 5% off, or R = 1 with its lambda) collapses at once.
- Core treatment prototype (scratchpad `item18_core_full3.py`): for Delta
  eigenvalues at 0 / 1, get_lambda_c puts the bath level at -+50 (bath
  empty / full), and after each cycle that orbital's lambda goes to (min
  eigenvalue of lambda on the other orbitals) - W - 50 (full) or max + W + 50
  (empty), W the half-bandwidth of h0_kin_k (bounds R eps R^dag, Z <= 1).
  Needs no mu or beta; idempotent. (Targets relative to the qp band extremes
  failed: the frozen flat level is itself the extreme, so it recedes by 50
  every cycle; holding it once at the edge collapsed again.) Results: all
  three trap starts with DIIS 11-17 cycles (linear warm-up then DIIS 18-24)
  to the one-band-equivalent solution, Z_1 0.86118, self-consistency error
  <= 2e-11; bilayer and complex controls unchanged.
- Implemented (user: on by default, scale relative to the model, explained
  only in code comments): `LatticeSolver.one_cycle` puts the bath level of
  quasiparticle orbitals with Delta eigenvalue 0 / 1 at -+scale after
  Lambda_c is built, and their lambda below (above) the other orbitals'
  lambda by W + scale after the update; scale = 10 (W + largest |h0_loc| and
  |lambda| on the other orbitals), W the half-bandwidth of h0_kin_k
  (computed once). The first version took max |lambda| over all orbitals,
  which includes the placed level, so the scale grew ~10x per cycle and it
  collapsed; leaving the edge directions out fixed it. `helpers.EDGE_TOL`
  (1e-12) is now public (shared by helpers and LatticeSolver). Decoupled
  full-shell model, starts diag(0, 4), (0, 6), (2, 7), eigsh and eigh: 10-16
  it, Z_1 0.86118, self-consistency error <= 2e-11. Test replaced: the
  completely full decoupled orbital leaves the one-band solution (both
  solvers, vs a one-band solve on the same mesh; fails without the change).
  Suite 91 passed (70 passed, 21 skipped without TRIQS).
- Open for item 18: escaping the trivial fixed point R = 0 (the trap), the
  frozen-orbital core treatment as an option (prototype works: exact
  one-band-equivalent fixed point found by DIIS), and why ghost-like
  solutions win (variational freedom of the spare quasiparticle orbital).

## Item 15 notes (Mott regime)

- Residual at the thermal Mott states (found with the self-consistency check
  of item 18): complex 2-orbital, U = 5, beta 40: max |f1| 4e-14, max |f2|
  0.105, all in the traceless part (Tr f2 = 0: impurity filling 2 = n_target;
  rho_f eig 0.440, 0.560 vs Delta eig 0.455, 0.545). At R = 0 (D = 0) the
  cycle maps lambda to itself, so the recursion stops with f2 != 0; the
  non-unique lambda - mu below is this. 3-orbital Kanamori Mott test with
  fockit: 0.42, also traceless (filling 3 = n_target). User: RISB need not
  "converge" onto the Mott state in the Lagrangian sense; the Mott state is
  the decoupled bath (atomic limit, Z -> 0), where f1 / f2 do not change the
  energy or observables. So "converge" means the numerical procedure, and
  stationarity of the Lagrangian is a separate question. Contrast: the item
  18 trap (R = 0 where a metal is expected) has Tr f2 = -1 (impurity filling
  2 vs n_target 3), wrong observables.

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

- Resolved (step 8, items 13 and 15): degenerate ground states, with
  `degenerate="warn"` / `"average"` and `degenerate_tol`.
- Resolved (step 7): `how-to/lattice_solver.md` stops scipy's root finders on
  the residual (`options={"fatol": ...}`).
- Resolved: `ArpackNoConvergence` is raised (k = 1 has no partial result);
  `how-to/impurity_solvers.md` says so, and `method="eigh"` or
  `method_options` are the user's remedies.
- Resolved (step 3): the `N`/`S2`/Kanamori helpers live in `risb.helpers_fockit`.
- Moot (item 27): AtomDiag is removed.

## Loose ends notes

- Dead commented-out code removed: `hubb_N` in `tests/test_impurity_solvers.py`
  (inlined, its hopping comment and unused arguments dropped) and two lines in
  `examples/decorated_honeycomb_trimer_basis.py`.
- np.int64 mode labels (from item 23), reproduced with fockit 0.4.2: a
  3-orbital Slater-like h_int with labels from np.nonzero raises "h_imp does
  not conserve a diagonal symmetry", int labels solve. Mechanism (fockit):
  `fockit.algebra.rules.mode_sort_key` matches `case int()`, which np.int64
  is not, so ("up", np.int64(0)) sorts by repr after every int label. But
  np.int64(0) == 0 with the same hash, so OpSum's term dict and the
  lru_cache of `ordering._insert` treat them as one mode: the same operator
  gets two normal orders that do not cancel (c+_0 c+_1 c_1 c_0 vs
  c+_1 c+_0 c_0 c_1 left in [h, Q]). Fix belongs in fockit:
  `case numbers.Integral(): return (0, int(mode))` (np.integer is
  registered as Integral). Checked by patching it in at runtime: the int64
  model then solves, with the int model's ground-state energy
  (-2.96967843716710).
- Frozen-shell model (item 18, decoupled full orbital, U = 2, J = 0, beta 40,
  n = 3), rerun now: starts lambda = 0 (default), (0, 4), (0, 6), (2, 7),
  (-4, 4) reach Z_1 = 0.86118 (error <= 2e-11); lambda = h0_loc = (-2, 2)
  still ends at R = 0 in 3 iterations, |f1|, |f2| = 0.6, with the
  RuntimeWarning about unsatisfied equations. Closed (user): known and warned.
- DummyImpuritySolver has no `entropy`: `[s.entropy for s in solvers]` fails
  with Dummy clusters (a unitary rotation does not change the entropy, so
  the referenced solver's value is the right one). Done: `entropy` is in
  the `ImpuritySolver` protocol, and DummyImpuritySolver.solve copies it.
