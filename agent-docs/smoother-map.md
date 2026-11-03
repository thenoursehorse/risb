# Smoother self-consistency map

Goal: improve reliable convergence; measure success across starts and final
physical residuals, not only iteration count. Keep prototypes outside the repository.

## Scope

- Analytic thermal-fit Jacobians and directional penalties apply to
  ThermalLatticeSolver only. LatticeSolver has no thermal fits.
- T = 0 spectral cutoffs and full/empty pinning are prototyped below.
  Ground-state crossings and consistent continuation remain open.
- Directional Tikhonov regularization belongs inside both nonlinear
  least-squares fits, not DIIS. It penalizes motion along weak response
  directions more than well-determined ones.
- Freeze the penalty matrix during one fit so its objective and Jacobian
  agree. This does not remove gauge freedom or guarantee zero bias.
- Exact gauge rotations preserve physics. Small but nonzero response
  directions can change physics; an anchor can bias them. Check the
  unregularized bath and impurity matching equations and reduce the penalty.

## Running to-do

1. [x] Record baseline suite and fit/cycle measurements.
2. [x] Prototype analytic matrix-Fermi derivatives in both thermal fits.
3. [x] Check real/complex rectangular cases, repeated levels, cold and
   high-temperature weak responses; compare with finite differences.
4. [x] Run the existing suite with the analytic prototype.
5. [x] Prototype smooth directional penalties; compare with the existing
   isotropic anchor and a moving reference.
6. [x] Compare 30 starts per variant; sweep alpha/tau on known targets;
   report all matching residuals and physical changes.
7. [x] Record results, limitations and recommended next step.
8. [x] Add analytic derivatives and response-weighted anchoring as
   optional features, preserving current defaults.
9. [x] Prototype T = 0 spectral regularization; reject the tested forms after physical regressions.
10. [x] Prototype smooth full/empty pinning and a smooth near-edge step cap.
11. [x] Add clip/smooth step-cap methods; compare both defaults across the full suite.
12. [x] Prototype bounded corrections using the original matching equations; compare core choices.
13. [ ] Find a variant that improves reliability across starts without relying on a branch change.
14. [x] Review current line-search/trust-region literature and prototype T=0 safeguards.
15. [x] Prototype matching-residual models and test the published Anderson algorithms with g(x)-x.
16. [x] Add and evaluate iteration-scheduled safeguarded Anderson as an optional DIIS policy.
17. [ ] Investigate the separate direct-variational T=0 route.
18. [x] Reject BianLi budgets that cannot complete a guarded step; keep
   hard cycle limits and verify continuation with a valid budget.
19. [ ] Investigate thermal stopping with BianLi when matching residuals
   are small but g(x)-x remains large; distinguish parameter drift from
   failure to satisfy the matching equations.
20. [ ] Investigate smearing options: methods, beta and k-point convergence;
   check the three-orbital Mott case before changing defaults.
21. [x] Prototype AutoBZCore integration of density and kinetic matrices on
   general lattices; check filling, derivatives and T=0 self-consistency.
22. [x] Add general k-integration interfaces, weighted NumPy sums, an optional
   AutoBZCore adapter, and temperature extrapolation over supplied beta values.
   Migrate the lattice solvers and remove SmearingKWeight.
23. [ ] Check adaptive detection of small pockets and sampling guides more
   broadly before changing any integration default.
24. [ ] Validate T=0 extrapolation when quasiparticle energy scales shrink.
   Colder confirmation and count checks are implemented, but warm samples
   can still falsely pass. Require evidence that the sampled temperatures
   resolve the relevant energy scales.
25. [x] Implement midpoint-first mu search in BandKIntegrator with
   use_midgap and electron_count_atol; retain Brent fallback and report
   midpoint availability. Require colder confirmation for extrapolation.
26. [x] Allow linear temperature terms in matrix and mu extrapolation.
   Report mu_temperature_error separately from integral convergence.
27. [x] Check the final electron count after solving mu. Tighten Brent's
   tolerance and return converged=False when the count misses
   electron_count_atol, including at floating-point precision limits.

28. [x] Keep the small mu correction when evaluating cold occupations.
   BandKIntegrator refines mu when the initial count misses tolerance.
   Results and extrapolation retain mu_reference and mu_correction;
   the final count check remains required.
29. [ ] Address weak level response near the atomic branch: the tested
   density residual scales as R^2 but the level update scales as R^4.
   Compare physical-residual stopping and response scaling. The common-level
   charge-response prototype helps recursion DIIS but is not a consistent
   improvement; the first abrupt failure also exposes item 24.
30. [ ] Treat atomic ground-state and band degeneracies consistently.
   Validate the lower-energy, nearly singular-R root using physical spectra
   and matching equations rather than requiring R^dag R=0.

31. [x] Prototype direct T=0 filling on supplied weighted k-meshes.
   All tested square/Bethe B=5 starts match, including doped square cases.
   Cubic Mott remains sensitive to numerical degeneracy handling; keep
   items 13 and 30 open. This does not address adaptive AutoBZ integration.

## Results

Prototypes: /tmp/risb-smoother-map/prototypes-3tjc2hu3/thermal_fit_prototype.py; runner:
/tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py. The prototype phase made no production changes.

Baseline full suite: 212 passed, 111 subtests passed, one failure, 120.94 s.
The failure is
test_complex_two_orbital_model_reaches_the_mott_state_with_a_thermal_impurity:
the expected self-consistency warning is not emitted. Reproduces alone
before installing either prototype.

Independent complex response check against scipy.linalg.expm_frechet:
relative errors 5.9e-16, 1.8e-15, 1.8e-15 for sizes 2, 6, 10.

At a repeated zero level, the exact thermal response is -beta * field / 4.
Central differences with step sqrt(machine epsilon):
- beta 3: relative error 4.5e-9.
- beta 1e8: 9.3e-2 (step too large for the thermal width).
- beta 1e-8: 1.42 (response lost to rounding).
The analytic result is exact in these cases. Rescaling the finite-difference
step to 1e-4 / beta reduces its error to 4.7e-10 in all three cases.
These deliberately extreme examples establish a derivative advantage,
not a convergence advantage on realistic problems.

## References

- [GEM Appendix A](https://arxiv.org/html/2609.32633v2#A1): analytic
  matrix-Fermi derivatives and isotropic proximal penalties.
- Existing experiments: fockit-migration.md, Items 37–38.

## Prototype details

Both fits use the same matrix packing, residual equations and scipy
least_squares tolerances as production. The analytic derivative uses the
Daleckii–Krein formula, including the product-rule terms for rho_qpf @ D and
R.T @ rho_qpf. Divided differences use expm1 and separate particle/hole
Fermi factors to avoid cancellation. Residual/Jacobian calls share one
diagonalization.

The weighted penalty is generalized (directional) Tikhonov regularization:
for J = U diag(s) V^T at the start of a fit, append

    sqrt(alpha) V diag(tau / sqrt(s^2 + tau^2)) V^T (y - y_ref)

to the residual. Freeze this matrix for that fit; recalculate for the next.
Here y and the kinetic residual can be scaled by energy_scale (1 in these
benchmarks); R is dimensionless. alpha=1e-8, tau=1e-3 unless stated.

- isotropic: analytic Jacobian with production's existing anchor.
- directional: replace that anchor's isotropic norm with the weighted norm;
  OnPlateau still decides when to activate it.
- moving: use the starting parameters of each inner fit as its reference,
  including after OnPlateau fires. This is a distinct experiment.

The weak-direction weights are continuous; there is no rank cutoff.
They do not identify which weak directions are pure gauge. They also do
not guarantee zero bias, uniqueness, contraction or convergence.

## Controlled inner fits

Seven cases, both fits, three seeds: 42 fits per derivative method.
Real 3x1, complex 3x2, larger complex 5x2, repeated levels, beta=1000,
beta=0.01, and a large common energy offset (mu=1e5, beta=100).
Targets are generated exactly from known quadratic embeddings. Starts
perturb the rectangular matrix by 5% and the level matrix by a small
Hermitian field. Compare original unregularized density/cross residuals,
not fitted parameters (gauge nonuniqueness).

Median density evaluations per fit:

| Case | 2-point | 3-point | Analytic |
| --- | ---: | ---: | ---: |
| Ordinary real | 146 | 286 | 36 |
| Ordinary complex | 198 | 3583 | 44 |
| Larger complex | 976 | 322335 | 2842 |
| Repeated levels | 50 | 204 | 38 |
| Cold | 98 | 274 | 44 |
| Hot | 56 | 95 | 24 |
| Common energy offset | 128 | 9466 | 74 |

At a physical residual threshold of 1e-8, 2-point passes 40/42,
3-point 36/42, analytic 41/42. Specifically:
- Cold 2-point failure has residual 1.7e-3; analytic passes all six.
- Analytic is not always cheaper: the larger complex case is worse than
  2-point, and repeated-level analytic overhead can exceed FD savings.
- Common offsets hurt relative finite-difference steps. Reparameterizing
  relative to mu is another remedy; this adversarial case does not prove
  analytic derivatives are necessary.
- Tuned finite-difference steps recover the extreme hot/cold response
  accuracy. Default 3-point is not a reliable blanket improvement.
- Fits can report scipy success with nonzero physical matching error.

The 2/3-point controls use the same residual, packing and tolerances in the
prototype. Counts include finite-difference perturbations, which scipy's
nfev excludes. Timings from overlapping experiments are not used for
speed claims.

## Penalty bias check

One known-target 3x1 embedding, beta=0.05, reference at the perturbed
starting parameters. At alpha=1e-8, tau=1e-3:

| Physical matching error | Isotropic | Directional |
| --- | ---: | ---: |
| Bath fit | 3.28e-8 | 2.31e-9 |
| QP fit | 1.66e-7 | 1.07e-8 |

Sweep: alpha=1e-6, 1e-8, 1e-10, 1e-12; directional tau=1e-4, 1e-3,
1e-2. Reducing alpha to 1e-10 brings both errors to about 1e-9, near this
fit's stopping floor. Weighting reduces bias here; does not eliminate it.

## Suite results

- Baseline: 212 pass, 111 subtests pass, one existing T=0 warning failure.
- Analytic: 211 pass, 111 subtests pass, that same failure plus the thermal
  entropy derivative test at T=0.1. Its self-consistency error is
  1.2048584352442049e-11; the test requires <1e-11 (solve tolerance 1e-13).
- Directional: all 65 thermal/helper tests and 2 subtests pass, including
  that tight entropy test. Full directional suite: 212 pass, 111 subtests
  pass, only the same existing T=0 warning failure. A threaded run was
  stopped; the completed rerun used one BLAS/OpenMP thread.
- Ruff, Ruff LSP and strict mypy pass (all 53 Python files). No tests were
  loosened or changed.

## Outer-loop comparison

Cold starts, ten cases, three starts each: seed 0 default, seeds 1–2
independent 1e-3 real perturbations of R and Hermitian Lambda.
400 Bethe points; B=3 uses U=2, B=5 U=2.4; eta=40;
outer tolerance 1e-10, limit 300 cycles. Doped case has fixed mu=-0.5.
The complex control has two physical orbitals, B=1, U=3, J=0.6 and a
128-point complex two-band dispersion (see build_model).

Median cycles across the three starts:

| Case | Baseline FD | Analytic | Analytic + directional |
| --- | ---: | ---: | ---: |
| B=3, T=0.01 | 14 | 16 | 16 |
| B=3, T=0.1 | 12 | 40 | 40 |
| B=3, T=0.5 | 42 | 7 | 7 |
| B=3, T=1 | 29 | 28 | 29 |
| B=3, T=10 | 25 | 16 | 16 |
| B=5, T=0.005 | 15 | 16 | 16 |
| B=5, T=0.03 | 32 | 62 | 62 |
| B=5, T=0.1 | 18 | 21 | 21 |
| Doped | 9 | 13 | 13 |
| Complex control | 50 | 63 | 63 |

All three methods report convergence in 30/30 runs. Including the original
bath matching equations, errors are below 1e-8 in 26/30 baseline runs and
30/30 for both analytic variants. All 30 baseline runs meet 1e-6.
At B=3, T=10, baseline bath errors are 2.9e-8 to 4.7e-7;
analytic errors are 1.2e-13 to 3.5e-11.

This is not evidence of universal improvement: analytic derivatives help
warm B=3, hurt B=5 at T=0.03, and the weighted fixed anchor adds almost
nothing to this cycle matrix. The tighter suite test is a separate benefit.

The complex control has multiple stationary solutions. At seed 1 the
baseline gives Omega=-4.87546042195, analytic/weighted -4.81243447678.
Both satisfy the matching equations. Do not interpret this difference as
regularization bias or assume changing Jacobians preserves the basin.
For the nine one-band cases the largest baseline-to-analytic shifts are
5.6e-7 in double occupancy and 4.4e-9 in Omega. Compare physical quantities
separately from ghost parameters.

Moving reference: the driver converges in 27/30 runs; 23/30 have all
matching errors <1e-8, and only 22/30 meet both conditions. There is a
300-cycle failure at B=3, T=0.5, and bath errors around 3e-7 despite small
impurity matching errors. At B=5, T=0.03, double occupancy shifts by about 1e-4 while
bath errors remain 7e-8 to 2e-7. A moving anchor does not automatically
solve the physical equations at the driver's stopping point.

## Reproduce

Run from the repository root. For comparable threading, prefix commands
with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1.

    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py response
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py fits
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py penalties
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py suite --variant baseline
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py suite --variant analytic
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py suite --variant analytic --regularization directional
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py cycles --variant baseline
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py cycles --variant analytic
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py cycles --variant analytic --regularization directional
    uv run python /tmp/risb-smoother-map/prototypes-3tjc2hu3/smoother_map.py cycles --variant analytic --regularization moving

Cycle controls: --cases, --seeds, --maxiter, --alpha and --tau.
Environment: Python 3.14.8, NumPy 2.5.4, SciPy 1.18.1, fockit 0.5.3.
Raw JSONL measurements from this session: /tmp/risb-smoother-map/.
The prototype phase left production defaults and existing tests unchanged.

## Sequential timing check

After the overlapping experiments finished, seed 0, one BLAS/OpenMP
thread, one run per setting (indicative, not a repeated timing study):

| Case | Baseline | Analytic | Analytic + directional |
| --- | ---: | ---: | ---: |
| B=3, T=0.5 | 0.98 s, 42 cycles | 0.49 s, 7 cycles | 0.48 s, 7 cycles |
| B=5, T=0.03 | 6.91 s, 29 cycles | 12.97 s, 60 cycles | 12.94 s, 60 cycles |

## Prototype assessment

Analytic derivatives are correct and sometimes
improve fit accuracy, matching accuracy and runtime, but do not uniformly
improve convergence and introduce a tight-test regression on their own.
The directional fixed anchor reduces known-target bias and restores that
test, but adds little across the outer-loop matrix. The moving reference
is less reliable. The user wants the useful variants as opt-in features;
current defaults should remain unchanged.

Freezing weights only means solving a fixed objective during one fit.
It does not fix gauge redundancy or remove physical bias. Reducing alpha
and checking the original matching equations are validation steps, not
guarantees supplied by the weighting formula.

## Scope correction

No additions to benchmarks. Prototypes are archived outside the repository
at /tmp/risb-smoother-map/prototypes-3tjc2hu3.
API parameters belong in docstrings/autodoc. The how-to stays short;
experiment results and development findings belong only in this note.
Moving-reference regularization is a separate experiment, not part of the
proposed analytic-derivative and weighted-anchor features.

## Feature API

Implemented selectors: density_matching_jacobian ("2-point" / "analytic") and
density_matching_regularization ("uniform" / "response-weighted").
The regularization changes the geometry of the existing fixed anchor;
it does not activate a penalty before OnPlateau anchors the solve.
Both choices pass from ThermalLatticeSolver.__init__ to update_bath and
next_r_lambda, then the two existing public matching helpers and _fit.

Moving reference means using the starting parameters of each inner solve
as its penalty target, rather than a fixed target captured once.
This remains a separate experiment.

Test scope revised by the user: one analytic-versus-finite-difference
regression case. Use a small known complex rectangular embedding from the
prototype, with identical inputs, starting parameters and regularization.
Exercise both matching helpers on that one model. Compare reconstructed
densities/correlations and original matching residuals, not gauge-dependent
parameter entries. Existing tests remain on their default path; no new
outer-loop solves, parameter grids or separate response test. The earlier
six-case proposal is withdrawn. No separate weighted-regularization test cases were added.

## Implementation

- Added get_fermi_density_response and optional analytic Jacobians to both
  public thermal matching helpers.
- Added uniform/response-weighted fixed-anchor geometry. The weight matrix
  is frozen per inner solve and uses the selected derivative method
  for the starting response.
- ThermalLatticeSolver forwards density_matching_jacobian,
  density_matching_regularization and density_matching_regularization_scale
  (default 1e-3). Defaults remain 2-point and uniform.
- API details are in docstrings/autodoc; the how-to has only a short example.
- No moving-reference feature, optimizer changes or benchmark additions.
- Added exactly one test:
  test_analytic_density_matching_agrees_with_finite_differences.
  One complex 3x2 embedding checks both public matching helpers against the
  finite-difference result and known physical target. Runtime: 0.03 s.
- Reviewed against principles and testing philosophy. Ruff, Ruff LSP and
  strict mypy pass. Full suite: 213 passed, 111 subtests passed,
  one unchanged pre-existing T=0 Mott-warning failure, in 84.18 s.

## Separate equations from solver choices

Both thermal matching helpers now provide the residual equations and a
factory for their analytic evaluator. They do not branch on the selected
Jacobian or anchor weighting. The factory builds the matrix perturbations
only if the shared solver needs them.

_fit selects the method, caches the analytic evaluation and applies the
anchor penalty. DensityMatchingJacobian and DensityMatchingRegularization
are defined once in types.py and reused by both helpers and the lattice
solver. Another numerical method changes that shared selection and its
option type, rather than both sets of physical equations.

The public options, defaults and single comparison test are unchanged.

## Coupling audit before the selected-provider refactor

Behavior at the time of that review: response-weighted anchoring worked with either public
Jacobian option. With 2-point, least_squares still uses finite differences,
but the anchor weights come from one analytic Jacobian at the start of
each anchored inner solve. Without an anchor no weights are built.
The weighting formula needs a Jacobian, not specifically an analytic one;
the analytic dependency is a policy in _fit.

Issues identified in that review:
- _fit requires an analytic-evaluator factory even for residual-only
  finite-difference solves. Derivative acquisition and penalty construction
  could use independent callables.
- Each matching helper computes its residual values both in residual()
  and in its analytic evaluator. Reuse one definition of those values.
  Analytic derivatives still necessarily follow the physical equations.
- The ordinary and response paths each reconstruct the Fermi density
  matrix. Keep that convention and calculation in one shared primitive.
- Diagnostics recover physical residuals by removing len(y0) penalty
  entries. That assumes the penalty has one row per parameter. Evaluate
  the original physical residual, or retain its length explicitly.
- New tuning parameters still pass through the lattice constructor,
  stored attributes, both calls and both helper signatures. A small
  options object could reduce this plumbing if more controls are added.
- Existing OnPlateau starts anchoring and changes the stopping error to
  parameter changes. Anchor activation and physical convergence checks
  are separate responsibilities; changing anchor behavior should not
  silently change what counts as a solution.

The last refactor removed duplicated solver choices from both physics
helpers, moved caching and penalty-Jacobian wrapping into _fit, and shared
their option types. It did not eliminate every dependency above.

## Selected derivatives and shared density calculation

Completed the four requested changes:
1. _select_matching_derivatives selects the residual, solver Jacobian and weight
   Jacobian once. Finite-difference solves retain SciPy's 2-point steps;
   their anchor weights use the same two-point difference rule, including
   the actual representable step. They never build the analytic evaluator.
   Analytic solves share a cached residual/Jacobian evaluation.
2. Each matching helper defines its residual values once in get_matching_residual().
   Both the ordinary and analytic paths call it.
3. _fermi_density is one shared calculation of the density matrix and
   eigensystem. get_h0_emb_densities and get_fermi_density_response use it.
   The response reuses that eigensystem; this adds no second diagonalization.
   _extract_embedding_densities defines the density blocks and bath-hole
   convention once. Caching remains separate in the selected evaluator.
4. Diagnostics call the original physical residual at the returned
   parameters. They no longer infer its length from penalty dimensions.

API docstrings now state that anchor weights use the selected derivative
method. No options grouping or OnPlateau changes were made.

Validation: the one agreed comparison test passes unchanged. All 70
relevant existing tests and 2 subtests pass. Ruff, Ruff LSP and strict mypy
pass. The same known complex embedding was also checked with weighted
anchoring under both derivative methods; its reported physical errors
agree with errors calculated directly from the density matrices.
No tests or benchmark files were added for this refactor.

## T = 0 prototypes (2026-11-01)

Prototypes only: /tmp/risb-smoother-map/zero_temperature.py and
/tmp/risb-smoother-map/zero_checks.py. No production changes, new suite
tests or benchmark additions.

### Spectral regularization

Tried two functions of K = n (1 - n):
- Floor: f(K) = sqrt(K + epsilon), inverse 1 / f(K).
- Taper: f(K) = K / sqrt(K + epsilon), with damped inverse
  K / (K + epsilon)^(3/2). This inverse is deliberately approximate.

Each gradient differentiates its corresponding forward function using
stable divided differences. The taper's divided difference is
(1 - n_i - n_j) (1 + epsilon / (a_i a_j)) / (a_i + a_j),
where a_i = sqrt(K_i + epsilon). It stays finite at repeated levels.
Finite-difference gradient checks: about 3e-8 relative error for the
real floor examples, 9e-8 for a complex repeated-level example, and
7e-8 for the taper. These are experiment checks, not added tests.

The tapered forward function and damped inverse do not multiply to one:
their product is K^2 / (K + epsilon)^2. Dropping them into the existing
closed-form R update therefore makes its fixed point incompatible with
exact matching for nonzero epsilon and correlations. A residual-correction
update would be a different experiment; tuning this taper alone is
insufficient. The floor keeps that inverse identity but changes the
exact-edge constraints.

Neither version is ready for production:
- The floor alone breaks exact-edge controls. Combining it with smooth
  pinning helps the custom controls but fails the existing pinned
  full-orbital regression: active Z changes from 0.86117659 to 0.91988525.
  The no-pinning negative control also changes its wrong-filling result.
- Floor epsilon = 1e-8 can leave original matching errors above 1e-8
  despite the modified equations converging. At epsilon = 1e-10 the
  custom controls pass, but reducing epsilon to 1e-12 leaves one of
  three noninteracting starts unconverged; 1e-14 leaves all three
  unconverged within 200 cycles. Pinning width was 100 epsilon in this
  sweep, so it does not isolate the source of that sensitivity.
- The taper preserves zero at exact edges but is worse overall: only
  one of three full-orbital starts and none of three metal starts
  converge within 200 cycles, with or without smooth pinning.
- Custom full-orbital controls use dense impurity diagonalization and
  force_real=True; the existing regression uses the usual solver options.
  Passing the custom controls did not establish equivalent behavior.

### Smooth pinning and step cap

For occupation spread K, use p = 1 - smoothstep(K / width), with the
quintic smoothstep clipped to [0, 1]. Replace eigenvector-by-eigenvector
diagonal changes with a matrix blend:
A Lambda A + V diag((2 p - p^2) target) V^dag, A = V diag(1 - p) V^dag.
This also damps couplings to pinned orbitals. Quasiparticle targets use
smooth bounds of A Lambda A; log-sum-exp replaces switching active
subspaces and hard extrema.

The step cap uses a tapered near-edge matrix Q and step
Q (Lambda - Lambda_in) Q. Multiply that step by
limit / sqrt(limit^2 + ||step||_F^2), rather than switching a spectral-norm
cap on at a threshold. Edge pinning and the cap can be tried separately.

Direct prescribed-matrix probes cross the old thresholds:
- Bath pinning: the old output jump remains 15 as the perturbation shrinks.
- QP pinning: the old jump remains 19.5.
- Near-edge cap: the old jump remains 3.
- The smooth versions' differences tend to zero in each probe.
These probes establish local continuity of those operations, not the
whole interacting map. The hard spectral cutoff remains with pinning alone.

### Convergence comparisons

Same starts per variant; default start and two 0.01 perturbations;
outer tolerance 1e-9, limit 200 cycles; one BLAS/OpenMP thread.
Model controls: full-orbital charge transfer, B=3 Hubbard metal and Mott
state, B=5 ground-state impurity, and B=3 noninteracting limit.
The criterion is optimizer convergence and original f1/f2 errors < 1e-8.

| Case | Baseline cycles | Smooth pinning cycles |
| --- | --- | --- |
| Full orbital | 14, 11, 12 | 14, 11, 11 |
| B=3 metal | 11, 14, 14 | 11, 14, 14 |
| B=3 Mott | 17, 14, 14 | 17, 14, 14 |
| B=5 ground state | all reach 200 | all reach 200 |
| B=3 noninteracting | 7, 7, 7 | 2, 3, 3 |

Both pass 12/15 starts. Smooth pinning does not rescue B=5 and can increase
its residual. Widths 1e-10, 1e-8 and 1e-6 retain convergence and matching
physics on the full-orbital and noninteracting controls.

Known two-orbital B=3 quarter-filling regression, 2000 Bethe points,
beta=2000, U=3, J=0.45, default start and seeds 1, 2, 4 with noise 0.05:
baseline cycles 26, 26, 24, 40; smooth pinning 20, 24, 26, 25.
All converge; energy differences are below 7e-10. At seed 4, cap-only
takes 25 cycles, edge-transition-only 40, and both 25. The speedup in
this case comes from the step cap. This is no improvement in success rate.

### Existing tests and assessment

Six existing physical cases: full-orbital pinned/unpinned, noninteracting
ghosts, published Hubbard metal/Mott results, and complex Lagrangian
stationarity. Baseline: 6 pass. Smooth pinning alone: 6 pass in 0.61 s.
Floor plus smooth pinning: 2 fail, 4 pass; floor alone: 3 fail, 3 pass.
No assertions or tolerances were changed. Ruff, Ruff LSP and strict mypy
pass for both external prototype files.

The smooth cap is the most promising feature candidate. The full/empty
transition removes a real switch but has no demonstrated general
convergence benefit. Reject the two spectral prototypes in their present
form. More smoothness alone did not improve reliable convergence here.

Reproduce from the repository root, with OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1:
- uv run python /tmp/risb-smoother-map/zero_temperature.py continuity
- uv run python /tmp/risb-smoother-map/zero_temperature.py cycles
- uv run python /tmp/risb-smoother-map/zero_temperature.py cycles --spectrum-model taper
- uv run python /tmp/risb-smoother-map/zero_temperature.py cycles --cases quarter --variants baseline,pinning --noise 0.05 --maxiter 400
- uv run python /tmp/risb-smoother-map/zero_temperature.py cycles --cases quarter --variants pinning --pinning-parts cap --seed-start 4 --seeds 1 --noise 0.05 --maxiter 400
- uv run python /tmp/risb-smoother-map/zero_checks.py matrix
- uv run python /tmp/risb-smoother-map/zero_checks.py scales
- uv run python /tmp/risb-smoother-map/zero_checks.py suite --variant pinning

Raw records: zero-starts.jsonl, zero-scales.jsonl, zero-taper.jsonl,
zero-quarter.jsonl, zero-quarter-starts.jsonl, zero-cap-only.jsonl,
zero-edges-only.jsonl and zero-pinning-widths.jsonl under
/tmp/risb-smoother-map/.

## Combining the thermal features with the T = 0 prototypes

These are different recurring update maps. ThermalLatticeSolver uses the
analytic/finite-difference density fits and optional Tikhonov anchors.
LatticeSolver uses the square-root updates and full/empty pinning; it
does not call those thermal fits.

One overlap: ThermalLatticeSolver.update_bath uses get_d and get_lambda_c
to seed the first bath fit when no previous bath parameters exist.
Spectral changes could therefore change the thermal starting guess.
They do not smooth its recurring fitted map. ThermalLatticeSolver never
calls FullEmptyPinning. No combined convergence improvement has been
established; the previous experiments must not be described as testing
all these features together in one loop.

## Residual corrections and core alternatives (2026-11-01)

Prototype: /tmp/risb-smoother-map/residual_correction.py.
Goal: bound the correction without replacing the matching target.

For S = sqrt(rho (1 - rho)), use
G_epsilon = V diag(K / (K + epsilon)^(3/2)) V^dag, K = n (1 - n).
The R correction is
R_next^T = R^T + (rho_cf - R^T S_f) G_epsilon(rho_f).
Using rho_f follows the current get_r algebra; once f2 = 0 it is
equivalent to the original f1 equation using rho_qp. The original f1/f2
remain the stopping residuals. An exploratory rho_qp correction was also
tried, but is a different update even without damping and was not used
for the final comparison.

The D correction is
D_next = D + G_epsilon(rho_qp) (ke^T - S_qp D).
D is explicit in the outer vector; it is not taken from hidden previous
fits. The expanded root vector includes the original bath equation,
scaled only for units. One baseline cycle seeds D; evaluation counts
include that extra impurity solve.

S, its gradient and the level equations are unchanged. Lambda_c is still
computed algebraically from current D, R, Lambda and rho_qp; Lambda still
uses the current get_lambda update. This is a partial smoothing of the
updates, not a removal of all edge and level discontinuities. G_epsilon
vanishes at exact edges, so physical residuals must be checked directly:
a small preconditioned correction alone is not convergence.

Both core policies were tried:
- zero: retain the current choice of zero R/D components at the edge;
- keep: let residual corrections retain those undetermined components.
The existing choice is a control, not a requirement or a claim of optimality.
User explicitly permits changing existing choices when that improves the
whole procedure. No variant is rejected just for differing from zero.

### Results

Same five models, three starts, tol 1e-9, 200-cycle limit, epsilon 1e-6,
existing DIIS settings, original f1/f2 and bath errors < 1e-8.
The following comparison uses the zero-core control.

| Case | Baseline | R correction | R and D corrections |
| --- | --- | --- | --- |
| Full orbital | 14, 11, 12 | 14, 12, 12 | 12, 15, 15 |
| B=3 metal | 11, 14, 14 | 11, 14, 14 | 15, 21, 19 |
| B=3 Mott | 17, 14, 14 | 17, 14, 14 | 24, 25, 23 |
| Noninteracting | 7, 7, 7 | 7, 7, 7 | 7, 7, 7 |
| B=5 | 0/3 match | 0/3 match | 1/3 match |

R+D needs one additional seeding solve beyond the cycles in this table.
The extra B=5 root is an atomic state: docc ~0, Z=0, energy -1.2, rather
than the metallic iterates with docc ~0.053 and energy near -1.26185.
It also has a large parameter recursion despite tiny physical residuals.
Report the additional matching root, but do not call it reliable recovery
of the metallic branch or a stable parameter fixed point.

B=5 controls:
- Expanded D with no damping (epsilon=0): 0/3 match.
- Epsilon=1e-4: 2/3 match, both atomic states.
- Epsilon=1e-8: 1/3 match, a low-Z branch with docc 0.02681594 and
  energy -1.25511949; another run stops at the impurity symmetry check.
- R correction plus smooth cap, epsilon=1e-6: 1/3 match, again the
  docc 0.02681594 branch.
- R+D plus smooth cap, epsilon=1e-6: 0/3 match; it can be much worse
  than either component alone.

Two-orbital quarter-filling case, seed 4, noise 0.05:
baseline 40 cycles; R correction 39; R+D 42 plus one seed solve.
Smooth cap alone 25; cap plus R correction 24; cap plus R+D 43 plus
one seed solve. Matched solutions have the same energy to ~1e-9.

Keeping core parameters does not improve these comparisons:
- Full-orbital controls still converge, but retaining R changes its
  undetermined full-orbital component; no physical equivalence is assumed.
- B=5: no starts match for R or R+D; one R+D run stops at the impurity
  symmetry check.
- Noninteracting R: 43, limit 200, 64 cycles; R+D: 70, limit 200, 194
  plus the seed solve. Each matches 2/3, versus 3/3 for the zero-core control.
This rejects blindly retaining everything as a robustness improvement,
not other possible treatments of core orbitals.

### Smooth cap clarified

It is a limiter for the proposed quasiparticle Lambda update inside
FullEmptyPinning.qp_levels, before DIIS. It does not cap DIIS extrapolation.
The existing limiter selects near-edge orbitals sharply, then limits
the spectral norm of their level change. The prototype uses occupation
weights that turn on gradually and rounds the norm clipping, using the
Frobenius norm. For one selected scalar step d with limit L, the smooth
step is d / sqrt(1 + (d/L)^2). With L=2, d=1 gives 0.894 and d=8 gives
1.940; the current cap gives 1 and 2.

Cap-only quarter-filling runs confirm both help and harm:
baseline 26, 26, 24, 40 cycles for starts 0, 1, 2, 4;
smooth cap 20, 24, 26, 25. It can shorten useful steps and change the basin.
An optional feature would be an alternative policy for the existing
FullEmptyPinning step limiter, not a new optimizer or a thermal solver option.
No production API has been added.

### Verification and assessment

The selected R residual correction passes the same six existing physical
test cases in 0.64 s. No test cases or assertions were added or changed.
Known matching inputs have zero correction; revisiting identical expanded
inputs after a different input gives identical maps and residuals.
Ruff, Ruff LSP and strict mypy pass for the external prototype.

Residual damping avoids the inconsistent-inverse flaw of the earlier
taper, but did not deliver a general reliability or speed improvement.
It can find additional matching roots and change their basin. Expanding
D is usually slower here, and combining modifications can make them worse.
Keep all of these as experiments until a more reliable variant is found.

Commands: use the same one-thread environment as above.
- uv run python /tmp/risb-smoother-map/residual_correction.py cycles
- uv run python /tmp/risb-smoother-map/residual_correction.py cycles --smooth-cap
- uv run python /tmp/risb-smoother-map/residual_correction.py cycles --cases full,b5,u0 --core-policy keep
- uv run python /tmp/risb-smoother-map/residual_correction.py cycles --cases b5 --variants rd --epsilon 0
- uv run python /tmp/risb-smoother-map/residual_correction.py cycles --cases quarter --seed-start 4 --seeds 1 --noise 0.05 --maxiter 400
- uv run python /tmp/risb-smoother-map/residual_correction.py checks
- uv run python /tmp/risb-smoother-map/residual_correction.py suite

Raw JSONL under /tmp/risb-smoother-map/: residual-starts,
residual-cap-starts, residual-keep-core, residual-b5-unfiltered,
residual-b5-stronger, residual-b5-weaker, residual-quarter,
residual-quarter-cap. Earlier exploratory runs are residual-initial.
Cap-only additional starts are zero-cap-starts.jsonl.

## Smooth step-cap feature and full-suite default comparison (2026-11-01)

Implemented in src/risb/pinning.py:
- FullEmptyPinning accepts keyword-only step_cap_method="clip" or "smooth".
- Smooth is the default. Clip retains the previous limiting calculation;
  step_cap=None still disables limiting.
- The constructor selects the limiter once. qp_levels reuses its
  eigensystem and calls the selected helper; both remain inside the
  existing pinning interface. No new LatticeSolver or DIIS arguments.
- Smooth uses the measured prototype's quintic occupation weights and
  rounded Frobenius-norm limit. Full/empty level pinning is unchanged.
- Option details are in the class docstring only. No how-to example,
  benchmark additions or new tests.

Full suites, run sequentially with one BLAS/OpenMP thread:

| Default | Passed | Subtests passed | Failed | Wall time |
| --- | --- | --- | --- | --- |
| Previous clipping | 213 | 111 | 1 | 82.74 s |
| Smooth | 213 | 111 | 1 | 81.27 s |

No new failures. Both fail
test_complex_two_orbital_model_reaches_the_mott_state_with_a_thermal_impurity:
the expected self-consistency warning is not emitted. The failure occurs
in the unchanged baseline before adding this feature.

Repeated the three largest relevant timing changes twice per method,
sequentially. Median call times and logged solve cycles:

| Existing test | Clip | Smooth | Cycles, clip -> smooth |
| --- | --- | --- | --- |
| Perturbed two-orbital ghosts, free gauge | 3.33 s | 2.24 s | 27 + 42 -> 21 + 26 |
| Same, Lambda eigenbasis | 2.79 s | 3.14 s | 23 + 36 -> 34 + 33 |
| Double-counting restart | 0.56 s | 0.41 s | 200 + 1 -> 146 + 1 |

The Lambda-eigenbasis slowdown is reproducible (~13%). Free gauge improves
~33%; double counting improves ~26%. These do not establish a universal
speedup. Smooth is retained as default because it passes the same physical
assertions without new failures and improves the net cost of these
affected tests. Clip remains selectable.

Complete per-test call-time comparison:
/tmp/risb-smoother-map/step-cap-suite/timings.csv.
Raw reports, logs, summary JSON and repeated timings are in the same
directory. Temporary runner: /tmp/risb-smoother-map/step_cap_suite.py.
Its method override changes constructor defaults only in the test process;
no test files or assertions were edited.

Ruff, Ruff LSP and strict mypy pass (51 project files). The safeguard
prototype is still separate future work: residual-based acceptance with
backtracking would be a line-search method, beyond fixed DIIS alpha.

## Current globalization literature and T = 0 safeguards (2026-11-01)

Reviewed current work before choosing a prototype. Line searches and trust
regions remain active frameworks; recent work changes the models,
acceptance rules, regularization and cost management rather than replacing
those frameworks solely because they are old.

Primary references:
- Herbst and Levitt, JCP 2022, "A robust and efficient line search for
  self-consistent field iterations":
  https://doi.org/10.1016/j.jcp.2022.111127 .
  Cost-aware SCF backtracking with an inexpensive energy model. Its
  energy-based setting is different from our auxiliary-variable root problem.
- Ouyang, Tao, Milzarek and Deng, JSC 2023, "Nonmonotone Globalization for
  Anderson Acceleration via Adaptive Regularization":
  https://doi.org/10.1007/s10915-023-02231-4 .
  Adjusts regularization using a nonmonotone actual/predicted reduction
  test. Useful design for a small history solve, not a guarantee for RISB.
- Bian and Li, September 2026 preprint, "A nonmonotone globalization
  framework for Anderson acceleration for contractive and nonexpansive
  fixed point problems":
  https://arxiv.org/abs/2609.28950 .
  Windowed residual acceptance with relaxed fixed-point fallback.
  Contractivity/nonexpansiveness assumptions are not established here.
- Greiner et al., 2025 preprint, OpenTrustRegion:
  https://arxiv.org/abs/2509.13931 .
  Reusable second-order orbital optimization using Hessian-vector products.
  Full Newton-style curvature is a larger implementation task; history-based
  trust-region ideas need no such full outer derivative.
- Bao and Shi, August 2026 preprint, AURORA-SCF:
  https://arxiv.org/abs/2608.07354 .
  Auxiliary curvature, target-level secant corrections and target acceptance.
  It requires a useful cheap response model; it is not a drop-in RISB method.
- Current SCS implementation:
  https://www.cvxgrp.org/scs/algorithm/acceleration.html .
  Uses AA regularization, residual checks, rejection and history resets.
  Safeguarding remains a practical current technique.
- Tagliente, Pasqua and Fabrizio, July 2026 preprint:
  https://arxiv.org/abs/2607.27156 .
  Directly compares iterative ghost-Gutzwiller embedding with variational
  minimization, finding Mott/Zeeman-field artifacts avoided by the latter.
  Direct minimization deserves a separate T=0 investigation.

The older Jacobian comment was too broad: the full orbital trust-region
schemes need curvature, but adaptive regularization of Anderson's small
history solve does not. Both derivative-free safeguards and history-based
regularization were therefore prototyped on LatticeSolver, not restricted
to ThermalLatticeSolver. The inner thermal Jacobian is not an outer RISB
Jacobian.

### Prototype

File: /tmp/risb-smoother-map/backtracking.py. No production changes.
No tests or benchmarks added.

Policies:
- plain: current DIIS, alpha=1, history=6, periodic mixing every 3 steps;
- monotone: backtrack the whole proposal displacement, checking reduction
  against the current physical residual norm;
- nonmonotone: the same bounded search, but against the maximum of the
  last five accepted norms;
- safeguard: a recent-AA-inspired windowed accept/reject rule with
  gamma_k = 1 - 1/(k+2); rejected proposals reset history and use a
  relaxed fixed-point step of size 0.2;
- regularized: an adaptation of the 2023 adaptive history-coefficient
  method, using a best-residual history anchor, residual-scaled ridge,
  actual/predicted reduction, and fixed-point fallback.

These are physical-residual adaptations. The cited AA papers use the
fixed-point residual g(x)-x and assumptions about their fallback maps.
Our default DIIS uses the physical matching equations instead. We do not
claim a faithful reproduction of their convergence theorems.

All runs use the current smooth level cap and the original matching
equations. Merit includes original f1/f2 and the bath equation
ke^T - sqrt(rho_qp (1-rho_qp)) D, with only energy-unit normalization.
The original residual vector drives DIIS. Acceptance does not add an
anchor or change a physical equation.

The bounded searches try the full step and five halvings, then a separately
backtracked 0.2 mixing direction. They stop as stalled if no step passes
within that bound; this is not proof that no smaller acceptable step exists.
A full trial uses the original proposed array, avoiding unnecessary
floating-point reconstruction differences.

Retries restore D, Lambda_c and rho_c from the same accepted state.
Rejected trials never enter DIIS history. Accepted trial results are
reused, rather than repeating every successful evaluation. The final cycle
is rerun from its original starting state to check the reported physical point.
Every impurity.solve call counts, including those at rejected guesses
and those used to recalculate the final cycle.

### Controlled T = 0 results

Six models, default start plus seeds 1 and 2 with noise 0.01. Equal budget
of 300 cycle evaluations, including recalculation of the final cycle. Stop
only when the outer residual norm is <1e-9 and every original matching
error is <1e-8.
Models: full-orbital charge transfer, B=3 metal/Mott, B=5 ground state,
noninteracting B=3, two-orbital quarter filling. All use LatticeSolver;
the existing B=3 metal/Mott controls use beta=200 impurity averaging as a
T=0 convergence aid. B=5, full-orbital and two-orbital cases use ground
states.

| Policy | Matching solutions / 18 | B=5 / 3 |
| --- | --- | --- |
| Plain DIIS | 15 | 0 |
| Monotone backtracking | 14 | 0 |
| Nonmonotone backtracking | 15 | 0 |
| Window safeguard | 16 | 1 |
| Adaptive regularization | 15 | 0 |

Monotone backtracking also stalls one noninteracting start. Both bounded
searches stall on all three B=5 starts. Counting fewer accepted iterations
would hide their rejected-trial cost. On paired successful cases, monotone
backtracking uses 277 solves versus 223 for plain DIIS.

The window safeguard's B=5 root has energy -1.261517085, docc 0.053344852
and Z 0.16424070, close to the B=3 solution; it does not establish recovery
of the lower-energy metallic iterates approached by the failed runs.
Adaptive regularization can be substantially worse on B=5, ending with
original residuals of order 0.35-0.6 after its budget.

Extended B=5 check, seeds 0 through 7:
- plain DIIS: 1/8 matching roots (seed 6, docc 0.02681594,
  energy -1.25511949);
- window safeguard: 1/8 (seed 0, the B=3-like solution above).
Different starts and different branches succeed. The initial 1/3 rescue
did not become an overall success-rate improvement in the larger sample.

Known difficult two-orbital start (seed 4, noise 0.05), impurity solves:

| Plain | Monotone | Nonmonotone | Window safeguard | Adaptive regularization |
| --- | --- | --- | --- | --- |
| 26 | 31 | 26 | 29 | 28 |

Counts include the final verification solve. Physical energies on paired
successful controls agree within 1.8e-9. Additional investigations were
run concurrently; their wall times are not used for speed claims.

### Assessment and verification

Do not ship simple monotone backtracking as a default. The nonmonotone
and adaptive ideas are relevant modern choices, but these particular
adaptations have not improved reliability enough for production adoption.
A DIIS direction need not be a descent direction for the physical
residual; shrinking it alone is not a complete remedy. A reliable fallback
or a better residual model remains unresolved.

Direct T=0 variational minimization is a separate research direction,
especially for degenerate Mott states, not an assumption that any energy
line search on the current unconstrained auxiliary variables is valid.

Existing optimizer tests: 20 passed in 0.28 s. Ruff, Ruff LSP and strict
mypy pass for the prototype. No production files changed.

Artifacts under /tmp/risb-smoother-map/:
- backtracking-controlled.jsonl: main four-policy comparison;
- backtracking-regularized.jsonl: adaptive history regularization;
- backtracking-b5-extra.jsonl: extended baseline/window starts;
- backtracking-quarter-hard.jsonl: known difficult two-orbital start;
- backtracking-summary.json and backtracking-comparison.csv: summaries.

Reproduce with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
VECLIB_MAXIMUM_THREADS=1:
- uv run python /tmp/risb-smoother-map/backtracking.py
- uv run python /tmp/risb-smoother-map/backtracking.py --policies regularized
- uv run python /tmp/risb-smoother-map/backtracking.py --cases b5 --policies plain,safeguard --seed-start 3 --seeds 5
- uv run python /tmp/risb-smoother-map/backtracking.py --cases quarter --policies plain,monotone,nonmonotone,safeguard,regularized --seed-start 4 --seeds 1 --noise 0.05

## Published Anderson algorithms and matching-residual models (2026-11-01)

The preceding safeguard experiments changed the papers' residual to the
matching equations. They were adaptations, not tests of the published
algorithms. The comparisons below keep those experiments separate.

### Residuals and current behavior

x packs Lambda and R. g(x) is one cycle, before DIIS extrapolation.
NewtonSolver.error_of already supports both:
- recursion: g(x) - x;
- root: the packed f2 and f1 from cycle(x), the default.

The matching equations are f2 = rho_f - rho_qp and
f1 = rho_cf - R.T sqrt(rho_qp (I-rho_qp)). Here rho_f follows the
embedding bath-hole convention. The bath check is
ke.T - sqrt(rho_qp (I-rho_qp)) D.

Small matching errors and small parameter updates are different criteria.
For example, the FD-initialized model in the full-orbital control reaches
max matching error 7e-11 while ||g(x)-x|| is 49.5. A completely occupied
auxiliary direction has R approximately 1; the map sets it to 0 and moves
its Lambda from approximately 0 to -35. The matching equations leave
parameters undetermined there. This does not decide which edge convention
is better.

### Published algorithms: use g(x)-x throughout

Prototype: /tmp/risb-smoother-map/paper_anderson.py.

- recursion-diis: existing DIIS(error_fun="recursion"), including its
  periodic mixing.
- pure-aa: unregularized Anderson with no periodic mixing.
- glob-aa: Bian and Li Algorithm 1, Eq. 2.26,
  gamma_k = 1 - max(1/(k+2), W_k/(2 W_0)), KM weight 0.5,
  and history reset after fallback.
- glob-aa-regularized: the same with the paper's Eq. 3.2 stabilization,
  eta = 1e-10.
- glob-aa-schedule: Algorithm 1 with the published Remark 2.6 schedule
  gamma_k = 1 - 1/(k+2), KM weight 0.5.
- lm-aa: Ouyang et al. Algorithm 1, best/newest residual anchor,
  lambda_k = mu_k ||g(x_anchor)-x_anchor||^2, gamma = 1e-4,
  p1 = 0.01, p2 = 0.25, eta1 = 2, eta2 = 0.25, mu0 = 1,
  c = 0.99. Rejection uses g(x_anchor), without an added history reset.

These implement the stated coefficient, acceptance and fallback equations,
not the authors' programs. The small least-squares problems use NumPy's
dense SVD solve; Bian and Li use LSQR. We choose an absolute update
tolerance of 1e-9 and a 300-evaluation budget for these RISB comparisons.
The matching equations are checked independently and do not alter paper
steps or stopping. Ouyang's returned g(x) is evaluated on stopping.

References:
- Bian and Li, September 2026 preprint:
  https://arxiv.org/html/2609.28950 .
- Ouyang et al., adaptive regularization:
  https://arxiv.org/html/2006.02559v3 ;
  published JSC paper https://doi.org/10.1007/s10915-023-02231-4 .

On B=5, Eq. 2.26 gives gamma <= 0 for all three initial starts.
Its assumed bound W_k <= W_0 fails here. Those runs stop with
gamma-outside-domain; we do not clip gamma and call that the paper's rule.
The iteration-based schedule remains defined. No convergence theorem is
claimed for the RISB map.

Positive control: reproduce section 3.1's contraction, n=200, m=1,
x0=2.1, KM weight 1. Plain AA cycles through its 501-evaluation budget;
the published iteration schedule converges in four iterations, agreeing
with the paper. This checks the safeguard outside RISB's unverified
assumptions.

### Separate matching-residual trust-region models

Prototype: /tmp/risb-smoother-map/physical_residual_trust.py.

Both minimize a local approximation of the original matching norm at
x + step, with actual/predicted squared-norm reduction for acceptance and
radius updates. Neither uses g(x)-x as its objective.

- history-trust: infer a response in the span of recent displacements;
  include probes along the map direction outside that span.
- fd-trust: initialize a full forward-difference outer Jacobian, then
  use Broyden secant updates; refresh after three rejected steps or
  twelve accepted steps. This is an FD-initialized model, not a fresh
  full Jacobian at every iteration.

Lambda is scaled by the model energy scale; R is dimensionless.
The linear trust-region subproblem uses an SVD and solves for the
multiplier that enforces the radius. Independent linear controls satisfy
the constrained and unconstrained optimality equations to <5e-16.
A bad result here does not rule out other trust-region models.

### Controlled results

Same six LatticeSolver models and starts as the preceding comparison:
default plus seeds 1 and 2, noise 0.01, budget 300 including probes,
rejects, Jacobian evaluations and recalculation of the final cycle.
Current smooth cap.
Full-orbital, B=5 and quarter-filled embeddings use ground states;
B=3 controls retain their beta=200 impurity averaging. There are no inner
thermal density fits.

"Matching" below means every f1/f2/bath entry <1e-8.
"Strict" additionally requires ||packed(f2,f1)|| <1e-9.
A point can pass matching while the algorithm reports a budget/stall.

| Method | Matching /18 | Strict /18 | Solves on 15 non-B=5 controls |
| --- | ---: | ---: | ---: |
| Current root DIIS | 15 | 15 | 231 |
| Existing recursion DIIS | 15 | 15 | 247 |
| Pure Anderson | 15 | 15 | 316 |
| Bian-Li Eq. 2.26 | 15 | 15 | 256 |
| Eq. 2.26 with stabilization | 15 | 15 | 256 |
| Bian-Li iteration schedule | 17 | 16 | 254 |
| Ouyang adaptive regularization | 15 | 15 | 312 |
| Matching history trust region | 12 | 11 | 1341 |
| Matching FD-initialized trust region | 9 | 8 | 2720 |

The two matching-residual models are worse than current DIIS on this
sample. Their counts include substantial derivative/probe overhead.
Neither rescues B=5. Do not adopt these prototypes.

Extended B=5, seeds 0 through 7, with both baselines repeated locally:
- root DIIS: matching 1/8, strict 1/8;
- recursion DIIS: matching 0/8;
- published iteration schedule: matching 5/8, strict 4/8.

The fifth scheduled-AA point has max error 5.75e-10 and packed norm
1.50e-9: a threshold difference, not a large hidden matching error.
Successful scheduled runs cost 42, 108, 189, 213 and 215 impurity solves.
One reaches the B=3-like branch (energy -1.261517085); four reach
energy approximately -1.261595334, docc 0.05341123, Z 0.16168111.
Baseline's one root has energy -1.255119491 and docc 0.02681594.
These results do not establish a unique root or the lowest-energy solution.

Known difficult quarter-filled start, seed 4, noise 0.05:
root DIIS 26 solves; recursion DIIS 36; pure AA 47;
Bian-Li variants 32; Ouyang 47. Both matching trust-region models
exhaust 300 evaluations without passing matching.

Recommendation: the published iteration-scheduled safeguard merits an
optional optimizer implementation. Keep the current default: it is about
10% cheaper across the 15 controls (231 versus 254 solves), and the new
method still fails some starts. Earlier matching-residual adaptations did
not test this algorithm and should not be used to dismiss it.

Only this running note changes in the repository. No production changes,
benchmark additions or permanent tests. Existing optimizer tests: 20 pass.
Ruff, Ruff LSP and strict mypy check the temporary prototypes.

Artifacts under /tmp/risb-smoother-map/:
- paper-anderson-controlled.jsonl, paper-anderson-schedule.jsonl;
- paper-anderson-b5-extra.jsonl, paper-anderson-b5-baselines.jsonl;
- paper-anderson-quarter-hard.jsonl;
- physical-model-controlled.jsonl, physical-model-quarter-hard.jsonl;
- model-controls.json, residual-model-summary.json.

Run the two prototype scripts with --help for cases, starts and budget;
set OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
VECLIB_MAXIMUM_THREADS=1. Wall times from overlapping investigations
are not used for speed comparisons.


## Bian-Li policy implementation

- DIIS owns accepted history; Pulay owns periodic mixing, damping, restart
  and residual selection. BianLi implements Algorithm 1 with the iteration,
  bounded-residual, constant and combined schedules. Pulay remains default.
- Preserve callbacks and SolveResult.x_next. Acceptance precedes convergence
  and best-point bookkeeping. Budgets count candidate and fallback evaluations,
  and cycles rerun to restore checkpoints. Stop before a step that cannot
  fit and retain x_next for continuation.
- Reconstruction checkpoints retain pre-cycle parameters and DC density.
  Thermal checkpoints add previous D/Lambda_c and anchor strength/references.
  Block2 ground state saves only its previous sector. Fockit, thermal Block2
  and Dummy add no independent state; no RNG snapshots or MPS copies.
- Existing DIIS Gram/pseudoinverse calculation is shared by both policies.
  This differs numerically from the temporary SVD prototype.
- Actual implementation, same 18 starts, 300-cycle budget: Pulay passes
  physical matching and stopping on 15/18, BianLi on 16/18. Non-B=5 controls
  cost 216 versus 266 cycles, including checkpoint restoration. Total costs:
  1116 versus 914.
  B=5 seed 0 converges in 50 evaluations with BianLi; seeds 1 and 2 fail.
  Do not carry over the SVD prototype's 17/18 or extended 5/8 results.
- Paper cycling example: ordinary AA fails within 500 evaluations; BianLi
  takes four transitions (five accepted evaluations including the initial point).
- Temporary stateful checks: T=0/DC converges in 29 evaluations; thermal/DC
  in 85. Restoring the thermal/DC/anchor checkpoint reproduces all compared
  arrays exactly.
  Budget pause/continuation agrees with uninterrupted runs for T=0 and thermal.
- No permanent tests added. Existing optimizer tests migrated: 20 pass.
  Full suite: 213 passed, 111 subtests passed, same pre-existing missing-warning
  failure in the complex two-orbital thermal-impurity Mott test (243.74 s).
  Block2 tests included. No benchmark changes.
- Local artifacts: /tmp/risb-smoother-map/implemented.py,
  implemented-results.jsonl and implemented-stateful.py.
- Public lattice solve entry points also checked with DC: T=0 converges
  in 29 evaluations; thermal with its default OnPlateau callback in 60,
  with physical error 1.32e-10 and unregularized bath error 1.33e-15.
- Final affected tests: 41 passed. Ruff, Ruff LSP, strict mypy (50 files),
  and strict Sphinx build pass. Fixed two unresolved docstring references.

## Full-suite BianLi comparison

- Temporary runner: /tmp/risb-smoother-map/bian_li_suite.py.
  All lattice DIIS solves use the iteration schedule and fallback 0.5;
  explicit Pulay optimizer checks and SciPy keep their own algorithms.
  Production defaults unchanged. Sequential runs with one BLAS/OpenMP thread.
- Pulay: 213 passed, 1 failed, 111 subtests; 81.16 s, 2651 lattice evaluations.
  BianLi: 210 passed, 4 failed, 108 subtests; 688.71 s, 5448 evaluations.
- New failures: thermal two-orbital Block2 exhausts its budget (998
  evaluations, 579.22 s for its Block2 solve, versus 21 and 11.90 s);
  both maxiter=1 continuation tests stall; thermal site-outside-cluster
  T=1 converges in 153 evaluations but exceeds the test's 100-evaluation cap.
- Block2 max physical residual is 5.47e-8 but recursion norm is 8.74e-5.
  Physical matching and stopping on fitted parameter changes differ sharply.
- The old missing-warning Mott test passes with BianLi, but its matching
  residual is 0.00352. This is not a physical-matching improvement.
- Of 71 lattice tests passing both runs: 10 use fewer evaluations, 41 more,
  20 unchanged. One apparent improvement is the deliberately unpinned
  negative control; exclude it from convergence successes.
- Clear gains: one-band thermal ghosts T=10, 25 to 4 evaluations,
  0.596 to 0.091 s; T=1, 29 to 20, 0.773 to 0.489 s.
  Clear losses: thermal B=5/B=3 comparison, 40 to 108, 5.02 to 14.50 s;
  thermal Hubbard example, 114 to 344, 2.72 to 7.24 s.
- Follow-up profiling inside solve_fixed_point preserves the native
  lattice solve caller/warning path: all other tests repeated, 210 passed,
  same three failures, one long Block2 case deselected (102.75 s).
- Keep BianLi optional. Full per-test tables, CSV, logs and residuals:
  /tmp/risb-smoother-map/bian-li-suite/.

## Gauge rules with Pulay and BianLi (2026-11-02)

T = 0 LatticeSolver, default FullEmptyPinning with the smooth cap, tol 1e-9
to 1e-10. Gauge rules through gauge_fix: free, risb.gauge.lambda_eigenbasis,
and a prototype Hermitian-R rule (R rotated to [P; 0] with P = (R^dag R)^(1/2),
Lambda diagonal and sorted on the quasiparticles R does not couple to).

Cycles to converge, x = not converged.

Double-counting ghost Hubbard (B = 3, U = 2, the restart test's model):

| Policy | Free | Lambda | Hermitian R |
| --- | --- | --- | --- |
| Pulay root | 146 | x (1e-7 at 1000) | x |
| Pulay recursion | 167 | 741 | x |
| BianLi | 142 | 152 | 104 |

Two-orbital B = 3 quarter filling (the perturbed-start test), default start
and seeds 0-5, noise 0.05, cap 1000:

| Policy | Free | Lambda | Hermitian R |
| --- | --- | --- | --- |
| Pulay root | 21-34 | 31-58 | x all 7 |
| Pulay recursion | 25-43 | 39-170 | x 5 of 7 |
| BianLi | 28-36 | 29-79 | x all 7 |

Two orbitals with a crystal field at half filling, Bethe lattice, B = 3,
density-density Kanamori, beta 100, Sz = 0 fockit impurity, up/dn averaged;
U = 3, J = 0.5 at 8 crystal fields and U = 4, J = 0.7 at 7, cap 300.
Lowest = converged with f1/f2 < 1e-8 onto the lowest energy found in any run.

| Policy | Gauge | Converged | Lowest |
| --- | --- | --- | --- |
| Pulay root | free | 8/15 | 8/15 |
| Pulay root | Lambda | 14/15 | 13/15 |
| Pulay root | Hermitian R | 12/15 | 12/15 |
| BianLi | free / Lambda / Hermitian R | 2/15 | 1-2/15 |

- Hermitian R fails every degenerate two-orbital start; rejected.
- Lambda eigenbasis helps the crystal-field model and only slows the easy
  ghost cases, but needs BianLi on the double-counting case.
- BianLi converges 2/15 crystal-field cases in 300 cycles, against 8/15 for
  Pulay without a gauge.
- No combination is best everywhere; lambda_eigenbasis stays opt-in.

## BianLi evaluation budgets (2026-11-02)

NewtonSolver.step_fits_budget now supplies the budget check for both solve
loops. The policy still declares max_candidate_evaluations; Checkpoint still
declares its restoration cost. Neither loop needs a policy-specific branch.

An impossible total budget raises ValueError. A valid budget with too few
evaluations left stops normally. BianLi needs at least two evaluations
without restoration, or three with the current lattice checkpoint.
The checkpoint-specific requirement is checked once a checkpoint exists,
so a fresh two-cycle lattice solve can accept its first point before raising.

One parameterized public-API regression checks both solve entry points,
with and without checkpoint restoration. The existing continuation test now
also checks BianLi at a valid budget, comparing accepted points and total
evaluations against an uninterrupted solve. No how-to or benchmark added.


Scratch continuation checks also cover T=0, thermal fitting and
density-dependent double counting. Chunked calls with maxiter=3 match the
uninterrupted accepted points and next guesses bitwise, using the same
30, 101 and 129 evaluations respectively.
Full suite: 330 passed in 85.31 s.

## Local auxiliary-basis alignment (2026-11-02)

Scratch only: /tmp/risb-smoother-map/basis_alignment.py.
Report and raw comparisons: /tmp/risb-smoother-map/basis-alignment/report.md.

Locally rotate each updated R and Lambda toward the current input basis.
Rotation invariants hold to roundoff. Compare free gauge, the existing
Lambda eigenbasis and alignment across four T=0 models, three starts and
both Pulay and BianLi: 72 runs at fixed model parameters and smearing.

Original matching equations pass in 12/24 free-gauge runs, 17/24 Lambda
runs and 14/24 aligned runs. Alignment rescues the high-spin crystal-field
case with Pulay relative to free gauge, but usually costs more than Lambda.
It also makes a quarter-filled BianLi run report convergence with matching
error 0.049. Such points are excluded from the success counts.

Matched high-spin roots agree in energy, density and spin but differ in
chemical potential and Z; full physical equivalence remains unresolved.
Do not ship alignment from this evidence. Item 13 stays open.

## AutoBZCore integration prototype

Scratch code and report: /tmp/risb-smoother-map/autobz/report.md.
Uses registered AutoBZCore 0.4.1 directly; no copied quadrature code.
No production solver, dependency, interface, how-to or benchmark changes.

Custom matrix integrands work with AutoPTR and IAI. Nine model/width
combinations cover complex 2D/3D bands, rectangular ghost matrices, a
pocket and an insulator. Against converged NumPy meshes, maximum matrix
discrepancies are 1.93e-11 for IAI and 2.64e-9 for AutoPTR at tolerance 1e-7.
Filling and quasiparticle band-energy derivative checks also pass.

Method selection matters: IAI helps several cold 2D integrands, but takes
28 s versus AutoPTR's 1.1 s on the smoother 3D metal. The scratch AutoPTR
rate estimate uses width, velocity and a conservative gap bound.

Plain IAI misses an off-centre pocket: it reports Success and essentially
zero filling, whereas dense meshes give 0.0003183935631. Adding a Lorentzian
near mu to the error-controlled outputs recovers the pocket within the
reference uncertainty, without changing occupations or matching equations.
That sampling guide needs broader checks; it is not a production default.

Square-lattice T=0 self-consistency uses pure Fockit ground states, Pulay,
target filling and unchanged bath equations. Ordinary metal and one-band
Mott controls agree between methods. Both fail the B=5 case at beta=200.
At beta=800, IAI converges from two starts in 30/69 cycles (78/328 s),
with refined matching errors 2.6e-11/2.0e-10. The 64x64 mesh reports
convergence in 35/39 cycles (5/9 s), but refined matching errors remain
3.3e-6/2.7e-6. Independent 256x256 and 512x512 sums confirm these findings.
This does not resolve the earlier three-orbital Mott failure.

Proceed with interface design, keeping fixed-mesh and adaptive methods
available. No full CSC charge reconstruction, nuclear-force calculation
or exactly zero-smearing integration was tested.

Existing production suite: 219 tests and 111 subtests passed (236.80 s).
AutoBZ checks ran separately in scratch space. Ruff, Ruff LSP and strict
mypy pass for the three Python scratch scripts.


## Fixed-input smearing errors

Report: /tmp/risb-smoother-map/autobz/smearing-limit/report.md.
47 samples at frozen R/Lambda; mu solved for the same filling at each
width. No outer self-consistency solves. Uses upstream Richardson.jl
power=2, as in AutoBZ's T=0 density recipe, applied to both matrix integrals.

At beta=800, density smearing errors are 2.1e-4 for B=5 and 0.23 for
the weak-ghost input, despite accurate quadrature. The frozen Mott input
already agrees with T=0 within 5.8e-11 at beta=50.

For a regular metal, extrapolation reaches measured error 2.7e-11 at
beta=1600; raw integration first meets 1e-8 at beta=25600. B=5 needs
beta=204800 for the extrapolator to certify 1e-8. Weak ghosts make its
default error-growth stop terminate above tolerance; allowing continued
cooling reaches measured error 4.6e-11 at beta=102400.

Every reported 1e-8 success meets the independent T=0 reference. Estimates
below input accuracy are not trustworthy; some unconverged extrapolations
also violate density bounds. Target per-evaluation T=0 matrix accuracy,
with separate quadrature and smearing checks. Items 20 and 22 remain open.

## General k-integration implementation

- `risb.kintegration` separates numerical integration from smearing, band
  diagonalization, filling, and RISB observables. The solvers share the
  projected density/kinetic integrands for every cluster and block.
- Weighted supplied meshes retain quasiparticle densities for CSC. The density
  and band-energy helpers accept nonuniform weights. Adaptive nodes are not
  exposed as a replacement DFT mesh.
- Temperature extrapolation uses a NumPy Richardson tableau in beta^-2.
  It checks the requested integrals and mu; propagated numerical integration
  errors enter stopping when available. Fixed-mesh discretization error remains
  unknown. Pointwise CSC-density error is not estimated.
- AutoBZCore is optional and runs in a persistent Julia worker. Generic Python
  callbacks cross the process boundary; this is slower than a native Julia
  integrand. No integration default changed.

- KIntegrationDiagnostics reports k-sum results and extrapolation errors without
  depending on a lattice solver. Debug output includes each beta, mu searches
  and timings. The existing extrapolated lattice test checks this output.
- The extrapolation docstrings cite Richardson (1911), the Richardson.jl
  sequence implementation, and AutoBZ's T=0 density example.

## Per-matrix extrapolation errors and hard T=0 cases

Report: /tmp/risb-smoother-map/hard-t0-extrapolation/report.md.
72 production-integrator comparisons: square B=3/B=5, original Bethe B=5,
three-orbital cubic Mott; three starts, Pulay/BianLi, original beta/beta800/
T=0 extrapolation. No integration defaults changed.

KIntegrationDiagnostics now reports Frobenius norm and largest absolute entry
of the extrapolation error estimate for every requested matrix. Results retain
these as temperature_errors. The existing total stopping rule is unchanged.
Unknown fixed-mesh errors remain unknown; AutoBZ's scalar quadrature estimate
is not presented as a per-matrix estimate.

Square B=5 BianLi matching successes: 1/3 at beta200, 3/3 at beta800, 3/3
with extrapolation. Extrapolation takes about five to seven times longer than
beta800. Pulay extrapolation fails its integration tolerance from all three
starts. Bethe extrapolation fails before its first completed cycle; mu can
veto convergence despite small matrix errors. The three-orbital Mott failure
is still unresolved.

Independent exact T=0 sums on the same mesh expose optimistic Richardson
estimates: converged square B=5 density has Frobenius error 9.87e-8 per spin,
but the estimate is 1.91e-9. This is separate from mesh discretization.
Colder frozen-state samples remove the bias; extending the outer solves'
beta sequence alone does not fix the failures.

Before trusting requested error tolerances, prototype a colder validation
sample, separate matrix tolerances from mu, and check cold filling accuracy.
Keep item 20 open. Production suite: 221 passed; no new test cases or benchmarks.

## Colder confirmation prototype

Report: /tmp/risb-smoother-map/colder-check/report.md.
No production code, tests, benchmarks or defaults changed.

Current acceptance takes the smallest preceding-table change, which can be
accidentally small and can select an entry omitting the newest samples.
The prototype uses the latest three samples and requires confirmation from
the next colder three-sample window. The total criterion still includes mu.

Across 16 frozen inputs and two beta budgets, confirmation removes all false
1e-8 acceptances found here. Latest-three extrapolation alone does not.
For matched square B=5 inputs, combined matrix norm error drops from 1.43e-7
to about 9e-10, using 12 beta samples instead of 10.

42 outer solves: square B=5 with confirmation and a 19-sample allowance passes
3/3 starts with both Pulay and BianLi. All 15 matched outputs checked against
exact same-mesh T=0 sums have per-matrix norm errors below 6.01e-10.
Bethe and cubic Mott remain unresolved. Confirmation can also exhaust a supplied
beta list before checking an accurate candidate. Item 24 remains open.

VASP MIDGAP starts its filling bisection at the gap centre, accepts it if the
count is already correct, otherwise continues the search. Next prototype this
policy inside band filling; preserve the filling constraint. Sources:
https://vasp.at/wiki/EFERMI and
https://vasp.at/wiki/index.php/Smearing_technique#Calculation_of_the_Fermi_energy.

## Lattice-side Lambda updates (2026-11-02)

Scratch: /private/tmp/claude-501/-Users-henry-GitHub-risb/a31c7647-62cb-47a3-8fca-704239bcdc3b/scratchpad/diis/level_solve.py.
No production code changed. Both keep get_r and replace part of get_lambda,
solving f2 on the lattice side with rho_f fixed, no extra impurity solves.
The k-integrator is a black box (finite differences); mu is re-solved per
evaluation ("joint") or kept at the cycle's mu ("fixed").

- linear: get_lambda, then each interior rho_f direction's level from one
  Newton step on its f2. inner: the interior block of Lambda solves
  logit(rho_qp) = logit(rho_f) by Newton with backtracking.
- Needed: only interior directions, n(1-n) > 1e-3 (full/empty have no
  finite level, so FullEmptyPinning stays); logit equations (Newton on n
  fails in the smearing tail); a step cap.
- Crystal-field model, 15 cases, Pulay root: default 9/15; linear joint
  9/15, fixed 10/15, both only low spin, missing lower-energy states;
  inner joint 6/15 (all orbitals in the cluster makes it overdetermined),
  fixed 7/15. Not better than get_lambda.
- Berlingreen beta=800 HS, from the 3.4e-3 state of the joint pin with root:
  linear diverged at cycle 4 (19 eV level step, sc 0.23). inner converges
  each solve (19 unknowns, 5-8 Newton steps, 100-170 band integrations,
  9-13 min against 2 min for block2) and reaches sc 3.0e-3, then 3.1e-2
  at cycle 4 as the Fe2 t2g ghost coupling grows 0.04 -> 0.13.
- The remaining berlingreen instability is in R (that ghost's coupling),
  not Lambda. The 0.05 pin threshold is crossed by the same drift.

## Midpoint-first chemical potential prototype

Report: /tmp/risb-smoother-map/midgap-policy/report.md.
No production code, tests, benchmarks or defaults changed.

The policy finds the filling boundary in the full weighted auxiliary spectrum,
tries its midpoint, and checks the finite-beta electron count. A target cutting
through a degenerate level has no gap; use that level only as a starting guess.
Ghost states remain in the count even when their physical spectral weight is
zero. A sampled gap is not proof of a bulk insulating gap.

128 frozen comparisons: with colder confirmation, midpoint + Brent accepts
15/16 inputs within the 19-sample budget, versus 10/16 for the original mu
search. No accepted input fails measured 1e-8 matrix accuracy. Midpoint +
bisection also accepts 15/16 but costs more count evaluations.

24 outer solves: square B=5 stays 3/3 for Pulay and BianLi; original Bethe B=5
improves from 0/3 to 3/3 (Pulay) and 2/3 (BianLi). All 14 matched outputs
checked against exact same-mesh T=0 sums have matrix norm errors <=5.99e-10.
The remaining Bethe start needs colder integration. Cubic Mott is unresolved.

Recommend implementing colder confirmation in extrapolation.py and a separate
midpoint-first band-filling option in integrators.py, retaining Brent fallback.
The VASP-style bisection variant was tested separately. Item 24 remains open
until production implementation and validation.

Remaining issue: a partially occupied flat level has
mu(beta)=E_g+log[p/(1-p)]/beta. Extrapolating mu in beta^-2 can reject accurate
matrices. A 75%-occupied flat-level control has density error 7.4e-13 while
mu still blocks 1e-8 acceptance. Midgap cannot remove this thermal shift.
Also control the filling residual at very large beta; a converged mu root
alone does not establish the required count accuracy.

## Cold filling accuracy

Report and data: /tmp/risb-smoother-map/cold-filling/.
No new test cases; extended the existing flat-level and weighted-filling checks.

Brent's default absolute mu tolerance can exceed the thermal width. A flat
level with target occupation 0.75 reports convergence with count error 0.077
at beta=1e12. Resolving mu to floating-point precision removes that error
when the level is at zero.

On 14 saved hard-case inputs at five beta values, the old search misses
1e-10 count accuracy in 15/70 integrations. Tightening Brent fixes 13 of
these. The remaining two miss by 1.67e-9 and 1.75e-9 at beta=1e12; checking
16 neighboring mu values on either side finds no better count. These now
return converged=False. All 68 accepted integrations meet the count tolerance;
the integrated density traces also agree with the returned electron counts.

BandKIntegrator uses electron_count_atol for both midpoint acceptance and
the final count, replacing midgap_electron_count_atol. The existing lattice
and extrapolation code already rejects failed integration results.
This checks the calculated count, not unknown k-mesh errors. It does not
resolve the three-orbital Mott case.

Verification: 221 tests and 111 subtests pass. Ruff, Ruff LSP and strict
mypy pass. No new test cases or benchmark changes.

## Three-orbital Mott diagnosis

Scratch report: /tmp/risb-smoother-map/mott-diagnosis/report.md.
No production code, tests, benchmarks or defaults changed.

All six original extrapolated trajectories first exhaust their beta lists.
One to three colder samples fix their frozen inputs, but longer outer runs
then expose count precision and weak level updates. Dense impurity solves
agree with the original spin-averaged densities at those first failures.

A relative-mu prototype retains the small correction in the Fermi function.
With the existing Pulay(error_fun="root"), two of three starts reach matched
roots; the default start remains unresolved. The unmodified production
extrapolator independently confirms both outputs below 3.1e-9 matching error.

Their energy is 3.5191354 per cell, below the atomic root's 3.6. Their nearly
singular R has nonzero R^dag R, but the band at mu has negligible physical
weight. Do not classify this root or reject it using R^dag R alone.

At the symmetric atomic point, averaging the 80-state ground space removes
a bath-density mismatch left by the ordinary pure-vector choice. This
does not fix every trajectory or prescribe an ensemble for general cases.
Items 20 and 28-30 remain open. Existing affected tests: 35 pass.

## Relative-mu implementation

BandKIntegrator now refines a failed count using a correction relative to
the approximate mu. Smearing functions keep their signatures; the final
electron-count tolerance still decides acceptance. Results expose the
reference and correction, and mu remains their rounded sum.

Temperature extrapolation retains the pair when fitting mu and estimating
its error. Diagnostics report it. No optimizer, lattice-solver, backend,
checkpoint or how-to changes were needed.

The existing flat-level test now checks a shifted level at beta=1e16 and
reconstructs its 75% occupations from the pair. Its extrapolation check
also resolves a correction smaller than a floating-point step in absolute
mu. No new test cases. Nineteen scratch checks across all built-in
smearings and adaptive AutoBZ meet the count tolerance.

On the three original Mott starts with the 27-sample beta allowance,
physical-residual Pulay matches 2/3 in 98 and 122 cycles. Independent
reevaluation confirms maximum matching errors <=3.1e-9 and energy about
3.5191354. Recursion Pulay and BianLi match 0/3. The remaining failures are
cooling-budget exhaustion or the previously identified stopping and
atomic-density issues, rather than failed finite-beta count checks.

Verification: 221 tests and 111 subtests pass. Ruff, Ruff LSP, strict mypy
and the warning-free Sphinx build pass. No benchmark changes.

## Berlingreen HS beta=800: optimizer, extrapolation and smearing (2026-11-03)

All from the joint-pin root run's cycle-5 input (root residual 3.6e-3, Fe1 HS),
no pin, 10 evaluations. Scratch driver berlingreen_ramp.py with risb_compat.py.

- One-cycle scan of Fe2's middle t2g ghost coupling c = |vR|, others fixed:
  smooth, near-linear map, slope ~4-6 through c* ~ 0.028 (a repelling fixed
  point of that slice). A plain mixing step multiplies the error ~x3.
- Pulay root, mixing_period=None: 3.6e-3 -> 2.8e-5 (cycle 10), filling -4e-4,
  E -105.034 eV. Jointly the ghost decouples (c -> 0, level -> mu, n = 1/2).
  Continued: plateau 3-5e-5, all in Fe2 t2g. Not DMRG: Fe2 at bond dim 1000,
  cutoff 1e-12 (discarded weight 2e-11 vs 1.9e-9) changes rho_f by 4e-7 and
  rho_cf by 2e-7 for the same input; the plateau is the iteration.
- Default mixing_period=3 bounced 4e-3 to 0.1 from the same kind of state.
- Bian-Li (recursion): 6.8e-2. Bian-Li with root acceptance: 8.6e-2; halving
  the rejected step instead of mixing is identical, as each rejection clears
  the history and the next guess is again g(x).
- T=0 extrapolation (betas 200..12800, unconfirmed results forced through):
  temperature errors 6e-3 to 0.6, unconfirmed with both the 1/beta fit
  (957a5c7) and the old beta**-2 fit; root residual 1e-2 -> 8e-2.
- Gaussian smearing, beta=800, Pulay mixing_period=None: 7.7e-3 -> 5.8e-4 in
  10 cycles, still decreasing (slower than Fermi from a Fermi-converged start).

## Common-level response prototype and warm extrapolation

Report: /tmp/risb-smoother-map/level-response/report.md.
No production code, tests, benchmarks or defaults changed.

A common-Lambda Newton correction based on the impurity charge response
agrees with finite differences. It preserves the original matching equations.
On three starts with direct T=0 sums, original physical-residual DIIS matches
3/3 and recursion DIIS 0/3. Bounded response corrections give 2/3 for each.
Uncapped corrections give the same robust counts; an apparent third atomic
success fails independent reevaluation. Some matched roots have slightly
higher energy. Keep item 29 open.

The first density jump is caused by an accepted but inaccurate kinetic
integral: estimated error 1.5e-13, actual same-mesh error 6.7e-6.
Correcting that integral changes max f2 from 0.235 to 5.4e-11 at the same
input. A separate analytic kinetic reference confirms the direct sum.

Even the coldest four supplied beta values falsely pass a narrower-band
control: estimated error 1.8e-9, actual error 6.7e-7. All those temperatures
are still warm compared with the band scale. Reopen item 24; another fixed
number of colder samples does not resolve the validity problem.

All regular matched outputs pass independent original-map checks. One
atomic output is excluded. Existing affected tests: 35 pass. Ruff, Ruff LSP
and strict mypy pass.

## Direct T=0 weighted filling prototype

Report: /tmp/risb-smoother-map/direct-t0/report.md.
No production code, permanent tests, benchmarks or defaults changed.

Fill weighted levels directly, sharing a partly occupied degenerate group.
Densities retain the supplied k-point order. Forty-five comparisons use the
same three starts and physical-residual Pulay DIIS for each method.

Direct T=0 and extrapolation match all square B=3/B=5, doped square B=5 and
Bethe B=5 starts. Square B=5 takes 15/29/29 direct cycles versus 33/29/33 at
beta=800; doped B=5 takes 62/65/50 versus 50/62/67. No consistent outer-cycle
advantage. Direct filling avoids the repeated temperature samples.

Both T=0 methods match two of three cubic Mott starts at E=3.5191354.
The default direct run exhausts 300 cycles; extrapolation fails at cycle 43.
Beta=800 reaches different roots at E=3.6 and E=3.5195291.

Changing the numerical degeneracy window from 8 to 64 eps lets the default
direct Mott run reach the lower-energy root in 100 cycles. The checked
matched final densities do not change, but the trajectory does. The earlier
3/3 Mott result is not robust; keep degeneracy handling and branch dependence open.

The narrow-band kinetic reference agrees within 8.5e-21. Weighted complex
and fractional-level controls pass. Existing affected tests: 35 pass.
Ruff, Ruff LSP and strict mypy pass. Direct mesh filling does not make
metallic level crossings smooth or certify an adaptive T=0 integral.
