# Smoother self-consistency map

Goal: improve reliable convergence; measure success across starts and final
physical residuals, not only iteration count. Keep prototypes outside the repository.

## Scope

- Further convergence work targets B>1. Set aside B=1 atomic-limit degeneracy.

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
20. [ ] Investigate smearing options: methods, beta and k-point convergence
   for B>1 failures. The B=1 atomic limit is not a convergence target.
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

32. [x] Prototype a coupled weak-ghost R/Lambda correction with fresh local
   derivatives. One of six continuations matches, the same count as unchanged
   restarts, on a different input. No consistent benefit; keep in scratch.
33. [x] Compare the same B=5 failures with ThermalLatticeSolver at T=1e-4.
   Unanchored thermal control matches 1/3; LatticeSolver with identical band
   beta=10000 matches 3/3. This is not the default thermal solve; see item 34.

34. [x] Investigate cold thermal B=5 convergence with both derivative methods.
   Corrected baseline uses .solve() defaults, including OnPlateau.
   Finite differences match 3/3 inputs; analytic matches 2/3.
   Fixing the basis inside both fits gives 3/3 with either derivative.
   Default FD takes 29/83/621 cycles; the prototype takes 20/49/30.
   Exact fit rotations and unchecked inaccurate fit outputs are identified.
35. [ ] Reassess the fixed-basis thermal-fit feature before implementing.
   Scratch multiorbital weak-ghost comparisons show regressions with both
   derivative methods. Complex phases remain unfixed; active anchoring can
   increase the matching error. Paramagnetism is preserved in the controls;
   other onsite/permutation symmetries remain unvalidated.
   Keep fit-accuracy handling separate from the coordinate choice.

36. [x] Inner lattice Lambda solve (always, plateau-triggered, with history
   reset). Rejected as a general fix; see the plateau-trigger section.
37. [x] Decoupled-ghost plateau: decompose the residual and recent DIIS steps
   by component (decoupled ghost Lambda/R row vs coupled R/Lambda) on the
   stuck model cases (U=3 d=1.0/1.25, U=4 d=1.6/1.8, fresh, no mixing).
38. [x] Same cases with ScipyRoot (lm, krylov): is the plateau DIIS-specific?
39. [x] Decoupled-ghost Lambda pin (prototype, model): see section below.
   Next: hysteresis on detection; berlingreen (multi-cluster, projectors).
40. [ ] Direct minimization (arXiv:2607.27156) for decoupled-ghost cases;
   see item 17. Last resort.
41. [ ] Optimizer schedules: spec in agent-docs/optimizer-schedule.md.
   R floor considered and not pursued: not a fixed point (residual ~eps),
   induces eps^2 exchange that can change the spin state, and band width
   eps^2 W stays far below T.

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

## Plateau, B=2, model grid and fresh starts (2026-11-03, later)

Berlingreen HS beta=800 from the mixing_period=None Pulay plateau (3e-5):
history 12, restart every 4 (keeping the newest entry; risb's restart_period
empties the history, which spincrossover's record cannot handle) and Fe2 at
bond dim 1000 all stay at 3.4e-5. The lattice-side inner Lambda solve
(scratch level_solve.py, analytic band-response Jacobian, 3-4 band
integrations per cycle) goes 3.8e-5 -> 4.1e-4 -> 5.3e-6 in 8 cycles, filling
1e-7, E -105.0316 eV.

The HS state has one decoupled ghost (|vR| = |Dv| = 0, level = mu, n = 1/2)
in every Fe1 block and Fe2 t2g block. With Slater + SU(2) block2, Fe1's
embedding sectors S = 0..5 are degenerate to 5e-6 eV: S = 5/2 plus five free
ghost spins. B=2 with Sz=1/2 (N odd), started from HS with the ghost removed,
converges to LS (E -100.51 eV), not HS.

Crystal-field model, 15 cases, default start: get_lambda with mixing_period
3/None converges 8/11 of 15 (None: the other four within 3e-6..5e-8, on the
HS branch); inner solve 7/8, LS only.

Berlingreen default start, beta=800, mixing_period=None: inner solve diverges
by cycle 4 (sc 0.6), get_lambda by cycle 10 (sc 0.33, mu +1.2 eV). The
mixing step is needed far from the solution and harmful near it.

## Coupled ghost correction and very cold thermal solver

Report: /tmp/risb-smoother-map/coupled-ghost/report.md.
No production code, defaults, benchmarks or permanent tests changed.

Three B=5 failure inputs, two restart policies, 200 impurity solves per
comparison. The local correction varies two weak modes' R components and
Lambda levels together, retaining every original matching residual.
Fresh finite-difference responses and rejected evaluations count toward its
60-solve allowance, which is deducted from the continuation budget.

The correction reduces the local norm but does not reach a root. After
checking the derivative step near the avoided crossing, the refined method
matches 1/6 continuations, versus 1/6 unchanged restarts. It rescues a
different input and loses the baseline success. Keep it out of production.

At beta=10000, physical-residual Pulay with ThermalLatticeSolver matches
one of three inputs in 72 impurity solves; two runs exhaust 200.
LatticeSolver with identical band smearing matches all three in 15/40/25
solves. Thermal uses analytic inner derivatives, no anchor, and all particle
numbers and spin sectors. Matched outputs pass fresh-solver checks and
agree in energy and physical observables.

The colder-band comparison changes the beta=200 equations; it does not
certify continuous-BZ accuracy. B=1 atomic-limit work is no longer a priority.
Existing affected tests: 40 pass. Ruff, Ruff LSP and strict mypy pass.

## Thermal fits: derivative comparison and basis freedom

Report: /tmp/risb-smoother-map/thermal-fit-diagnosis/report.md.
Scratch only; no production code, defaults, benchmarks or permanent tests changed.

The original comparison below disabled OnPlateau and anchoring and used a
200-solve budget. It is a controlled unanchored experiment, not the library
default. The corrected .solve() comparison follows it.

At beta=10000 on the same three B=5 failure inputs, the unanchored analytic and
finite-difference methods each match 1/3. The successful FD solve takes 25
impurity solves versus 72 analytic. Analytic uses fewer fit evaluations,
but is not an outer convergence improvement by itself.

Both fits have ten exact rotation freedoms in the real one-band B=5 case.
The analytic Jacobian resolves them correctly; forward differences introduce
spurious response. The two methods can fit identical observables in different
bases, which DIIS then mixes. A frozen exact fit fails at residual 1.17e-3
with either derivative method despite reporting optimizer success.

Removing those rotations inside the fit reduces 20 variables to ten while
retaining all physical variables and matching equations. At the original
2000-evaluation fit limit, analytic converges in 15/181/51 impurity solves;
finite differences in 15/42/27. All six match the same lower-energy branch
and pass fresh-solver checks. Post-fit alignment and a bad-bath-fit retry were
also tested; these help some inputs but can select higher-energy branches.

Keep both derivative options. Explicit fit gauge choice and checks of actual
matching error are the promising changes, not a blanket analytic default.
Complex/multiorbital/degenerate cases remain unvalidated. Existing affected
tests: 39 pass. Ruff, Ruff LSP and strict mypy pass.

## Thermal fits: corrected default-workflow comparison

Scratch: /tmp/risb-smoother-map/thermal-fit-diagnosis/default_workflow.py.
Same three inputs and beta=10000; call solver.solve(x0=x) without overriding
optimizer, callbacks, tolerance or iteration limit. Finite differences are
the library default; analytic is a separate option. Both retain OnPlateau,
conditional uniform anchoring, tol=1e-12 and maxiter=1000.

Cycles before one additional validation cycle per run:

| Method | Input | Default | Existing outer lambda gauge | Fit-basis prototype |
| --- | --- | ---: | ---: | ---: |
| FD | Pulay seed 1 | 29 | 18 | 20 |
| FD | BianLi seed 1 | 83 | fails at 1000 | 49 |
| FD | Pulay seed 2 | 621 | 33 | 30 |
| Analytic | Pulay seed 1 | 75 | 23 | 20 |
| Analytic | BianLi seed 1 | fails at 1000 | 35 | 182 |
| Analytic | Pulay seed 2 | 311 | 41 | 51 |

Success requires f1/f2/bath max errors below 1e-8 and filling within 1e-7,
both at termination and on another cycle. Every successful default or fit-basis
run reaches E=-1.2623464126. The outer-gauge analytic BianLi input instead
reaches E=-1.2551201043. Its FD counterpart stalls with bath error 4.18e-8.

The default callback activates anchoring in analytic inputs 1 and 3, at
cycles 54 and 292. Neither the default FD runs nor the fit-basis runs activate
it; they still use the default callback. Thus this sample does not compare
both active penalties inside the prototype, or show that anchoring is
insufficient. Default FD already succeeds on all three. The prototype's
observed benefit is fewer cycles, especially 621 to 30 on input 3.

Existing lambda_eigenbasis rotates the completed outer R/Lambda proposal.
The prototype instead removes rotations from both inner fit searches and
returns matrices in their input coordinates. No DIIS algorithm changes.
Returning coordinates does not enforce onsite or permutation symmetries:
those must constrain the fit itself. The prototype covers only the real
one-band case. Do not treat it as a general symmetry-compatible feature.

The previous unanchored 1/3 result must not be used as the default-library
baseline. All 18 corrected runs and their callback actions, fits and matrices
are saved in default-comparison.jsonl and per-run JSON/NPZ files.

## Does the inner Lambda solve hurt solutions without a decoupled ghost? (2026-11-03)

Scratch: diis/near_bench.py, diis/level_solve.py. Both variants continue from
the first state with max |f1|,|f2| < 1e-3 under Pulay(mixing_period=None).

Crystal-field model, cycles to 1e-9 (default / inner):
- one decoupled ghost: U=4 d=0.3 23/7, U=4 d=0.9 24/10, U=3 d=0.0 25/37,
  U=3 d=0.5 neither (7e-3 / 2e-2, inner drifts to a higher-E state);
- all ghosts coupled: U=3 d=1.5 20/26, d=1.75 22/22;
- integer ghost occupations: U=3 d=2.0 9/157; four others 3-5 both.
Same E and S^2 wherever both converge.

Berlingreen LS, beta=800, from fresh-bianli-lambda entry 22 (all ghosts
coupled), 10 cycles each: default 2.6e-4 -> 1.7e-7, inner 2.6e-4 -> 7.2e-8
with a jump to 4.3e-4 at cycle 2. Both E = -101.0056 eV.

The inner solve is not neutral: mixed on decoupled cases, and one
non-decoupled case is 17x slower. Not suitable as an always-on default.

## Plateau-triggered inner solve (2026-11-03)

Scratch: diis/plateau_trigger.py. get_lambda until no 10x error drop for 8
cycles AND a ghost with |n-1/2| < 0.05 and coupling < 1e-2; then inner for
the rest. DIIS history kept across the switch. Pulay(mixing_period=None).

Never fired without a decoupled ghost: those runs are identical to default.
Where it fired it mostly did damage:
- near-solution grid: 8/11 (default 10/11). U=4 d=0.9 diverged, U=4 d=0.3
  raised (h_imp symmetry), U=3 d=0 fine (switched at 21, same 25 cycles).
- fresh grid: 9/15 (default 11/15, always-inner 8/15). Fired at cycles
  20-69; U=3 d=0.5, U=4 d=0.3/0.9 went from converged to diverged. The
  default's stuck cases (U=3 d=1.0/1.25, U=4 d=1.6, sc 1e-6..1e-8) diverged
  after the switch; U=4 d=1.8 converged, but to LS E=2.20 (default HS 1.846).
Unchecked: whether the switch with stale DIIS history, rather than the inner
solve itself, causes the damage.

Same trigger with TRIGGER_NEAR=1e-4 (best error so far) and TRIGGER_RESET=1
(clear DIIS history, restart from best_x at the switch):
- near grid: 10/11, same as default; U=4 d=0.9 now fine (25 vs 24), U=3 d=0
  23 vs 25. Stale history was part of the earlier damage.
- fresh grid: 12/15 converged but only 10 at the lowest-energy state
  (default 11). On the default's four stuck HS cases (sc 1e-6..1e-8): U=3
  d=1.0 diverged, U=4 d=1.6 raised (h_imp symmetry), U=3 d=1.25 -> E=1.500
  S^2=2.000 (HS 1.427), U=4 d=1.8 -> LS E=2.200 (HS 1.846). U=3 d=0.5,
  converged by default, raised.
The inner solve never took a stuck model case to the HS solution; the
berlingreen HS break-through (I3) has no model counterpart.

## Decoupled-ghost plateau: diagnosis and ScipyRoot (2026-11-03)

Scratch: diis/plateau_diag.py, diis/scipy_bench.py. Stuck model cases U=3
d=1.0/1.25, U=4 d=1.6/1.8, fresh, Pulay(mixing_period=None), root error.

At the plateau each spin block has two decoupled ghosts (one per orbital):
n=0.5000, |R row| ~1e-8, level-mu ~1e-9, exactly degenerate. The other four
are coupled (|R|=0.70, n~0/1). DIIS error swings 1e-7..1e-2, not a stall.
Residual is ~all f2 inside the decoupled 2x2 block (10-1000x the rest),
i.e. rho_qp != 1/2 from level-mu offsets times flat-band stiffness. Steps
spread over all of Lambda and R at 1e-3..1e-2, 10-100x the residual:
get_lambda sets that block from ~1e-8 R/D rows, and the noise moves everything.

ScipyRoot (root error), max|f1|,|f2|:
- default start: lm, hybr, krylov all fail on all four (DIIS gets to ~1e-7).
- from DIIS best_x: lm and hybr stay HS, stop at 2e-8..1e-7 (likely the FD
  Jacobian step). krylov reaches HS at 1e-14 (U=3 d=1.25, 1388 evals) and
  1e-11 (U=4 d=1.6, 372 evals); the other two drift (5.7e-5 HS; U=4 d=1.8 to
  an S^2=0 state at 8e-4).
An exact HS root exists and is reachable; the plateau is the near-singular
decoupled block, not a missing solution. Supports item 39.

## Decoupled-ghost Lambda pin (2026-11-03)

Scratch: diis/decoupled_pin.py. Each cycle, detect ghosts with |n-1/2| < 0.05
and R row < 1e-3 (rho_f eigenbasis, input R); P their projector, Q = 1-P.
Input and output Lambda -> Q Lambda Q + mu P; bands living on P hold
occupation 1/2 during the mu search. R, x, residual and DIIS unchanged, so a
growing R row releases the pin at the next detection.

Crystal-field model, converged to 1e-9 (default / pin):
- fresh, Pulay(mixing_period=None): 11/15 -> 15/15.
- fresh, Pulay(mixing_period=3): 8/15 -> 14/15 (U=4 d=2.1 fails in both).
- near-solution grid: 10/11 -> 11/11; decoupled cases 23-25 -> 5-9 cycles.
- from the four DIIS plateau states: 4/4 in 4-6 cycles.
Every converged run ends at the lowest-energy state; no-decoupled runs are
unchanged. Path effects: U=3 d=2.5 and U=4 d=2.6 (mix 3) held for one cycle
then released and still reached LS. U=4 d=0.3 (no mix) toggled on/off 33 of
228 cycles: 228 vs 38 cycles. Detection without hysteresis chatters.

ScipyRoot cost for reference: lm/hybr build FD Jacobians (n+1 evals each).
Berlingreen x has 180 entries; at ~6 min per HS cycle that is ~18 h per
Jacobian. On the model, lm used 494-2031 and hybr ~145 evals from the
plateau without reaching 1e-9; all failed from the default start.

Berlingreen HS beta=800, from A entry 9 (err 1.6e-4), Pulay no mixing, 10
cycles with the pin: err 1.6e-4 -> 1-9e-3, E -105.034 -> -105.033. Held
count chatters 16/10/16/4/10. Near-1/2 ghost couplings per cycle:
Fe1 eg (up_0,3) stay 4e-8; Fe1 t2g grow 7e-6 -> 2e-3 (x1.8 per cycle;
unpinned A2 stays 6e-6..2e-5); Fe2 t2g jump 5e-5 -> 5e-3 at the first step
(fresh history, so a plain mixing step: the known repelling direction) and
hover around the 1e-3 threshold. Input levels pinned at the previous mu sit
~0.01 meV off. Filling error -4..-8e-4, as unpinned A2 (-5e-4); I3 1e-7.
The model result does not carry over: weakly coupled, not decoupled, ghosts
cross the threshold, and the Fe1 t2g R rows are repelled under the pin.

## Thermal fit basis: multiorbital weak-ghost validation

Scratch: /tmp/risb-smoother-map/thermal-basis-validation/report.md.
No implementation or permanent tests added by this investigation.

The B=5 621-cycle slowdown is not a decoupled-ghost example: the half-filled
pole coupling starts at 0.339 and ends at 0.373; no saved cycle has a
half-filled pole coupled below 1e-2. The prototype removes redundant fit
rotations there, but that does not establish a weak-ghost convergence fix.

Selected two-orbital U=4, J=0.7, crystal field 0.3, B=3. Its T=0 plateau
has two half-filled weak modes per block, coupled at 1.8e-6 and 2.6e-8;
other modes couple at 0.70. Thermal beta=10000 comparisons retain .solve
defaults and stop after a 600-second safety limit. In a common 60-cycle
prefix, best max f1/f2/bath is:
- unrestricted FD 8.6e-6; fixed FD 1.6e-6;
- unrestricted analytic 4.0e-9; fixed analytic 1.2e-6.
Unrestricted analytic first physically matches at cycle 50 and later uses
the default anchor. Fresh checks pass with and without its anchor, but
the 1e-12 optimizer stopping target is not reached. These runs leave the
initially decoupled regime.

A cleaner local control first converges the four-active-mode thermal root,
adds two exactly free modes per spin, then perturbs their couplings by 1e-6.
With the same 60-cycle budget and a 300-second safety limit:
- unrestricted FD/analytic physically match by cycle 4 and remain matched
  in the common first 13 cycles;
- fixed FD never matches in 60 cycles;
- fixed analytic first matches at 25, then loses the match by 28.
Completed cycles / actual impurity calls: full FD 13/14, full analytic
22/23, fixed FD 60 plus validation/61, fixed analytic 28/29.
None reaches the optimizer stopping target. Timed-out runs are not
completed default solves.

A frozen update confirms the intended mechanism: fixed coordinates reduce
the same quasiparticle step from 1.65e-5 to 1.09e-7 at matching error 3.8e-12.
It still does not improve the full map in the weak-ghost control.

The complex fit retains six independent row phases. Frozen repeated-level
controls fit accurately, but active uniform/response-weighted anchors can
increase complex qp matching error from about 3e-9 to 1.4e-8 when the
eigenbasis is fixed. Do not assume the penalized objective is gauge invariant.

At saved six-mode points, Fockit retains 32 thermal states near weak ghosts,
versus two at the coupled root; one-probe impurity times are 13.0 versus
1.44 seconds. Fit work is about 0.049 seconds in either probe. Near weak
ghosts, unnecessary outer cycles are especially costly.

Do not implement the current prototype yet. Separate redundant rotations,
additional insensitive ghost parameters, and the stopping criterion.
Existing affected tests: 39 pass. Scratch Ruff, Ruff LSP and strict mypy pass.

Same start, PIN_COUPLING_TOL=1e-6 (Fe1 eg only, 4 held; 10 at cycle 9 when
Fe1 t2g rows dipped below 1e-6): err 1.6e-4 -> 3.5e-4 (empty-history mixing
step) -> 1.2e-4 (cycle 8) -> 2.0e-4. Same as unpinned A2 (1.9e-4 -> 1.6e-4).
Harmless but no gain: the berlingreen plateau is not in the truly decoupled
Fe1 eg ghosts. Next: decompose the berlingreen residual by block/ghost as
plateau_diag did for the model.

### Berlingreen plateau breakdown (diis/berlin_diag.py, saved history only)

A2 plateau (err 1.6-2.4e-4): residual spread evenly, ~4e-5 per decoupled
ghost on all 16 (Fe1 eg, Fe1 t2g, Fe2 t2g); steps on them 1e-8..1e-6.
All decoupled levels share one offset, level - mu = -0.2 ueV, so each flat
band has filling n_qp = 0.50004 against n_f = 0.500000 (beta/4 x 2e-7 eV = 4e-5).
Summed over the 16: +6-7e-4 electrons in the flat bands, matching the stuck
filling error (-5e-4) of the physical density. get_lambda gives no update to
that common offset, and the coupled sector's count error is buffered by the
flat bands rather than corrected. I3 (inner solve) cut the filling error to
1e-7, consistent with it fixing this offset jointly with mu.
P2 (Fe1 eg held at 1/2): the other 12 still carry +4-7e-4; same plateau.

### Common-offset Newton step (diis/offset_pin.py)

One offset delta = level - mu for all decoupled ghosts, set each cycle by a
Newton step on the flat-band count with mu re-solved: g_eff = g_flat g_rest /
(g_flat + g_rest). Output Lambda only.
- Model: fresh no-mix 10/15, near 8/11 (pin 15/15, 11/11). Failures are mu
  search errors: gapped rest (g_rest/g_flat ~1e-7) makes delta slaved to mu.
- Berlingreen (density interaction, from A entry 9): g_flat 3200/eV, g_rest
  82/eV. Step 0.3: err 1.6e-4 -> 1e-3 by cycle 2 -> 4e-2 at cycle 10; step 1:
  -> 1.1e-1. Fe2 t2g leave the decoupled set at cycle 2 (held 16 -> 10) and
  return/leave again; flat-band excess swings +6e-4 .. -1.6e-2. DIIS
  extrapolation overrides the output offset (input -0.32 ueV after +0.29).
Rejected. Note: spincrossover ee8bb01 made Slater the berlingreen default;
pass INTERACTION=density to compare with earlier runs.

Summary for berlingreen HS beta=800 from A entry 9: only the inner Lambda
solve (I3) went below the 1.6e-4 plateau. Every flat-band/charge lever
(hold 1/2, offset step) kicks the Fe2 t2g ghosts, which are weakly coupled
(|vR| 1e-5..1e-2), not decoupled, and sit on the known repelling direction.

## Thermal multiorbital comparison at beta=100 (2026-11-04)

Scratch: /tmp/risb-smoother-map/thermal-basis-validation/beta-100/report.md.
Same weak-ghost U=4, J=0.7, crystal-field 0.3, B=3 starting state.
Both lattice and impurity beta=100. Use committed snapshot f36e53f to exclude
concurrent optimizer/callback edits. Retain .solve() defaults and a
600-second safety limit for each variant.

No variant returns a converged solve or passes all physical checks.
Completed cycles / impurity calls:
- unrestricted FD 368/369: max matching 4.88e-8, filling error 2.25e-7;
- unrestricted analytic 81/82: matching 6.28e-4, filling error 4.49e-5;
- fixed FD 88/89: matching 2.03e-4, filling error 7.39e-5;
- fixed analytic 178/179: ends fully filled (4, target 2), despite vanishing
  matrix residuals. Not a valid solution.
A fresh default .solve() restart from the FD plateau also fails to recover
within 180 seconds/100 impurity calls; it reanchors after 22 cycles.

The initial ghost dispersion scale is roughly R_ghost^2 W ~1e-12 or less.
Both temperatures are large on that scale. In a frozen response check,
warming reduces the remaining weak responses by about 100 while the leading
response stays near 0.11. Warmer does not mean better-determined ghost fits.

First cold near-root divergence: the bath fit matches to 1e-12 in either
basis, but its weak bath-hole occupations differ (0.477/0.622 unrestricted;
0.920/0.998 fixed, FD). This is a bath eigenvalue choice, not simply a basis
rotation. Fixed eigenvectors do not remove this ambiguity.

Next candidate, untested: retain full matrix fits and use the existing
response-weighted penalty from the first fit, referenced to the parameters
entering that fit. Freeze references/weights inside the local fit; update
references on the next fit. This differs from the current persistent anchor
and need not impose its bias at a true fixed point. Verify original
f1/f2/bath/filling and branch energy under reduced regularization.
No implementation changes made. Matching committed tests: 39 pass.
Warm scratch scripts: Ruff, Ruff LSP and strict mypy pass.

## Weak-ghost model scan (diis/weak_scan.py, 2026-11-04)

The script is diis/weak_scan.py. Setup:
- the crystal-field model with doping, B=3;
- U/J = 3/0.5 or 4/0.7;
- doping 0.3, 0.6 or 1.0;
- δ from 0 to 2;
- fresh Pulay(mixing_period=None), maxiter 300.

Findings:
- **Weakly coupled ghost (n=½, |vR| 2–4e-3, coupled ghosts fractional).** Seen at U=4 dop=1.0 δ=0.5/1.0/1.5 and U=4 dop=0.6 δ=0.5. None of these converge in 300 cycles (sc 4e-2..2e-1). The ghost always sits in the orbital at n=0.5, not in a doped one (berlingreen Fe2 t2g is at 0.664). U=3 dop=1.0 δ=1.5 has a weak ghost at 2e-6 with sc 4e-3.
- **Doped charge-sink state.** At U=3 δ=0.5/1.0 and U=4 dop=0.3, n_c = [0.5, 0.5] for every doping. There are 2 exactly decoupled ghosts, the same E=1.427 (U=3) at every doping, and sc 5e-2..2e-1. The doping charge sits in the flat decoupled-ghost bands. This is the berlingreen flat-band excess mechanism, but at full size.
- **Converged cases.** Undoped-like converged cases with decoupled ghosts (U=4 dop=0.6 δ=1.0/1.5) take 196–282 cycles.

Candidate test bed: U=4 J=0.7 dop=1.0 δ=1.0 (weak ghost 4e-3, coupled ghosts 0.18/0.68).

### Weak-ghost and charge-sink cases: default vs pin vs inner (diis/weak_bench.py)

Setup: fresh Pulay(mixing_period=None), maxiter 300. `fill` from observe is relative to the undoped count, so the correct value is fill = doping.

Charge-sink cases:
- U=3 dop 0.6 δ 0.5/1.0;
- U=4 dop 0.3 δ 0.

Weak-ghost cases:
- U=4 dop 1.0 δ 0.5/1.0/1.5;
- U=4 dop 0.6 δ 0.5.

| case | default | pin | inner |
|---|---|---|---|
| sink U3 d.6 δ.5 | frozen 8e-2, fill 0 | 5e-3 (best 1e-3), fill .6 | 7e-2, fill .4 |
| sink U3 d.6 δ1 | frozen 1e-1, fill 0 | 1e-2, fill .6 | 3e-1, fill 2 (wrong) |
| sink U4 d.3 δ0 | frozen 6e-2, fill 0 | 1e-1, fill .2 | conv 231, fill .3 |
| weak U4 d.6 δ.5 | 4e-2 (best 4e-3) | 2e-2 | 3e-1, fill 2 (wrong) |
| weak U4 d1 δ.5 | 8e-2 | conv 239, n_c=[1,.5], ghost→dec | 7e-1, fill -2 (wrong) |
| weak U4 d1 δ1 | 2e-1 | 1e-1 | 3e-1, fill 2 (wrong) |
| weak U4 d1 δ1.5 | 2e-1 | 8e-2 | 2e-1, fill 0 (wrong) |

Findings:
- **Charge sink.** Default DIIS sits exactly frozen there: sc is constant and the doping charge stays in the decoupled flat bands. The pin always releases the charge but converges in none of the 3 sink cases.
- **Weak-ghost cases.** Under default DIIS these wander between 1e-2 and 5e-1. They are chaotic, not a smooth plateau like berlingreen (1.6e-4). The pin converges 1 of 4.
- **Inner solve.** It ends at the wrong filling (|fill−doping| up to 3) in 5 of 7 cases, and converges 1 of 7. Rejected for doped systems.
- **Pin overall.** Best of the three (1/7 converged, 6/7 with the charge released), but not robust.
- **Open question.** Whether a fixed point with a weak (not decoupled) ghost exists in these cases is unknown. The one pin convergence ends with the ghost fully decoupled.

### Do these cases have fixed points? (diis/root_check.py)

Method: ScipyRoot hybr (plain get_lambda, maxfev 3000), started from the best DIIS x (default or pin start, 300 cycles).

Results:
- Root found in 2/14 runs (sc ≤ 1e-10), both from pin starts:
  - U3 dop.6 δ.5: n_c=[.8,.5], E=4.132;
  - U4 dop1 δ.5: n_c=[1,.5], E=8.218 (already converged by the pin).
  Both are orbital-selective: orbital 2 at exactly ½, its ghost decoupled (|vR| ~1e-11), the other ghosts 0/1.
- The remaining 12 runs stop at 1e-3..8e-2 after 150–900 evals. Most sit in the same orbital-2 = ½ state with a weak ghost (2e-4..7e-3) whose coupling shrinks along the way. So "weak ghost" here looks like a transient towards a decoupled fixed point, where the Jacobian is singular. It is not a separate weak-ghost solution, and so it is unlike berlingreen Fe2 t2g (0.664).

Pin detection bug (U3 dop.6 δ.5):
- The pin held 0 ghosts for all of the last 80 cycles, yet hybr found a root there with 2 decoupled ghosts.
- Cause: the two near-½ ghosts are degenerate in rho_f, so eigh returns an arbitrary mix of the coupled (2e-2) and decoupled (1e-5) directions, and both exceed COUPLING_TOL.
- Fix: within the near-½ eigenspace, take the SVD of the R rows and hold the directions with singular value < COUPLING_TOL.

Pin chatter: detection still flips 0↔2 every few cycles (U4 dop.6 δ.5, U4 dop1 δ1/δ1.5). It needs hysteresis.

### Pin: SVD detection and hysteresis (diis/decoupled_pin.py v2; v1 kept as decoupled_pin_v1.py)

Changes:
- Detection is now the SVD of the R rows within the rho_f eigenspace near ½.
- Hysteresis: a block keeps up to its previous held count while the singular values stay below PIN_RELEASE_TOL (default 10×COUPLING_TOL; set it to 1e-3 to turn hysteresis off).

| | v1 eig | v2 SVD+hyst | v2 SVD only |
|---|---|---|---|
| undoped no-mix (15) | 15/15 | 14/15 (U4 δ2.1 fails) | 14/15 (U4 δ2.1 fails) |
| doped (7) converged | 1 | 0 | 0 |
| doped best sc | 3e-10,1e-3,8e-3,... | 2e-4,3e-3,1e-6,4e-3,1e-2,2e-2,4e-2 | 1e-3,5e-7,3e-6,2e-2,5e-5,1e-2,4e-2 |

Findings:
- No clear win. Per-case outcomes change a lot from one variant to the next, so the 7+15 sample is dominated by chaos.
- Doped pin runs reach the hybr roots (U3 dop.6 δ.5: E=4.1318, matching hybr), down to 5e-7..5e-5. Then they bounce back up and do not finish in 300 cycles.
- The held set still flips 2→0 occasionally even near the root (U4 dop1 δ1 at 2e-6, coupling 6e-12). That suggests a DIIS extrapolation jump transiently raises the coupling. It is not detection noise.
- Hysteresis keeps the U4 dop.3 sink held (4 ghosts), which preserves the sink.
- Remaining failure mode: DIIS fails to finish near a decoupled solution, because each change of the held set changes the map and the history mixes both maps.

### Pin + history reset on held-set change (decoupled_pin.reset_on_change, PIN_RESET=1, SVD, no hysteresis)

Mechanism: when the per-block held counts change, the callback takes a plain step to the last output and clears the DIIS history.

Undoped no-mix:
- 15/15 converged, but 5 end on a higher-energy state than v1. For example, U3 δ.5 gives E=2.000 vs 1.427; also U3 δ1/δ1.75 and U4 δ.3/δ1.8.

Doped:
- 2/7 converged:
  - U4 d1 δ1: E=7.9682, decoupled ghost, 91 cycles;
  - U3 d.6 δ1: E=3.9836, a coupled-ghost solution, 118 cycles.
- The others reset 10–42 times. Most of them reach 2e-7..8e-6 and are then kicked into fill = 2 or [½,½] states.

Rejected: the plain steps after each reset jump branches and lose the lowest-energy solution. Overall: pin v1 (eig detection, no reset) is still the best on the undoped model, and nothing tested is robust on the doped model.

## Ligand model (diis/ligand_model.py, ligand_scan.py, 2026-11-04)

Model:
- the crystal-field model plus a ligand orbital per spin, with dispersion e_k+eps_L, hybridized by V with both correlated orbitals;
- n_target=3, U/J = 4/0.7, B=3;
- fresh Pulay(mixing_period=None), maxiter 300;
- scan: eps_L 0.5..2, V 0.3/0.6, δ 0..2.

Findings:
- **Berlingreen-like plateaus.** 9 of the 40 runs end unconverged at sc 2e-7..6e-4 with a weak ghost (|vR| 1e-7..1e-3, or "dec" at 1e-7), in an orbital at n=0.50–0.51 (not exactly ½), with the other orbital fractional (0.02–0.25). Examples:
  - eps_L=2 V=.3 δ=1: 5e-4, |vR| 2e-5;
  - eps_L=2 V=.3 δ=1.5: 5e-4, |vR| 9e-6;
  - eps_L=1.5 V=.3 δ=1: 6e-4, |vR| 1e-5.
- **Other failures.** A few runs end at 1e-2..3e-1 with a coupled ghost near ½ (|vR| ~0.1–0.5).
- **Converged runs.** The ones that converge have a ghost at exactly ½ with |vR| 1e-11 ("dec"), a coupled ghost at ½ (~0.5), or no ghost near ½.
- **Comparison with the doped model.** Unlike the doped model, the stuck runs here sit close to a solution, as berlingreen does. The weak-ghost orbital is still near ½, though.

### Ligand plateaus: default vs pin v1 vs hybr (diis/ligand_bench.py)

Measured at the final state: excess = Σ_flat w(f−½), offset = mean decoupled level − μ.

**Easy class: 7 cases** (eps_L 0.5/1.0)
- Default stalls at 9e-8..1e-5, and its excess/offset = −50 = −2·β/4 in every case. This is exactly the berlingreen signature: a common offset charging the flat bands.
- The pin converges 7/7 in 50–178 cycles, to the same energy (ΔE ≤ 7e-5), with the ghost decoupled (1e-10..1e-13).
- Hybr stalls at 1e-9..1e-5 after ~145 evals, because the Jacobian is singular.

**Hard class: 5 cases** (eps_L 1.5/2.0, V 0.3; the coupled orbital at 0.10–0.32)
- Default reaches 5e-4..3e-2, with the weak ghost at |vR| 1e-5..9e-4 and an excess of 5e-4..4e-2.
- The pin reaches 1e-3..1e-1 (best 2e-4..9e-4).
- Hybr reaches 1e-4..8e-4.
- Nothing converges. This matches berlingreen: plateau 1.6e-4, excess 6–7e-4, and the all-16 pin failed there too.

**Controls**
- Default and hybr converge all 3.
- The v1 pin fails eps_L=2 δ=0 (6e-5), which has 2 degenerate decoupled ghosts. It holds only one of them (2e-3 / 1e-5), which is the eigh-mixing detection bug that the v2 SVD detection addresses.

### Ligand hard class: temperature and SVD pin (diis/hard_ligand.py, ligand_bench.py pin2)

**Temperature** (default DIIS, fresh, then hybr from its best x)

| β | default converged | hybr from best |
|---|---|---|
| 100 | 0/5 (best 1e-4..2e-3) | stalls at the same point (145 evals, no progress) |
| 200 | 4/5 | 1e-14 |
| 400 | 5/5 | 1e-14 |

- The converged solutions have the ghost decoupled (|vR| ≤ 1e-10), except eps_L 1.5 δ1.5, where it is coupled (0.3).
- The solutions found at β=200/400 have energies close to the β=100 stuck points. For example, eps_L 1.5 δ1: 2.21983 (β 200) vs 2.22069 (β 100 stuck).
- At β=100 excess/offset = −50 = −2β/4 again. The residual is split comparably between the near-½ ghosts and the rest (e.g. 5e-4 / 8e-4), so it is not only a decoupled-ghost problem.
- Reading: at β=100 hybr makes no progress from a non-root point, and the residual has a nonzero minimum. That looks like a fold (saddle-node) where the decoupled-ghost branch stops existing as T rises. Untested.

**SVD pin** (with hysteresis / without)
- Easy class 7/7 both ways, with the same energies.
- Hard class 0/5 both ways.
- Control eps_L 2 δ0 fails both ways (3e-4 / 2e-5), although default converges it.
- The held ghosts there have coupling 1e-3 and 4e-4, i.e. not yet decoupled. Pinning a ghost whose R row is ~1e-3 changes the map away from its fixed point, so COUPLING_TOL=1e-3 is too loose for this case.

### β continuation of the hard class (diis/beta_cont.py) and pin threshold 1e-5

**Continuation:** default DIIS at β=400, then hybr at each lower β from the previous x.

The 4 decoupled-ghost cases:
- Follow the solution to β=115 with sc ≤ 5e-8 (mostly 1e-13), coupling ≤ 3e-7, and E smooth.
- At β=100 all 4 fail together: sc 1e-4..6e-4, and the coupling grows to 4e-5..1.2e-3.
- That the parameters differ yet the β at which they fail is shared suggests a common mechanism. A candidate is a pitchfork, where the decoupled ghost starts to couple: a weakly coupled branch R ~ sqrt(β_c−β) with a near-singular Jacobian. That would be like berlingreen Fe2 t2g. Untested.
- No temperature threshold was found in the k-integration code.

The fifth case, eps_L 1.5 δ1.5 (coupled ghost, 0.3), fails from β=200 down (sc 1.5e-3..8e-3).

**Pin with COUPLING_TOL 1e-5**

| | easy 7 | controls 3 | hard 5 |
|---|---|---|---|
| with hysteresis | 6/7 | 1/3 | 0/5 |
| no hysteresis | 7/7 | 2/3 (eps_L 2 δ0 at 6e-7) | 0/5 |

- No hysteresis at 1e-5 is the best pin setting so far, but it is slower than 1e-3 (62–232 vs 50–178 cycles).

### Fine and warm β (beta_cont.py FINE=1; hard_ligand.py at β 80/60/40)

**Continuation** (3 cases): the 1-decoupled-ghost branch converges to 1e-14 down to β≈104–110 (case dependent; hybr is erratic near the edge). Warmer, every case fails, and the stuck residual grows steadily with T: 1e-4 at 100, then 1e-3..6e-3 at 40. The coupling at the stuck point stays 1e-5..1e-3, with no sqrt growth.

**Fresh DIIS + hybr**
- eps_L 2 δ0.5 and δ1 converge at β 80/60 (δ0.5 also at 40) to solutions with **2** decoupled ghosts.
- eps_L 2 δ1 at β 40: 2 decoupled ghosts plus 1 coupled ghost near ½.
- Fillings are close to the cold branch (e.g. [0.502, 0.24] vs [0.501, 0.249]).
- eps_L 1.5 δ1/δ1.5 and eps_L 2 δ1.5: no solution at β ≤ 80 (hybr best 8e-4..4e-3).

**Reading**
- Not a pitchfork to a weakly coupled solution.
- The 1-decoupled-ghost representation stops being a solution near β≈105 (fold-like: the minimal residual grows from 0 as T rises).
- Warmer, the same physical state needs a different number of decoupled ghosts. Neither DIIS nor hybr moves between those representations.
- Not yet tested: whether decoupling one more ghost from the stuck β=100 state gives the solution.

### Decouple-one-more-ghost move (diis/decouple_test.py, eps_L 2 δ1)

Ghosts in the stuck state vs the converged solution (up block; n/|vR|):

| | ghost occupations (coupling) |
|---|---|
| stuck β80 | .011(.65) .026(.57) .30(.63) .5(8e-4) .92(.34) .99(.67) |
| solution β80 | .012(.65) .107(.72) .5(1e-11) .5(9e-12) .64(.47) .99(.66) |

So the second decoupled ghost does not come from one ghost moving to ½. The .03/.30/.92 ghosts reorganize into .11/.5/.64.

Move: decouple ghost g of the stuck state (R row → 0, Λ → QΛQ + μP), then hybr. Tried on every ghost not near ½ (5 per β):
- **β80:** only ghost 1 works (n=.026, coupling .57, i.e. strongly coupled). It reaches 2e-14 at the reference solution (E=3.03990, identical). The other 4 end at 7e-3..2e-1.
- **β60:** none of the 5 works (4e-3..1e-1).

Not a general fix: 1 of 10 tries worked, and the right ghost is neither the weakest coupled nor the one nearest ½.

### Hard class with B=2 (hard_ligand.py, env B=2)

- Default DIIS converges 14/15 (β 100/80/60 × 5 cases). The exception, β80 eps_L 2 δ1.5, is at 3e-3, and hybr finishes it at 5e-13.
- Every solution has 1 decoupled ghost, and excess/offset = −β/2 at the DIIS end, as before.
- B=2 energies are higher than B=3 by ~0.015:
  - eps_L 2 δ1 β80: 3.05491 vs the B=3 2-ghost solution 3.03990;
  - eps_L 1.5 δ1 β100: 2.23476 vs B=3 2.22053 at β115.
- Fillings differ by ~0.015 too (second orbital 0.128 vs 0.143).
- So the hard class comes from B=3 ghost redundancy. B=2 is robust here but less accurate.
- B=2 is not available for berlingreen: 5 orbitals per spin block give 5 ghosts per spin (odd), so there is no Sz=0 half-filled state.

### Check: B=3 started from a B=1/B=2 solution (diis/embed_test.py)

This is a diagnostic only. It is not a general fix, because B=2 is not a valid metal (g-GA paper) and B=1 is plain RISB.

Setup: the extra ghosts get level μ and R rows seeded at 1e-2. Then default DIIS runs, followed by hybr if DIIS does not converge.

**From B=2:** 12/15 converge with DIIS, including 4/5 at β=100, where fresh B=3 fails.
- All have ≥1 decoupled ghost.
- At β80 eps_L 2 δ1 this reaches the known 2-ghost solution, E=3.03990.
- β=100 solutions, e.g. eps_L 2 δ1 3.03847 and eps_L 1.5 δ1.5 2.02854, lie above the fresh-B=3 stuck near-misses (3.02766 at eps_L 2 δ1), which are not solutions.

**From B=1:** 12/15 converge, but mostly to another B=3 solution.
- That solution has a coupled ghost near ½ (|vR| 0.4–0.6) and an energy 0.02–0.04 higher. For example, eps_L 2 δ1 β80 gives 3.07569 vs 3.03990.
- 3/15 reach the low branch.

**Reading:** B=3 has several self-consistent solutions, which differ in ghost arrangement, and the start picks the branch. A B=1 start, the only small-B start possible for berlingreen, tends to pick a higher-energy branch.

### Exact impurity chemical potential loop (diis/imp_mu.py, imp_mu_bench.py)

Scheme:
- qp bands are filled at a fixed chemical potential 0;
- the physical μ acts only on physical orbitals: −μ on the impurity physical orbitals (h0_loc_matrix) and −μ on the ligand in the lattice;
- each cycle, Brent solves μ so that Σ impurity Tr ρ_c + ligand density = 3. Every trial is a full set of impurity solves.

Why it is equivalent: at fixed embedding particle number, −μ n_c is the same as Λc → Λc − μ. So this is the standard scheme with μ absorbed into Λ, and it has the same fixed points (confirmed: identical energies where both converge).

**μ placement**
- In zero-T LatticeSolver, mu=None by default, so the embedding gets no μ, which is standard.
- If mu is passed, −μ N_total in the fixed half-filled sector is a constant and has no effect.
- ThermalLatticeSolver puts μ in H_qp and also −μ N_total on the impurity, whose ensemble spans N±1.., where it is not a constant. By the argument above, the standard scheme has no μ in the embedding, so this looks suspicious. Unverified against the thermal derivation.

**Results** (Pulay, mixing_period=None, 300 cycles)

| | default | pin v1 | imp μ |
|---|---|---|---|
| β100 easy (7) | 0/7 (9e-8..1e-5) | 7/7 | 5/7 (others 4e-4, 3e-6) |
| β100 hard (5) | 0/5 | 0/5 | 0/5 (1e-3..1e-2) |
| β100 controls (3) | 3/3 | 2/3 | 2/3 (eps_L 2 δ0 at 2e-2) |
| β80 hard (5) | 2/5 | — | 2/5 (same solutions; eps_L 2 δ1.5 at 8e-5) |

- Cost: ~8–10 impurity solves per cycle (bracket plus Brent), against 1 for default.
- Reading: taking the decoupled flat bands out of the filling fixes most easy plateaus, but less well than the pin and at ~9× the cost. It does not touch the hard class, as expected if there is no solution on that branch, and it breaks one control.

### Impurity μ as an optimizer unknown (diis/imp_mu_diis.py)

Same scheme as the exact loop, but μ is appended to x, as ThermalLatticeSolver does: the update is μ − (n − n_target)/SLOPE, with residual n − n_target. One impurity solve per cycle, plus ~13 for an exact starting μ.

| β100 | default | pin v1 | imp μ exact | μ in x, slope 1 | μ in x, slope 0.3 |
|---|---|---|---|---|---|
| easy (7) | 0/7 | 7/7 | 5/7 | 3/7 (+2e-8, 6e-7, 6e-6) | 5/7 (one at 3e-10 counted) |
| hard (5) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| controls (3) | 3/3 | 2/3 | 2/3 | 2/3 | 2/3 |
| solves/cycle | 1 | 1 | ~9 | 1 | 1 |

- Failures:
  - eps_L 1 δ1.5 (easy) ends in a coupled-ghost state at 4e-2..5e-2;
  - control eps_L .5 δ.5 ends at 3e-3/1e-2 (default converges it).
- β80 hard, slope 1: 2/5, the same solutions as default; eps_L 2 δ1.5 at 4e-7.
- Reading: as good as the exact loop at 1/9 of the cost, with two regressions. Still no effect on the hard class. The pin remains better on the easy class.

### Where μ goes (checked against the code and arXiv:2603.20559 Eq. 27)

- The quadratic embedding H0_emb (get_h0_emb) has −μ on both the quasiparticles f and the bath b. The paper's λ and λ^c include μ; the code pulls it out.
- The impurity bath must equal H0_emb's bath, so it also gets −μ n_b. The physical c replace f and get −μ n_c. So −μ N_total on the thermal impurity is correct.
- Zero T: the fixed half-filled sector makes −μ N_total a constant, so LatticeSolver drops it and μ acts only via f(E−μ). This is the same physics.
- "μ on physical orbitals only" is right only in the gauge where μ is absorbed into Λ (qp chemical potential fixed), as in imp_mu*.py.
- Thermal tests pass: low-T equals zero-T, fixed filling equals fixed μ, and N = −∂Ω/∂μ.

### Hard class near T=0 (hard_ligand.py at β 2000, 1e5)

- Default DIIS converges all 5 hard cases (sc ≤ 2e-10; hybr then reaches 1e-14).
- Mostly they reach a different solution: the ghost near ½ is coupled (|vR| 0.18–0.33) and no ghost is decoupled. The energy is below the β=400 decoupled solutions (eps_L 1.5 δ1: 2.21737 vs 2.21966).
- The exception, eps_L 2 δ1 at β2000, still has a decoupled ghost.
- So the hard class exists only near β≈100 (the fold). At T=0, where the benchmarks/ energy formulation is valid, there is nothing hard here to test it on.

### Correction: β in the ligand and berlingreen sections is band smearing

- Every ligand-model run and berlingreen use LatticeSolver: the impurity ground state in the half-filled sector, with β only Fermi-smearing the qp bands.
- "Temperature", "warmer" and "colder" above mean smearing width. The fold near β≈105 means the decoupled-ghost solution of the smeared equations disappears as the smearing grows.
- The thermal saddle in benchmarks/adiis_hard.md belongs to ThermalLatticeSolver (thermal impurity) and does not apply here.
- The zero-T energy formulation applies with smearing if the band energy becomes the Mermin form E_band − T S_smear.

### Energy objective on the ligand model (diis/energy_ligand.py, energy_gradcheck.py)

Port of benchmarks/adiis.py:
- y = (D, upper Λc) of the up block, 33 coordinates, no gauge fixing;
- impurity ground state, then R = get_r, then an inner Λ fit (hybr, 21 unknowns) to ρ_qp = ρ_f in the smeared, fixed-N bands;
- F = Σ f E − Tr Λρ_qp − S/β + ⟨h_int + h0_loc⟩ (Mermin form);
- mapped fields from get_d/get_lambda_c, and the adjoint gradient as in adiis.py (MINRES on the 4900-state Sz=0 sector).

No optimizer comparison was possible:
1. **Default start:** the inner fit is not representable (the "representable target" of the note). The impurity gives n_c = 0.40 per spin, so the fixed total N needs a ligand charge of 1.1 per spin; the fit floor is 0.024. This needs the note's canonical μ extension, which is not implemented.
2. **Decoupled-ghost solution** (control eps_L .5 δ.5): the inner fit fails (8e-2) even at the converged solution.
3. **Hard stall point** (eps_L 2 δ1, DIIS best): the inner fit is unique for a fixed y, but F jumps by 0.06–0.1 for |δy| = 1e-5. Finite differences grow like 1/h, and the adjoint does not match.
   - Cause: the impurity ground state is degenerate. In the half-filled Sz=0 sector the two lowest levels differ by < 5e-7, the next by 0.067.
   - At a regular coupled-ghost solution (eps_L 1.5 V.6 δ1) the gap is 0.069, and the adjoint and finite differences agree to the noise (~5e-8, at a stationary point).
4. **Reading:** a ghost at ½ with R row ~0 makes the embedding ground state (quasi-)degenerate, with gap ~ coupling². A single ground state then gives ρ_f for that ghost anywhere in the degenerate pair, so the map and the energy are discontinuous there. This affects the root DIIS map too, and may be why the stalls are plateaus.
   - An exactly decoupled ghost with ρ_f = ½ needs an equal-weight ensemble of the pair, which the zero-T solver does not form.
   - Next check: FockitImpuritySolver(beta=...) weights states within the half-filled sector. It would average the pair.

### gem on the ligand cases (diis/gem_ligand.py; gem cloned to scratchpad/gem-src, pure Python + numba)

How gem's zero-T loop (example E3) works:
- bands are Fermi-smeared (Tsmearing = 1/β) at a fixed chemical potential;
- μ sits on the impurity physical orbitals, refitted (mode 'imp', regula falsi) when |n − n_target| > 1e-4;
- impurity: ground state in the half-filled Sz=0 sector;
- linear mixing of Λ and R with 0.2 old weight (no DIIS);
- convergence measure: the change in the Λ eigenvalues and in |U R| (U = Λ eigenvectors) between input and output, i.e. a gauge-invariant step, with tol 1e-4.

Model setup: the ligand is a U=0, B=1 fragment, and the interaction is density-density Kanamori, built by hand. On the control eps_L 1.5 V.6 δ1 it reproduces risb (μ 2.31550 vs 2.31551, the same n_c).

Results at β=100, 300 iterations, run to a step of 1e-10:
- easy 7/7 (30–95 iterations, ghost |vR| ~1e-15);
- hard 4/5 (82–202 iterations; 2 decoupled ghosts, e.g. eps_L 2 δ1 n_c [0.5016, 0.2432]); eps_L 1.5 δ1.5 at 2e-2;
- controls 2/3 (eps_L .5 δ.5 at 3e-4, which risb default converges);
- 78–444 impurity solves.

So the hard-case solutions exist at β=100 and gem reaches them. Candidate reasons, untested:
1. linear mixing instead of DIIS;
2. impurity μ (alone with DIIS it gave 0/5 hard);
3. gem's start (tanh levels, R = D = 1/√B);
4. the ligand as a fragment.

### ThermalLatticeSolver on the ligand cases (diis/thermal_ligand.py, stopped early)

Setup: FockitThermalImpuritySolver (lowest, delta 2, eta 8), production default solve, β=100. About 6–30 s per cycle with 14 parallel jobs. Stopped after 3 of 15 finished.

All 3 converged:
- eps_L .5 δ.5: 63 cycles, n_c [0.488, 0.053], no ghost near ½;
- eps_L 1.5 V.6 δ1: 55 cycles, matches zero-T n_c [0.497, 0.181];
- hard case eps_L 2 δ1.5: 30 cycles, n_c [0.511, 0.185], no ghost near ½ (others 0.14/0.65).

So the thermal formulation does not need decoupled ghosts here, and its one finished hard case converged quickly.

### Impurity in-sector Boltzmann smearing (FockitImpuritySolver beta; IMP_BETA in ligand_model.py)

| ligand cases, band β100 | default GS | default imp β1000 | pin GS | pin imp β1000 | default imp β100 (14/15 run) | pin imp β100 (7/15 run) |
|---|---|---|---|---|---|---|
| easy (7) | 0/7 | 2/7 | 7/7 | 5/7 | 3/7 | 6/7 (7th at 4e-5) |
| hard (5) | 0/5 | 1/5 (eps_L 2 δ.5, 2 dec. ghosts) | 0/5 | 0/5 | 0/5 | — |
| controls (3) | 3/3 | 2/3 | 2/3 | 1/3 | 1/2 | — |

- Not a fix: some easy cases end in coupled-ghost states, and the pin gets worse.
- β=100 impurity smearing is slow, because Lanczos needs every state up to 32/β. The runs were stopped early.

### Are gem's converged states risb fixed points? (diis/check_gem_fixed_point.py)

Method: gem's final R, Λ, μ go into risb with the same scheme (qp μ = 0, μ on the impurity physical orbitals and the ligand), then one cycle.

| case | max|f1| | max|f2| | filling error |
|---|---|---|---|
| control eps_L 1.5 V.6 δ1 (no decoupled ghost) | 1e-11 | 8e-12 | −1e-5 |
| easy eps_L .5 δ1 | 2e-12 | 1.5e-2 | 4e-5 |
| hard eps_L 2 δ1 | 5e-12 | 1.8e-2 | −2e-5 |
| hard eps_L 1.5 δ1 | 3e-11 | 1.5e-2 | −6e-5 |

f2 is ~1e-11 on every coupled ghost. All of the 1–2e-2 is on the decoupled ghosts (n=½, |vR| ≤ 2e-11):
- gem leaves their levels at +5e-4..8e-4 above the qp chemical potential, so the flat bands have filling 0.48 instead of ½;
- gem's step measure does not see this, because Λ for a decoupled ghost stops changing and its R row is 0;
- the filling error ≤ 1e-4 is gem's ntol.

So gem "converges" by not requiring f2 = 0 on decoupled ghosts. Those ghosts carry no physical weight, and the physical results agree with risb's pin (easy: n_c [0.5041, 0.0198] vs [0.504, 0.020]).

In risb the qp count fixes μ and includes these flat bands, so exact ½ is enforced. Near β≈100 that may be what has no solution (the fold). This is untested: a risb residual that excludes decoupled-ghost directions, or the impurity μ plus such a residual.

### Berlingreen history: gem's step measure and observables (diis/berlin_gem_measure.py)

Method: Λ_out and R_out are recomputed per entry from the stored D, Λc, ρ_f, ρ_cf; gem measure = max over up blocks of the change in eig Λ and in |UᵀR|.

| run | root error | gem measure (all / without n≈½, \|vR\|<1e-3 ghosts) | E (eV) | Fe2 n | Fe2 S² |
|---|---|---|---|---|---|
| A entry 9 | 1.6e-4 | 1.6e-2 / 1.6e-2 | −105.03380 | 5.44599 | 2.2668 |
| A2 plateau (7 entries) | 1.3–2.4e-4 | 2.9e-3..1.5e-2 / same | −105.0339..−105.0344 | 5.44593–6 | 2.2669–70 |
| I3 end | 2.8e-5 | 5.0e-4 / 5.0e-4 | −105.03165 | 5.44610 | 2.2665 |

- gem would **not** call berlingreen converged: its step stays ≥ 5e-4, above gem's 1e-4, even where the root error is 2.8e-5.
- The step is not on the decoupled ghosts. Excluding them leaves it unchanged, so it comes from the Λ of coupled or weakly coupled ghosts, which get_lambda determines poorly.
- Physical spread between the 1.6e-4 plateau and I3 (2.8e-5): Fe occupations 1.5e-4, Fe2 S² 4e-4, total energy 2.4 meV; the energy also varies by 0.5 meV within the plateau.
- So the berlingreen plateau is not the gem-type "f2 only on decoupled ghosts" state. Its leftover residual moves the energy at the meV level.

### risb with gem's choices, gem-like stop (diis/compare_gem.py, STOP=coupled)

Stop rule: the residual with decoupled ghosts excluded (n within .05 of ½, |vR| < 1e-6; their f2 rows/columns and f1 columns dropped) < 1e-9. β=100, 300 cycles.

| variant | easy 7 | hard 5 | controls 3 | cycles (converged) | solves/cycle | s/solve |
|---|---|---|---|---|---|---|
| risb DIIS (baseline) | 4 | 0 | 3 | 59–179 | 1 | 0.08–0.12 |
| DIIS + imp μ (exact loop) | 5 | 0 | 2 | 19–264 | ~9 | 0.05–0.09 |
| DIIS + imp μ + gem start + warm eigsh | 4 (+1 error) | 0 | 1 | 29–285 | ~9 | 0.05–0.08 |
| recursion error + lambda_eigenbasis | 0 (spurious) | 0 (spurious) | 1 | — | 1 | 0.1 |
| lin 0.2 + imp μ + gem start + warm (all gem) | 0 (coupled 2e-7..1e-3) | 4 | 2 | 35–132 | ~8 | 0.05–0.07 |
| gem itself | 7 | 4 | 2 | 30–202 iterations | 2–3 | 0.17–0.25 |

Notes:
- **All-gem risb, hard cases:** it reaches gem's solutions (E 3.14398, 3.03883; also eps_L 2 δ1.5 2.89326, and eps_L 1.5 δ1.5 2.02854, which gem itself fails). It misses eps_L 1.5 δ1, which gem gets.
- **All-gem risb, easy cases:** it does not finish in 300 cycles, though gem does in 30–95. Unexplained. Candidates: the ligand as a fragment in gem; gem refits μ only when it is off by > 1e-4; symmetrization.
- **Recursion + lambda_eigenbasis** (μ from the qp count): with decoupled-ghost f2 free, charge moves into the flat ghost bands. It "converges" to wrong states, e.g. eps_L 2 δ1 at E 4.54 vs 3.04, with full residual 0.1–0.2. So ignoring decoupled-ghost f2 is only safe when μ acts on the impurity, as in gem.
- **What the hard cases need:** linear mixing + impurity μ + the gem-like stop. DIIS + impurity μ does not get there, and linear mixing with qp-count μ is unstable.
- **Speed:** risb per solve is ~2× faster than gem. The exact Brent μ loop costs ~8 solves per cycle against gem's 2–3.
- **Energies:** gem's hard-case solutions lie above risb's stalled near-misses. E.g. eps_L 2 δ1: 3.03883 vs 3.02646 stalled, the stall not being a solution.

### Lazy μ refit and ligand as a fragment (compare_gem.py lazy / LIGAND_FRAGMENT=1, all on top of lin 0.2 + gem start + warm eigsh, STOP=coupled)

| variant | easy 7 | hard 5 | controls 3 | solves/cycle |
|---|---|---|---|---|
| exact μ, ligand uncorrelated (previous) | 0 | 4 | 2 | ~8 |
| lazy μ (refit when off > 1e-4) | 2 (+3 at 1e-9..3e-8) | 4 | 2 | ~3–4 |
| exact μ, ligand as U=0 B=1 impurity | 0 (+1 Brent bracket error) | 3 | 2 | ~8 |
| lazy μ + ligand impurity | 1 | 3 | 2 | ~3–4 |
| gem | 7 | 4 | 2 | 2–3 |

- Neither change explains gem's easy-case success (30–95 iterations in gem). risb's gem-like runs leave the residual without decoupled ghosts at 1e-9..1e-3 after 300 cycles.
- Lazy μ cuts the cost to ~3–4 solves per cycle and keeps hard 4/5. It leaves up to 1e-4 filling error: on the control, the energy is 1.78901 vs 1.78890.
- The Λ formulas match: gem calc_Lambda and risb get_lambda are both −Λcᵀ − ∂Tr[√(ρ(1−ρ)) D Rᵀ]/∂ρ. For a decoupled ghost (D and R rows 0) the map is the identity on its level, so the level never moves and f2 stays nonzero. This happens in both codes.

### risb's gem-like iteration started at gem's converged state (diis/from_gem.py)

Setup: lin 0.8, warm eigsh, impurity μ, ligand uncorrelated, starting from gem's R, Λ, μ; 50 cycles.

| start | μ policy | residual without decoupled ghosts | μ | drift of R / Λ |
|---|---|---|---|---|
| easy eps_L .5 δ1 | lazy (1e-4) | 1e-11..1e-12 throughout | frozen 1.662987 (fill error 3.6e-5) | 2e-8 / 5e-8 |
| hard eps_L 2 δ1 | lazy | → 1e-15 | frozen (fill error −1.8e-5) | 5e-11 / 4e-10 |
| easy eps_L .5 δ1 | exact every cycle | 4e-5 at cycle 1, then ~1%/cycle decay (1e-5 at 50) | drifts 1.6630 → 1.6498 | 4e-3 / 1.6e-2 |

Findings:
- gem's state is a stable fixed point of risb's map when μ is lazy.
- Requiring exact filling moves μ and drags the solution along a very slow mode of linear mixing (contraction ~0.99 per cycle).
- So gem's fast easy-case convergence rests on two tolerances:
  1. f2 on decoupled ghosts is ignored;
  2. a filling error up to 1e-4 is accepted, which freezes μ and removes that slow mode.
- The fresh-start lazy risb runs (easy 2/7) differ in the path: each refit above 1e-4 kicks the slow mode.
- DIIS can handle the slow mode: DIIS + exact impurity μ gave easy 5/7, but hard 0/5.

### Summary table: ligand cases, β=100 band smearing, 300 cycles (2026-11-05)

Conventions:
- "conv" = residual without decoupled ghosts (f1/f2) < 1e-9, so false stops by recursion/canonical criteria are excluded.
- "best E" = converged at the lowest energy any method found for that case, within ~3e-4 (allowing for ≤1e-4 filling errors).

Lowest known energies:
- easy: .35553, .11156, −.13516, .03653, −.42204, .94124, .73049;
- hard: 2.22757, 2.02854, 3.14398, 3.03883, ~2.888;
- controls: .59440, 1.78890, 3.19304.

| method | easy conv (best E) /7 | hard conv (best E) /5 | ctrl conv /3 | filling at end | solves/case | basis safe |
|---|---|---|---|---|---|---|
| gem (lin 0.2, imp μ lazy, ligand fragment) | 7 (7) | 4 (2) | 2 | ≤1e-4 | 78–444 | yes |
| risb DIIS (qp-count μ), coupled stop | 4 (4) | 0 | 3 | exact | 59–179 | yes |
| risb DIIS + pin v1 (full f1/f2) | 7 (7) | 0 | 2 | exact | 50–178 | yes |
| DIIS + exact imp μ, coupled stop | 5 (5) | 0 | 2 | exact | ~9/cycle | yes |
| lin + lazy imp μ (risb "gem"), canonical stop | 1 (1) | 4 (3) | 2 | ≤1e-4 | 140–1060 | yes |
| lin→DIIS, root error, coupled stop | 2 (1) | 4 (3) | 1 | exact | 180–2600 | yes |
| lin→DIIS, recursion + lambda_eigenbasis | 5 (3) | 5 (4) | 2 (+1 false stop) | exact | 200–1100 | no (rotates iterates) |
| lin→DIIS, canonical error | 5 (1) | 5 (3) | 2 | exact | 220–1350 | yes |
| DIIS→lin, root error, coupled stop | 2 (2) | 4 (3) | 2 | ≤1e-4 | 200–890 | yes |
| DIIS→lin, recursion + lambda_eigenbasis | 3 (2) | 5 (4) | 2 (+1 false stop) | ≤1e-4 | 190–1100 | no |
| DIIS→lin, canonical error | 4 (2) | 5 (3) | 2 | ≤1e-4 | 200–1100 | yes |
| DIIS only, canonical error | 0 | 0 | 2 | exact | — | yes |

Findings:
- **Several solutions per case.** The hard case eps_L 1.5 δ1 has at least 3 solutions (2.22757, 2.23249, 2.23353–2.23366), and eps_L 2 δ1.5 has 2 (≈2.888, 2.89326). Easy-case "conv" runs on another branch sit 4e-4..9e-4 higher. Each method picks its own branch.
- **Easy kind:** solved best, at the lowest energies, by DIIS with qp-count μ plus the pin (7/7) and by gem (7/7, with ≤1e-4 filling error).
- **Hard kind:** needs impurity μ plus linear mixing at the start. Both lin→DIIS variants converge all 5 at exact filling.
- **Crashes:**
  - Brent bracket failure in the impurity-μ loop;
  - "h_imp does not conserve a diagonal symmetry" in two lin→DIIS 1e-6 runs.

## Berlingreen lin→DIIS schedule (imp_schedule.py)

Start: A-pulay-nomix entry 9. Stage 1 runs linear mixing with α 0.8 and the lazy impurity μ (1e-4, secant refit). The switch to canonical DIIS is set at a canonical error below 1e-4, with a 40-cycle cap.
- Never switched: the lowest canonical error was 4.5e-4 (cycle 36), and it ended at 1.9e-3. The coupled residual ended at 4.2e-5. The run used 144 solves in 4.35 h (~3.6 solves per cycle).
- Cycles 5–11 went off course: canonical up to 0.40 and coupled up to 5.6e-2. After that it settled into a 4e-4..3e-3 oscillation.
- It ended on another branch: Fe2 has Sz 0.000 and S² 2.233, against Sz 0.52 and S² 2.267 for A2 and I3. Fe1 is HS (S² 7.650).
- Filling error 3e-9. μ = μ0 + δμ = −0.40820.
- Energy: spincrossover 8528e4d changed E_dc, so the absolute values shift by ~12 eV. With the old E_dc (76.5) the energy is −105.0461, against −105.0341 for A2 and −105.0316 for I3. That is 12 meV lower, but the run is not converged.

Continuation (S3-imp-diis): start from S-imp-lin2diis entry 35, with switch threshold 2e-3.
- **Restart needs the old μ0 and δμ.** The stored x is Λ' = Λ − δμ. Restarting with a fresh qp μ fit (S2) gave a canonical error of 1.3e-2 and diverged. With SCHED_MU0 = −0.407549 and SCHED_DMU = −6.08e-4 it reproduced 4.7e-4.
- **Canonical-error DIIS from cycle 2 met the stop at cycle 22.** Canonical 8.8e-7, coupled 2.8e-8, filling −5.4e-7 (lazy 1e-6). 35 solves in 1.4 h.
- **What's left is on decoupled ghosts only.** run_risb's full f1/f2 is 8.7e-5 and its root error 3.1e-4.
- **Solution:** Fe1 HS (S² 7.650, n 5.263); Fe2 Sz 0, S² 2.233, n 5.447.
- **Energy:** E (old E_dc) = −105.0452 eV, 11 meV below A2 (−105.0341) and 14 meV below I3 (−105.0316). Those runs were Fe2 Sz 0.52 and unconverged. The energy uses Tr[H_KS ρ_lattice], so δμ does not enter.

Fresh DIIS only (T-fresh-diis): default start, canonical-error DIIS (Pulay α 0.6, no mixing period), lazy impurity μ (1e-6), 40 cycles.
- **Spin state:** Fe1 Sz 0 from cycle 1 (S² 1.43). Run A, also from the default start, had Fe1 HS (Sz 2.29) at cycle 1. The cause is unknown; the log headers are identical.
- **End:** Fe1 S² 1.608, n 5.398; Fe2 S² 2.143; μ −0.2259 (δμ −0.189). E (old E_dc) = −101.006 eV, 4.04 eV above S3.
- **Convergence:**
  - coupled 2.5e-7, root error 1.3e-6, full f1/f2 2.4e-7;
  - canonical error stuck at 2–4e-4 from cycle 25 on, so the stop never fired;
  - the canonical floor is likely an artifact of near-degenerate Λ levels against the grouping tolerance 1e-3 (unchecked).
- **Cost:** 259 solves, 4.4 h. Refits took up to 20 solves while δμ moved by 0.19.

Why run A started HS, and a fresh HS start (2026-11-06):
- **A's start:** A was not fresh. berlingreen_ramp.py set R and Λ from the joint-pin run's cycle 5. run_risb still logs "Starting from the default R and Lambda" because the script sets them itself. The true default start gives Fe1 LS at cycle 1.
- **Seeds:**
  - A spin-split seed does nothing: run_risb's equivalence symmetrizer averages up and dn, and Fe1 HS is paramagnetic here (equal up and dn densities, S² 7.65).
  - An orbital seed works. SEED=1 sets the cold-start Λ shift (run_risb's initial_potential) to −1 on eg (blocks 0, 3) and +2/3 on t2g. Cycle 1 then gives Fe1 S² 7.69.
- **T2 (seeded, DIIS only, canonical error, α 0.6, lazy impurity μ):** lost HS at cycle 2 (S² 1.59) and never settled. Over 36 cycles canonical stayed at 2–37 and coupled at 0.01–0.9, δμ swung between −2.7 and +0.2, and Fe1 S² ranged 0.8–7.6. 489 solves in 10 h; stopped by hand.
- **Canonical floor in T:** it is not degeneracy, since the levels are well separated and canonical equals raw. It is Fe1's nearly full ghost (n 0.999996, level −9.43): its level moves 3–8e-4 per cycle because f1/f2 hardly depend on it. The 3.93 level (n 1.6e-4) moves ~1e-5. A canonical stop needs to drop or down-weight nearly full and empty ghosts.

## Density feedback (ac7bd0a) and PhysicalMuLatticeSolver (bb17a49..7f4d3f2) on the ligand cases (2026-11-06)

feedback_bench.py, c = 4/β = 0.04, DIIS Pulay(mixing_period=None), default start, 300 cycles. Stop at risb's full convergence (tol and self-consistency 1e-9, decoupled ghosts included).
- fb: LatticeSolver with qp-count μ.
- fblazy: imp_mu_gauge with lazy refit 1e-4.
- phys: PhysicalMuLatticeSolver(n_target=3).

| variant | easy full conv (best E) /7 | hard /5 | ctrl /3 | solves/case (conv) | filling at end |
|---|---|---|---|---|---|
| fb | 1 (1), +1 coupled only | 0 | 1 | 55–270 | ≤1e-3 unconverged |
| fblazy | 6 (6) | 0 | 3 | 54–788 | ≤5e-5 |
| phys | 1 (1) | 0 | 2 | 31–85 | exact when converged |

- With the feedback, the decoupled ghosts converge too: fblazy runs end at full f1/f2 ≤ 3e-10.
- Neither hard cases nor (1.0, 0.6, 1.5) converge with any of the three. In fblazy, (1.0, 0.6, 1.5) and (1.5, 0.3, 1.5) blow up to a residual of 0.3–0.7.
- spincrossover passes no density_feedback, so berlingreen gets the default c = 0.04, which is 8× 4/β at β = 800.

### Linear mixing → DIIS with the feedback, and a secant μ step for PhysicalMuLatticeSolver (sched_bench.py, secant_mu.py)

- **Common settings:** c = 4/β; gem start; warm eigsh; linear mixing 0.8 until the canonical error (+ μ entry) is below 1e-4, then DIIS on it.
- **Stop:** full f1/f2 < 1e-9 (decoupled ghosts included) and |filling error| < 1e-8; 300 cycles. "best E" is within 3e-4 of the lowest known.
- **Measured dn/dμ** at fixed Λ′ and R ranges 0.06–1.9 across cases, so the unit step contracts slowly at 0.06 and is close to unstable at 1.9.
- **physsec** steps by (N − n)/κ, with κ the secant of successive cycles clipped to [0.02, 5].

| run | easy /7 (best E) | hard /5 (best E) | ctrl /3 | solves/case |
|---|---|---|---|---|
| fb lin→DIIS | 3 (0) | 3 (3), incl. 2.22757 | 2 | 94–280 |
| fblazy lin→DIIS | 6 (4) | 2 (2) | 2 | 64–699 |
| phys lin→DIIS | 5 (4) | 3 (3) | 2 | 28–266 |
| physsec lin→DIIS | 6 (5) | 3 (3) | 2 | 32–254 |
| physsec DIIS only | 1 (1) | 0 | 2 | 28–249 |

- The secant μ step does nothing for DIIS only. With the schedule it adds one easy case.
- Unsolved by every schedule run: (0.5, 0.3, 1.5) except via fblazy, (1.5, 0.3, 1.0) except via fb, (2.0, 0.3, 1.5) except via fb, and (0.5, 0.3, 0.5) in most.
- The easy-case misses land on a branch 4e-4..9e-4 higher, at E .03710, .73134 and −.42164.

Full grid, same stop (full f1/f2 < 1e-9 and exact filling). "root" is risb's default DIIS error. "canonical" is canonical_error plus the μ entry; it is also the switch test (root runs switch on max(full residual, filling error) < 1e-4). DIIS-only rows use the default start; schedule rows and "DIIS only, canonical" use the gem start and warm eigsh.

| solver (feedback c = 4/β) | DIIS only, root | DIIS only, canonical | lin→DIIS, root | lin→DIIS, canonical |
|---|---|---|---|---|
| LatticeSolver, qp-count μ | 1(1)/0/1 | 0/0/1 | 3(0)/0/1 | 3(0)/3(3)/2 |
| LatticeSolver + lazy imp μ | 6(6)/0/3 | 3(3)/1(1)/2 | — | 6(4)/2(2)/2 |
| PhysicalMuLatticeSolver | 1(1)/0/2 | 1(1)/0/2 | 5(4)/3(3)/3 | 5(4)/3(3)/2 |
| PhysicalMu + secant μ step | — | 1(1)/0/2 | 6(5)/3(3)/2 | 6(5)/3(3)/2 |

- For PhysicalMuLatticeSolver the error makes no difference.
- Root uses fewer solves per case: 24–189 against 28–266.
- For qp-count μ, only the canonical error reaches the hard cases.

## PhysicalMuLatticeSolver on the test suite, and a window on the quasiparticle count (2026-11-06)

swap_plugin.py runs every test that solves a LatticeSolver (93) with PhysicalMuLatticeSolver (n_target = band target − ghost electrons) and/or other optimizers. Passes /93: LatticeSolver 93; PhysicalMu 81; PhysicalMu, no mixing period 75; PhysicalMu, BianLi 73; LatticeSolver + lin→DIIS 84; PhysicalMu + lin→DIIS 77. Some failures come from the harness: x has μ appended, a test expects band-μ log lines, optimizer-history tests.

Causes of the PhysicalMu failures:
- kagome and the trimer basis: DIIS extrapolates μ past the bands, every band fills, FullEmptyPinning moves the levels to ±147 / −29.5, dn/dμ = 0, and μ random-walks.
- Site basis: the same overfilling. The degenerate ρ_qp gives pinning eigenvectors that break C3, and the h_imp symmetry check raises.
- Néel: feedback with Pulay mixing_period=3. It passes without feedback (77) or without the mixing period (68).
- Hubbard example at U = 4.75: μ = Λ = U/2 throughout and Z wanders 0.1–0.3. Cause unknown; BianLi converges it.
- Thermal-impurity Mott case with LatticeSolver + lin→DIIS: R = 0 makes the update the identity, linear mixing stalls at f1/f2 = 2.4e-3, and the switch at 1e-4 never comes.

window_mu.WindowedMuLatticeSolver: if the quasiparticle filling at μ is more than 0.5 from n_target + ghost electrons, μ is moved to where it is 0.5 off (one extra band solve). Tests: 84/93. It fixes kagome (17), trimer (17), site basis (17) and Néel (37). The Mott test converges to an orbitally split solution (Λ 4.40/4.50/4.60, Z 0.08) instead of Z = 0. Rounds on common passes 0.98× LatticeSolver. Ligand benchmark (root error): lin→DIIS 5(4)/3(3)/2 against 5(4)/3(3)/3 without the window; DIIS only 3(3)/0/2 against 3(3)/1(0)/2.

Window δ sweep: 93 tests passed / impurity rounds on the 80 passing everywhere / wall time on those.

| δ | tests passed | rounds | time | ligand lin→DIIS | ligand DIIS only |
|---|---|---|---|---|---|
| none | 81 | 2534 | 15 s | 5(4)/3(3)/3 | 3(3)/1(0)/2 |
| 0.25 | 85 | 2479 | 50 s | 5(3)/2(2)/2 | 3(3)/0/2 |
| 0.5 | 84 | 2575 | 20 s | 5(4)/3(3)/2 | 3(3)/0/2 |
| 0.75 | 85 | 2583 | 18 s | 5(4)/3(3)/2 | 4(4)/0/2 |

- δ = 0.5 sends the Mott test to an orbitally split metal (Z 0.1–0.4, f1/f2 ~1e-11). δ = 0.25 and 0.75 give the Mott state.
- The LatticeSolver Mott test is not converged: it stops at tol 1e-8 with max f1/f2 = 0.36. The recursion error vanishes at R = 0 because the update is the identity there. At tol 1e-12 it converges after 310 cycles to Λ = −32/5/42 (one orbital full, one empty), and δ = 0.75 gets there in 19.
- ThermalLatticeSolver, examples at β = 40 as a real temperature, impurity symmetries dropped:
  - trimer: μ runs away to −6000 with every band empty, the same failure;
  - kagome: μ is stable but R blows up (Z = 12) and f1/f2 stays at 1.
- Implemented max_qp_filling_deviation = 0.75 in PhysicalMuLatticeSolver. This also fixes test_physical_mu_finds_the_neel_antiferromagnet. 272 tests pass.

δ sweep with the implemented window (swap_plugin WINDOW_DELTA; inf = no window). Tests passed /93, impurity rounds on the 80 passing everywhere, ligand lin→DIIS and DIIS only (root error):
- inf: 81, 2297; 5(4)/3(3)/3, 3(3)/1(0)/2.
- 0.75: 85, 2346; 5(4)/3(3)/2, 4(4)/0/2.
- 0.9: 83 (Mott → other solution, Néel fails), 2336; 5(4)/3(3)/2, 4(4)/0/1.
- 0.99: 84 (Néel fails), 2336; 5(4)/3(3)/2, 2(2)/0/1.

ThermalLatticeSolver + the same window (thermal_window.py), on the examples:
- kagome: μ → +2600 with physical filling error +3, so the impurities are full.
- trimer: μ is stuck at 14.18 with filling error +4, f1/f2 8e-10.
- The thermal solver does not shift Λ with each μ step. Λ follows μ, so the window on the quasiparticle filling does not stop the impurity filling from saturating. In the trimer each μ step down puts the band filling outside the window and the window moves μ back.

ThermalLatticeSolver with the Λ shift on each μ step (and Λc the other way), thermal_window.py:
- The 19 tests in test_thermal_latticesolver.py pass with the shift, with and without the 0.75 window, in about the same time.
- The examples still fail, and R blows up in every run:
  - kagome without window: μ stable, Z → 3454;
  - kagome with window: μ → 457, filling error +3, Z → 79;
  - trimer without window: f1/f2 6.8e-6 at cycle 2, then a jump to Z 13 and runaway;
  - trimer with window: Z up to 1e14.
- The thermal failure on these models is the quasiparticle fit (R unbounded), not only μ.

Clip of R's singular values to ≤ 1 (clip_r.py; on R entering a cycle and on R from the update):
- Full suite: 272 pass.
- The 93 LatticeSolver tests: 93 pass, impurity rounds 0.995×. Clipped 127 times on input, 19 on output.
- Same tests with PhysicalMu (window 0.75): 85 pass either way, rounds 1.004×.
- Thermal examples with the clip (no shift, no window): R stays bounded (max eigenvalue of R†R ≤ 1), but neither converges.
  - kagome: μ 4.97, filling exact, f1/f2 stuck at 1.0.
  - trimer: f1/f2 8.5e-3 at cycle 8, then μ → −1.2e5 with all bands empty.
