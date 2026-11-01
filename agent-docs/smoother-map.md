# Smoother self-consistency map

Goal: improve reliable convergence; measure success across starts and final
physical residuals, not only iteration count. Keep prototypes outside the repository.

## Scope

- Analytic thermal-fit Jacobians and directional penalties apply to
  ThermalLatticeSolver only. LatticeSolver has no thermal fits.
- T = 0 follow-up: inverse square roots, full/empty pinning, ground-state
  crossings, and consistent continuation. Not covered by these prototypes.
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
9. [ ] Investigate T = 0 smoothing separately.

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
