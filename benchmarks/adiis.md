# Reduced-energy ADIIS prototype

[adiis.py](adiis.py) preserves the zero-temperature objective and dense
adjoint-response calculation previously explored in temporary scripts.
It uses current RISB/fockit APIs and has no dependency on those scripts.
The derivation is in [the energy-minimization note](../notes/energy_minimization.md).

Run the comparison and physical checks from the repository root:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python benchmarks/adiis.py --repeats 3 --maxiter 150
    uv run pytest benchmarks/test_adiis.py -q

The driver saves complete records and ADIIS energy/residual traces to
_build/adiis/results.json. These generated records are ignored by Git;
the source, tests, and summary here are durable.

## Algorithm

Q-Chem's [ADIIS documentation](https://manual.q-chem.com/5.4/sub_adiis.html)
uses L-BFGS for the small coefficient optimization, with
c_i = t_i² / sum(t²). This does not impose a line search on the actual
SCF energy. BFGS on the physical energy and L-BFGS on stored mixing
coefficients are different operations.

For input fields y_i = (D_i, Lambda_c_i), compute the constrained energy
and its actual adjoint gradient g_i. Relative to the newest sample,
form s_i = y_i - y_n, u_i = g_i - g_n and minimize

    g_n · sum(c_i s_i) + 0.5 sum(c_i c_j s_i · u_j)

over nonnegative coefficients summing to one. The irrelevant constant
energy is omitted. The prototype uses two L-BFGS-B starts, also checking
the vertices; it does not establish a global optimum of an indefinite model.

Each evaluation supplies the expected embedding fields from helpers.get_d
and helpers.get_lambda_c, transformed to the same sorted-bath,
positive-hybridization gauge. The next proposal mixes these **mapped
outputs**, rather than only the input fields. This generates new states
outside the input-history hull, analogous to Q-Chem modelling densities
but mixing Fock outputs. It is a proposed RISB adaptation, not the standard
SCF ADIIS functional applied unchanged.

There is **no actual-energy rejection or backtracking** in this ADIIS path.
Failed inner density matching aborts that run and is reported as a failure.
ADIIS-half mixes halfway from the combined inputs to combined outputs;
it also has no energy rejection rule. The default ADIIS uses the full output.

Each reduced evaluation requires one impurity ground-state solve, an inner
noninteracting band-density fit, and one dense adjoint linear solve.
The inner fit does not call the interacting impurity solver. The production
DIIS comparison requires no adjoint calculation.

## Local results

Measured 4 November 2026: half-filled paramagnetic Bethe model, half bandwidth
1, U=1, 400 equally weighted samples, B=1 or 3, dense impurity ground states.
B=1 has one field coordinate; B=3 has (d0, d1, b). The impurity Hilbert
spaces have dimensions 6 and 70. All starts are distortions of a known
metallic solution, not independent searches for unknown phases.

Methods start from the same physical matrices. Every method repeat starts
fresh numerical history; band fits use the common perturbed initial state,
not the reference solution. One numerical thread, three serial repeats in
rotating method order; no concurrent checks ran during these measurements.
Reference generation, initial matching, operator-cache construction,
gradient checks, and final observation are outside the timing/counts.

Each table entry is **median milliseconds / impurity solves**:

| B | Initial field factors | DIIS | BFGS | ADIIS |
|---|---|---:|---:|---:|
| 1 | (1.03) | 7.96 / 8 | 4.86 / 6 | 7.05 / 6 |
| 1 | (1.30) | 8.82 / 9 | 5.58 / 7 | 8.68 / 7 |
| 3 | (1.03, 0.97, 1.03) | 29.55 / 9 | 178.04 / 20 | 101.27 / 11 |
| 3 | (1.30, 0.70, 1.30) | 36.18 / 11 | 377.62 / 25 | 207.70 / 13 |
| 3 | (0.50, 0.50, 1.50) | 35.97 / 11 | 644.95 / 42 | 234.96 / 15 |

Each BFGS/ADIIS impurity solve also needs one adjoint solve; DIIS needs none.
The JSON's reduced-band-evaluation counter is not instrumented for
production DIIS and is recorded as null there.

All 60 method runs passed the common full-field residual threshold 1e-7.
ADIIS stops at that threshold; BFGS uses gradient gtol=1e-8, and DIIS
uses root tolerance 1e-10, so final precisions are not identical.
Across these runs, energy errors were below 1.7e-13, double-occupancy
errors below 1.6e-8, and quasiparticle-weight errors below 3.7e-8
against the reference solution. Central energy differences checked the
adjoint gradient to better than 3.5e-10 at every starting point.

The distorted B=3 ADIIS run took one energy-increasing step, approximately
1.51e-7, and still converged in all repeats. The other full-output starts
had no increases above 1e-12. Half mixing also converged, but took
21/24/30/35/32 impurity solves for the five starts and was slower than the
full-output method.

Here ADIIS beat BFGS for B=3, BFGS was cheaper for B=1, and production DIIS
was fastest for B=3. This does not isolate nonmonotonicity as the cause of
the ADIIS/BFGS difference: the proposal directions, histories, and stopping
rules also differ. It establishes neither a general advantage nor a cure
for Mott, degenerate, multiorbital, thermal, or charge-self-consistent cases.
No production optimizer API was changed.

Harder B=3 near-Mott and thermal cases, including failures, are in
[the hard-case report](adiis_hard.md).

The [energy-space hybrid report](adiis_hybrid.md) compares ADIIS followed
by gradient DIIS on the same reduced energy, including harder distorted B=3
starts and Q-Chem-like switching thresholds.
