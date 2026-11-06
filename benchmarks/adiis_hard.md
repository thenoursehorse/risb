# Small hard g-RISB cases

[adiis_hard.py](adiis_hard.py) extends the standalone ADIIS experiment.
All new runs use **B=3**: a one-band impurity with eight fermion modes.
The zero-temperature sector has 70 states; the full thermal Fock space
has 256 states, diagonalized in particle-number sectors. No large or
multiorbital cases were run.

## Cases and method

The finite-temperature target comes from
[non-unique ghost solutions](../notes/non-unique-ghost-solutions.md):
half-filled Bethe lattice, U=2, T=1, with nearly flat ghost directions.
T=0.5 is a lower-temperature control. U=2.4, 2.8, and 3.0 at zero
temperature probe the near-Mott difficulty discussed in
[preconditioning](../notes/preconditioning.md) and the
[energy note](../notes/energy_minimization.md).

Thermal runs below are stationarity diagnostics, not validated unrestricted
energy-minimization experiments: the full reduced functional remains a saddle.
The [energy-space hybrid report](adiis_hybrid.md) gives the corrected
ADIIS-to-gradient-DIIS comparison and restricted-curvature checks.

The thermal objective is the note's reduced grand potential after solving
quadratic lattice/reference stationarity. Its first gradient is an
impurity/reference expectation-value difference; no interacting adjoint
solve is needed at finite temperature. The ADIIS proposal first fits
quasiparticle fields to the impurity densities, then embedding fields to
the new lattice densities, using only quadratic calculations.

Two coordinate choices are reported separately:

- **PH**: three embedding coordinates constrained by particle-hole symmetry.
- **Full**: all six real coordinates after fixing only the bath rotation:
  three positive hybridizations and three independent bath levels.
  This retains bath-charge directions removed by PH symmetry.

Proposal fits may be inexact, as in the ordinary thermal SCF cycle;
their largest residual is recorded. Every evaluated reduced state must
still satisfy the quadratic lattice/reference equations within 1e-8.
Success requires all original unregularized density and hybridization
matching equations, including the bath density/kinetic equations, below
1e-7. Thermodynamic agreement alone is insufficient.

All methods receive the same physical initial matrices. Two deterministic
random starts perturb a known solution by 15% of each field's magnitude,
with a scale floor of 0.2. These are harder local restart tests, not fresh
phase searches. The common perturbed inner solution is reused as the
fixed starting guess for subsequent inner solves; they are not seeded
from the exact reference on every evaluation.

One numerical thread, serial method runs, maximum 150 outer iterations
and 20 seconds per method. These are single-run diagnostics, not repeated
timing statistics. Preparation, finite-difference gradient checks, and
final observation are outside the reported solver cost. Native stopping
criteria differ: DIIS root tolerance 1e-10, BFGS gradient tolerance 1e-8,
and ADIIS full-field residual 1e-7. BFGS's iteration limit can include
multiple function evaluations per iteration.

DIIS-default retains the production OnPlateau callback and anchoring.
DIIS-no-anchor uses the same solver with callbacks disabled.

## Results, 4 November 2026

Counts below are actual interacting solves. Zero-temperature BFGS and
ADIIS additionally use one adjoint solve per objective evaluation.
Thermal BFGS and ADIIS need no adjoint solves.

| Case | DIIS-default | BFGS | ADIIS-inspired |
|---|---|---|---|
| T=0, U=2.4 | 2/2 pass; 9 solves each | 2/2 pass; 37–76 | 2/2 pass; 30–32 |
| T=0, U=2.8 | 2/2 pass; 11–14 | 1/2 pass | 1/2 pass; 143 on the passing start |
| T=1, U=2, PH control | 2/2 pass; 29 each | 2/2 pass; 23–65 | 2/2 pass; 4 each |
| T=1, U=2, full coordinates | 2/2 pass; 29 each | 0/2 complete | 0/2 complete |

At U=2.8 the unsuccessful ADIIS start reaches residual 1.254e-7 after
150 evaluations, just above the strict threshold. The other reaches
9.956e-8 after 143. BFGS's unsuccessful start reports precision loss
with residual 1.092e-7 after 67 evaluations; its successful start takes
20. DIIS residuals are about 2.5e-11. Small energy errors therefore do
not imply comparable equation accuracy.

For full thermal T=1, ADIIS fails a quadratic inner solve on one start
(residual 5.91e-7) and reaches the 20-second limit after 91 interacting
evaluations on the other. BFGS also fails quadratic inner solves.
These are limitations of the complete reduction/proposal pipeline,
not isolated failures of the coefficient minimizer.

The PH control is much easier because it removes difficult directions.
Its four-evaluation ADIIS result is not evidence that the full hard
case has been solved. Unanchored production DIIS fails both PH-control
starts at 150 evaluations, with residuals 1.10e-5 and 4.10e-6. In the
full-coordinate case it passes one start in 67 evaluations and misses
the other at 150, with residual 4.19e-4. Default anchoring matters.

Matched full-coordinate DIIS runs agree with reference energy and double
occupancy within about 8e-9 and 3e-9, respectively, while sampled
self-energies differ by as much as 1.74e-3. This is consistent with the
note's distinction between stable thermodynamics and weakly determined
ghost parameters/spectra.

## Failures before an optimizer can run

- **T=0, U=3.0:** the constrained objective cannot match its impurity target
  to the sharp-band lattice density at the reference. No ADIIS/BFGS
  comparison is claimed.
- **T=0.5, full coordinates:** both perturbed starts fail the quadratic
  inner solve, at residuals 2.87e-7 and 1.23e-6. All methods are marked
  not run for this primary comparison.

A diagnostic full-matrix least-squares recovery was also tried, using
three-point differences, Jacobian scaling, at most 300 function evaluations,
and unchanged physical acceptance thresholds. One T=0.5 start could then
be evaluated, but ADIIS reached the 20-second limit after only seven
impurity solves and over 31,000 quadratic evaluations. The other start
timed out during preparation. The fallback is not retained in the source;
its diagnostic results are saved separately.

## The thermal objective is still a saddle

At the full T=1 reference, varying all three bath levels together gives
negative reduced curvature. Centered differences of the exact first
gradient give -0.005400, -0.005405, and -0.005388 for steps 1e-3, 3e-4,
and 1e-4. Thus the full reduced grand potential is not a local minimum
in all six coordinates. This confirms the qualification already present
in the energy note: auxiliary maximizing directions need further
elimination before ordinary physical-energy minimization is justified.

These tests do not support ADIIS as a general improvement over current
DIIS. The useful result is the boundary: restricted thermal mixing can
be fast, but the full hard problem still needs a reliable reduced
formulation and inner inversion. Allowing uphill steps alone does not
resolve it.

## Reproduce

From the repository root, select one case:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python benchmarks/adiis_hard.py --case thermal-1.0-full --maxiter 150 --seconds 20
    uv run pytest benchmarks/test_adiis.py benchmarks/test_adiis_hard.py -q

Other case names: zero-2.4, zero-2.8, zero-3.0, thermal-0.5-full,
thermal-1.0-ph. Records are written under _build/adiis with the case name;
they retain preparation failures, solver failures, physical errors,
thermodynamics, and completed ADIIS traces. A trace's energy field is
the evaluated functional: physical energy at T=0, grand potential at T>0.
Generated JSON is ignored by Git; this summary and the source/tests persist.
