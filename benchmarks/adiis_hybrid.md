# ADIIS → DIIS on the same reduced energy

[adiis_hybrid.py](adiis_hybrid.py) changes the coefficient rule while retaining
the same embedding coordinates, objective, inner constraints, and mapped
outputs. Both phases use actual reduced-energy gradients. It never hands
the iteration back to the production saddle-point solver. Root DIIS is
a separate baseline, starting from the same physical state.

ADIIS uses the local energy model described in [adiis.md](adiis.md).
The DIIS phase chooses unrestricted coefficients with sum(c)=1 to minimize

    ||sum_i c_i gradient(f(y_i))||²

and mixes the same mapped output fields. Six history samples carry across
the switch. Neither phase enforces monotonic physical energy, and neither
uses BFGS on the physical objective. Gradient DIIS can also converge to
nonminimum stationary points; the objective's minimum formulation remains
essential.

## Switching and comparison

Q-Chem 5.4 documents an [ADIIS-to-DIIS switch](https://manual.q-chem.com/5.4/sub_adiis.html)
at error below 10^(-THRESH_ADIIS_SWITCH), default exponent 3, or a maximum
30 ADIIS cycles. It recommends exponents 3 or 4.
Its [DIIS error](https://manual.q-chem.com/5.4/sect_diis.html) is an orbital
commutator, with a maximum-element convergence measure by default.
Here the analogous measure is the maximum absolute reduced-energy gradient;
1e-3 and 1e-4 are numerical analogies, not equivalent physical accuracies.
Field units and scaling affect these thresholds.

All runs below use B=3, the one-band half-filled Bethe model, half bandwidth 1,
400 band samples, and a 70-state zero-temperature impurity sector.
Both energy-mixing phases require one impurity solve and one adjoint solve
per evaluation. Root DIIS needs no adjoint solves.
Thermal traces include all 256 Fock states across particle-number sectors.

Energy methods must satisfy both gradient maximum <1e-8 and remaining
matching residual <1e-7. Inner matching must be below 1e-8. Root DIIS uses
its native root 2-norm tolerance 1e-10, and its original physical matching
equations are checked below 1e-7. These are different stopping measures;
root DIIS generally finishes at substantially tighter residuals.
The common budget is 150 evaluations or 20 seconds, one numerical thread.
These are single-run local restart diagnostics, not repeated timing statistics
or independent phase searches. Reference preparation is excluded.
The common perturbed inner state is held as the initial guess for inner
solves; subsequent evaluations do not restart from the exact reference.

## Zero-temperature results, 4 November 2026

Each entry gives interacting solves for starts 1 / 2. F denotes failure.
Nearby starts use deterministic seeds 1 and 2, with 15% perturbations and
a field scale floor of 0.2. Distorted starts multiply reference coordinates
(d0,d1,b) by (0.5,0.5,1.5) and (1.6,0.4,1.6).

| U and starts | Root DIIS | Pure ADIIS | Pure gradient DIIS | Hybrid 1e-3 | Hybrid 1e-4 |
|---|---:|---:|---:|---:|---:|
| 1.0, nearby | 11 / 8 | 11 / 12 | 8 / 8 | 9 / 8 | 9 / 8 |
| 2.4, nearby | 9 / 9 | 41 / 40 | 11 / 10 | 13 / 12 | 16 / 15 |
| 2.8, nearby | 14 / 11 | F150 / F150 | 13 / 11 | 29 / 16 | 36 / 33 |
| 1.0, distorted | 11 / 11 | 15 / 14 | 10 / 11 | 11 / 10 | 12 / 12 |
| 2.8, distorted | 38† / 15 | F150 / F150 | F14 / 17 | 39 / 36 | 39 / 36 |

Root DIIS uses the production solver and default callbacks, with the same
sharp fixed-half-filling occupations as the reduced energy on this equal-weight
grid. Its independent final reduced-objective validation costs one additional
impurity solve, excluded from the table and recorded separately.

† The 38-solve root run satisfies the original matching equations to 5.0e-12
but reaches a different, insulating stationary branch: Z is about zero,
E=-1.44627909 and double occupancy=0.01789303.
The hybrid reaches the metallic reference with Z=0.02630337,
E=-1.44647642 and double occupancy=0.02288225.
The insulating root is about 1.97e-4 higher in energy. Its reduced-objective
validation fails inner inversion; it is not counted as reaching the metallic
minimum. Root convergence alone does not establish minimum stability.

The original root benchmark's band smearing of 1e-6 is also retained as a
separate root-DIIS-smeared record. Counts and reference agreement are identical
in the other nine starts. On distorted U=2.8 start 1 it instead takes 24 solves,
reaches a nearly insulating state, and fails the sharp-energy gradient
criterion (gradient maximum 2.5e-4). This is why matching band occupations
matters for that start.

At U=2.8 distorted start 1, pure gradient DIIS leaves the region where the
inner lattice target can be matched. Both hybrids finish with gradient
2.5e-10 and matching residual 5.2e-11. This is one small example supporting
ADIIS as an initial accelerator: pure ADIIS stalls, pure gradient DIIS
fails, and the combination succeeds. Near the solution, pure gradient DIIS
is consistently faster than the hybrid.

Both distorted U=2.8 hybrids switch at the 30-evaluation cap, for both
thresholds. Their success does not distinguish 1e-3 from 1e-4.
For nearby U=2.8 starts, 1e-3 triggers at evaluations 23 / 12; 1e-4 instead
hits the cap. For U=2.4 the switches occur at 9 / 7 and 15 / 14 respectively.
On these limited tests, switching at 1e-3 finishes earlier than waiting for
1e-4; the cap prevents a long ADIIS tail.

The stricter common gradient criterion explains why pure ADIIS counts differ
from [the earlier hard-case report](adiis_hard.md), which stopped ADIIS on
matching residual alone. U=3.0 fails during reference density matching;
no optimizer comparison is available there.

## Thermal runs are stationarity diagnostics

Eliminating the quadratic variables does **not** establish a thermal minimum.
For full six-coordinate U=2, T=1, the reduced grand potential has negative
curvature about -0.0054 when all bath levels shift together. These runs
therefore cannot validate ordinary minimization of that functional. One
possible minimum construction eliminates auxiliary maximizing fields, then
assesses the remaining curvature. A coupled saddle algorithm is another
possibility; it does not require this particular nested construction.
Such an extra interacting inner solve may require multiple impurity solves;
the existing quadratic inner solve does not.

Particle-hole restriction removes this bath-charge direction but does not
certify a strict minimum. At the reference, finite differences with steps
1e-3, 3e-4, 1e-4 give smallest Hessian eigenvalues
2.25e-6, 1.89e-7, -6.58e-8. The other eigenvalues are about 0.00112 and 3.10;
the smallest sign is below numerical resolution. A nearly flat mode remains.

For completeness, on nearby PH starts, pure ADIIS takes 5 / 4 solves,
pure gradient DIIS 4 / 4, and both hybrids 4 / 4 to reach stationarity.
Broader PH starts expose failures: factors (1.3,0.7,1.3) converge with
pure ADIIS in 23 solves, while gradient DIIS and both hybrids fail inner
matching; factors (0.5,0.5,1.5) fail for all methods.
One hybrid reaches matching 4.8e-9 but gradient 1.9e-8, narrowly misses the
common criterion, then wanders and fails. It is still counted as failure.

The root baseline also ran from these same initial thermal matrices:

| T=1, U=2 diagnostic | Root DIIS | Pure ADIIS | Pure gradient DIIS | Hybrid 1e-3 | Hybrid 1e-4 |
|---|---:|---:|---:|---:|---:|
| PH, nearby | 29 / 29 | 5 / 4 | 4 / 4 | 4 / 4 | 4 / 4 |
| PH, distorted | 27 / 31 | 23 / F150 | F85 / F31 | F9 / F35 | F9 / F35 |
| Full, nearby | 29 / 29 | F1 / F104 | F1 / F13 | F1 / F20 | F1 / F33 |

Root DIIS is not confined to the three-coordinate PH ansatz and can move among
nearly degenerate ghost representations. Its original full matching residuals
pass 1e-7, and grand potentials agree with the reference within 4e-9.
Projecting its final fields back to the reduced ansatz or selecting a different
quadratic inner branch can produce a nonzero reduced gradient. Consequently
this thermal table compares original-equation stationarity versus reduced
stationarity, not interchangeable minimum searches or identical final states.
The 104-solve ADIIS failure is a wall-clock timeout, not a fixed count limit.

Full-coordinate T=1 energy-mixing runs fail through inner inversion or the time
limit. Both T=0.5 full-coordinate starts fail before optimization.
These diagnostics motivate a valid minimum reduction and better inner
inversion; changing mixing coefficients alone does not supply either.

## Reproduce

From the repository root:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python benchmarks/adiis_hybrid.py --case zero-2.8-distorted --budget 150 --seconds 20
    uv run pytest benchmarks/test_adiis.py benchmarks/test_adiis_hard.py benchmarks/test_adiis_hybrid.py tests/test_optimize.py -q

Other cases are listed by --help. Records are saved as
_build/adiis/energy-hybrid-<case>.json, retaining traces, switching reasons,
physical checks, and failures. Generated records are ignored by Git;
this report, source, and physical regression tests persist.

The [finite-temperature saddle prototype](thermal_saddle.md) now tests
actual nested auxiliary maxima and simultaneous Newton/extragradient updates,
with analytic Gibbs responses, full matrix checks, and production DIIS baselines.
