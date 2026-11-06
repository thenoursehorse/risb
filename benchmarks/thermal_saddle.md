# Finite-temperature nested maxima and coupled saddle prototypes

[thermal_saddle.py](thermal_saddle.py) tests three local nested reductions,
coupled Newton, and signed-preconditioned extragradient, against production
thermal root DIIS. All use the implemented grand-canonical thermal functional.
No production optimizer or impurity-solver API changed.

## Functional, coordinates, and derivatives

Write L(x,y)=Omega_imp(y)+A(x,y), with x=(R,Lambda) and
y=(D,Lambda_c). The one-band B=3 problem has six real coordinates in each
group: three couplings and three diagonal potential levels. The quadratic
quasiparticles and impurity bath have independent rotation gauges. Thus
diagonalizing Lambda and Lambda_c separately is legitimate; R and D rotate
with their own potentials. An initial full-matrix check exposed three
quadratic rotation zero modes, not three missing physical variations.

The quadratic stationary branch satisfies A_x=0 and defines G(y)=L(x*(y),y).
Every derivative uses the actual Gibbs functional, including reference
constants. The first impurity derivative uses expectation values.
The Hessian uses analytic matrix-Fermi responses and the complete Gibbs
spectral covariance, including nonlinear R terms. It needs no displaced
impurity solves. All 256 impurity states are retained explicitly in these
tiny experiments; the largest particle-number sector has 70 states.
The first truncated-trace diagnostic was rejected rather than using an
incomplete response.

After splitting y=T_u u+T_v v, a nested evaluation solves both

    L_x=0,  T_v^T L_y=0

with u held fixed. It checks that G_vv is negative definite. The inner
Newton iteration solves x and v together, so it does not fully repeat a
quadratic fit before every intermediate v step. New v values still require
new interacting Gibbs solves.

The remaining objective is F(u)=L(x*(u),T_u u+T_v v*(u)).
Its gradient is T_u^T L_y and its Hessian is the Schur complement over
w=(x,v):

    H_F = L_uu - L_uw inverse(L_ww) L_wu.

An implicit predictor follows the inner branch as u changes. Outer Newton
steps use the actual reduced potential for acceptance, allowing a 1e-13
numerical energy tolerance. Inner stationarity is checked to 2e-9 and
quadratic matrix matching to 1e-8. The construction is refused if the
maximizing block is not concave or resolved negative outer curvature remains.

## Three nested splits

**Common shift:** maximize only the common shift of all bath levels.
This removes one negative direction but leaves others.

**Local spectral split:** at the initial quadratically matched state, use
negative eigenvectors of H_G (below -1e-8) as the maximizing coordinates;
hold that basis fixed during the run. This is a local stationary-point
construction, not a physical phase-selection principle.

**PH split:** keep three particle-hole preserving combinations as outer
coordinates: central hybridization, symmetric outer hybridizations, and
antisymmetric outer bath levels. Maximize over outer-hybridization asymmetry
and the two particle-hole breaking bath-level combinations. The initial
states are not constrained to PH symmetry; these three coordinates are
solved, not manually set to zero.

At T=0.2, U=2, the PH maximizing block at the reference has eigenvalues about

    -1.28630, -0.00750264, -0.0000294611

and the remaining Hessian has eigenvalues about

    0.000501426, 0.299647, 2.84431.

This establishes a strict local minimum after that particular reduction.
It does not establish a global variational upper bound or a valid partition
for other phases, fillings, or temperatures. Maximizing arbitrary negative
modes can hide a physical instability.

## What “coupled saddle” means here

**Coupled Newton** solves the twelve equations grad(L)=0 simultaneously:

    H_L delta = -grad(L).

Signed curvature gives descent in locally convex modes and ascent in
locally concave modes. Numerically unresolved modes are omitted from the
inverse, steps are capped at 0.25 per coordinate, and a line search reduces
the stationarity norm. It does not require the actual saddle functional
to decrease. It can converge to an unstable stationary branch.

**Coupled extragradient** first predicts a state, evaluates its gradient,
then corrects from the original state using that predicted gradient:

    z_predict = z - alpha M grad(L(z))
    z_next    = z - alpha M grad(L(z_predict)).

M is the signed inverse Hessian at the current accepted state, reused
within the predictor/corrector pair and refreshed each iteration.
The same stationarity merit controls backtracking. This is a local
preconditioned adaptation, not a claim that RISB meets convex-concave
convergence assumptions. The initial frozen-preconditioner variant stalled
in 41/95 solves on the small T=0.2 starts; those diagnostic records are retained.

The predictor/corrector idea is discussed in
[Mokhtari, Ozdaglar, and Pattathil](https://arxiv.org/abs/1901.08511).
For broader Newton approaches to local minimax problems, see
[Zhang et al.](https://arxiv.org/abs/2006.14592).
Our coupled Newton is a direct stationary-equation Newton prototype, not
a reproduction of that paper's algorithms.

These methods avoid nested auxiliary maximization. Production root DIIS
also seeks stationarity, but accelerates its existing nonlinear matching
cycle and uses its default plateau/anchoring callbacks, rather than the
analytic Hessian of L.

## Experiments, 4 November 2026

All cases: half-filled paramagnetic one-band Bethe model, U=2, B=3,
half bandwidth 1, 400 equally weighted band samples.
Two deterministic seeds perturb all twelve initial fields by 1% or 15%,
with a field magnitude floor of 0.2. These are local restarts around a
known solution, not searches for unknown phases. Each method starts from
the same perturbed matrices. Nested projection costs are included.

Every successful result must pass maximum gradient <1e-8 and all original
matrix matching residuals <1e-7. Nested quadratic projection uses 1e-9.
Production root DIIS uses native root tolerance 1e-10 and then passes the
same fresh gradient/matrix checks. Its separate validation impurity solve
is excluded from table counts; the JSON records both counts.

Each method receives at most 200 interacting solves and 30 seconds.
All rejected and inner impurity trials count. Runs are serial with one
numerical thread. Reference generation is excluded.
Times and response counts are saved, but these are single-run diagnostics,
not repeated performance statistics. Each analytic Hessian also has a
quadratic/spectral response cost; root DIIS needs none.

Each entry below is solves for seeds 1 / 2.

| T, perturbation | Root DIIS | Nested PH maximum | Coupled Newton | Coupled extragradient |
|---|---:|---:|---:|---:|
| 0.2, 1% | 6 / 8 | 12 / 13 | 7 / 8 | 135 / 97 |
| 0.2, 15% | 8 / 8 | C42 / Q0 | 16 / F43 | 151 / F200 |
| 0.5, 1% | 7 / 30 | Q0 / Q0 | F200 / F200 | F200 / F200 |
| 1.0, 1% | 29 / 30 | Q0 / Q0 | F200 / F200 | F200 / F200 |
| 1.0, 15% | 28 / 29 | Q0 / Q0 | F200 / F200 | F200 / F200 |

F denotes failure after that many impurity solves: F43 is a stationarity
line-search stall; F200 exhausts the solve budget.
C42 reaches a stationary auxiliary block that is not a maximum.
Q0 fails the quadratic projection before an interacting solve; it is not
a zero-cost success.

For T=0.2, 1% starts, the common-shift reduction is refused after 4/5 solves:
the remaining minimum curvature is -0.0269/-0.0318.
The spectral split exhausts 200/200 solves, despite potential errors of
only 2.3e-11/3.5e-8; matching residuals remain 2.9e-6/3.3e-5.
The PH split converges with matching below 8.2e-11, potential errors below
1e-15, and double-occupancy errors below 3.2e-11.
Its largest maximizing curvature is about -2.95e-5 and smallest outer
curvature about 5.01e-4 at both final states.

At T=0.2, 15%, the spectral split loses concavity after 17 solves on seed 1;
common-shift maximization leaves negative outer curvature after 9.
All nested versions fail the initial quadratic projection on seed 2,
at matrix residual 8.04e-5.

At T=0.5, initial quadratic projection stalls at 1.95e-7/1.52e-7.
At T=1 it stalls at 1.08e-6/2.78e-7 for 1% and
5.39e-6/9.68e-7 for 15%. These fail the inner tolerances without changing them.

The coupled methods bypass that projection, but do not solve the high-T
difficulty in this prototype. At T=1, 1%, Newton finishes at matching
1.22e-7/2.40e-7 and gradient 4.86e-7/9.58e-7.
Extragradient finishes at matching 9.66e-7/1.14e-7 and gradient
3.86e-6/4.57e-7. Tiny potential errors do not qualify these as converged.

The local minimum construction is therefore possible on a controlled
finite-T branch. The tested nested and coupled methods do not improve
robustness over default production root DIIS on these harder starts.
No general thermodynamic minimum formulation has been established.

## Reproduce and checks

From the repository root:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 uv run python benchmarks/thermal_saddle.py --temperature 0.2 --perturbation 0.01 --budget 200 --seconds 30
    uv run python -m pytest benchmarks/test_thermal_saddle.py -q

Use temperature 0.5 or 1.0, or perturbation 0.15, for the other cases.
Complete records and traces are written to
_build/adiis/thermal-saddle-T<temperature>-p<perturbation>.json.
Generated records are ignored by Git; this report, source, and tests persist.

Six physical checks cover interacting potential gradients, analytic Gibbs
responses, the matched nested minimum and its curvature, both coupled
methods, and the residual negative curvature after common-shift elimination.
