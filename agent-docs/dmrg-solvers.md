# DMRG impurity solvers

Explore DMRG impurity solvers for d (5 orbitals) and f (7 orbitals) shells with
ghosts, past fockit ED. One step at a time; stop after each step for the user to
review. Plan: `~/.claude/plans/this-repo-currently-have-piped-star.md`.

Decided (user): block2, ITensor (juliacall), TeNPy prototypes; MPToolkit later;
yaqs and quimb dropped. Collinear first, SOC later. One solver per code
(`Block2ImpuritySolver`, `ITensorImpuritySolver`, `TeNPyImpuritySolver`), each an
optional dependency. Bond-dimension extrapolation inside each impurity solve
(against the variance), only when the truncation error at M_max is above a
tolerance; extrapolation of converged RISB results as a comparison.

## Steps

1. [x] Embedding Hamiltonian as integrals (see Step 1 notes).
2. [x] block2 prototype + validation vs fockit ED (see Step 2 notes).
3. [x] ITensor prototype + validation (see Step 3 notes).
4. [x] TeNPy prototype + validation (see Step 4 notes).
5. [x] d- and f-shell benchmark, B = 1 and 3 (see Step 5 notes).
6. [ ] LatticeSolver test with the best candidate: inner vs outer extrapolation.
7. [ ] Implement the kept solvers, optional deps, tests.

## Step 1 notes

- Scratch code (not in the package): `integrals.py`, `check_integrals.py` in the
  session scratchpad `dmrg/` folder.
- `to_integrals(op, index)` -> `const, h1, V` with
  `op = const + sum h1[p,q] c_p^dag c_q + sum V[p,q,r,s] c_p^dag c_q^dag c_r c_s`,
  V antisymmetrized, spin-orbital modes in `FockitImpuritySolver` order
  (physical blocks, then `bath_` blocks). Round trip exact (<= 2.2e-16) for
  Kanamori n_orb = 2, 5, 7 with B = 1, 3 and random Lambda_c, D, h0_loc.
- The converter only reads fockit OpSum monomials (algebra, no basis). The test
  models build `h_imp` with fockit operators, same terms as `set_h_imp`.
- For step 7: the h_imp terms (h0_loc, bath, hybridization) are built inside
  `_FockitBase`; the DMRG solvers need the same terms, so make that a public
  function rather than duplicate it.

## Step 2 notes

- Scratch: `block2_solver.py`, `check_block2.py`, `probe_thrds.py`,
  `probe_extrapolation.py`. block2 0.5.4 (cp314 arm64 wheel) via
  `uv run --with block2==0.5.4`, pyproject untouched.
- SZ mode, spatial sites = physical orbitals then bath orbitals, N = n_sites,
  2Sz = 0. MPO from `expr_builder` with the `to_integrals` terms (c/d up, C/D
  down); constant added after. 1-RDM from `get_1pdm` (up, dn).
- vs fockit ED (Kanamori U = 4, J = 0.8, random Lambda_c, D, h0_loc): energy
  to <= 5e-14 for n_orb = 2 B = 1, 3; n_orb = 3 B = 1; d (n_orb = 5) B = 1
  (M = 1000, 1.5 s; ED 4.9 s).
- 1-RDM error scales as sqrt(Davidson thrd) (thrd 1e-10, 1e-12, 1e-14, 1e-16
  -> 6e-8, 1.6e-8, 2.2e-9, 2.3e-10), not with truncation. Need thrd ~1e-16.
- Truncation, d B = 1 (exact needs ~1000):

  | M | dE | discarded weight | variance | max 1-RDM error |
  |---|---|---|---|---|
  | 60 | 1.1e-1 | 3.5e-3 | 1.7 | 2.6e-2 |
  | 120 | 1.1e-2 | 2.8e-4 | 2.5e-1 | 4.2e-3 |
  | 240 | 3.8e-5 | 1.5e-6 | 1.2e-3 | 9.1e-5 |
  | 400 | 2.1e-7 | 7.5e-9 | 7.3e-6 | 3.3e-6 |

- Variance from |H psi> fitted at 4M or 8M: same to 2 digits.
- Reverse schedule (M_max x 1, 0.9, 0.8, 0.7, 2 sweeps each), linear fit:
  energy extrapolation helps (M_max 240: 3.8e-5 -> 9e-7 in discarded weight,
  -3.6e-5 in variance; 400: 2.1e-7 -> 7e-8). 1-RDM extrapolation does not
  (240: 9.1e-5 -> 1.2e-4 / 1.4e-4; 400: 3.3e-6 -> 6e-6 / 5e-6); the 1-RDM
  error is not smooth in M (400, 360, 320: 3.3e-6, 4.1e-6, 2.4e-5). With a
  wide schedule (down to 0.4 M_max) both get worse.

## Step 3 notes

- Scratch: `itensor_solver.py`, `check_itensor.py`, `probe_itensor.py`,
  `probe_itensor_threads.py`; shared `SPINS`, `Result`, `sites` moved to
  `integrals.py`. Julia 1.13.2, ITensors 0.9.32, ITensorMPS 0.4.2 via
  juliacall/juliapkg into the scratchpad (`julia_env/`, `juliapkg.json` next to
  the scripts, `PYTHON_JULIAPKG_PROJECT`). First resolve + precompile 9 min,
  load 2 s, first DMRG call (JIT) ~20 s per process.
- "Electron" sites with N, Sz conserved, OpSum from the `to_integrals` terms,
  random MPS in the Sz = 0 half-filled sector, `correlation_matrix` for the
  1-RDMs, exact variance `inner(H, psi, H, psi) - E^2`.
- vs fockit ED (same models as step 2): energy <= 3e-14, 1-RDMs 1e-12 to 1e-16
  at `eigsolve_tol` 1e-12 or 1e-14 (block2 needs thrd 1e-16 for 2e-10).
  Exception n_orb = 3 B = 1: 16 sweeps leave drho 6e-6 to 2e-5 (3 seeds), 40
  sweeps 1e-13. Fixed sweep counts are not enough; a solver needs a
  convergence test.
- Time, d B = 1, M = 1000, 16 sweeps: 17.9 s (1 Julia thread, BLAS 4); 26.5 s
  with threaded block-sparse (4 threads, Strided off). block2: 1.5 s, but its
  `dmrg` stops early on energy tol, so not the same sweeps; fair timing in
  step 5. Threads need `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` (silent crash
  without).
- Truncation, d B = 1, exact variance: same picture as block2.

  | M | dE | variance | max 1-RDM error |
  |---|---|---|---|
  | 240 | 3.3e-5 | 9.4e-4 | 1.3e-4 |
  | 400 | 1.9e-7 | 5.3e-6 | 4.6e-6 |

  Reverse schedule (1, 0.9, 0.8, 0.7) M_max, linear in variance: energy
  240: 3.3e-5 -> -3.7e-5, 400: 1.9e-7 -> -7e-8; 1-RDM 240: 1.3e-4 -> 1.5e-4,
  400: 4.6e-6 -> 2.3e-6. 1-RDM error again not monotonic in M (400, 360:
  4.6e-6, 3.6e-6).

## Step 4 notes

- Scratch: `tenpy_solver.py`, `check_tenpy.py`. physics-tenpy 1.1.1 (cp314
  arm64 wheel) via `uv run --with physics-tenpy`.
- `SpinHalfFermionSite(cons_N, cons_Sz)` on a finite `Chain`, terms through
  `add_local_term` (handles Jordan-Wigner; `add_multi_coupling_term` does
  not), `TwoSiteDMRGEngine` with mixer, `chi_list`, and its own convergence
  test (`max_E_err`, `max_S_err`). 1-RDMs from `correlation_function`
  (Jordan-Wigner automatic). Exact variance from `H_MPO.variance`.
- vs fockit ED (same models): energy <= 4e-14, 1-RDMs 1e-11 to 1e-15 for every
  case, including n_orb = 3 B = 1 (where ITensor's fixed 16 sweeps fell short),
  at E_tol 1e-12 and 1e-14. Correct on the first run.
- Time, d B = 1, M = 1000: 37 s (E_tol 1e-12), 46 s (1e-14). Reverse schedule
  with exact variances: 92 s (M_max 240), 83 s (400).
- Truncation and extrapolation: same numbers as block2 to 2 digits (M = 240:
  dE 3.8e-5, drho 9.1e-5; 400: 2.1e-7, 3.8e-6). Extrapolated in variance: 1-RDM
  240: 9.1e-5 -> 1.3e-4, 400: 3.8e-6 -> 5.6e-6; energy 2.1e-7 -> -7.7e-8.

## Step 5 notes

- Scratch: `bench.py` (grid, one process per point, 30 min cap), `bench_one.py`,
  `analyze.py`, `check_gaps.py`; results in the scratchpad `bench/`.
  Rule per point: warm-up (M/4, M/2, then M; noise off after 6 sweeps), then
  sweeps at M until |dE| < 1e-12, max 40. 4 BLAS threads.
- Models (U = 4, J = 0.8): structured = per-orbital ghosts (particle-hole
  symmetric lambda, d = 0.5 / 1), crystal field; dense = random all-to-all
  Lambda_c, D, h0_loc. Reference: ED (d B = 1), else lowest block2 energy.
- Early timeout runs left the python under `uv` running (fixed: process group);
  block2 timings of dense d B = 3 fiedler 500 onward are inflated by that.
- block2, structured, interleaved ordering (physical orbital then its ghosts)
  is best everywhere:

  | size | converged at | discarded weight | time | max RSS |
  |---|---|---|---|---|
  | d B = 1 | 250 (ED: dE 3e-14, drho 3e-9) | 7e-20 | 0.3 s | 0.2 GB |
  | d B = 3 | 500 | 4e-11 | 4.5 s | 1.2 GB |
  | f B = 1 | 250 | 7e-14 | 3 s | 0.4 GB |
  | f B = 3 | 500 | 1e-10 | 49 s | 1.4 GB |

  Fiedler second (f B = 1 fine, d B = 3 needs 1000). Block ordering (all
  physical, then all ghosts) hits 40 sweeps from d B = 3 / f B = 1 on, dE
  1e-2 to 1 at M = 1000.
- Excited-state trap: 6 B = 3 runs (interleaved d 250, 2000, f 250, 1000;
  fiedler d 250, 2000) converged to E_0 + 1.2 (the first gap), with small
  discarded weight. Variance and discarded weight cannot catch it (eigenstate).
  For RISB: warm start from the previous iteration's MPS, or several starts and
  keep the lowest energy.
- Dense: only d B = 1 converges (M = 1000). d B = 3 and f B = 1 hit 40 sweeps at
  every M <= 1000 (f B = 1 fiedler converges at 2000, 9 sweeps, 516 s), with
  discarded weight 1e-3 to 1e-2. Random all-to-all Lambda_c, D is a worst
  case; symmetry usually makes them block diagonal per orbital. Dense f B = 3
  not run; ITensor and TeNPy on structured only.
- ITensor, TeNPy, structured, interleaved, M <= 1000 (time / sweeps, dE, max
  1-RDM error; block2 for comparison):

  | size, M | block2 | ITensor | TeNPy |
  |---|---|---|---|
  | d B = 1, 250 | 0.3 s / 8, 3e-9 | 17 s / 8, 3e-9 | 2.6 s / 9, 2e-11 |
  | d B = 3, 500 | 4.5 s / 9, 6e-9 | 64 s / 12, 3e-7 | 120 s / 18, 5e-9 |
  | f B = 1, 250 | 3 s / 9, 1e-9 | 24 s / 9, 2e-8 | 12 s / 12, 2e-9 |
  | f B = 3, 500 | 49 s / 11, ref | 209 s / 20, 2e-7 | 237 s / 14, 2e-9 |
  | f B = 3, 1000 | 546 s / 13, trapped | 1494 s / 40, trapped | 470 s / 17, 1.5e-8 |

  ITensor times include ~15 s JIT per process; its RSS starts at ~2 GB.
  TeNPy f B = 3 M = 1000 is 6e-9 below the block2 M = 500 reference.
- Trap by code: block2 and ITensor (random initial MPS) land on E_0 + 1.2 in
  4 and 3 B = 3 runs; TeNPy (alternating up/down product state + mixer) in
  none.
- ITensor 1-RDMs are 1e-7 to 1e-8 at the same energy rule (others 1e-9).

## Trap check (before step 6)

- Scratch: `check_trap.py`, `check_su2.py`. Structured d B = 3, interleaved,
  block2, M = 250, 4 seeds (block2's `rand_seed` does not make runs
  reproducible; seed 0 gave different states in two runs).
- Ground state: S = 0, N_a = 4 in every orbital block (orbital + its ghosts),
  all P_a = (-1)^N_a = +1. Stuck states: an electron moved between two blocks
  (N_a = 5, 3; P_a = -1 on both), S = 0 or 1 nearly degenerate (+1.219,
  +1.213 for blocks 0, 4; +1.23 for 1, 3 or 2, 3); two pairs moved, S = 2,
  +2.45. P_a is conserved because Lambda_c, D are orbital diagonal and
  Kanamori pair hopping moves 2 electrons.
- Hits on the ground state out of 4 starts: SZ default noise 1, SZ
  Wavefunction (random) noise 2, SZ `occs` hints 2, SU2 S = 0 0, SU2 S = 0 +
  parities 4 (9 sweeps, 1 s each; SZ needs M = 500, 4.5 s).
- Spin alone does not help: the low excitations are inter-block charge
  transfers with S = 0 and 1 almost degenerate. Unrestricted-parity SU2 S = 1
  also landed at +1.226, not the S = 1 minimum +1.213 (parity sector 0b10001).
- Parities in block2: `initialize_system(orb_sym=..., pg_irrep=...)` with site
  irrep 2^a for block a (XOR combination); works with 5 bits in SU2 mode, f
  (7 bits) untested. Hamiltonian through `get_qc_mpo(h1e, g2e)` with
  (ij|kl) = 4 V[i up, k dn, l dn, j up] (spin-free; checked equal to the
  expr_builder MPO).
- Which symmetries exist depends on the RISB problem (block structure of
  Lambda_c, D); per-orbital parity is the orbital-diagonal case.

## Step 6 notes (in progress)

- Symmetries (user): implied by the gf_struct blocks (N, Sz for up/dn blocks)
  are enforced and their sectors scanned (lowest kept, as ED). Value-based ones
  (SU2 from equal up and dn; per-orbital parities from orbital-diagonal
  Lambda_c, D) are opt-in only: enforcing them prevents spontaneous symmetry
  breaking.
- Literature: the trap is known for DMRG impurity solvers with a bath per
  orbital. F. A. Wolf, A. Go, I. P. McCulloch, A. J. Millis, U. Schollwoeck,
  Phys. Rev. X 5, 041032 (2015), Appendix A1 (arXiv:1507.08650): noise built
  from H cannot mix sectors H does not connect (e.g. three-band Kanamori);
  their fix, sweeps with extra symmetry-mixing hoppings of 10 %, then 1 %, then
  the exact H. Must be cited wherever it is used.
- Scratch `b2_mix.py` (copied to `b2_risb.py`): Wolf et al. warm-up, 6 sweeps
  at 10 %, 6 at 1 % of the largest one-body coupling, random hoppings between
  every pair of physical orbitals and between chain neighbours; spin dependent
  in SZ mode (breaks S^2), spin independent in SU2 mode, only within a group
  with parities enforced. Chain neighbours alone were not enough (d B = 3:
  charge transfer between distant orbitals stayed stuck, 2/4 to 3/4).
- Validation vs ED (Kanamori U = 3, J = 0.6, n_orb = 2, 3, B = 1, 3, risb
  ghost layout), all settings with mixing: right sector, dE <= 7e-9, drho
  <= 4e-8. Without mixing, n_orb = 3 B = 3 picks Sz = +1 (+1.0).
- d B = 3 (risb layout), ground state out of 4 seeds, M = 250:

  | setting | M_scan 60 | M_scan 100 |
  |---|---|---|
  | SZ, no mixing | 1 | |
  | SZ, mixing | 2 | 4 |
  | SU2, mixing | 2 | 4 |
  | SU2 + parities, mixing | 4 | |

  At M_scan = 60 the warm-up runs at M = 15, 30, too small (Sz = 0 alone: 2/4
  at M = 60, 4/4 at M = 100). Scan needs M_scan >= 100.
- SU2 mode averages up and dn one-body terms (LatticeSolver's differ beyond
  1e-8 at n_orb = 3 B = 3), warns above 1e-6.
- Case a (Kanamori J = 0.2U, Bethe, U = 1.5, block2 SU2 + parities, M = 200,
  RISB tol 1e-6), Z block2 / ED: 2 orb B = 1 0.7607156 / 0.7607156; B = 3
  0.96039622 / 0.96039622; 3 orb B = 1 0.6756161 / 0.6756163. 3 orb B = 3: no
  convergence with either solver (DIIS, 200 cycles, error 3e-4 block2, 1.4e-4
  ED in 99 min); block2 map deterministic there (repeat solves 2e-13, sector gap
  0.29), so a RISB-side problem.
- Case b (3 orb B = 1, U 1 -> 4 -> 1, warm started): same branches with ED and
  block2: metal up to U = 1.5, Z ~ 5e-5 at U = 2, Mott above; on the way down
  both stay at R = 0 (the atomic-limit fixed point of the iteration). In the
  Mott phase the bath decouples and many sectors are exactly degenerate: the
  rescan now keeps the current sector unless another is lower by > 1e-8.
- Case d (2 orb B = 3, SZ default, ED Z = 0.9603962208): no RISB run converged
  to 1e-6 at M = 20 to 50; the error stalls at the truncation level (fixed M:
  dZ 1.5e-3, 1.8e-3, 2.9e-4, 1.0e-4; stalled error 2e-4, 7e-4, 3e-4, 4e-5).
  Repeat solves agree to 1e-10, so not noise: the truncated map is not smooth.
  Inner extrapolation is worse (dZ 6.8e-3, 3.3e-3, 3.6e-4, 4.4e-5; stalled
  error 2 to 70 times larger). Outer (Z linear in the variance of unconverged
  fixed-M runs): 1.1e-4 (40, 50), 6.6e-5 (30 to 50). So the gate should raise
  M until the truncation is below the RISB tolerance, not extrapolate.
- Case c (d B = 3) with opt-in parities: Lambda_c, D acquire inter-orbital
  elements up to 8e-3 (degenerate orbitals: ghost rotations between orbitals
  are a gauge freedom), which the groups dropped. Opt-in parities need a
  matching LatticeSolver symmetrizer. Rerun with SU2 only, no extrapolation,
  M = 400.
  Result: not converged in 100 cycles (34 min, ~20 s per solve); error
  falls to ~4e-3 by cycle 33, then stalls at 3e-3 to 6e-3. Z = 0.9713, sector
  S = 0 every scan (gap 0.42 to S = 1), dw 6e-8. R off-diagonals ~1e-4 to 1e-3
  (no orbital symmetrizer). Same picture as case d (truncation floor) plus
  possibly the B = 3 RISB-side problem of 3 orb B = 3 ED; not separated.

## Decisions after step 6
- M as large as the machine allows; if that is not enough, extrapolate in an
  outer loop (several converged fixed-M RISB solves). No inner extrapolation.
- Symmetries are user supplied (opt-in), never detected automatically.
- block2 only; ITensor and TeNPy dropped.
- Later: 3 orb B = 3 RISB non-convergence (both solvers). Case c not rerun.

## Step 7 design (agreed)
- `Block2ImpuritySolver(h_int, gf_struct, bond_dim, symmetries={}, spin_blocks, mix=True, *, bath_size)`.
- N always half filling, never user-set.
- `symmetries: Mapping[str, float | None]`, keys "sz", "s" (SU2 mode), later
  "parities"; value targets that sector, None solves every sector of it and
  keeps the lowest (ties within 1e-8 keep the previous sector). "s" with "sz"
  is an error. `{}` conserves N only (block2 SGF).
- No walk-outwards heuristic: E_min(S) need not be monotone. Spin-rotation
  symmetric H: Sz = 0 (or 1/2) already holds the ground state.
- No n_threads (OMP_NUM_THREADS), no scratch arg (tempdir, follows TMPDIR),
  no scan bond dim, no rescan().
- All MPOs via get_qc_mpo (MPI later). helpers_fockit.get_integrals public.
- CI installs the block2 extra.
- N-only check (d B = 3, M = 500, mixing, 4 seeds): all reach the S = 0 energy
  (dE 3.7e-9 N only, 3.6e-9 Sz = 0, truncation level); time 18 s N only, 8 s
  Sz = 0, 6 s S = 0.
- Default `symmetries={}` (user). "s" with "sz" allowed: SU2 solve, then
  mps_change_to_sz (needs align_mps_center(ket, ref=0) first, and the
  projected state has norm 1/(2S+1)).
- Done: src/risb/impurity_solvers/block2.py, helpers_fockit.Op_to_tensors,
  lazy import, `block2` extra, mypy override, CI/CD `--extra block2`,
  tests/test_block2.py (8 tests, 5 s), how-to section, about and README.
  Package check vs ED (2 orb B 1 and 3, 3 orb B 1, Zeeman h = 0.3): every
  mode dE <= 4e-13, drho <= 8e-9 (2.2e-7 for the projected S = 1 state).
- Later: MPI (block2-mpi, mpi=True; single-rank speed unknown, Linux-only
  wheels), parities, finite T, SOC, outer extrapolation helper.
