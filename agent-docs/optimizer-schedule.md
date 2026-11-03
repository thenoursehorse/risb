# Optimizer schedules

Keep it simple. Keep modules uncoupled; prefer functions over stateful classes.
Use existing optimizer and diagnostic interfaces and follow repository names.

## Workflow

Implement one checklist item at a time. Describe the files, behavior and
signatures before each item, then wait for approval. Approval covers that item
only. Review the code, run LSP, lint, type checks and relevant tests, update
the checklist, and stop for review.

## Design

`OptimizerSchedule` wraps `NewtonSolver` optimizers. Each `SchedulerPhase`
is one stage of the iteration, with an optimizer, switching condition,
destination and history choice. Standalone optimizers remain supported.

| Component | Responsibility |
| --- | --- |
| Schedule | Choose optimizers, check conditions and count progress. |
| NewtonSolver / DIIS | Keep iterates, calculate guesses and accept or reject trials. |
| FixedPointLoop | Evaluate the problem, enforce maxiter and track convergence. |
| Diagnostics | Report progress and switches. |

`DIIS.policy` still selects the numerical method. `Pulay` controls mixing
and periodic history resets; `BianLi` controls acceptance and damped steps
after rejection. DIIS delegates to `DIISPolicy` without checking concrete
policy types. Policies share `extrapolate()` in `diis.py`.

Usage and examples: [DIIS guide](../docs/how-to/diis.md#scheduling-optimizers).

## History and switching

History stores accepted `x`, `g_x` and root residuals, newest first.
Both `get_history()` and `set_history()` copy arrays. Loading history keeps
up to `history_size` entries, recomputes errors using the receiving
optimizer's `error_fun`, and starts its method counter at one.

The destination chooses `history="carry"` to reuse available iterates or
`"clear"` to keep only the iterate at the switch. Changing residuals does
not itself require discarding iterates.

After each step:

1. The current optimizer accepts the trial or evaluates a replacement.
2. Check the condition using that accepted iterate.
3. If switching, load the selected history and calculate the next guess.

The existing step method also calculates an outgoing guess that may be
unused. Switching adds no problem evaluation. If periodic clearing emptied
the history, use the accepted switching point to start the next optimizer.
Calling `update_x()` alone does not trigger Pulay's periodic clearing.

Without `next_phase`, a condition selects the next listed stage. An integer
or callable can select any index, including the current or an earlier stage.
Re-entry starts a fresh method run and resets local counts; total counts
continue. At most one switch occurs per accepted step. A stage without a
condition runs until the solve stops. A final stage with a condition needs
an explicit destination.

For BianLi's `residual` and `combined` rules, the fixed reference is
`||g(x)-x||` at the switch, rather than at the original R and Lambda.
It remains fixed during that stage, including after rejection clears the
rolling history. The `iteration` rule does not use it. An invalid acceptance
factor still raises an error. BianLi formulas remain inside its policy.

## Residuals and progress

Conditions are functions of immutable `SchedulerState` snapshots:
`after_steps`, `residual_below`, `reduced_by`, `plateau`, `any_of`, `all_of`.
Residual conditions use Euclidean norms and can select root or recursion
errors. By default they follow the active optimizer.

`scope="phase"` measures progress during the current stage;
`scope="solve"` measures progress across the schedule, including continued
solve calls. Root and recursion norm histories are kept separately, oldest
first. Stage histories include the switching point without counting it twice.

| Counter | Meaning |
| --- | --- |
| `n_steps` | Accepted iterations, including the first evaluation. |
| `n_evaluations` | All cycle calls, including rejected trials and state restoration. |
| `n_phase_steps`, `n_phase_evaluations` | Counts during the current stage. |
| `scheduler_phase_index` | Current zero-based stage index. |

The active optimizer's checks enforce one `maxiter` across all stages in
each call. Rejected trials do not advance the schedule independently.
Convergence after a switch uses the new optimizer's residual.

`FixedPointLoop` keeps separate best iterates for root and recursion errors.
`get_error`, `get_best_error` and `get_best_x` select either residual or follow
the active optimizer. `SolveResult.history_error_funs` labels the values in
`history`, recorded before callbacks. Callback changes recheck convergence
without changing earlier history labels or adding an evaluation.

## Resets and plateau recovery

- `reset_active_optimizer()` clears the active method's history and counter,
  keeping scheduler position and progress.
- `reset()` clears all optimizers and starts the schedule again.
- `clear_norm_history=True` also discards norms of a previous fixed-point map.

`OnPlateau` resets the active method when progress stalls. Near a solution it
can anchor a problem at its best iterate and switch to recursion errors.
It monitors cycle evaluations; the schedule's `plateau` condition counts
accepted iterations.

By default the callback follows the active residual and starts a new
comparison when that definition changes. An explicit `error_fun` fixes the
monitoring residual. `reset_on_scheduler_phase_entry=True` starts a new
comparison on each entry, including re-entry; the default is False.

Anchoring discards old numerical history, residual norms and best iterates
while retaining scheduler position and counts. The next cycle establishes
the new reference. Continuing with the same schedule and `result.x_next`
retains the current stage.

## Diagnostics

`report_step(observer, *, error, converged)` reports the accepted iteration
before callbacks. The schedule also emits `optimizer.switched`, including
self-transitions, through the existing observer. Lattice reporters show
switches at INFO with stage indices, history choice, optimizer and residual.
Diagnostic cycle numbers refer to the current solve call.

## Validation

The full suite passed: 254 tests and 222 subtests. Ruff, mypy and LSP checks
are clean; the strict Sphinx build passes. The 27 added test cases took
5.7 seconds in that full run, including 3.1 seconds for the thermal comparison.

Cycle evaluations at tol=1e-12 with no recovery callbacks:

| Model | Pulay | BianLi | Scheduled carry | Scheduled clear |
| --- | ---: | ---: | ---: | ---: |
| One band | 18 | 18 | 20 | 21 |
| Bilayer | 17 | 15 | 19 | 18 |
| Kanamori | 17 | 17 | 19 | 20 |

Schedules use four linear-mixing steps with alpha=0.1, then default BianLi.
All methods agree on chemical potential, Lambda and quasiparticle weights,
with self-consistency error below 1e-10. The illustrative schedules take
slightly more evaluations on these default starts.

Split runs preserve accepted iterates to absolute tolerance 1e-12, residual
labels and total evaluation counts, including rejected trials and state
restoration. Diagnostics retain the correct switch counts and cycle numbers.
The anchored thermal ghost solution agrees with the standalone result in
filling, double occupancy and grand potential.

Physical tests live beside their model fixtures in `test_latticesolver.py`
and `test_thermal_latticesolver.py`; reporting tests are in
`test_lattice_diagnostics.py`.

## References

- [Q-Chem hybrid SCF](https://manual.q-chem.com/6.3/sub_custom_algorithm.html):
  algorithm sequences with iteration and error thresholds. Its documentation
  does not establish whether ADIIS-to-DIIS switching discards history.
- [Psi4's implementation](https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/diis.py):
  ADIIS and DIIS can use the same stored samples and blend coefficients.
- [Brune et al.](https://arxiv.org/abs/1607.04254): nonlinear solver composition.
- [Bian and Li](https://arxiv.org/html/2609.28950v1): the acceptance rules used by
  the existing BianLi policy.

## Deferred

Persistent optimizer serialization needs a separate API. Continuation keeps
the optimizer in memory; existing problem checkpoints restore rejected trials.

## Implementation checklist

1. [x] Record the agreed design, names and approval workflow in this file.
2. [x] Implement accepted-sample storage and history import in
   `solver_newton.py` and `diis.py`; preserve standalone behavior.
3. [x] Review and redesign policy ownership: remove concrete-policy dispatch
   from DIIS; define initialization, rejection and periodic history clearing.
   Outline the design and obtain approval before implementation.
4. [x] Add `schedule.py` with `OptimizerSchedule`, `SchedulerPhase`,
   conditions, counters and explicit/callable transition targets; export the
   public API in `optimize/__init__.py`.
5. [x] Integrate metric-aware progress, callbacks and diagnostics in
   `fixed_point.py`, `callbacks.py` and `diagnostics/lattice.py`.
6. [x] Complete physical scheduling tests in `tests/test_latticesolver.py`,
   `tests/test_thermal_latticesolver.py` and `tests/test_lattice_diagnostics.py`;
   compare representative physical problems, update `docs/how-to/diis.md`,
   and run the full suite, static checks and documentation build.
