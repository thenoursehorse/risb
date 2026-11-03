# Optimizer schedules

A general, opt-in way to run a sequence of mixing policies in one
`risb.optimize.DIIS` solve, switching between them when conditions hold.

## Motivation

What helps a fixed-point solve depends on where it is. Far from a solution,
plain linear mixing or periodic mixing steps move the guess out of a bad
starting region and refresh a stale extrapolation subspace. Close to a
solution, extrapolation alone converges fastest, and mixing steps can undo its
progress. A single fixed policy has to compromise; a schedule does not.

Making a step conditional on lowering the residual is not a substitute: the
point of a periodic mixing step is to move the guess even when that costs a
temporary increase.

## Framework

A schedule is an ordered list of phases. Each phase has a policy (an existing
`Pulay` or `BianLi`, or any future policy) and an exit condition. The solve
starts in the first phase and moves to the next when the current phase's exit
condition holds. The last phase has no exit condition.

Conditions should be small composable objects evaluated after each step, for
example:

- a number of cycles spent in the phase;
- the residual norm below a threshold;
- the residual norm fallen by a factor since the phase started;
- no progress for some number of cycles (a plateau, as `OnPlateau` detects);
- `any_of` / `all_of` combinations of the above.

Policies receive the history, including the current residual, in
`update_x(x, g_x, error, iteration)`, so residual conditions need no new
plumbing. State which norm a residual condition uses.

A possible API, following the existing policy dataclasses:

```python
DIIS(
    policy=Scheduled(
        [
            (Pulay(alpha=1.0, mixing_period=1), Cycles(5)),
            (Pulay(mixing_period=3), ResidualBelow(1e-2)),
            (Pulay(mixing_period=None), None),
        ]
    )
)
```

`Pulay(mixing_period=1)` takes the plain mixing step every cycle, so a
linear-mixing phase needs no new policy. The numbers are illustrative only;
the framework should have no tuned defaults.

## Requirements

- Opt-in: `DIIS()` and its current policies behave as now.
- Phases only advance. No automatic return to an earlier phase.
- One `error_fun` across all phases, or a clear error at construction if
  phases disagree.
- Keep the DIIS history across phase switches; do not empty it implicitly.
  An empty history makes the next extrapolation a plain mixing step, and
  callers may read the latest history entry after every step.
- Policies that can reject steps (`BianLi`) must keep their contract inside a
  phase, including how rejected evaluations count against the budget.
- Report the active phase and each switch through the diagnostics.
- Checkpoint and restore must include the schedule state (the phase and its
  counters), so a restored solve continues in the right phase.

## Tests

Few and high level, through the public solve API:

- A schedule ending in an extrapolation-only phase converges on a test
  problem where a fixed policy with periodic mixing stalls.
- A schedule starting with mixing converges on a test problem where an
  extrapolation-only policy diverges from its default start.
- Phase switches happen at the stated conditions and are reported.
