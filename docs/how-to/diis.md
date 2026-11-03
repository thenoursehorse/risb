# Using `DIIS`

This guide shows you how to use and customize {py:class}`~risb.optimize.DIIS`. This is an
implementation of algorithms from Ref. [^Chupin2021].

## Truncating history

{{DIIS}} combines previous iterates to reduce the residual. Limit the history
to control memory use and avoid keeping iterates far from the solution.

To specify how big the history should be

```python
from risb.optimize import DIIS, Pulay

optimize = DIIS(history_size=...)
```

If you want to reset the entire history after $n$ iterations

```python
optimize = DIIS(policy=Pulay(restart_period=...))
```

The history is kept between solves, so passing the same optimizer to another
`solve` continues it, e.g., one cycle per step of charge self-consistency

```python
result = S.solve(optimize, maxiter=1, x0=result.x_next)
```

To clear it

```python
optimize.reset()
```

## Step size

Each new guess moves a fraction $\alpha$ of the way from the extrapolated
guess toward the extrapolated fixed-point function. A smaller step is slower
but more stable, e.g., when the error grows or oscillates from one iteration to
the next. It is the mixing parameter of Pulay mixing in DFT codes
(`mixing_beta` in Quantum ESPRESSO)

```python
optimize = DIIS(policy=Pulay(alpha=...))
```

## Linear mixing step

Our implementation takes a single linear mixing step, with the same $\alpha$,
every `mixing_period` iterations.

To change how frequently a linear mixing step is taken, or to never take one
with `None`

```python
optimize = DIIS(policy=Pulay(mixing_period=...))
```

## Only linear mixing

If you just want to use linear mixing with nothing special

```python
from risb.optimize import DIIS, Pulay

optimize = DIIS(history_size=1, policy=Pulay(alpha=..., error_fun="recursion"))
```

## Arguments to `solve()`

```
optimize.solve(fun = function returning (g_x, residual),
               x0 = initial guess for x,
               args = args for fun,
               tol = stop solver when the error is less than this,
               maxiter = maxium number of iterations,
)
```

The error is the supplied root residual by default. To use the change
`g_x - x` instead

```python
optimize = DIIS(policy=Pulay(error_fun="recursion"))
```

## DIIS policies

`Pulay` combines DIIS with optional linear-mixing steps and history resets.
`BianLi` uses Anderson acceleration with an acceptance check: a rejected trial
is replaced by a damped fixed-point step. It requires `error_fun="recursion"`.

Custom mixing rules implement {py:class}`~risb.optimize.DIISPolicy`.
The DIIS extrapolation is also available as
{py:func}`~risb.optimize.extrapolate`.

## Scheduling optimizers

For RISB, the iterate $x$ contains $R$ and $\Lambda$. The `"root"` residual
measures the self-consistency equations $f_1$ and $f_2$; `"recursion"` measures
the change $g(x)-x$ after one iteration.

{py:class}`~risb.optimize.OptimizerSchedule` switches optimizers during a solve.
Each {py:class}`~risb.optimize.SchedulerPhase` describes one stage:

```python
from risb.optimize import (
    DIIS,
    BianLi,
    OptimizerSchedule,
    Pulay,
    SchedulerPhase,
    after_steps,
)

optimizer = OptimizerSchedule(
    [
        SchedulerPhase(
            DIIS(policy=Pulay(alpha=0.1, mixing_period=1)),
            until=after_steps(4),
        ),
        SchedulerPhase(DIIS(policy=BianLi()), history="carry"),
    ]
)
result = S.solve(optimizer, tol=1e-12, maxiter=1000)
```

This takes four linear-mixing steps, then switches to BianLi. The parameters
are illustrative. A stage without `until` runs until the solve stops.

### When to switch

| Condition | Switch when |
| --- | --- |
| `after_steps(5)` | Five iterations have been accepted. |
| `residual_below(1e-4, error_fun="root")` | The root-residual norm is below the threshold. |
| `reduced_by(10, error_fun="recursion")` | The recursion norm has fallen by a factor of ten. |
| `plateau(10, factor=2)` | Ten accepted iterations pass without a factor-two error reduction. |
| `any_of(a, b)` / `all_of(a, b)` | Any / all conditions hold. |

Import these helpers from `risb.optimize`. Norms are Euclidean; the default
`error_fun=None` follows the active optimizer. `after_steps`, `reduced_by`
and `plateau` use the current stage by default (`scope="phase"`).
`scope="solve"` uses progress across the whole schedule, including continued
solve calls.

The default destination is the next stage. `next_phase` can be a zero-based
index or a function of {py:class}`~risb.optimize.SchedulerState`:

```python
from risb.optimize import SchedulerState


def select_next_optimizer(state: SchedulerState) -> int:
    return 1 if state.recursion_norms[-1] < 1e-3 else 0
```

Setting `next_phase=select_next_optimizer` on the first stage above repeats
linear mixing until the recursion norm is small enough. Re-entry restarts
that stage using its history choice; total counts continue.
A final stage with an exit condition needs an explicit destination.

### Reusing iterates

`history="carry"` reuses accepted iterates, up to the next optimizer's
history size. `history="clear"` keeps only the iterate at the switch.
Errors are recomputed using the next optimizer's residual definition.

For BianLi's `residual` and `combined` rules, the fixed reference is
$\lVert g(x)-x\rVert$ at the switch. It stays fixed during that stage,
including after rejection clears the rolling history. Switching requires
no extra lattice cycle, and convergence uses the new optimizer's residual.

### Continuing and resetting

`maxiter` limits lattice-cycle calls across all stages, including rejected
trials and calls needed to restore the last accepted state.

```python
result = S.solve(optimizer, maxiter=50)
if not result.converged:
    result = S.solve(optimizer, x0=result.x_next, maxiter=50)
```

`optimizer.get_state()` reports total and current-stage iteration counts
and residual norms. Iterations count accepted steps; evaluations count all
cycle calls. The iterate at a switch is not counted twice.

- `reset_active_optimizer()` clears the current optimizer's history and
  method counter, keeping the stage and scheduler progress.
- `reset()` clears all optimizers and starts the schedule again.
- `reset_active_optimizer(clear_norm_history=True)` also discards saved
  norms, for use after changing the self-consistency iteration.

### Plateaus and diagnostics

`plateau(...)` switches stages. {py:class}`~risb.optimize.OnPlateau` resets
the active optimizer when progress stalls; near a solution it can also
anchor a thermal fit. It counts cycle calls rather than accepted iterations.

`OnPlateau(error_fun="root", reset_on_scheduler_phase_entry=True)` monitors
the root residual and starts a new comparison on each stage entry.
By default it follows the active residual and retains progress across
switches unless the residual definition changes.
Anchoring discards errors and iterates from the previous fixed-point map
while keeping scheduler position and total counts.

`result.history_error_funs` labels the residual used for each value in
`result.history`, recorded before callbacks. Best iterates are tracked
separately for root and recursion errors.

Existing lattice diagnostics report `optimizer.switched` at INFO, including
the old and new stage indices, history choice and active optimizer.
Diagnostic cycle numbers refer to the current solve call.

## Using with `LatticeSolver`

:::{seealso}
[Using `LatticeSolver`](lattice_solver.md#choosing-the-optimizer).
:::

If you want to use your customized {{DIIS}} solver

```python
S.solve(optimizer=DIIS(policy=Pulay(alpha=...)), tol=..., maxiter=...)
```

[^Chupin2021]:
    [M. Chupin, M.-S. Dupuy, G. Legendre and É. Séré,
    _Convergence analysis of adaptive DIIS algorithms with application to electronic ground state calculations_,
    ESAIM: M2AN **55**, 6, 2785-2825 (2021)](https://doi.org/10.1051/m2an/2021069)
