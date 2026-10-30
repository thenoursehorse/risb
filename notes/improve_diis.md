# Improving DIIS

Three guards that `DIIS` lacks. Each deals with a different way a single DIIS
step goes wrong. See also `preconditioning.md`.

## 1. Damping (β < 1)

Write `ḡ = Σ cᵢ g(xᵢ)` for what DIIS proposes now, and `x̄ = Σ cᵢ xᵢ` for the
same combination of past inputs. Damping means

```
x_next = x̄ + β (ḡ − x̄)        # = (1−β) x̄ + β ḡ
```

- β = 1 is the current code: jump all the way to `ḡ`.
- β = 0.3 moves only 30% of the way toward `ḡ`.

In directions the history has not seen yet, DIIS has no information and just
trusts `g`. If `g` overshoots in some direction (e.g., Λ reacts strongly),
β = 1 overshoots too, and the next cycle swings back the other way. Damping
trades speed for stability. This β is what QE calls `mixing_beta` and VASP
calls `AMIX`.

It matters most at the start and right after any history reset. With one
vector in the history, `c = [1]`, so the step is just `x → g(x)` with no
damping.

## 2. Backing off when the residual grows

At the moment every proposed step is accepted. Backing off means checking the
result first:

```
x_try = DIIS proposal
run cycle at x_try → error_try
if error_try > 2 × error_prev:            # step made things clearly worse
    x_try = x_prev + 0.5 (x_try − x_prev)  # halve the step, run again
    (repeat a few times)
```

This is a crude line search. Each retry costs an extra cycle, i.e., another
impurity solve. A milder version does not redo the cycle: lower β for later
steps whenever the error jumps, and raise it again when things go well.
DFTK's adaptive damping is a more careful version of this.

What we have now:

- `FixedPointLoop` remembers `best_x`.
- `OnPlateau` acts only after 20 cycles without progress, and it is not on by
  default in `LatticeSolver`.

Nothing reacts to a single bad step.

## 3. Restarting when the history is ill-conditioned

DIIS solves a small linear system for `cᵢ` from the matrix `B[i,j] = eᵢ·eⱼ`.
When the stored error vectors become almost linearly dependent, B is nearly
singular. That happens during stagnation, when the error goes back and forth
between similar states, or with noisy `g`. Then the `cᵢ` blow up with
alternating signs, for example

```
c = [+812, −1630, +819]      # sums to 1, but huge
```

The extrapolation `Σ cᵢ g(xᵢ)` then lands far from every point actually
computed, and it amplifies any noise in `g` by about 1000×. It shows up as a
sudden big jump in Λ.

`DIIS.extrapolate` uses `pinv(B)`. Its default cutoff only discards singular
values near machine precision, so in practice it gives almost no protection.

Guards:

- Check `max |cᵢ|` or the condition number of B each step.
- If it is too large, drop the oldest history vectors until B is
  well-conditioned again, or clear the history completely (like `OnPlateau`'s
  `reset()`, but triggered by conditioning rather than a plateau).
- The literature calls this filtering, e.g., Pollock & Rebholz, or Walker & Ni
  on QR-based Anderson.

Damping keeps each step small, backing off undoes a step that made things
worse, and the conditioning check stops DIIS from extrapolating wildly.
