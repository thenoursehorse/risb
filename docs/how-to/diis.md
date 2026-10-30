# Using `DIIS`

This guide shows you how to use and customize {py:class}`~risb.optimize.DIIS`. This is an
implementation of algorithms from Ref. [^Chupin2021].

## Truncating history

Like other quasi-Newton methods, {{DIIS}} uses the history of previous
guesses for the vector $x$ that minimizes the loss/root function $f(x)$. If
your system requires many iteraions to find a solution this history can become
computationally expensive. Sometimes the history also includes old and bad
guesses that have too much influence on new gusses for $x$.

To specify how big the history should be

```python
from risb.optimize import DIIS

optimize = DIIS(history_size=...)
```

If you want to reset the entire history after $n$ iterations

```python
optimize = DIIS(n_restart=...)
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
optimize = DIIS(alpha=...)
```

## Linear mixing step

Our implementation takes a single linear mixing step, with the same $\alpha$,
every `n_period` iterations.

To change how frequently a linear mixing step is taken, or to never take one
with `None`

```python
optimize = DIIS(n_period=...)
```

## Only linear mixing

If you just want to use linear mixing with nothing special

```python
from risb.optimize import LinearMixing

optimize = LinearMixing(alpha=...)
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
optimize = DIIS(error_fun="recursion")
```

## Using with `LatticeSolver`

:::{seealso}
[Using `LatticeSolver`](lattice_solver.md#choosing-the-optimizer).
:::

If you want to use your customized {{DIIS}} solver

```python
S.solve(optimizer=DIIS(alpha=...), tol=..., maxiter=...)
```

[^Chupin2021]:
    [M. Chupin, M.-S. Dupuy, G. Legendre and É. Séré,
    _Convergence analysis of adaptive DIIS algorithms with application to electronic ground state calculations_,
    ESAIM: M2AN **55**, 6, 2785-2825 (2021)](https://doi.org/10.1051/m2an/2021069)
