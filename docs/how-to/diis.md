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

## Linear mixing step

Our implementation takes a single linear mixing step every `n_period`
iterations.

To change how frequently a linear mixing step is taken

```python
optimize = DIIS(n_period=...)
```

The size of the linear mixing step $\alpha$ is specified when creating the
solver

```python
optimize = DIIS(alpha=...)
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

The error is the change `g_x - x` by default. If you want it to be the
residual instead

```python
optimize = DIIS(error_fun="root")
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
