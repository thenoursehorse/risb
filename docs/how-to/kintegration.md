# K-space integration

Use equal weights for precomputed Hamiltonian arrays:

```python
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator

kintegrator = BandKIntegrator(
    WeightedKPointIntegrator(),
    beta=200,
    n_target=n_target,
)
S = LatticeSolver(h0_k, gf_struct, impurity_solver, kintegrator)
```

Supply mesh weights or another smearing function:

```python
from risb.kintegration import gaussian_function

kintegrator = BandKIntegrator(
    WeightedKPointIntegrator(kpoints, weights),
    smearing=gaussian_function,
    beta=200,
    mu=mu,
)
```

Extrapolate each cycle to T=0 using a supplied sequence of inverse temperatures:

```python
from risb.diagnostics import KIntegrationDiagnostics
from risb.kintegration import TemperatureExtrapolatedKIntegrator

with KIntegrationDiagnostics(level="debug", output="kintegration.out") as diagnostics:
    kintegrator = TemperatureExtrapolatedKIntegrator(
        kintegrator,
        betas=[20, 40, 80, 160, 320, 640],
        atol=1e-8,
        rtol=1e-6,
        diagnostics=diagnostics,
    )
    S = LatticeSolver(h0_k, gf_struct, impurity_solver, kintegrator)
    S.solve()
```
