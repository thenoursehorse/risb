# K-space integration

Use equal weights for precomputed Hamiltonian arrays:

```python
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator

band_integrator = BandKIntegrator(
    WeightedKPointIntegrator(),
    beta=200,
    n_target=n_target,
)
S = LatticeSolver(h0_k, gf_struct, impurity_solver, band_integrator)
```

Supply mesh weights or another smearing function:

```python
from risb.kintegration import gaussian_function

band_integrator = BandKIntegrator(
    WeightedKPointIntegrator(kpoints, weights),
    smearing=gaussian_function,
    beta=200,
    mu=mu,
)
```

Extrapolate each cycle to T=0 using a supplied sequence of inverse temperatures:

```python
from risb.diagnostics import KIntegrationDiagnostics
from risb.kintegration import TemperatureExtrapolatedBandKIntegrator

with KIntegrationDiagnostics(level="debug", output="kintegration.out") as diagnostics:
    band_integrator = TemperatureExtrapolatedBandKIntegrator(
        band_integrator,
        betas=[20, 40, 80, 160, 320, 640],
        atol=1e-8,
        rtol=1e-6,
        diagnostics=diagnostics,
    )
    S = LatticeSolver(h0_k, gf_struct, impurity_solver, band_integrator)
    S.solve()
```
