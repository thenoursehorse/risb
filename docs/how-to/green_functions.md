# Constructing Green's functions

This guide shows you how to construct Green's functions from a {{RISB}}
solution with the functions in {py:mod}`risb.helpers`. They return numpy
arrays in each block, indexed as `[k, frequency, orbital, orbital]` for the
lattice and `[frequency, orbital, orbital]` for a cluster.

:::{tip}
In `examples/green_functions.py` we provide an example.
:::

Below `S` is a solved `LatticeSolver` and the chemical potential $\mu$ is
`kweight.mu`. The frequencies $\omega$ are relative to $\mu$, e.g., Matsubara
frequencies or real frequencies with a broadening $\eta$

```python
import numpy as np

iw = 1j * (2 * np.arange(n_iw) + 1) * np.pi / beta
w = np.linspace(-10, 10, 2001) + 1j * eta
```

The functions take $\mathcal{R}$ and $\lambda$ in the blocks of the lattice,
so they work with projectors, several clusters and ghost quasiparticles.
Without projectors or ghosts these are

```python
from risb import helpers

R_full, Lambda_full = helpers.upfold_r_lambda(S.R, S.Lambda, S.h0_kin_k)
```

With projectors, pass them as the fourth argument, and with ghosts pass the
projectors onto the quasiparticles from
{py:func}`~risb.helpers.get_ghost_projectors` instead. `S.R_full` and
`S.Lambda_full` are the same, but from the start of the last cycle of the
solver, one step behind `S.R` and `S.Lambda`.

## Non-interacting Green's function

The non-interacting Green's function
$G_0(k, \omega) = \left[ \omega + \mu - H_0(k) \right]^{-1}$ is

```python
G0_k_iw = helpers.get_g0_k_w(iw, h0_k, mu)
```

## Self-energy

The local self-energy of block `bl` of cluster `i` is

```python
Sigma_iw = helpers.get_sigma_w(
    iw, S.R[i][bl], S.Lambda[i][bl], mu, S.h0_loc_matrix[i][bl]
)
```

Without ghosts it is

$$
\Sigma(\omega) = (\mathbb{1} - Z^{-1}) (\omega + \mu)
+ \mathcal{R}^{-1} \lambda \mathcal{R}^{-\dagger} - H_0^{\mathrm{loc}},
$$

where $Z = \mathcal{R}^{\dagger} \mathcal{R}$. With ghosts it has poles, see
[the self-energy and the quasiparticle weight](../explanations/self_energy.md).

## Interacting Green's function

The interacting Green's function of {{RISB}} is
$G(k, \omega) = \mathcal{R}^{\dagger} G^{\mathrm{qp}}(k, \omega)
\mathcal{R}$, with the Green's function of the quasiparticles
$G^{\mathrm{qp}}(k, \omega) = \left[ \omega + \mu - H^{\mathrm{qp}}(k)
\right]^{-1}$ and $H^{\mathrm{qp}}(k) = \mathcal{R} H_0^{\mathrm{kin}}(k)
\mathcal{R}^{\dagger} + \lambda$,

```python
G_k_iw = helpers.get_g_k_w(iw, R_full, Lambda_full, S.h0_kin_k, mu)
G_qp_k_iw = helpers.get_g_qp_k_w(iw, R_full, Lambda_full, S.h0_kin_k, mu)
```

It is also the Green's function from Dyson's equation
$G(k, \omega) = \left[ G_0^{-1}(k, \omega) - \Sigma(k, \omega) \right]^{-1}$,
with the self-energy of each cluster upfolded with
{py:func}`~risb.helpers.upfold`. For one cluster without projectors

```python
G_k_iw = {
    bl: np.linalg.inv(np.linalg.inv(G0_k_iw[bl]) - Sigma_iw[bl][None]) for bl in G0_k_iw
}
```

Without ghosts the interacting Green's function from {{RISB}} only describes
the quasiparticles: its spectral weight is $Z = \mathcal{R}^{\dagger}
\mathcal{R}$ instead of 1, because the incoherent part (e.g., the Hubbard
bands) is not described at the saddle point.

## Local Green's function

The local Green's function of each cluster,
$G_i^{\mathrm{loc}}(\omega) = \frac{1}{\mathcal{N}} \sum_k P_i(k) G(k, \omega)
P_i^{\dagger}(k)$, is

```python
G_iw_loc = helpers.get_g_w_loc(G_k_iw, projectors)
```

as a list with an entry for each cluster. Without projectors there is one
entry, the average over $k$ of the whole unit cell.

## Green's functions of TRIQS

The arrays are easily converted to the Green's function objects of the
{{TRIQS}} library, e.g.,

```python
from triqs.gfs import BlockGf, MeshImFreq

mesh = MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw)
Sigma = BlockGf(mesh=mesh, gf_struct=gf_struct)
for bl, gf in Sigma:
    w = np.fromiter(gf.mesh.values(), dtype=complex)
    gf.data[...] = helpers.get_sigma_w(
        w, S.R[0][bl], S.Lambda[0][bl], mu, S.h0_loc_matrix[0][bl]
    )
```
