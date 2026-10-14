# Constructing Green's functions

This guide shows you how to construct Green's functions from a {{RISB}}
solution. The functions are in {py:mod}`risb.helpers_triqs` and use the
Green's function objects of the {{TRIQS}} library, which must be installed.

:::{tip}
In `examples/green_functions.py` we provide an example.
:::

## Meshes

The lattice Green's functions are on a product of a $k$-space mesh and a
frequency mesh

```python
from triqs.gfs import MeshImFreq, MeshProduct
from triqs.lattice.tight_binding import BravaisLattice, BrillouinZone, MeshBrZone

# The k-space mesh
k_mesh = MeshBrZone(BrillouinZone(BravaisLattice(units=...)), n_k=...)

# The Matsubara frequency mesh
iw_mesh = MeshImFreq(beta=..., statistic="Fermion", n_iw=...)

k_iw_mesh = MeshProduct(k_mesh, iw_mesh)
```

`h0_k` and `h0_kin_k` must be given at the $k$-points of `k_mesh`, in the same
order. If you want real frequencies instead

```python
from triqs.gfs import MeshReFreq

w_mesh = MeshReFreq(window=(..., ...), n_w=...)
```

Below `S` is a solved `LatticeSolver`, and the chemical potential $\mu$ is
`kweight.mu`.

## Non-interacting Green's function

The non-interacting Green's function

$$
G_0(k, \omega) = \left[ \omega + \mu - H_0(k) \right]^{-1}
$$

is constructed as

```python
from risb.helpers_triqs import get_g0_k_w

G0_k_iw = get_g0_k_w(gf_struct=gf_struct, mesh=k_iw_mesh, h0_k=h0_k, mu=mu)
```

## Self-energy

The local self-energy from {{RISB}} is

$$
\Sigma(\omega) = (\mathbb{1} - Z^{-1}) (\omega + \mu)
+ \mathcal{R}^{-1} \lambda \mathcal{R}^{-\dagger} - H_0^{\mathrm{loc}},
$$

where $Z = \mathcal{R}^{\dagger} \mathcal{R}$. It is constructed as

```python
from risb.helpers_triqs import get_sigma_w

Sigma_iw = get_sigma_w(
    mesh=iw_mesh,
    gf_struct=gf_struct,
    Lambda=S.Lambda[0],
    R=S.R[0],
    h0_loc=S.h0_loc_matrix[0],
    mu=mu,
)
```

## Interacting Green's function

The interacting Green's function from Dyson's equation

$$
G(k, \omega) = \left[ G_0^{-1}(k, \omega) - \Sigma(\omega) \right]^{-1}
$$

is constructed as

```python
from risb.helpers_triqs import get_g_k_w

G_k_iw = get_g_k_w(g0_k_w=G0_k_iw, sigma_w=Sigma_iw)
```

## Quasiparticle Green's function

The quasiparticle Green's function

$$
G^{\mathrm{qp}}(k, \omega) = \left[ \omega - H^{\mathrm{qp}}(k) \right]^{-1},
$$

where $H^{\mathrm{qp}}(k) = \mathcal{R} H_0^{\mathrm{kin}}(k)
\mathcal{R}^{\dagger} + \lambda - \mu$, is constructed as

```python
from risb.helpers_triqs import get_g_qp_k_w

G_qp_k_iw = get_g_qp_k_w(
    gf_struct=gf_struct,
    mesh=k_iw_mesh,
    h0_kin_k=S.h0_kin_k,
    Lambda=S.Lambda[0],
    R=S.R[0],
    mu=mu,
)
```

The interacting Green's function is also

$$
G(k, \omega) = \mathcal{R}^{\dagger} G^{\mathrm{qp}}(k, \omega) \mathcal{R},
$$

which is constructed as

```python
G_k_iw = get_g_k_w(g_qp_k_w=G_qp_k_iw, R=S.R[0])
```

## Local Green's function

The local Green's function

$$
G^{\mathrm{loc}}(\omega) = \frac{1}{\mathcal{N}} \sum_k G(k, \omega)
$$

is constructed from any of the lattice Green's functions above as

```python
from risb.helpers_triqs import get_g_w_loc

G_iw_loc = get_g_w_loc(G_k_iw)
```

The electron filling is `G_qp_iw_loc.total_density()` from the quasiparticle
Green's function. The interacting Green's function from {{RISB}} only has the
quasiparticle part, so its filling is reduced by $Z$.
