# Using `SmearingKWeight`

This guide shows how to customize the options for the smearing $k$-space
integration methods.

## Electron filling

If you want the chemical potential $\mu$ fixed to a specific value

```python
from risb.kweight import SmearingKWeight

# Width of smearing method
beta = ...

# Fixed chemical potential
mu = ...

kweight = SmearingKWeight(beta = beta, mu = mu)
```

If you want the electon filling per unit cell $n$ fixed to a specific value

```python
from risb.kweight import SmearingKWeight

# Width of smearing method
beta = ...

# Fixed filling
n_target = ...

kweight = SmearingKWeight(beta = beta, n_target = n_target)
```

The fixed electron filling method uses :py:func:`scipy.optimize.brentq` to
find a $\mu$ that gives the correct $n$.

## Getting integration weights

```python
# A dict[list] of energies
energies = ...

weight = kweight.update_weights(energies = energies)
```

The method sets the appropriate $\mu$ and returns the integration weight
at each $k$-point as a `dict[list]`, in the same structure as `energies`.

## Smearing method

There are three smearing functions implemented. Below $\xi$ is an energy and
$\mu$ is the chemical potential.

```python
kweight = SmearingKWeight(...,
                          method = 'fermi' or 'gaussian' or 'methfessel-paxton',
)
```

The `fermi` method is the weighting function (Fermi-Dirac)

$$
f(\xi) = \frac{1}{\exp \left( \beta (\xi - \mu) \right) + 1 }.
$$

The `gaussian` method is the weighting function

$$
f(\xi) = \frac{1}{2} \left[ 1 - \mathrm{erf} (\beta (\xi - \mu)) \right],
$$

where $\mathrm{erf}$ is the error function, and $f(\xi)$ is the
integral of the Gaussian

$$
\delta(\xi) = \frac{\beta}{\sqrt{\pi}} \exp \left[ - (\beta (\xi - \mu))^2 \right].
$$

The `methfessel-paxton` method is the $N=1$ one outlined in
Ref. [^Methfessel1989]. You should cite this reference if you use it.

## Using own smearing function

If you want to change the smearing function

```python
from risb.kweight import SmearingKWeight

def my_smearing_function(energies : numpy.ndarray,
                         beta : float,
                         mu : float,
) -> ndarray:
...
    return an array of weights

kweight = SmearingKWeight(beta = ..., mu or n_target = ...)

kweight.smear_function = my_smearing_function
```

[^Methfessel1989]:
    [M. Methfessel and A. T. Paxton,
    _High-precision sampling for Brillouin-zone integration in metals_,
    Phys. Rev. B **40**, 3616 (1989)](https://doi.org/10.1103/PhysRevB.40.3616)
