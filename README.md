# risb - Rotationally invariant slave bosons

Documentation at [https://thenoursehorse.github.io/risb](https://thenoursehorse.github.io/risb)

Copyright (C) 2016-2023 H. L. Nourse and B. J. Powell, 2016-2022 R. H. McKenzie

<!-- INDEX-START -->

## What is risb?

Tools to solve strongly correlated many-body electronic problems using
rotationally invariant slave-bosons (RISB), an auxilliary particle method.
RISB is like dynamical mean-field theory (DMFT), but solves problems in a
fraction of the time, with hopefully not a fraction of the accuracy.

## Where to start?

If you want to learn how to solve some common strongly correlated lattice
models, and how RISB is implemented, then start with the
[tutorials](https://thenoursehorse.github.io/risb/tutorials).

If you want to quickly see a calculation, then start with the `examples/`
folder in this repository and refer to the
[how-to guides](https://thenoursehorse.github.io/risb/how-to/).

<!-- INDEX-END -->

<!-- CITATION-START -->

## Citation

We kindly request that you cite the following paper if your project uses our code:

[H. L. Nourse, Ross H. McKenzie, and B. J. Powell Phys. Rev. B **103**, L081114 (2021)](https://doi.org/10.1103/PhysRevB.103.L081114)

If the default root DIIS method is used in the `Solver` classes you should also cite:

[M. Chupin, M.-S. Dupuy, G. Legendre and É. Séré, ESAIM: M2AN **55**, 6, 2785-2825 (2021)](https://doi.org/10.1051/m2an/2021069)

If ghost quasiparticles (`bath_size`) are used you should also cite:

[N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević, Phys. Rev. B **96**, 195126 (2017)](https://doi.org/10.1103/PhysRevB.96.195126)

[N. Lanatà, Phys. Rev. B **105**, 045111 (2022)](https://doi.org/10.1103/PhysRevB.105.045111)

<!-- CITATION-END -->

Lastly, the appropriate original theory outlined in the [documentation](https://thenoursehorse.github.io/risb/about.html#original-theory) should be cited.

<!-- INSTALL-START -->

## Dependencies

- [python](https://www.python.org/) version `>= 3.14`
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [fockit](https://github.com/thenoursehorse/fockit)

Optional

- [block2](https://block2.readthedocs.io) for `Block2ImpuritySolver` and
  `Block2ThermalImpuritySolver`, installed with `pip install "risb[block2]"`,
  or `uv add "risb[block2]"` in a [uv](https://docs.astral.sh/uv) project
- [mpi4py](https://mpi4py.readthedocs.io) to split the block2 solvers over MPI
  ranks, built against the MPI of the machine, installed with
  `pip install --no-binary mpi4py "risb[mpi]"`, or with uv
  `uv pip install --no-binary mpi4py "risb[mpi]"`

## Installation

(Optional) Create a
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

Install

```shell
pip install risb
```

or add it to a [uv](https://docs.astral.sh/uv) project

```shell
uv add risb
```

### Uninstall

```shell
pip uninstall risb
```

or

```shell
uv remove risb
```

## Tests

Clone source

```shell
git clone https://github.com/thenoursehorse/risb
cd risb
```

Install risb and the test prerequisites with [uv](https://docs.astral.sh/uv)

```shell
uv sync
```

Tests are run with

```shell
uv run pytest
```

Type checks are run with

```shell
uv run mypy
```

## Documentation

Clone source

```shell
git clone https://github.com/thenoursehorse/risb
cd risb
```

Install the documentation prerequisites

```shell
uv sync --group docs
```

Build the documentation and set up a local server

```shell
uv run sphinx-autobuild -b html docs docs/_build
```

Access through a browser at `http://127.0.0.1:8000`.

<!-- INSTALL-END -->
