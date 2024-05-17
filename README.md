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

The [TRIQS](https://triqs.github.io/triqs) library should also be cited if any of their library is used:

[O. Parcollet, M. Ferrero, T. Ayral, H. Hafermann, I. Krivenko, L. Messio, and P. Seth, Comp. Phys. Comm. 196, 398-415 (2015)](https://doi.org/10.1016/j.cpc.2015.04.023)

If the default root DIIS method is used in the `Solver` classes you should also cite:

[M. Chupin, M.-S. Dupuy, G. Legendre and É. Séré, ESAIM: M2AN **55**, 6, 2785-2825 (2021)](https://doi.org/10.1051/m2an/2021069)

<!-- CITATION-END -->

Lastly, the appropriate original theory outlined in the [documentation](https://thenoursehorse.github.io/risb/about.html#original-theory) should be cited.

<!-- INSTALL-START -->

## Dependencies

- [python](https://www.python.org/) version `> 3.10`
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [TRIQS](https://triqs.github.io/) version `3.2.x` if using :class:`EmbeddingAtomDiag`

## Installation

(Optional) Create a
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

Install

```shell
pip install risb
```

### Uninstall

```
pip uninstall risb
```

### Docker

There is a `Dockerfile` and `docker-compose.yml` inside the `docker` folder.
The `Dockerfile` will pull the
[TRIQS docker image](https://hub.docker.com/r/flatironinstitute/triqs)
from the hub and install risb. Using the image will be the same as outlined in
the [install instructions](https://triqs.github.io/triqs/latest/install.html#docker).
To connect to the [Jupyter](https://jupyter.org/) notebook it is

```shell
localhost:8888/?token=put/token/here
```

You can find the token by attaching a shell to the container
and running

```shell
jupyter server list
```

There is also a development `Dockerfile.dev` and the corresponding
`docker-compose-dev.yml` in order to have a container to develop code. It
installs [TRIQS](https://triqs.github.io/) from source, and works on
Apple M1/M2 (arm64, aarch64), and any amd64 system.

## Tests

The tests require a working [TRIQS](https://triqs.github.io/) installation.

Clone source

```shell
git clone https://github.com/thenoursehorse/risb
cd risb
```

Install the prerequisites

```shell
pip install -e .[test]
```

Tests are run with

```shell
pytest
```

## Documentation

Clone source

```shell
git clone https://github.com/thenoursehorse/risb
cd risb
```

Install the prerequisites

```shell
pip install -e .[docs]
```

Build the API

```shell
sphinx-apidoc -o docs/api --module-first --no-toc --force --separate src/risb
```

Build the documentation and set up a local server

```shell
sphinx-autobuild -b html docs docs/_build
```

Access through a browser at `http://127.0.0.1:8000`.

<!-- INSTALL-END -->
