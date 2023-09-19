# risb - Rotationally invariant slave bosons

<!-- INDEX-START -->

Rotationally invariant slave-bosons (RISB) is a non-perturbative method to 
approximately solve many-body fermionic problems. 

Source at [https://github.com/thenoursehorse/risb](https://github.com/thenoursehorse/risb)

<!-- INDEX-END -->

Documentation at [https://thenoursehorse.github.io/risb](https://thenoursehorse.github.io/risb)

<!-- INSTALL-START -->

## Dependencies

* [TRIQS](https://triqs.github.io/) `v3.2.x` if using :class:`.EmbeddingAtomDiag`
and for the tests
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)

## Installation

(Optional) Create a 
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

Clone source

```shell
git clone https://github.com/thenoursehorse/risb
cd risb
```

### With [pipx](https://pypa.github.io/pipx/)

```shell
pipx install .
```

To develop code without reinstalling

```shell
pipx install --editable .
```

### With pip

Install from local (-e allows to develop code without reinstalling, omit if
not editing the source code)

```shell
python3 -m pip install -e ./
```

### Uninstall

```
python3 -m pip uninstall risb
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

Tests are run with

```shell
python3 -m pytest
```

## Documentation

Install the prerequisites

```shell
python3 -m pip install -e .[docs]
```

Build the API

```shell
sphinx-apidoc -o docs/api --module-first --no-toc --force --separate src/risb
```

Serve

```shell
sphinx-autobuild -b html docs docs/_build
```

Access through a browser at `http://127.0.0.1:8000`.

<!-- INSTALL-END -->

## Examples

See `test_one_band_cubic.py` and `test_two_band_cubic_bilayer.py` in 
the `tests` folder for examples.

## To do

* Add EmbeddingAtomDiag tests
* Add kweights tests
* Sort out basic_functions tests
* Make h0_k all h0 hoppings, and work out h0_loc and add to h_int to make a 
h_loc = h0_loc + h_int, and then h0_kin_k = h0_k - h_0_loc
* sort out projectors/multiple clusters
* as always, very sensitive to initial R, Lambda guess, make more robust
* Use static type hints for functions (learning work in progress)