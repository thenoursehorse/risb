# risb - Rotationally invariant slave bosons

Rotationally invariant slave-bosons (RISB) is a non-perturbative method to 
approximately solve many-body fermionic problems. 

Documentation at [https://thenoursehorse.github.io/risb](https://thenoursehorse.github.io/risb)

Source at [https://github.com/thenoursehorse/risb](https://github.com/thenoursehorse/risb)

<!-- SPHINX-START -->

## Dependencies

* [TRIQS](https://github.com/TRIQS/triqs) v3.2.x if using `EmbeddingAtomDiag`
and for the tests
* numpy
* scipy

## Installation

Update packaging software

```
python3 -m pip install --upgrade pip setuptools wheel
```

Clone source

```
git clone https://github.com/thenoursehorse/risb
cd risb
```

### With pipx

```
pipx install .
```

To develop code without reinstalling

```
pipx install --editable .
```

### With pip

(Optional) Create a 
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

Install from local (-e allows to develop code without reinstalling, omit if
not editing the source code)

```
python3 -m pip install -e ./
```

### Uninstall

```
python3 -m pip uninstall risb
```

### Docker

There is a Dockerfile and docker-compose.yml inside ./docker/. The Dockerfile will 
pull the TRIQS [docker from the hub](https://hub.docker.com/r/flatironinstitute/triqs) 
and install risb. Using the Docker image will be the same as the instructions 
for TRIQS (e.g., Jupyter). To connect to the Jupyter notebook it is 

```
localhost:8888/?token=put/token/here
```

You can find the token by attaching a shell to the container 
and running

```
jupyter server list
```

There is also a development Dockerfile.dev and the corresponding 
docker-compose-dev.yml in order to have a container to develop code. It 
installs TRIQS from source, and works on Apple M1/M2 (arm64), and any amd64 
system.

## Tests

Tests are run with

```
python3 -m pytest -v
```

or

```
nox -s pytest -v
```

For developers, coverage can be determined with `pytest-cov` using

```
python3 -m pytest -ra --cov=risb --cov-branch
```

## Examples

See test_one_band_cubic.py and test_two_band_cubic_bilayer.py in 
test/python for examples.

## Embedding solvers

* Sparse exact diagaonalization ([embedding_ed](https://github.com/thenoursehorse/embedding_ed))

## k-space integrators

* Linear tetrahedron ([k-int](https://github.com/thenoursehorse/kint))

## To do

* Add EmbeddingAtomDiag tests
* Add kweights tests
* Sort out basic_functions tests
* Make h0_k all h0 hoppings, and work out h0_loc and add to h_int to make a 
h_loc = h0_loc + h_int, and then h0_kin_k = h0_k - h_0_loc.
* Use static type hints for functions (learning work in progress)