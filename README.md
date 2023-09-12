# risb - Rotationally invariant slave boson mean-field theory

## Dependencies

* [TRIQS](https://github.com/TRIQS/triqs) v3.2.x if using `EmbeddingAtomDiag`
and for the tests.
* numpy
* scipy

## Installation

1. Update packaging softare
    ```
    python3 -m pip install --upgrade pip setuptools wheel
    ```

1. (Optional) Create a 
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

1. Clone source
    ```
    git clone https://github.com/thenoursehorse/risb
    ```

1. Install from local (-e allows to develop code without reinstalling, omit if
not necessary)
    ```
    cd risb
    python3 -m pip install -e ./
    ```

### Uninstall

```
python3 -m pip uninstall risb
```

### Docker

There is a Dockerfile and compose.yml inside ./docker/. The Dockerfile will 
pull the TRIQS [docker from the hub](https://hub.docker.com/r/flatironinstitute/triqs) 
and install risb. Using the Docker image will be the same as the instructions 
for TRIQS (e.g., Jupyter). To connect to the Jupyter notebook it is probably

```
http://localhost:8888/?token=3aa53ba98d29e7605ad59941937907aa610f27a35e334bb2
```

but if the token is wrong you can find it by attaching a shell to the container 
and running

```
jupyter server list
```

There is also a development Dockerfile.dev and the corresponding 
compose-dev.yml in order to have a container to develop code. It installs 
TRIQS from source, and works on Apple M1/M2 (arm64), and any amd64 system.

## Tests

Tests are run with
```
python3 -m pytest
```

## Examples

See test_one_band_cubic.py and test_two_band_cubic_bilayer.py in 
test/python for examples.

## Embedding solvers

* Sparse exact diagaonalization ([embedding_ed](https://github.com/thenoursehorse/embedding_ed))

## k-space integrators

* Linear tetrahedron ([k-int](https://github.com/thenoursehorse/kint))

## To do

* Fix docs to work with newer TRIQS.
* Add kweights tests
* Sort out basic_functions tests
* Fix passing emb_parameters and kweight_parameters to stuff