# risb - Rotationally invariant slave boson mean-field theory

Dependencies
-------------
* [TRIQS](https://github.com/TRIQS/triqs) v3.2.x if using `EmbeddingAtomDiag`
and for the tests.
* numpy
* scipy

Installation
---------------

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

To uninstall
```
python3 -m pip uninstall risb
```

Tests
---------------
Tests are run with
```
python3 -m pytest
```

Examples
---------------
See test_one_band_cubic.py and test_two_band_cubic_bilayer.py in 
test/python for examples.

Embedding solvers
---------------
* Sparse exact diagaonalization ([embedding_ed](https://github.com/thenoursehorse/embedding_ed))

k-space integrators
---------------
* Linear tetrahedron ([k-int](https://github.com/thenoursehorse/kint))

To do
-------------
* Fix docs to work with newer TRIQS.
* Add kweights tests
* Sort out basic_functions tests
* Fix passing emb_parameters and kweight_parameters to stuff