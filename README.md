# risb - Rotationally invariant slave boson mean-field theory

Dependencies
-------------
* [TRIQS](https://github.com/TRIQS/triqs) v3.2.x
* Impurity solver for RISB embedding problem ([embedding_ed](https://github.com/thenoursehorse/embedding_ed))
* k-space integrator ([k-int](https://github.com/thenoursehorse/kint))

Installation
---------------
* The build and install process is the same as the one outlined [here](https://triqs.github.io/app4triqs/unstable/install.html). Also see the Dockerfile.

Examples
---------------
* Look at ed_one_band_cubic.py and one_band_cubic.py in test/python for examples.

To do
-------------
* Finish porting to TRIQS 3.2.x.
* Tidy up classes to work with them more easily.
* I think it is possible to remove all c++ dependency?
* Fix docs to work with newer TRIQS.