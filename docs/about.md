# About

## Authors

Maintained by

[@thenoursehorse](https://github.com/thenoursehorse) H. L. Nourse (Okinawa Institute of Science and Technology)

Originally authored by

H. L. Nourse

with advisors

[B. J. Powell](https://people.smp.uq.edu.au/BenPowell/) (The University of Queensland)

[R. H. McKenzie](https://condensedconcepts.blogspot.com/) (The University of Queensland)

for the PhD thesis

[_Strongly correlated electrons on the decorated honeycomb lattice studied with
rotationally invariant slave-boson mean-field theory_ (2020)](https://doi.org/10.14264/uql.2020.169)

:::{include} ../README.md
:start-after: <!-- CITATION-START -->
:end-before: <!-- CITATION-END -->
:::

You may also find the following papers relevant:

[H. L. Nourse, Ross H. McKenzie, and B. J. Powell Phys. Rev. B **104**, 075104 (2021)](https://doi.org/10.1103/PhysRevB.104.075104)

[H. L. Nourse, Ross H. McKenzie, and B. J. Powell Phys. Rev. B **105**, 205119 (2022)](https://doi.org/10.1103/PhysRevB.105.205119)

% :download:`BibTeX file of citations <risb.bib>`

## Literature of original theory

Slave-bosons as introduced by Kotliar and Ruckenstein[^Kotliar1986] was
extended to the {{RISB}} formalism by Lechermann,
et al.[^Lechermann2007]. The implementation in this project uses the
embedding construction as introduced by
Lanatà, et al.[^Lanata2015] [^Lanata2017]. The above papers should also be
appropriately cited.

## Embedding solvers

% Fix class links

1. `EmbeddingAtomDiag` uses the [TRIQS](https://triqs.github.io/)[^Parcolett2015] library.

%1. `EmbeddingED` uses [TRIQS](https://triqs.github.io/)[^Parcolett2015] and
%[arpack-ng](https://github.com/opencollab/arpack-ng) with the
%[ezARPACK](https://krivenko.github.io/ezARPACK/) wrapper[^Krivenko2022].

## License

[GNU General Public License, version 3](http://www.gnu.org/licenses/gpl.html)

[^Kotliar1986]:
    [G. Kotliar and A. E. Ruckenstein,
    _New Functional Integral Approach to Strongly Correlated Fermi Systems:
    The Gutzwiller Approximation as a Saddle Point_,
    Phys. Rev. Lett. **57**, 1362 (1986)](https://doi.org/10.1103/PhysRevLett.57.1362)

[^Lechermann2007]:
    [F. Lechermann, A. Georges, G. Kotliar, and O. Parcollet,
    _Rotationally invariant slave-boson formalism and momentum dependence of the
    quasiparticle weight_,
    Phys. Rev.B **76**, 155102 (2007)](https://doi.org/10.1103/PhysRevB.76.155102)

[^Lanata2015]:
    [N. Lanatà, Y.-X. Yao, C.-Z. Wang, K.-M. Ho, and G. Kotliar,
    _Phase Diagram and Electronic Structure of Praseodymium and Plutonium_,
    Phys. Rev. X **5**, 011008 (2015)](https://doi.org/10.1103/PhysRevX.5.011008)

[^Lanata2017]:
    [Lanatà, Y.-X. Yao, X. Deng, V. Dobrosavljević, and G. Kotliar,
    _Slave Boson Theory of Orbital Differentiation with Crystal Field Effects:
    Application to UO<sub>2</sub>_,
    Phys. Rev. Lett. **118**, 126401 (2017)](https://doi.org/10.1103/PhysRevLett.118.126401)

[^Parcolett2015]:
    [O. Parcollet, M. Ferrero, T. Ayral, H. Hafermann, I. Krivenko,
    L. Messio, and P. Seth, _TRIQS: A toolbox for research on interacting quantum systems_,
    Comp. Phys. Comm. **196**, 398-415 (2015)](https://doi.org/10.1016/j.cpc.2015.04.023)

[^Krivenko2022]:
    [I. Krivenko, _ezARPACK - ezARPACK - a C++ ARPACK-NG wrapper
    compatible with multiple matrix/vector algebra libraries: Release 1.0_,
    10.5281/zenodo.3930203 (2022).](https://doi.org/10.5281/zenodo.3930202)
