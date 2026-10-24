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
Lanatà, et al.[^Lanata2015] [^Lanata2017]. Ghost quasiparticles were
introduced by Lanatà, et al.[^LanataGhost2017] and formulated as ghost
{{RISB}} by Lanatà[^Lanata2022]. The above papers should also be
appropriately cited.

## Impurity solvers

% Fix class links

1. `FockitImpuritySolver` uses the [fockit](https://github.com/thenoursehorse/fockit) library.
1. `Block2ImpuritySolver` uses the [block2](https://block2.readthedocs.io)
   library[^Zhai2023], and the mixing sweeps of Wolf, et al.[^Wolf2015].
1. `Block2ThermalImpuritySolver` uses the [block2](https://block2.readthedocs.io)
   library[^Zhai2023], and the purification of Feiguin and White[^Feiguin2005].

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

[^LanataGhost2017]:
    [N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević,
    _Emergent Bloch excitations in Mott matter_,
    Phys. Rev. B **96**, 195126 (2017)](https://doi.org/10.1103/PhysRevB.96.195126)

[^Lanata2022]:
    [N. Lanatà,
    _Operatorial formulation of the ghost rotationally invariant slave-boson
    theory_,
    Phys. Rev. B **105**, 045111 (2022)](https://doi.org/10.1103/PhysRevB.105.045111)

[^Zhai2023]:
    [H. Zhai, H. R. Larsson, S. Lee, Z.-H. Cui, T. Zhu, C. Sun, L. Peng, R. Peng,
    K. Liao, J. Tölle, J. Yang, S. Li, and G. K.-L. Chan,
    _Block2: A comprehensive open source framework to develop and apply
    state-of-the-art DMRG algorithms in electronic structure and beyond_,
    J. Chem. Phys. **159**, 234801 (2023)](https://doi.org/10.1063/5.0180424)

[^Wolf2015]:
    [F. A. Wolf, A. Go, I. P. McCulloch, A. J. Millis, and U. Schollwöck,
    _Imaginary-Time Matrix Product State Impurity Solver for Dynamical
    Mean-Field Theory_,
    Phys. Rev. X **5**, 041032 (2015)](https://doi.org/10.1103/PhysRevX.5.041032)

[^Feiguin2005]:
    [A. E. Feiguin and S. R. White,
    _Finite-temperature density matrix renormalization using an enlarged
    Hilbert space_,
    Phys. Rev. B **72**, 220401(R) (2005)](https://doi.org/10.1103/PhysRevB.72.220401)
