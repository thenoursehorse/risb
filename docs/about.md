# About

:::{tip}
Let's give readers a helpful hint!
:::

## Authors

Originally authored by 

H. L. Nourse (The University of Queensland)

during PhD with advisors

B. J. Powell (The University of Queensland), 

R. H. McKenzie (The University of Queensland)

Maintained by

[@thenoursehorse](https://github.com/thenoursehorse) H. L. Nourse (Okinawa Institute of Science and Technology)

Contributors

<ul class="list-style-none">
{% for contributor in site.github.contributors %}
  <li class="d-inline-block mr-1">
     <a href="{{ contributor.html_url }}"><img src="{{ contributor.avatar_url }}" width="32" height="32" alt="{{ contributor.login }}"/></a>
  </li>
{% endfor %}
</ul>

## Original theory

Slave-bosons as introduced by Kotliar and Ruckenstein [[1]](#ref1) was 
extended to the rotationally invariant (RISB) case by Lechermann, 
et al. [[2]](#ref2). The implementation in this project uses the 
embedding construction as introduced by 
Lanatà, et al. [[3]](#ref3)[[4]](#ref4). 

## Embedding solvers

1. `EmbeddingAtomDiag` uses [TRIQS](https://triqs.github.io/) [[5]](#ref5).

1. `EmbeddingED` uses [TRIQS](https://triqs.github.io/) [[5]](#ref5) and 
[arpack-ng](https://github.com/opencollab/arpack-ng) with the
[ezARPACK](https://krivenko.github.io/ezARPACK/) [[6]](#ref6) wrapper.

## License

[GNU General Public License, version 3](http://www.gnu.org/licenses/gpl.html)

## References

1\. <a id="ref1"> [G. Kotliar and A. E. Ruckenstein, 
*New Functional Integral Approach to Strongly Correlated Fermi Systems: 
The Gutzwiller Approximation as a Saddle Point*, 
Phys. Rev. Lett. **57**, 1362 (1986)](https://doi.org/10.1103/PhysRevLett.57.1362)

2\. <a id="ref2"> [F. Lechermann, A. Georges, G. Kotliar, and O. Parcollet, 
*Rotationally invariant slave-boson formalism and momentum dependence of the 
quasiparticle weight*, 
Phys. Rev.B **76**, 155102 (2007)](https://doi.org/10.1103/PhysRevB.76.155102)

3\. <a id="ref3"> [N. Lanatà, Y.-X. Yao, C.-Z. Wang, K.-M. Ho, and G. Kotliar, 
*Phase Diagram and Electronic Structure of Praseodymium and Plutonium*, 
Phys. Rev. X **5**, 011008 (2015)](https://doi.org/10.1103/PhysRevX.5.011008)

4\. <a id="ref4"> [Lanatà, Y.-X. Yao, X. Deng, V. Dobrosavljević, and G. Kotliar, 
*Slave Boson Theory of Orbital Differentiation with Crystal Field Effects: 
Application to UO<sub>2</sub>*, 
Phys. Rev. Lett. **118**, 126401 (2017)](https://doi.org/10.1103/PhysRevLett.118.126401)

5\. <a id="ref5"> [O. Parcollet, M. Ferrero, T. Ayral, H. Hafermann, I. Krivenko, 
L. Messio, and P. Seth, *TRIQS: A toolbox for research on interacting quantum systems*, 
Comp. Phys. Comm. **196**, 398-415 (2015)](https://doi.org/10.1016/j.cpc.2015.04.023)

6\. <a id="ref6"> [I. Krivenko, *ezARPACK - ezARPACK - a C++ ARPACK-NG wrapper 
compatible with multiple matrix/vector algebra libraries: Release 1.0*, 
10.5281/zenodo.3930203 (2022).](https://doi.org/10.5281/zenodo.3930202)