# The self-energy and the quasiparticle weight

In {{RISB}} the Green's function of the electrons is that of the
quasiparticles, renormalized by $\mathcal{R}$,

$$
\mathbf{G}(k, z) = \mathcal{R}^{\dagger}
\left[ z - \mathbf{H}^{\mathrm{qp}}(k) \right]^{-1} \mathcal{R},
\qquad
\mathbf{H}^{\mathrm{qp}}(k) = \mathcal{R} \mathbf{H}_0^{\mathrm{kin}}(k)
\mathcal{R}^{\dagger} + \lambda,
$$

with $z = \omega + \mu$. With $M$ quasiparticles and $\nu$ orbitals in a
cluster, $\mathcal{R}$ is $M \times \nu$. Without ghosts $M = \nu$ and
$\mathcal{R}$ is usually invertible, but with ghosts $M > \nu$ and it is not.
Below is one cluster; the last section is about several.

## The self-energy

Write $\boldsymbol{\varepsilon} = \mathbf{H}_0^{\mathrm{kin}}(k)$ and
$\mathbf{g}(z) = \mathcal{R}^{\dagger} (z - \lambda)^{-1} \mathcal{R}$. The
Woodbury identity for $(z - \lambda - \mathcal{R} \boldsymbol{\varepsilon}
\mathcal{R}^{\dagger})^{-1}$ gives

$$
\mathbf{G} = \mathbf{g} + \mathbf{g}
(\boldsymbol{\varepsilon}^{-1} - \mathbf{g})^{-1} \mathbf{g}
= \mathbf{g} (\boldsymbol{\varepsilon}^{-1} - \mathbf{g})^{-1}
\boldsymbol{\varepsilon}^{-1},
\qquad
\mathbf{G}^{-1} = \mathbf{g}^{-1} - \boldsymbol{\varepsilon},
$$

which holds by continuity when $\boldsymbol{\varepsilon}$ is not invertible.
Comparing with $\mathbf{G}^{-1} = z - \boldsymbol{\varepsilon} -
\mathbf{H}_0^{\mathrm{loc}} - \Sigma(z)$, the self-energy is the same at every
$k$,

$$
\Sigma(z) = z - \mathbf{H}_0^{\mathrm{loc}} - \left[ \mathcal{R}^{\dagger}
(z - \lambda)^{-1} \mathcal{R} \right]^{-1},
$$

which is Eq. 11 of Ref. [^Makaresz2026]. Evaluated this way it is singular
wherever $z$ is an eigenvalue of $\lambda$, although $\Sigma$ need not be.

## The poles of the self-energy

Split the quasiparticles into the $\nu$ that $\mathcal{R}$ reaches, an
orthonormal basis $\mathbf{V}_p$ of the columns of $\mathcal{R}$, and the
$M - \nu$ ghosts that it does not, an orthonormal basis $\mathbf{V}_g$. Then
$\mathcal{R} = \mathbf{V}_p \mathcal{R}_0$ with $\mathcal{R}_0 =
\mathbf{V}_p^{\dagger} \mathcal{R}$ invertible, and in this basis

$$
z - \lambda = \begin{pmatrix} z - \lambda_0 & -\lambda_1 \\
-\lambda_1^{\dagger} & z - \lambda_2 \end{pmatrix},
\qquad
\lambda_0 = \mathbf{V}_p^{\dagger} \lambda \mathbf{V}_p, \quad
\lambda_1 = \mathbf{V}_p^{\dagger} \lambda \mathbf{V}_g, \quad
\lambda_2 = \mathbf{V}_g^{\dagger} \lambda \mathbf{V}_g.
$$

Only its block on $\mathbf{V}_p$ enters $\mathbf{g}$, and by block inversion
that block of $(z - \lambda)^{-1}$ is the inverse of the Schur complement
$z - \lambda_0 - \lambda_1 (z - \lambda_2)^{-1} \lambda_1^{\dagger}$. So

$$
\mathbf{g}^{-1} = \mathcal{R}_0^{-1} \left[ z - \lambda_0 - \lambda_1
(z - \lambda_2)^{-1} \lambda_1^{\dagger} \right] \mathcal{R}_0^{-\dagger},
$$

and with $\mathbf{S} = \mathcal{R}^{\dagger} \mathcal{R} =
\mathcal{R}_0^{\dagger} \mathcal{R}_0$,

$$
\Sigma(z) = (1 - \mathbf{S}^{-1}) z
+ \mathcal{R}_0^{-1} \lambda_0 \mathcal{R}_0^{-\dagger}
+ \mathcal{R}_0^{-1} \lambda_1 (z - \lambda_2)^{-1} \lambda_1^{\dagger}
\mathcal{R}_0^{-\dagger} - \mathbf{H}_0^{\mathrm{loc}},
$$

which is Eq. B10 of Ref. [^Makaresz2026]. Since $\mathcal{R}_0^{-1}
\mathbf{V}_p^{\dagger} = \mathbf{S}^{-1} \mathcal{R}^{\dagger}$, it does not
depend on the choice of bases. Diagonalize $\lambda_2 = \mathbf{W}
\operatorname{diag}(e_j) \mathbf{W}^{\dagger}$, and let the columns of
$\mathbf{B} = \mathcal{R}^{\dagger} \lambda \mathbf{V}_g \mathbf{W}$ be the
residue vectors of the poles $e_j$. Then

$$
\Sigma(z) = (1 - \mathbf{S}^{-1}) z
+ \mathbf{S}^{-1} \mathcal{R}^{\dagger} \lambda \mathcal{R} \mathbf{S}^{-1}
+ \mathbf{S}^{-1} \mathbf{B}
\operatorname{diag}\left( \frac{1}{z - e_j} \right)
\mathbf{B}^{\dagger} \mathbf{S}^{-1}
- \mathbf{H}_0^{\mathrm{loc}},
$$

which is Eq. B7 of Ref. [^Makaresz2026] with
$\mathbf{V}_g \mathbf{V}_g^{\dagger} = Q$. There the step from Eq. B2 to B3
drops a sign, which Eqs. B3 to B6 and the second term of B7 carry. It is
only singular at the $M - \nu$ poles $e_j$ of the ghosts. Without
ghosts there are no poles and $\Sigma(z) = (1 - \mathbf{Z}^{-1}) z +
\mathcal{R}^{-1} \lambda \mathcal{R}^{-\dagger} - \mathbf{H}_0^{\mathrm{loc}}$
with $\mathbf{Z} = \mathcal{R}^{\dagger} \mathcal{R}$, as in Eq. 51 of the
supplemental of Ref. [^Lanata2017].
{py:func}`risb.helpers.get_sigma_w` uses this form.

## The quasiparticle weight

The quasiparticle weight is
$\mathbf{Z} = [1 - \partial_{\omega} \Sigma(\omega)|_{\omega = 0}]^{-1}$, with
the derivative at $z = \mu$. Only the first and the last term of $\Sigma$
depend on $z$, so

$$
1 - \partial_{\omega} \Sigma = \mathbf{S}^{-1}
+ \mathbf{S}^{-1} \mathbf{B} \mathbf{D} \mathbf{B}^{\dagger} \mathbf{S}^{-1}
= \mathbf{S}^{-1} (\mathbf{S} + \mathbf{B} \mathbf{D} \mathbf{B}^{\dagger})
\mathbf{S}^{-1},
\qquad
\mathbf{D} = \operatorname{diag}\left( \frac{1}{(\mu - e_j)^2} \right),
$$

and $\mathbf{Z} = \mathbf{S} (\mathbf{S} + \mathbf{B} \mathbf{D}
\mathbf{B}^{\dagger})^{-1} \mathbf{S}$. The Woodbury identity for the middle
inverse,
$(\mathbf{S} + \mathbf{B} \mathbf{D} \mathbf{B}^{\dagger})^{-1} =
\mathbf{S}^{-1} - \mathbf{S}^{-1} \mathbf{B} (\mathbf{D}^{-1} +
\mathbf{B}^{\dagger} \mathbf{S}^{-1} \mathbf{B})^{-1} \mathbf{B}^{\dagger}
\mathbf{S}^{-1}$, gives

$$
\mathbf{Z} = \mathbf{S} - \mathbf{B} \left[ \operatorname{diag}
\left( (\mu - e_j)^2 \right) + \mathbf{B}^{\dagger} \mathbf{S}^{-1}
\mathbf{B} \right]^{-1} \mathbf{B}^{\dagger}.
$$

It stays finite when a pole is at $\mu$, where $\mathbf{D}$ is not: in a Mott
insulator a pole with weight is at $\mu$ and $\mathbf{Z} = 0$ in its
direction. Without ghosts $\mathbf{B}$ has no columns and $\mathbf{Z} =
\mathbf{S} = \mathcal{R}^{\dagger} \mathcal{R}$. This is
{py:func}`risb.helpers.get_z`.

With ghosts $\mathcal{R}^{\dagger} \mathcal{R}$ is not the quasiparticle
weight. It is the spectral weight of $\mathbf{G}$ summed over all frequencies
(Eq. 49 of the supplemental of Ref. [^Lanata2017]), which is 1 if
$\mathcal{R}$ is an isometry.

## Several clusters

With several clusters, projectors that depend on $k$, or orbitals in no
cluster, $\mathcal{R}(k)$ and $\lambda(k)$ are upfolded, and they do not mix
the clusters in the basis of their orbitals. So $\mathbf{g}$ is block
diagonal, and the self-energy of the lattice is the upfolded self-energy
$\Sigma_i$ of each cluster,

$$
\Sigma(k, z) = \sum_i \mathbf{P}^{\dagger}_i(k) \Sigma_i(z)
\mathbf{P}^{}_i(k),
$$

which is zero on the orbitals in no cluster, where $\mathcal{R} = 1$ and
$\lambda = 0$. The self-energy and quasiparticle weight above are those of
each cluster, a matrix on its orbitals. E.g., for a cluster of two sites the
eigenvalues of $\mathbf{Z}_i$ are the weights of its bonding and antibonding
orbitals, and a quasiparticle weight at each $k$ of the original lattice
needs a choice of how to periodize.

[^Makaresz2026]:
    [E. M. Makaresz, O. Gingras, T.-H. Lee, N. Lanatà, B. J. Powell, and
    H. L. Nourse, _Accelerating dynamical mean-field theory convergence by
    preconditioning with computationally cheaper quantum embedding methods_,
    arXiv:2601.16401 (2026)](https://arxiv.org/abs/2601.16401)

[^Lanata2017]:
    [N. Lanatà, T.-H. Lee, Y.-X. Yao, and V. Dobrosavljević, _Emergent Bloch
    excitations in Mott matter_, Phys. Rev. B **96**, 195126
    (2017)](https://doi.org/10.1103/PhysRevB.96.195126)
