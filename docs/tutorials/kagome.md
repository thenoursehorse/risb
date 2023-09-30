# Hubbard model on kagome lattice

The tight-binding model on the kagome lattice can be written as 

$$
\hat{H}^0 = \sum_{k\sigma} 
\vec{c}_{k\sigma}^{\dagger}
\mathbf{H}_{\sigma}(k) 
\vec{c}_{k\sigma},
$$

where the second-quantized operators are written as vectors as

$$
\vec{c}^{\dagger}_{k\sigma} = 
\left( \hat{c}^{\dagger}_{kA\sigma}, 
\hat{c}^{\dagger}_{kB\sigma}, 
\hat{c}^{\dagger}_{kC\sigma} \right),
$$

where $\hat{c}^{\dagger}_{k\alpha\sigma}$ creates an electron on site 
$\alpha \in \{A,B,C\}$ with spin $\sigma$ within a unit cell, and the dispersion 
matrix is given by 

$$
\mathbf{H}_{\sigma}(k) = \begin{pmatrix}
0                & - 2 \cos(k_1/2)  & -2 \cos(k_2/2) \\
-2 t \cos(k_1/2) & 0                & -2 \cos(k_3/2) \\
-2 t \cos(k_2/2) & -2 t \cos(k_3/2) & 0,
\end{pmatrix} 
$$

where $t$ is the hopping amplitude. For lattice vectors in Cartesian 
coordinates given by $\vec{a}_1 = (1, 0)$ and $\vec{a}_2 = (1/2, \sqrt{3}/2)$,
the variables within the cosine functions are given by 
$k_1 = k_x$, 
$k_2 = k_x/2 + \sqrt{3} k_y / 2$ and 
$k_3 = - k_x / 2 + \sqrt{3} k_y / 2$.
