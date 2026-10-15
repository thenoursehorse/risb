# Copyright (c) 2023 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

"""Solvers for rotationally invariant slave-boson mean-field theory on a lattice."""

import warnings
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Literal, TypeIs

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from risb import helpers
from risb.optimize import DIIS
from risb.types import (
    BlockMatrix,
    BlockMatrixLike,
    BlockRealArray,
    GfStruct,
    ImpuritySolver,
    RootFinder,
    Symmetrize,
    UpdateWeights,
)

# Part of R^dag R = 1 that the ghost copies of an orbital start with
_GHOST_WEIGHT = 0.18


# The structure and methods here have been modeled
# after github.com/TRIQS/hartree_fock
class LatticeSolver:
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with a local interaction on each cluster.

    Parameters
    ----------
    h0_k : risb.types.BlockMatrixLike
        Single-particle dispersion between local clusters. Each key
        represents a single-particle symmetry.
    gf_struct : risb.types.GfStruct or list of risb.types.GfStruct
        Structure of the matrices. For each cluster, it must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example for a single cluster: ``[ ('up', 3), ('down', 3) ]``.
    impurity_solvers : risb.types.ImpuritySolver or sequence of risb.types.ImpuritySolver
        The solver of the impurity problem for each cluster. It must
        already store the interacting Hamiltonian ``h_int`` of the physical
        orbitals on a cluster, and implement :class:`risb.types.ImpuritySolver`.
        See class :class:`.FockitImpuritySolver`.
    update_weights : risb.types.UpdateWeights
        The function that gives the integral weights at each k-point on the
        lattice. It is called as ``update_weights(energies)``,
        where the energies are a dictionary with each key a list.
        See class :class:`.SmearingKWeight`.
    root : risb.types.RootFinder, optional
        The function that drives the self-consistent procedure. It is called
        as ``root(fun, x0, **kwargs)``, where ``x0`` is the initial
        guess vector, and ``fun`` is the function to minimize,
        where ``fun = self._target_function``.
        Defaults to :meth:`.DIIS.solve` method of :class:`.DIIS`.
    projectors : sequence of risb.types.BlockMatrixLike, optional
        The projectors onto the subspace of each cluster. ``projectors[i][bl]``
        has a row for each orbital of block ``bl`` of cluster ``i`` and a
        column for each orbital of the block of `h0_k` it maps to, and can
        depend on k (indexed as k, row, column). At every k-point they must
        have orthonormal rows and not overlap between clusters, which is not
        checked unless `check_projectors` is True. Orbitals in no cluster keep
        their bands, with ``R = 1`` and ``Lambda = 0``.
    gf_struct_mapping : sequence of mapping, optional
        The mapping from the symmetry blocks of each cluster
        to the symmetry blocks of `h0_k`. Default assumes the keys in
        all clusters are the same as the keys in `h0_k`.
    symmetrizers : sequence of risb.types.Symmetrize, optional
        Functions that symmetrize the mean-field matrices. The argument of
        each function must take a list of all clusters.
        E.g., ``[R_cluster1, R_cluster2, ...]``.
    force_real : bool, optional
        True if the mean-field matrices are forced to be real
    error_fun : {'root', 'recursion'}, optional
        At each self-consistent cycle, whether the returned error function is
        the f1 and f2 root functions (``'root'``) or the difference between
        consecutive :attr:`Lambda` and :attr:`R` (``'recursion'``).
        Defaults to ``'recursion'``.
    return_x_new : bool, optional
        Whether to return a new guess for ``x`` and the ``error`` at each iteration or
        only the ``error``. :func:`scipy.optimize.root` should only use the ``error``.
    check_projectors : bool, optional
        Raise a ValueError if `projectors` do not have orthonormal rows or
        overlap. Default is False.
    double_counting : callable, optional
        A double-counting potential, called each cycle as
        ``double_counting(i, rho_c)`` with the density matrix
        :attr:`rho_c` of cluster ``i`` from the previous cycle (its
        quasiparticle density matrix :attr:`rho_qp` in the first cycle). It
        returns a matrix for each block of the cluster, which is subtracted
        from the local terms of cluster ``i`` in its impurity problem.
    bath_size : sequence of mapping, optional
        Ghost RISB (`10.1103/PhysRevB.96.195126
        <https://doi.org/10.1103/PhysRevB.96.195126>`__,
        `10.1103/PhysRevB.105.045111
        <https://doi.org/10.1103/PhysRevB.105.045111>`__): the number of
        quasiparticles of each block of each cluster, which is the number of
        bath orbitals of its impurity problem, e.g., ``[{"up": 3, "dn": 3}]``
        for three per orbital. It must be a multiple of the size of the block.
        The quasiparticle bands then hold :attr:`n_ghost_electrons` more
        electrons than the physical ones, which ``update_weights`` must include
        in its filling. Default is the size of each block, i.e., RISB without
        ghosts.

    """

    def __init__(
        self,
        h0_k: BlockMatrixLike,
        gf_struct: GfStruct | list[GfStruct],
        impurity_solvers: ImpuritySolver | Sequence[ImpuritySolver],
        update_weights: UpdateWeights,
        root: RootFinder | None = None,
        projectors: Sequence[BlockMatrixLike] | None = None,
        gf_struct_mapping: Sequence[Mapping[str, str]] | None = None,
        symmetrizers: Sequence[Symmetrize] = (),
        force_real: bool = False,
        error_fun: Literal["root", "recursion"] = "recursion",
        return_x_new: bool = True,
        check_projectors: bool = False,
        double_counting: Callable[[int, BlockMatrix], BlockMatrixLike] | None = None,
        *,
        bath_size: Sequence[Mapping[str, int]] | None = None,
    ) -> None:
        #: risb.types.BlockMatrixLike : Non-interacting Hamiltonian in k-space.
        self.h0_k = h0_k

        # FIXME is this the best way to make sure gf_struct is a list of gf_struct?
        self.gf_struct: list[GfStruct]
        if _is_single_gf_struct(gf_struct):
            self.gf_struct = [gf_struct]
        else:
            self.gf_struct = gf_struct

        #: int : Number of correlated clusters per supercell on the lattice.
        self.n_clusters = len(self.gf_struct)

        #: list of risb.types.ImpuritySolver : Solver of the impurity problem
        #: for each cluster.
        self.impurity_solvers: list[ImpuritySolver]
        if isinstance(impurity_solvers, Sequence):
            self.impurity_solvers = list(impurity_solvers)
        else:
            self.impurity_solvers = [impurity_solvers]
        if len(self.impurity_solvers) != self.n_clusters:
            msg = f"Need an impurity solver (got {len(self.impurity_solvers)}) for each cluster (got {self.n_clusters}) !"
            raise ValueError(msg)

        self._update_weights = update_weights

        if root is None:
            self.optimize = DIIS()
            root = self.optimize.solve
        self._root: RootFinder = root

        self.projectors = projectors
        if (self.projectors is not None) and (len(self.projectors) != self.n_clusters):
            msg = f"Need a projector (got {len(self.projectors)}) for each cluster (got {self.n_clusters}) !"
            raise ValueError(msg)

        self.gf_struct_mapping: Sequence[Mapping[str, str]]
        if gf_struct_mapping is None:
            self.gf_struct_mapping = [
                {bl: bl for bl in h0_k} for i in range(self.n_clusters)
            ]
        else:
            self.gf_struct_mapping = gf_struct_mapping
        if len(self.gf_struct_mapping) != self.n_clusters:
            msg = f"Need a gf_struct_mapping (got {len(self.gf_struct_mapping)}) for each cluster (got {self.n_clusters}) !"
            raise ValueError(msg)
        if check_projectors and self.projectors is not None:
            _check_projectors(h0_k, self.projectors, self.gf_struct_mapping)

        self.double_counting = double_counting

        self.symmetrizers = symmetrizers
        self.force_real = force_real
        if error_fun not in {"root", "recursion"}:
            msg = "Unrecognized error functions for root !"
            raise ValueError(msg)
        self.error_fun = error_fun
        self.return_x_new = return_x_new

        # The Hermitian basis of each size of square matrix
        self._H_basis: dict[int, list[NDArray[np.inexact]]] = {}

        #: list of dict : Number of quasiparticles of each block of each
        #: cluster, which is the number of bath orbitals of its impurity problem.
        self.bath_size: list[dict[str, int]] = [
            dict(gf_struct_i) for gf_struct_i in self.gf_struct
        ]
        if bath_size is not None:
            self.bath_size = [dict(sizes) for sizes in bath_size]
        for i in range(self.n_clusters):
            for bl, bl_size in self.gf_struct[i]:
                if self.bath_size[i][bl] % bl_size:
                    msg = f"bath_size {self.bath_size[i][bl]} of block {bl} of cluster {i} must be a multiple of its size {bl_size} !"
                    raise ValueError(msg)

        #: float : Number of electrons per unit cell that the quasiparticle
        #: bands hold beyond the physical ones, ``sum (bath_size - size) / 2``
        #: over the blocks of the clusters.
        self.n_ghost_electrons = sum(
            (self.bath_size[i][bl] - bl_size) / 2
            for i in range(self.n_clusters)
            for bl, bl_size in self.gf_struct[i]
        )

        # Projectors onto the quasiparticles of each cluster, the same as the
        # projectors without ghosts
        self._projectors_qp: Sequence[BlockMatrixLike] | None = self.projectors
        if self.n_ghost_electrons > 0:
            self._projectors_qp = helpers.get_ghost_projectors(
                h0_k, self.bath_size, self.projectors, self.gf_struct_mapping
            )

        #: risb.types.BlockMatrix : :attr:`h0_k` with the local couplings of
        #: each cluster removed, so with the local terms of the orbitals that
        #: are in no cluster.
        self.h0_kin_k = helpers.get_h0_kin_k(h0_k, projectors, gf_struct_mapping)

        #: list of risb.types.BlockMatrix : Matrices of non-interacting local hopping
        #: terms and energies in :attr:`h0_k`.
        self.h0_loc_matrix = self._get_h0_loc_matrix()

        # The largest energy of the bands of h0_kin_k from zero, an energy
        # scale of the quasiparticle bands
        self._kinetic_half_bandwidth = max(
            float(np.max(np.abs(np.linalg.eigvalsh(h0))))
            for h0 in self.h0_kin_k.values()
        )

        #: list of risb.types.BlockMatrix : Renormalization matrix
        #: from c- to f-electrons at the mean-field level for each cluster.
        self.R: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Correlation potential matrix of the quasiparticles
        #: for each cluster.
        self.Lambda: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        # R starts at 1 and Lambda at 0. With ghosts each orbital also has
        # B - 1 ghost copies, which start with part of the weight in R and with
        # levels spread over the kinetic bandwidth, because ghosts that start
        # with R = 0 stay decoupled
        dtype = float if self.force_real else complex
        for i in range(self.n_clusters):
            for bl, bl_size in self.gf_struct[i]:
                n_copies = self.bath_size[i][bl] // bl_size - 1
                r, levels = [1.0], [0.0]
                if n_copies:
                    r = [np.sqrt(1 - _GHOST_WEIGHT)]
                    r += [np.sqrt(_GHOST_WEIGHT / n_copies)] * n_copies
                    levels += list(
                        np.linspace(-1, 1, n_copies) * self._kinetic_half_bandwidth
                    )
                eye = np.eye(bl_size)
                self.R[i][bl] = np.kron(np.array(r)[:, None], eye).astype(dtype)
                self.Lambda[i][bl] = np.kron(np.diag(levels), eye).astype(dtype)

        #: list of risb.types.BlockMatrix : Bath coupling of the impurity problem for each cluster.
        self.Lambda_c: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Hybridization of the impurity problem for each cluster.
        self.D: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Density matrix of quasiparticles for each cluster.
        self.rho_qp: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Lopsided kinetic energy of quasiparticles for each cluster.
        self.lopsided_ke_qp: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Kinetic energy of quasiparticles for each cluster.
        self.ke_qp: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Density matrix of cluster for each cluster.
        self.rho_c: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Density matrix of f-electrons in the impurity problem of each cluster.
        self.rho_f: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : Hybridization density matrix between the c-
        #: and f-electrons in the impurity problem for each cluster.
        self.rho_cf: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : The first root function of the self-consistent loop.
        self.f1: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: list of risb.types.BlockMatrix : The second root function of the self-consistent loop.
        self.f2: list[BlockMatrix] = [{} for i in range(self.n_clusters)]

        #: float : Largest absolute entry of :attr:`f1` and :attr:`f2` after
        #: :meth:`solve`. It is zero when the RISB self-consistency equations
        #: are satisfied.
        self.self_consistency_error: float

        #: risb.types.BlockRealArray : k-space integration weights of the
        #: quasiparticles in each band.
        self.kweights: BlockRealArray = {}

        #: risb.types.BlockRealArray : Band energy of quasiparticles.
        self.energies_qp: BlockRealArray = {}

        #: risb.types.BlockMatrix : Bloch band vectors of quasiparticles.
        self.bloch_vector_qp: BlockMatrix = {}

        #: risb.types.BlockMatrix : :attr:`R` of the clusters in the blocks of
        #: :attr:`h0_k`, from :func:`risb.helpers.upfold_r_lambda`, set at the
        #: start of each cycle. After :meth:`solve` it is from the :attr:`R`
        #: that the last cycle started from, not the :attr:`R` it returned.
        self.R_full: BlockMatrix = {}

        #: risb.types.BlockMatrix : :attr:`Lambda` of the clusters in the
        #: blocks of :attr:`h0_k`, set at the start of each cycle like
        #: :attr:`R_full`.
        self.Lambda_full: BlockMatrix = {}

        self.iteration = 0

    def root(
        self, *args: object, **kwargs: object
    ) -> NDArray[np.float64] | OptimizeResult:
        """
        Root function that drives the self-consistent procedure. It is called the same as :func:`scipy.optimize.root`.

        Returns
        -------
        numpy.ndarray or scipy.optimize.OptimizeResult

        """
        return self._root(*args, **kwargs)

    def update_weights(self, energies: BlockRealArray) -> BlockRealArray:
        """
        Update function that gives k-space integration weights.

        Parameters
        ----------
        energies : risb.types.BlockRealArray
            Energies at each k-point in each block.

        Returns
        -------
        risb.types.BlockRealArray
            Integration weights at each k-point in each block.

        """
        return self._update_weights(energies)

    @staticmethod
    def _hermitian_basis(N: int, is_real: bool = False) -> list[NDArray[np.inexact]]:
        """
        Return a basis of Hermitian matrices of size N.

        Parameters
        ----------
        N : int
            The dimension of the Hermitian matrics
        is_real : bool
            If True spans only symmetric matrices.

        Returns
        -------
        list of numpy.ndarray
            A list of matrices that define an orthonormal basis that spans all
            Hermitian matrices of dimension `N`.

        """
        H_rs: list[NDArray[np.inexact]]
        if is_real:
            n_basis = int(N + N * (N - 1) / 2)
            H_rs = [np.zeros([N, N]) for _ in range(n_basis)]
        else:
            n_basis = int(N + N * (N - 1) / 2 + N * (N - 1) / 2)
            H_rs = [np.zeros([N, N], dtype=complex) for _ in range(n_basis)]

        # Construct orthogonal basis
        i = 0
        for r, s in product(range(N), range(N)):
            if r == s:
                H_rs[i][r, r] = 1
                i = i + 1
            elif r < s:
                H_rs[i][r, s] = 1
                H_rs[i][s, r] = 1
                i = i + 1
            elif r > s:
                if not is_real:
                    H_rs[i][r, s] = 1j
                    H_rs[i][s, r] = -1j
                    i = i + 1

        # Normalize
        for i in range(len(H_rs)):
            H_rs[i] = H_rs[i] / np.sqrt(np.einsum("ij,ji->", H_rs[i].conj().T, H_rs[i]))
        return H_rs

    @staticmethod
    def _initialize_block_mf_matrix(
        gf_struct: list[GfStruct], is_real: bool
    ) -> list[BlockMatrix]:
        n_clusters = len(gf_struct)
        A: list[BlockMatrix] = [{} for i in range(n_clusters)]
        for i in range(n_clusters):
            for bl, bsize in gf_struct[i]:
                if is_real:
                    A[i][bl] = np.zeros((bsize, bsize))
                else:
                    A[i][bl] = np.zeros((bsize, bsize), dtype=complex)
        return A

    def _hermitian_basis_of(self, N: int) -> list[NDArray[np.inexact]]:
        if N not in self._H_basis:
            self._H_basis[N] = self._hermitian_basis(N, self.force_real)
        return self._H_basis[N]

    def _flatten_matrix(self, A: list[BlockMatrix], is_coeff_real: bool) -> list[float]:
        x: list[float] = []
        for i in range(self.n_clusters):
            for bl, _bl_size in self.gf_struct[i]:
                A_bl = A[i][bl]
                if A_bl.shape[0] != A_bl.shape[1]:
                    # A rectangular matrix, e.g., R with ghosts, is stored by
                    # its entries
                    coeffs = list(A_bl.ravel())
                else:
                    # To extract a coefficient of a basis that spans a vector
                    # space it is the inner product <basis_vec, vec> = coeff
                    # if the basis_vec is orthonormal. For matrices the inner
                    # product is <A, B> = Tr(A^+, B).
                    # Note that einsum is order mag faster than tr(a,b) and
                    # stores no intermediate array
                    # Could also vectorize both matrices as flatten('F') to
                    # column-major order and then take normal inner product
                    coeffs = [
                        np.einsum("ij,ji->", h.conj().T, A_bl)
                        for h in self._hermitian_basis_of(A_bl.shape[0])
                    ]
                for coeff in coeffs:
                    x.append(float(coeff.real))
                    if (not is_coeff_real) and (not self.force_real):
                        x.append(float(coeff.imag))
        return x

    def _unflatten_matrix(
        self,
        x: Sequence[float] | NDArray[np.float64],
        is_coeff_real: bool,
        like: list[BlockMatrix],
        offset: int = 0,
    ) -> tuple[list[BlockMatrix], int]:
        # Each block has the shape of the same block of like
        dtype = float if self.force_real else complex
        A: list[BlockMatrix] = [{} for i in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for bl, _bl_size in self.gf_struct[i]:
                shape = like[i][bl].shape
                if shape[0] != shape[1]:
                    basis = [
                        np.eye(shape[0] * shape[1])[n].reshape(shape)
                        for n in range(shape[0] * shape[1])
                    ]
                else:
                    basis = self._hermitian_basis_of(shape[0])
                A[i][bl] = np.zeros(shape, dtype=dtype)
                for h in basis:
                    # x stores the coefficients of the basis
                    if is_coeff_real or self.force_real:
                        A[i][bl] = A[i][bl] + x[offset] * h
                        offset = offset + 1
                    else:
                        A[i][bl] = A[i][bl] + (x[offset] + 1j * x[offset + 1]) * h
                        offset = offset + 2
        return A, offset

    def _project(self, A: list[BlockMatrix], is_coeff_real: bool) -> list[BlockMatrix]:
        # Onto the matrices that x can hold, e.g., Hermitian or real
        A_projected, _ = self._unflatten_matrix(
            self._flatten_matrix(A, is_coeff_real), is_coeff_real, like=A
        )
        return A_projected

    @staticmethod
    # Not really used
    def _make_hermitian(A: NDArray[np.inexact]) -> NDArray[np.inexact]:
        A_hermitian: NDArray[np.inexact] = 0.5 * (A + A.conj().T)
        return A_hermitian

    def _get_h0_loc_matrix(self) -> list[BlockMatrix]:
        h0_loc_matrix = self._initialize_block_mf_matrix(
            self.gf_struct, self.force_real
        )

        for i in range(self.n_clusters):
            for bl, _bl_size in self.gf_struct[i]:
                bl_full = self.gf_struct_mapping[i][bl]
                if self.projectors is not None:
                    h0_loc_matrix[i][bl] = helpers.get_h0_loc_matrix(
                        self.h0_k[bl_full], self.projectors[i][bl]
                    )
                else:
                    h0_loc_matrix[i][bl] = helpers.get_h0_loc_matrix(self.h0_k[bl_full])

        h0_loc_matrix = self._project(h0_loc_matrix, is_coeff_real=True)
        for symmetrize in self.symmetrizers:
            h0_loc_matrix = symmetrize(h0_loc_matrix)

        return h0_loc_matrix

    def _target_function(
        self, x: NDArray[np.float64]
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.Lambda, offset = self._unflatten_matrix(
            x, is_coeff_real=True, like=self.Lambda
        )
        self.R, _ = self._unflatten_matrix(
            x, is_coeff_real=False, like=self.R, offset=offset
        )
        self.Lambda, self.R, self.f1, self.f2 = self.one_cycle()
        x_new: NDArray[np.float64] = np.array(
            self._flatten_matrix(self.Lambda, is_coeff_real=True)
            + self._flatten_matrix(self.R, is_coeff_real=False)
        )

        x_error: NDArray[np.float64]
        if self.error_fun == "root":
            x_error = np.array(
                self._flatten_matrix(self.f2, is_coeff_real=True)
                + self._flatten_matrix(self.f1, is_coeff_real=False)
            )
        else:
            x_error = x_new - x

        if self.return_x_new:
            return x_new, x_error
        return x_error

    def one_cycle(
        self,
    ) -> tuple[
        list[BlockMatrix], list[BlockMatrix], list[BlockMatrix], list[BlockMatrix]
    ]:
        """
        Single iteration of the RISB self-consistent cycle.

        Returns
        -------
        Lambda : list of risb.types.BlockMatrix
            The new guess for the correlation potential matrix on each cluster.
        R : list of risb.types.BlockMatrix
            The new guess for the renormalization matrix on each cluster.
        f1 : list of risb.types.BlockMatrix
            The return of the fixed-point function that matches the
            quasiparticle density matrices on each cluster.
        f2 : list of risb.types.BlockMatrix
            The return of the fixed-point function that matches the
            hybridzation density matrices on each cluster.

        """
        for symmetrize in self.symmetrizers:
            self.R = symmetrize(self.R)
            self.Lambda = symmetrize(self.Lambda)

        # Make R, Lambda in supercell basis from basis of the clusters. The
        # orbitals in no cluster keep their bands: R = 1, Lambda = 0 there
        self.R_full, self.Lambda_full = helpers.upfold_r_lambda(
            self.R,
            self.Lambda,
            self.h0_kin_k,
            self._projectors_qp,
            self.gf_struct_mapping,
        )

        h0_k_R = {}
        # R_h0_k_R = dict()
        for bl in self.h0_kin_k:
            h_qp = helpers.get_h_qp(
                self.R_full[bl], self.Lambda_full[bl], self.h0_kin_k[bl]
            )
            self.energies_qp[bl], self.bloch_vector_qp[bl] = np.linalg.eigh(h_qp)
            h0_k_R[bl] = helpers.get_h0_kin_k_R(
                self.R_full[bl], self.h0_kin_k[bl], self.bloch_vector_qp[bl]
            )
            # R_h0_k_R[bl] = helpers.get_R_h0_kin_kR(R_full[bl], self.h0_kin_k[bl], self.bloch_vector_qp[bl])

        self.kweights = self.update_weights(self.energies_qp)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                bl_full = self.gf_struct_mapping[i][bl]
                P = None if self.projectors is None else self.projectors[i][bl]
                P_qp = None
                if self._projectors_qp is not None:
                    P_qp = self._projectors_qp[i][bl]
                self.rho_qp[i][bl] = helpers.get_rho_qp(
                    self.bloch_vector_qp[bl_full], self.kweights[bl_full], P_qp
                )
                self.lopsided_ke_qp[i][bl] = helpers.get_ke(
                    h0_k_R[bl_full],
                    self.bloch_vector_qp[bl_full],
                    self.kweights[bl_full],
                    P,
                    P_qp,
                )
                # self.ke_qp[i][bl] = helpers.get_ke(R_h0_k_R[bl_full], self.bloch_vector_qp[bl_full], self.kweights[bl_full], self.projectors[i][bl])

        # FIXME More expensive to do this than just storing and working with the coefficients, but
        # nothing compared to solving the impurity problem so this is likely fine
        # Enforce matrix structure from symmetries
        self.rho_qp = self._project(self.rho_qp, is_coeff_real=True)
        # self.lopsided_ke_qp, _ = self._unflatten_matrix( self._flatten_matrix( self.lopsided_ke_qp, is_coeff_real=False ), is_coeff_real=False )
        for symmetrize in self.symmetrizers:
            self.rho_qp = symmetrize(self.rho_qp)
            # self.lopsided_ke_qp = symmetrize(self.lopsided_ke_qp) # FIXME can I do it to ke as well? It should have same symm as D
            # self.ke_qp = symmetrize(self.ke_qp)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.D[i][bl] = helpers.get_d(
                        self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                    ).real
                    # self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl]).real
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(
                        self.rho_qp[i][bl],
                        self.R[i][bl],
                        self.Lambda[i][bl],
                        self.D[i][bl],
                    ).real
                else:
                    self.D[i][bl] = helpers.get_d(
                        self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]
                    )
                    # self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl])
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(
                        self.rho_qp[i][bl],
                        self.R[i][bl],
                        self.Lambda[i][bl],
                        self.D[i][bl],
                    )

        # Enforce matrix structure from symmetries
        self.Lambda_c = self._project(self.Lambda_c, is_coeff_real=True)
        self.D = self._project(self.D, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.Lambda_c = symmetrize(self.Lambda_c)
            self.D = symmetrize(self.D)

        # A completely full (empty) quasiparticle orbital has no bath partner.
        # Its D is zero, and the limit of the gradient term that would put its
        # bath orbital empty (full) is not finite, so put the bath level far
        # away instead. Otherwise the bath orbital competes for electrons and
        # the impurity problem can end at R = 0 with the wrong filling
        core_scale = self._core_scale()
        for i in range(self.n_clusters):
            for bl in self.Lambda_c[i]:
                self.Lambda_c[i][bl] = _place_core_baths(
                    self.Lambda_c[i][bl], self.rho_qp[i][bl], core_scale
                )

        for i in range(self.n_clusters):
            h0_loc_matrix = self.h0_loc_matrix[i]
            if self.double_counting is not None:
                rho = self.rho_c[i] if self.rho_c[i] else self.rho_qp[i]
                potential = self.double_counting(i, rho)
                h0_loc_matrix = {
                    bl: h0_loc_matrix[bl] - potential[bl] for bl in h0_loc_matrix
                }
            self.impurity_solvers[i].set_h_imp(
                self.Lambda_c[i], self.D[i], h0_loc_matrix
            )
            self.impurity_solvers[i].solve()
            for bl, _ in self.gf_struct[i]:
                self.rho_f[i][bl] = self.impurity_solvers[i].get_rho_f(bl)
                self.rho_cf[i][bl] = self.impurity_solvers[i].get_rho_cf(bl)
                self.rho_c[i][bl] = self.impurity_solvers[i].get_rho_c(bl)

        ## Enforce matrix structure from symmetries
        self.rho_f = self._project(self.rho_f, is_coeff_real=True)
        self.rho_cf = self._project(self.rho_cf, is_coeff_real=False)
        self.rho_c = self._project(self.rho_c, is_coeff_real=True)
        for symmetrize in self.symmetrizers:
            self.rho_f = symmetrize(self.rho_f)
            self.rho_cf = symmetrize(self.rho_cf)
            self.rho_c = symmetrize(self.rho_c)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                self.f1[i][bl] = helpers.get_f1(
                    self.rho_cf[i][bl], self.rho_qp[i][bl], self.R[i][bl]
                )
                self.f2[i][bl] = helpers.get_f2(self.rho_f[i][bl], self.rho_qp[i][bl])

        # Enforce matrix structure from symmetries
        self.f2 = self._project(self.f2, is_coeff_real=True)
        self.f1 = self._project(self.f1, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.f1 = symmetrize(self.f1)
            self.f2 = symmetrize(self.f2)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.Lambda[i][bl] = helpers.get_lambda(
                        self.R[i][bl],
                        self.D[i][bl],
                        self.Lambda_c[i][bl],
                        self.rho_f[i][bl],
                    ).real
                    self.R[i][bl] = helpers.get_r(
                        self.rho_cf[i][bl], self.rho_f[i][bl]
                    ).real
                else:
                    self.Lambda[i][bl] = helpers.get_lambda(
                        self.R[i][bl],
                        self.D[i][bl],
                        self.Lambda_c[i][bl],
                        self.rho_f[i][bl],
                    )
                    self.R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl])

        # Enforce matrix structure from symmetries
        self.Lambda = self._project(self.Lambda, is_coeff_real=True)
        self.R = self._project(self.R, is_coeff_real=False)
        for symmetrize in self.symmetrizers:
            self.Lambda = symmetrize(self.Lambda)
            self.R = symmetrize(self.R)

        # The quasiparticle level of a full (empty) orbital is only fixed to be
        # below (above) the chemical potential, so keep it below (above) every
        # other quasiparticle band. This finds the solution where the orbital
        # stays decoupled. A solution with a lower energy, where the
        # quasiparticle orbital is not exactly full (empty) and is an extra
        # channel for the other orbitals, can also exist
        for i in range(self.n_clusters):
            for bl in self.Lambda[i]:
                self.Lambda[i][bl] = _place_core_levels(
                    self.Lambda[i][bl],
                    self.rho_f[i][bl],
                    self._kinetic_half_bandwidth + core_scale,
                )

        self.iteration += 1

        return self.Lambda, self.R, self.f1, self.f2

    def solve(
        self,
        one_shot: bool = False,
        self_consistency_tol: float = 1e-6,
        **kwargs: object,
    ) -> NDArray[np.float64] | OptimizeResult:
        """
        Solve for the renormalization matrix :attr:`R` and correlation potential matrix :attr:`Lambda`.

        Sets the self-consistent solutions :attr:`Lambda` and :attr:`R`.

        Parameters
        ----------
        one_shot : bool, optional
            True if the calcualtion is just one shot and not self consistent.
            Default is False.
        self_consistency_tol : float, optional
            Warn if :attr:`self_consistency_error` is above this after a
            self-consistent solve. The root finder can stop where the guess of
            :attr:`Lambda` and :attr:`R` no longer changes but the
            self-consistency equations are not satisfied, e.g., at
            :attr:`R` = 0. In the atomic limit (:attr:`R` = 0 at a Mott
            insulator) the equations are not satisfied either, but there the
            bath of the impurity problem decouples and the observables do not
            depend on it. Default is ``1e-6``.
        **kwargs
            kwarg options to pass to :meth:`root`.

        Returns
        -------
        numpy.ndarray or scipy.optimize.OptimizeResult
            The flattened x vector of :attr:`Lambda` and :attr:`R`. If using
            :func:`scipy.optimize.root` the :class:`scipy.optimize.OptimizeResult`
            object will be returned.

        """
        x: NDArray[np.float64] | OptimizeResult
        if one_shot:
            self.Lambda, self.R, _, _ = self.one_cycle()
            x = np.array(
                self._flatten_matrix(self.Lambda, is_coeff_real=True)
                + self._flatten_matrix(self.R, is_coeff_real=False)
            )

        else:
            x0 = np.array(
                self._flatten_matrix(self.Lambda, is_coeff_real=True)
                + self._flatten_matrix(self.R, is_coeff_real=False)
            )
            x = self.root(fun=self._target_function, x0=x0, **kwargs)
            self.self_consistency_error = max(
                float(np.max(np.abs(f[i][bl])))
                for f in (self.f1, self.f2)
                for i in range(self.n_clusters)
                for bl in f[i]
            )
            if self.self_consistency_error > self_consistency_tol:
                msg = (
                    "The RISB self-consistency equations are not satisfied: "
                    f"max |f1|, |f2| = {self.self_consistency_error:.1e}. In the "
                    "atomic limit (R = 0, e.g., a Mott insulator) the bath of the "
                    "impurity problem decouples and this does not change the "
                    "observables. Otherwise the result is not a solution."
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return x

    def _core_scale(self) -> float:
        # An energy far from every energy of the model, leaving out the
        # levels of full or empty orbitals, which are themselves set far away
        largest = max(
            float(np.max(np.abs(A[i][bl]), initial=0))
            for A in (self.h0_loc_matrix,)
            for i in range(self.n_clusters)
            for bl in A[i]
        )
        for i in range(self.n_clusters):
            for bl in self.Lambda[i]:
                _, vectors, edge = _edge_orbitals(self.rho_qp[i][bl])
                active = vectors[:, ~edge].conj()
                if active.size:
                    levels = np.linalg.eigvalsh(
                        active.conj().T @ self.Lambda[i][bl] @ active
                    )
                    largest = max(largest, float(np.max(np.abs(levels))))
        return 10 * (self._kinetic_half_bandwidth + largest)

    @property
    def Z(self) -> list[BlockMatrix]:
        """
        Returns Z = R^dag R of each cluster, in the basis of the physical orbitals, as a list of block matrices.

        It is the spectral weight of each orbital in the Green's function
        ``R^dag G_qp R`` summed over all frequencies (Eq. 49 of the
        supplemental of `10.1103/PhysRevB.96.195126 <PRB195126_>`__). Without
        ghosts it is the quasiparticle weight. With ghosts (``bath_size``) the
        ghost poles carry weight too, so it is the sum rule of that Green's
        function, which is 1 if R is an isometry, and the quasiparticle weight
        is :func:`risb.helpers.get_z`.

        .. _PRB195126: https://doi.org/10.1103/PhysRevB.96.195126
        """
        Z: list[BlockMatrix] = [{} for i in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                Z[i][bl] = self.R[i][bl].conj().T @ self.R[i][bl]
        return Z


def _edge_orbitals(
    rho: NDArray[np.inexact],
) -> tuple[NDArray[np.float64], NDArray[np.inexact], NDArray[np.bool_]]:
    # Eigenvalues, eigenvectors, and which eigenvalues are 0 or 1
    eigenvalues, vectors = np.linalg.eigh(rho)
    return eigenvalues, vectors, eigenvalues * (1 - eigenvalues) < helpers.EDGE_TOL


def _place_core_baths(
    Lambda_c: NDArray[np.inexact], rho_qp: NDArray[np.inexact], scale: float
) -> NDArray[np.inexact]:
    eigenvalues, vectors, edge = _edge_orbitals(rho_qp)
    # The bath term is f Lambda_c f^dag, so filling the bath orbital costs
    # -Lambda_c: empty for a full quasiparticle orbital, full for an empty one
    for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
        level = -scale if delta > 0.5 else scale
        Lambda_c = Lambda_c + (level - np.real(v.conj() @ Lambda_c @ v)) * np.outer(
            v, v.conj()
        )
    return Lambda_c


def _place_core_levels(
    Lambda: NDArray[np.inexact], rho_f: NDArray[np.inexact], margin: float
) -> NDArray[np.inexact]:
    eigenvalues, vectors, edge = _edge_orbitals(rho_f)
    if not edge.any():
        return Lambda
    # rho_qp is the transpose of n_F(lambda - mu), so these are the
    # eigenvectors of lambda
    vectors = vectors.conj()
    active = vectors[:, ~edge]
    others = (
        np.linalg.eigvalsh(active.conj().T @ Lambda @ active)
        if active.size
        else np.zeros(1)
    )
    for delta, v in zip(eigenvalues[edge], vectors[:, edge].T, strict=True):
        level = others.min() - margin if delta > 0.5 else others.max() + margin
        Lambda = Lambda + (level - np.real(v.conj() @ Lambda @ v)) * np.outer(
            v, v.conj()
        )
    return Lambda


def _check_projectors(
    h0_k: BlockMatrixLike,
    projectors: Sequence[BlockMatrixLike],
    gf_struct_mapping: Sequence[Mapping[str, str]],
) -> None:
    rows: dict[str, list[NDArray[np.inexact]]] = {bl: [] for bl in h0_k}
    for i, P in enumerate(projectors):
        for bl, P_bl in P.items():
            bl_full = gf_struct_mapping[i][bl]
            # The same shape at every k-point, whether or not P depends on k
            n_k = np.shape(h0_k[bl_full])[0]
            P_k = np.broadcast_to(P_bl, (n_k, *np.shape(P_bl)[-2:]))
            rows[bl_full].append(P_k)
    for bl_full, P_list in rows.items():
        if not P_list:
            continue
        Q = np.concatenate(P_list, axis=-2)
        n_rows = Q.shape[-2]
        if n_rows > Q.shape[-1] or not np.allclose(
            Q @ np.swapaxes(Q.conj(), -1, -2), np.eye(n_rows), atol=1e-10
        ):
            msg = (
                f"The projectors onto block {bl_full} do not have orthonormal "
                "rows, or overlap between clusters, at some k-point."
            )
            raise ValueError(msg)


def _is_single_gf_struct(
    gf_struct: GfStruct | list[GfStruct],
) -> TypeIs[GfStruct]:
    return isinstance(gf_struct[0][0], str)
