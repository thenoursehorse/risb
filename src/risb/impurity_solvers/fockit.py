# Copyright (c) 2026 H. L. Nourse
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

"""Impurity solver using exact diagonalization from fockit."""

from collections.abc import Sequence
from typing import override

import numpy as np
from fockit.algebra import OpSum, c, cdag, hc, nc, zero
from fockit.hilbert import Basis, FermionSpace, build_basis, expectation, to_matrix
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from risb.types import BlockMatrix, BlockMatrixLike, GfStruct, ImpuritySolver


class FockitImpuritySolver(ImpuritySolver):
    """
    Impurity solver using exact diagonalization from fockit.

    Parameters
    ----------
    h_int : fockit.algebra.OpSum
        Interaction Hamiltonian of the physical orbitals. Its modes are
        ``(block, index)``, e.g. ``nc("up", 0) * nc("dn", 0)``.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """

    def __init__(self, h_int: OpSum, gf_struct: GfStruct) -> None:
        #: fockit.algebra.OpSum : Interaction Hamiltonian.
        self.h_int = h_int

        #: risb.types.GfStruct : Block matrix structure of the physical orbitals
        #: (c-electrons).
        self.gf_struct = gf_struct

        #: risb.types.GfStruct : Block matrix structure of the bath orbitals
        #: (f-electrons).
        self.gf_struct_bath = [(_bath(bl), bl_size) for bl, bl_size in gf_struct]

        modes = [
            (bl, i)
            for bl, bl_size in self.gf_struct + self.gf_struct_bath
            for i in range(bl_size)
        ]
        N = zero()
        for mode in modes:
            N += nc(*mode)

        #: fockit.hilbert.Basis : Half-filled particle sector of the impurity
        #: problem.
        self.basis: Basis = build_basis(
            {FermionSpace(): modes}, diagonal_symmetries={N: len(modes) // 2}
        )

        #: numpy.ndarray : Ground state of the impurity problem in
        #: :attr:`basis`, set by :meth:`solve`.
        self.gs_vector: NDArray[np.complex128]

        #: float : Ground state energy of the impurity problem, set by
        #: :meth:`solve`.
        self.gs_energy: float

        #: fockit.algebra.OpSum : Impurity Hamiltonian. It is the sum of
        #: :attr:`h0_loc`, :attr:`h_int`, :attr:`h_hybr`, and :attr:`h_bath`.
        self.h_imp: OpSum = zero()

        #: fockit.algebra.OpSum : Single-particle quadratic couplings of
        #: c-electron terms in :attr:`h_imp`.
        self.h0_loc: OpSum = zero()

        #: fockit.algebra.OpSum : Bath terms in :attr:`h_imp`.
        self.h_bath: OpSum = zero()

        #: fockit.algebra.OpSum : Hybridization terms in :attr:`h_imp`.
        self.h_hybr: OpSum = zero()

        #: risb.types.BlockMatrix : f-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_f: BlockMatrix = {}

        #: risb.types.BlockMatrix : c-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_c: BlockMatrix = {}

        #: risb.types.BlockMatrix : Density matrix of hybridization terms
        #: (c- and f-electrons), set by :meth:`solve`.
        self.rho_cf: BlockMatrix = {}

    def set_h0_loc(self, h0_loc_matrix: BlockMatrixLike) -> None:
        """
        Set the single-particle quadratic couplings of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h0_loc_matrix : risb.types.BlockMatrixLike
            Quadratic terms as a matrix. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        self.h0_loc = zero()
        for bl, bl_size in self.gf_struct:
            self.h0_loc += _bilinear(
                [cdag(bl, i) for i in range(bl_size)],
                h0_loc_matrix[bl],
                [c(bl, i) for i in range(bl_size)],
            )

    def set_h_int(self, h_int: OpSum) -> None:
        """
        Set the interaction terms of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h_int : fockit.algebra.OpSum
            Interaction Hamiltonian of the physical orbitals.

        """
        self.h_int = h_int

    def set_h_bath(self, Lambda_c: BlockMatrixLike) -> None:
        """
        Set the bath terms in the impurity Hamiltonian.

        Parameters
        ----------
        Lambda_c : risb.types.BlockMatrixLike
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        self.h_bath = zero()
        for bl, bl_size in self.gf_struct:
            self.h_bath += _bilinear(
                [c(_bath(bl), a) for a in range(bl_size)],
                Lambda_c[bl],
                [cdag(_bath(bl), a) for a in range(bl_size)],
            )

    def set_h_hybr(self, D: BlockMatrixLike) -> None:
        """
        Set the hybridization terms in the impurity Hamiltonian.

        Parameters
        ----------
        D : risb.types.BlockMatrixLike
            Hybridization coupling. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        self.h_hybr = zero()
        for bl, bl_size in self.gf_struct:
            tmp = _bilinear(
                [cdag(bl, alpha) for alpha in range(bl_size)],
                D[bl],
                [c(_bath(bl), a) for a in range(bl_size)],
            )
            self.h_hybr += tmp + hc(tmp)

    @override
    def set_h_imp(
        self,
        Lambda_c: BlockMatrixLike,
        D: BlockMatrixLike,
        h0_loc_matrix: BlockMatrixLike | None = None,
        mu: float | None = None,
    ) -> None:
        """
        Set the terms of the impurity Hamiltonian.

        Parameters
        ----------
        Lambda_c : risb.types.BlockMatrixLike
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        D : risb.types.BlockMatrixLike
            Hybridization coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        h0_loc_matrix : risb.types.BlockMatrixLike, optional
            Single-particle quadratic couplings of the c-electrons. Each key
            in dictionary must follow :attr:`gf_struct`.
        mu : float, optional
            Chemical potential, subtracted as ``mu * n`` on the physical
            orbitals.

        """
        if h0_loc_matrix is not None:
            self.set_h0_loc(h0_loc_matrix)
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        self.h_imp = self.h0_loc + self.h_int + self.h_bath + self.h_hybr
        if mu is not None:
            for bl, bl_size in self.gf_struct:
                for alpha in range(bl_size):
                    self.h_imp -= mu * nc(bl, alpha)

    @override
    def solve(self) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the impurity problem.

        Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block.

        Raises
        ------
        ValueError
            If :attr:`h_imp` is not Hermitian.

        """
        H = to_matrix(self.h_imp, self.basis)
        if abs(H - H.conj().T).max() > 1e-10:
            msg = "h_imp is not Hermitian."
            raise ValueError(msg)
        if self.basis.dim < 3:
            energies, states = np.linalg.eigh(H.toarray())
        else:
            v0 = np.random.default_rng(0).normal(size=self.basis.dim)
            energies, states = eigsh(
                H, k=1, which="SA", v0=v0, ncv=min(self.basis.dim, 40)
            )
        self.gs_energy = float(energies[0])
        self.gs_vector = np.asarray(states[:, 0], dtype=complex)
        for bl, bl_size in self.gf_struct:
            bl_bath = _bath(bl)
            sites = range(bl_size)
            self.rho_f[bl] = np.array(
                [
                    [
                        self.overlap(c(bl_bath, b) * cdag(bl_bath, a), False)
                        for b in sites
                    ]
                    for a in sites
                ]
            )
            self.rho_cf[bl] = np.array(
                [
                    [
                        self.overlap(cdag(bl, alpha) * c(bl_bath, a), False)
                        for a in sites
                    ]
                    for alpha in sites
                ]
            )
            self.rho_c[bl] = np.array(
                [
                    [
                        self.overlap(cdag(bl, alpha) * c(bl, beta), False)
                        for beta in sites
                    ]
                    for alpha in sites
                ]
            )

    @override
    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        """
        Return f-electron density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The f-electron density matrix :attr:`rho_f` from the impurity problem.

        """
        return self.rho_f[bl]

    @override
    def get_rho_c(self, bl: str) -> NDArray[np.inexact]:
        """
        Return c-electron density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The c-electron density matrix :attr:`rho_c` from the impurity problem.

        """
        return self.rho_c[bl]

    @override
    def get_rho_cf(self, bl: str) -> NDArray[np.inexact]:
        """
        Return the cf hybridization (off-diagonal) density matrix.

        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The c,f-electron hybridization density matrix :attr:`rho_cf` from
            the impurity problem.

        """
        return self.rho_cf[bl]

    def overlap(self, Op: OpSum, force_real: bool = True) -> float | complex:
        """
        Calculate the expectation value of an operator against the ground state of the impurity problem.

        Parameters
        ----------
        Op : fockit.algebra.OpSum
            Operator to take expectation of. The bath modes are
            ``("bath_" + block, index)``.
        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        float or complex
            Expectation value.

        """
        res = expectation(to_matrix(Op, self.basis), self.gs_vector)
        if force_real:
            return res.real
        return res


def _bath(bl: str) -> str:
    return "bath_" + bl


def _bilinear(
    left: Sequence[OpSum], A: NDArray[np.inexact], right: Sequence[OpSum]
) -> OpSum:
    out = zero()
    for i, j in np.ndindex(A.shape):
        out += complex(A[i, j]) * left[i] * right[j]
    return out
