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

"""Impurity solver using TRIQS AtomDiag."""

from itertools import product
from typing import override

import numpy as np
from numpy.typing import NDArray
from triqs.atom_diag import AtomDiag, act
from triqs.operators import Operator, c, c_dag, dagger

from risb.helpers_triqs import get_C_Op
from risb.types import BlockMatrix, BlockMatrixLike, GfStruct, ImpuritySolver


class AtomDiagImpuritySolver(ImpuritySolver):
    """
    Impurity solver using :class:`triqs.atom_diag.AtomDiag` from TRIQS.

    Parameters
    ----------
    h_int : triqs.operators.Operator
        Interaction Hamiltonian of the physical orbitals.
    gf_struct : risb.types.GfStruct
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """

    def __init__(self, h_int: Operator, gf_struct: GfStruct) -> None:
        #: triqs.operators.Operator : Interaction Hamiltonian.
        self.h_int = h_int

        #: risb.types.GfStruct : Block matrix structure of the physical orbitals
        #: (c-electrons).
        self.gf_struct = gf_struct

        # Set structure of bath and impurity problem
        # FIXME if doing ghost bath and loc are not the same size

        #: risb.types.GfStruct : Block matrix structure of the bath orbitals
        #: (f-electrons).
        self.gf_struct_bath = [
            (self._bl_loc_to_bath(bl), bl_size) for bl, bl_size in self.gf_struct
        ]

        #: risb.types.GfStruct : Block matrix structure of the entire impurity
        #: problem (physical and bath orbitals).
        self.gf_struct_imp = self.gf_struct + self.gf_struct_bath

        # Set of fundamental operators
        self.fops = self._fops_from_gf_struct(self.gf_struct)
        self.fops_bath = self._fops_from_gf_struct(self.gf_struct_bath)
        self.fops_imp = self._fops_from_gf_struct(self.gf_struct_imp)

        # Do gf_struct as a map
        self.gf_struct_dict = self._dict_gf_struct(self.gf_struct)
        self.gf_struct_bath_dict = self._dict_gf_struct(self.gf_struct_bath)
        self.gf_struct_imp_dict = self._dict_gf_struct(self.gf_struct_imp)

        #: numpy.ndarray : Ground state of the impurity problem in the
        #: eigenbasis of :attr:`ad`, set by :meth:`solve`.
        self.gs_vector: NDArray[np.inexact]

        #: The TRIQS AtomDiag instance, set by :meth:`solve`. See
        #: :class:`triqs.atom_diag.AtomDiag` in the TRIQS manual.
        self.ad: AtomDiag

        #: triqs.operators.Operator : Impurity Hamiltonian. It is the sum of
        #: :attr:`h0_loc`, :attr:`h_int`, :attr:`h_hybr`, and :attr:`h_bath`.
        self.h_imp: Operator = Operator()

        #: triqs.operators.Operator : Single-particle quadratic couplings of
        #: c-electron terms in ::attr:`h_imp`.
        self.h0_loc: Operator = Operator()

        #: triqs.operators.Operator : Bath terms in :attr:`h_imp`.
        self.h_bath: Operator = Operator()

        #: triqs.operators.Operator : Hybridization terms in :attr:`h_imp`.
        self.h_hybr: Operator = Operator()

        #: risb.types.BlockMatrix : f-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_f: BlockMatrix = {}

        #: risb.types.BlockMatrix : c-electron density matrix, set by
        #: :meth:`solve`.
        self.rho_c: BlockMatrix = {}

        #: risb.types.BlockMatrix : Density matrix of hybridization terms
        #: (c- and f-electrons), set by :meth:`solve`.
        self.rho_cf: BlockMatrix = {}

    @staticmethod
    def _bl_loc_to_bath(bl: str) -> str:
        return "bath_" + bl

    @staticmethod
    def _bl_bath_to_loc(bl: str) -> str:
        return bl.replace("bath_", "")

    @staticmethod
    def _fops_from_gf_struct(gf_struct: GfStruct) -> list[tuple[str, int]]:
        return [(bl, i) for bl, bl_size in gf_struct for i in range(bl_size)]

    @staticmethod
    def _dict_gf_struct(gf_struct: GfStruct) -> dict[str, int]:
        return dict(gf_struct)

    def set_h0_loc(self, h0_loc_matrix: BlockMatrixLike) -> None:
        """
        Set the single-particle quadratic couplings of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h0_loc_matrix : risb.types.BlockMatrixLike
            Quadratic terms as a matrix. Each key in dictionary must follow
            :attr:`gf_struct`.

        """
        C_Op = get_C_Op(self.gf_struct, dagger=False)
        C_dag_Op = get_C_Op(self.gf_struct, dagger=True)
        self.h0_loc = Operator()
        for bl, _bl_size in self.gf_struct:
            self.h0_loc += C_dag_Op[bl] @ h0_loc_matrix[bl] @ C_Op[bl]

    def set_h_int(self, h_int: Operator) -> None:
        """
        Set the interaction terms of the physical orbitals in the impurity Hamiltonian.

        Parameters
        ----------
        h_int : triqs.operators.Operator
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
        C_Op = get_C_Op(self.gf_struct_bath, dagger=False)
        C_dag_Op = get_C_Op(self.gf_struct_bath, dagger=True)
        self.h_bath = Operator()
        for bl_bath, _bl_bath_size in self.gf_struct_bath:
            bl = self._bl_bath_to_loc(bl_bath)
            self.h_bath += C_Op[bl_bath] @ Lambda_c[bl] @ C_dag_Op[bl_bath]

    def set_h_hybr(self, D: BlockMatrixLike) -> None:
        """
        Set the hybridization terms in the impurity Hamiltonian.

        Parameters
        ----------
        D : risb.types.BlockMatrixLike
            Hybridization coupling, indexed as ``D[bl][a, alpha]`` with ``a`` a
            bath orbital and ``alpha`` a physical orbital. The term is
            ``sum D[bl][a, alpha] c^dag(bl, alpha) c(bath, a) + h.c.``. Each key
            in dictionary must follow :attr:`gf_struct`.

        """
        C_Op = get_C_Op(self.gf_struct_bath, dagger=False)
        C_dag_Op = get_C_Op(self.gf_struct, dagger=True)
        self.h_hybr = Operator()
        for bl, _loc_size in self.gf_struct:
            bl_bath = self._bl_loc_to_bath(bl)
            tmp = C_dag_Op[bl] @ np.asarray(D[bl]).T @ C_Op[bl_bath]
            self.h_hybr += tmp + dagger(tmp)

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

        # For operators equal is copy, not a view
        self.h_imp = self.h0_loc + self.h_int + self.h_bath + self.h_hybr

        # NOTE h_bath must have + mu*np.eye()*f_a*f_a^dag to remove mu's
        # contribution
        if mu is not None:
            for bl, bl_size in self.gf_struct:
                for alpha in range(bl_size):
                    self.h_imp -= mu * c_dag(bl, alpha) * c(bl, alpha)

    # TODO other restrictions, like none, for testing
    # but it has been tested against sparse exact diagonalization and is the same answer
    # TODO what about superconductivity, ghosts?
    @override
    def solve(self) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the impurity problem.

        Sets the density matrices :attr:`rho_f`, :attr:`rho_cf`, and
        :attr:`rho_c` of every block.

        """
        M = int(len(self.fops_imp) / 2)
        # term is an array holding info of term as monomial, last index is its value
        if any(np.iscomplex(term[-1]) for term in self.h_imp):
            self.ad = AtomDiag(self.h_imp, self.fops_imp, n_min=M, n_max=M)
        else:
            self.ad = AtomDiag(self.h_imp.real, self.fops_imp, n_min=M, n_max=M)
        self.gs_vector = self.ad.vacuum_state.copy()
        self.gs_vector[0] = 1

        for bl, loc_size in self.gf_struct:
            bl_bath = self._bl_loc_to_bath(bl)
            bath_size = self.gf_struct_bath_dict[bl_bath]

            self.rho_f[bl] = np.zeros([bath_size, bath_size], dtype=complex)
            for a, b in product(range(bath_size), range(bath_size)):
                Op = c(bl_bath, b) * c_dag(bl_bath, a)
                self.rho_f[bl][a, b] = self.overlap(Op, force_real=False)

            self.rho_cf[bl] = np.zeros([loc_size, bath_size], dtype=complex)
            for alpha, a in product(range(loc_size), range(bath_size)):
                Op = c_dag(bl, alpha) * c(bl_bath, a)
                self.rho_cf[bl][alpha, a] = self.overlap(Op, force_real=False)

            self.rho_c[bl] = np.zeros([loc_size, loc_size], dtype=complex)
            for alpha, beta in product(range(loc_size), range(loc_size)):
                Op = c_dag(bl, alpha) * c(bl, beta)
                self.rho_c[bl][alpha, beta] = self.overlap(Op, force_real=False)

    @override
    def get_rho_f(self, bl: str) -> NDArray[np.inexact]:
        """
        Return f-electron densitym atrix.

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

    def overlap(self, Op: Operator, force_real: bool = True) -> float | complex:
        """
        Calculate the expectation value of an operator against the ground state of the impurity problem.

        Parameters
        ----------
        Op : triqs.operators.Operator
            Operator to take expectation of.
        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        float or complex
            Expectation value.

        """
        res = complex(self.gs_vector @ act(Op, self.gs_vector, self.ad))
        if force_real:
            return res.real
        return res

    @property
    def gs_energy(self) -> float:
        """Return ground state energy of impurity problem."""
        return float(self.ad.gs_energy)
