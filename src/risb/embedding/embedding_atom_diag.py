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

import numpy as np
from itertools import product
from typing import TypeAlias, TypeVar
from numpy.typing import ArrayLike
from triqs.atom_diag import AtomDiag, act
from triqs.operators import Operator, c, c_dag, dagger
from risb.helpers_triqs import get_C_Op

GfStructType : TypeAlias = list[tuple[str,int]]
OpType = TypeVar('OpType')
MFType : TypeAlias = dict[ArrayLike]

class EmbeddingAtomDiag:
    """
    Impurity solver of embedding space using :class:`triqs.atom_diag.AtomDiag` 
    from TRIQS.

    Parameters
    ----------
    h_loc : triqs.operators.Operator
        Local Hamiltonian including interactions and quadratic terms.
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """

    def __init__(self, 
                 h_loc : OpType, 
                 gf_struct : GfStructType) -> None:
        
        #: triqs.operators.Operator : Local Hamiltonian.
        self.h_loc = h_loc

        self.gf_struct = gf_struct

        # Set structure of bath and embedding space
        # FIXME if doing ghost bath and loc are not the same size    
        self.gf_struct_bath = [(self._bl_loc_to_bath(bl), bl_size) for bl, bl_size in self.gf_struct]
        self.gf_struct_emb = self.gf_struct + self.gf_struct_bath

        # Set of fundamental operators
        self.fops = self._fops_from_gf_struct(self.gf_struct)
        self.fops_bath = self._fops_from_gf_struct(self.gf_struct_bath)
        self.fops_emb = self._fops_from_gf_struct(self.gf_struct_emb)
        
        # Do gf_struct as a map
        self.gf_struct_dict = self._dict_gf_struct(self.gf_struct)
        self.gf_struct_bath_dict = self._dict_gf_struct(self.gf_struct_bath)
        self.gf_struct_emb_dict = self._dict_gf_struct(self.gf_struct_emb)

        #: triqs.atom_diag.AtomDiag vacuum : Ground state of the embedding problem.
        self.gs_vector = None
        
        #: The TRIQS AtomDiag instance. See :class:`triqs.atom_diag.AtomDiag` in the TRIQS manual.
        self.ad = None

        #: triqs.operators.Operator : Embedding Hamiltonian. It is the sum of
        #: :attr:`h_loc`, :attr:`h_hybr`, and :attr:`h_bath`.
        self.h_emb : OpType = Operator()
        
        #: triqs.operators.Operator : Bath terms in :attr:`h_emb`.
        self.h_bath : OpType = Operator()
        
        #: triqs.operators.Operator : Hybridization terms in :attr:`h_emb`.
        self.h_hybr : OpType = Operator()


    @staticmethod
    def _bl_loc_to_bath(bl : str) -> str:
        return 'bath_'+bl
    
    @staticmethod
    def _bl_bath_to_loc(bl : str) -> str:
        return bl.replace('bath_', '')
    
    @staticmethod
    def _fops_from_gf_struct(gf_struct : GfStructType) -> list[tuple[str, int]]:
        return [(bl,i) for bl, bl_size in gf_struct for i in range(bl_size)]
    
    @staticmethod
    def _dict_gf_struct(gf_struct : GfStructType) -> dict[str, int]:
        return {bl: bl_size for bl, bl_size in gf_struct}
    
    def set_h_bath(self, Lambda_c : MFType) -> None:
        """
        Sets the bath terms in the impurity Hamiltonian.
        
        Parameters
        ----------
        Lambda_c : dict of ndarray, optional
            Bath coupling. Each key in dictionary must follow ``gf_struct``.
        """
        C_Op = get_C_Op(self.gf_struct_bath, dagger=False)
        C_dag_Op = get_C_Op(self.gf_struct_bath, dagger=True)
        self.h_bath : OpType = Operator()
        for bl_bath, bl_bath_size in self.gf_struct_bath:
            bl = self._bl_bath_to_loc(bl_bath)
            self.h_bath += C_Op[bl_bath] @ Lambda_c[bl] @ C_dag_Op[bl_bath]

    def set_h_hybr(self, D : MFType) -> None:
        """
        Sets the hybridization terms in the impurity Hamiltonian.
        
        Parameters
        ----------
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow ``gf_struct``.
        """
        C_Op = get_C_Op(self.gf_struct_bath, dagger=False)
        C_dag_Op = get_C_Op(self.gf_struct, dagger=True)
        self.h_hybr : OpType = Operator()
        for bl, loc_size in self.gf_struct:
            bl_bath = self._bl_loc_to_bath(bl)
            tmp = C_dag_Op[bl] @ D[bl] @ C_Op[bl_bath]
            self.h_hybr += tmp + dagger(tmp)
        
    def set_h_emb(self, 
                  Lambda_c : MFType, 
                  D : MFType, 
                  mu : float | None = None) -> None:
        """
        Sets the terms in the impurity Hamiltonian to solve the embedding problem.
        
        Parameters
        ----------
        Lambda_c : dict[numpy.ndarray]
            Bath coupling. Each key in dictionary must follow ``gf_struct``.
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow ``gf_struct``.
        """
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        
        # For operators equal is copy, not a view
        self.h_emb = self.h_loc + self.h_bath + self.h_hybr

        # NOTE h_bath must have + mu*np.eye()*f_a*f_a^dag to remove mu's contribution
        if mu is not None:
            for bl, bl_size in self.gf_struct:
                for alpha in range(bl_size):
                    self.h_emb -= mu * c_dag(bl,alpha) * c(bl,alpha)

    # TODO other restrictions, like none, for testing
    # but it has been tested against sparse embedding and is the same answer
    # TODO what about superconductivity, ghosts?
    def solve(self) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the 
        embedding problem.
        """
        M = int(len(self.fops_emb) / 2)
        # term is an array holding info of term as monomial, last index is its value
        if any(np.iscomplex(term[-1]) for term in self.h_emb):
            self.ad = AtomDiag(self.h_emb, self.fops_emb, n_min=M, n_max=M)
        else:
            self.ad = AtomDiag(self.h_emb.real, self.fops_emb, n_min=M, n_max=M)
        self.gs_vector = self.ad.vacuum_state
        self.gs_vector[0] = 1

    def get_nf(self, bl : str) -> np.ndarray:
        """
        Parameters
        ----------
        bl : str
            Which block in ``gf_struct`` to return.

        Returns
        -------
        numpy.ndarray
            The f-electron density matrix :attr:`Nf` from impurity.
        """
        bl_bath = self._bl_loc_to_bath(bl)
        bl_size = self.gf_struct_bath_dict[bl_bath]
        Nf = np.zeros([bl_size, bl_size])
        for a, b in product(range(bl_size), range(bl_size)):
            Op = c(bl_bath, b) * c_dag(bl_bath, a)
            Nf[a,b] = self.overlap(Op, force_real=True)
        return Nf
    
    def get_nc(self, bl : str) -> np.ndarray:
        """
        Parameters
        ----------
        bl : str
            Which block in ``gf_struct`` to return.
        
        Returns
        -------
        numpy.ndarray
            The c-electron density matrix :attr:`Nc` from impurity.
        """
        bl_size = self.gf_struct_dict[bl]
        Nc = np.zeros([bl_size, bl_size])
        for alpha, beta in product(range(bl_size), range(bl_size)):
            Op = c_dag(bl, alpha) * c(bl, beta)
            Nc[alpha,beta] = self.overlap(Op, force_real=True)
        return Nc
    
    def get_mcf(self, bl : str) -> np.ndarray:
        """
        Parameters
        ----------
        bl : str
            Which block in ``gf_struct`` to return.

        Returns
        -------
        numpy.ndarray
            The c,f-electron hybridization density matrix :attr:`Mcf` from impurity.
        """
        bl_bath = self._bl_loc_to_bath(bl)
        bath_size = self.gf_struct_bath_dict[bl_bath]
        loc_size = self.gf_struct_dict[bl]
        Mcf = np.zeros([loc_size, bath_size], dtype=complex)
        for alpha, a in product(range(loc_size), range(bath_size)):
            Op = c_dag(bl, alpha) * c(bl_bath, a)
            Mcf[alpha,a] = self.overlap(Op, force_real=False)
        return Mcf
    
    def overlap(self, Op : OpType, force_real : bool = True) -> float | complex:
        """
        Calculate the expectation value of an operator against the ground state of 
        the embedding problem.

        Parameters
        ----------
        Op : triqs.operators.Operator
            Operator to take expectation of.

        force_real : bool, optional
            Whether the result should be real or complex.

        Returns
        -------
        triqs.operators.Operator
            Expectation value.
        """
        res = self.gs_vector @ act(Op, self.gs_vector, self.ad)
        if force_real:
            return res.real
        else:
            return res
        
    @property
    def Nf(self) -> MFType:
        """
        dict[numpy.ndarray] : f-electron density matrix.
        """
        Nf = dict()
        for bl, bl_size in self.gf_struct:
            Nf[bl] = self.get_nf(bl)
        return Nf

    @property
    def Nc(self) -> MFType:
        """
        dict[numpy.ndarray] : c-electron density matrix.
        """
        Nc = dict()
        for bl, bl_size in self.gf_struct:
            Nc[bl] = self.get_nc(bl)
        return Nc
    
    @property
    def Mcf(self) -> MFType:
        """
        dict[numpy.ndarray] : Density matrix of hybridization terms (c- and f-electrons).
        """
        Mcf = dict()
        for bl, bl_size in self.gf_struct:
            Mcf[bl] = self.get_mcf(bl)
        return Mcf
    
    @property
    def gs_energy(self) -> float:
        """
        float : Ground state energy of impurity problem.
        """
        return self.ad.gs_energy