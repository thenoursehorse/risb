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
from triqs.atom_diag import AtomDiag, act
from triqs.operators import Operator, c, c_dag

class EmbeddingAtomDiag:
    """
    Impurity solver of embedding space using atom_diag from TRIQS."

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
                 h_loc, gf_struct):
        
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

        #: triqs.atom_diag vacuum : Ground state of the embedding problem.
        self.gs_vec = None
        
        #: The TRIQS AtomDiag instance. See atom_diag in the TRIQS manual.
        self.ad = None

        #: triqs.operators.Operator : Embedding Hamiltonian.
        self.h_emb = None
        
        #: triqs.operators.Operator : Bath terms in `h_emb`.
        self.h_bath = None
        
        #: triqs.operators.Operator : Hybridization terms in `h_emb`.
        self.h_hybr = None


    @staticmethod
    def _bl_loc_to_bath(bl):
        return 'bath_'+bl
    
    @staticmethod
    def _bl_bath_to_loc(bl):
        return bl.replace('bath_', '')
    
    @staticmethod
    def _fops_from_gf_struct(gf_struct):
        return [(bl,i) for bl, bl_size in gf_struct for i in range(bl_size)]
    
    @staticmethod
    def _dict_gf_struct(gf_struct):
        return {bl: bl_size for bl, bl_size in gf_struct}
    
    def set_h_bath(self, Lambda_c):
        """
        Sets the bath terms in the impurity Hamiltonian.
        
        Parameters
        ----------
        Lambda_c : optional, dict of ndarray
            Bath coupling. Each key in dictionary must follow `gf_struct`.
        """
        self.h_bath = Operator()
        for bl_bath, bl_bath_size in self.gf_struct_bath:
            bl = self._bl_bath_to_loc(bl_bath)
            for a,b in product(range(bl_bath_size), range(bl_bath_size)):
                self.h_bath += Lambda_c[bl][a,b] * c(bl_bath,b) * c_dag(bl_bath,a)

    def set_h_hybr(self, D):
        """
        Sets the hybridization terms in the impurity Hamiltonian.
        
        Parameters
        ----------
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow `gf_struct`.
        """
        self.h_hybr = Operator()
        for bl, loc_size in self.gf_struct:
            bl_bath = self._bl_loc_to_bath(bl)
            bath_size = self.gf_struct_bath_dict[bl_bath]
            for a, alpha in product(range(bath_size), range(loc_size)):
                self.h_hybr += D[bl][a,alpha] * c_dag(bl,alpha) * c(bl_bath,a)
                self.h_hybr += np.conj(D[bl][a,alpha]) * c_dag(bl_bath,a) * c(bl,alpha)

    def set_h_emb(self, Lambda_c, D, mu=None):
        """
        Sets the terms in the impurity Hamiltonian to solve the embedding problem.
        
        Parameters
        ----------
        Lambda_c : dict[numpy.ndarray]
            Bath coupling. Each key in dictionary must follow `gf_struct`.
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow `gf_struct`.
        """
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        
        # For operators equal is copy, not a view
        self.h_emb = self.h_loc + self.h_bath + self.h_hybr
        
        if mu is not None:
            for bl, bl_size in self.gf_struct:
                for alpha in range(bl_size):
                    self.h_emb -= mu * c_dag(bl,alpha) * c(bl,alpha)

    # TODO other restrictions, like none, for testing.
    def solve(self, fixed='half'):
        """
        Solve for the groundstate in the half-filled number sector of the 
        embedding problem.

        Parameters
        ----------
        fixed : {'half'}
            How the Hilbert space is restricted. For fixed = 'half' 
            atom_diag will be passed n_min = n_max = half-filled.
        """
        if fixed == 'half':
            # FIXME for ghost does this need to be different?
            M = int(len(self.fops_emb) / 2)
            self.ad = AtomDiag(self.h_emb, self.fops_emb, n_min=M, n_max=M)
            self.gs_vec = self.ad.vacuum_state
            self.gs_vec[0] = 1

        else:
            raise ValueError('Unrecognized fixed particle number !')
        
    def get_nf(self, bl):
        """
        Parameters
        ----------
        bl : str
            Which block in `gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The f-electron density matrix from impurity.
        """
        bl_bath = self._bl_loc_to_bath(bl)
        bl_size = self.gf_struct_bath_dict[bl_bath]
        Nf = np.zeros([bl_size, bl_size])
        for a, b in product(range(bl_size), range(bl_size)):
            Op = c(bl_bath, b) * c_dag(bl_bath, a)
            Nf[a,b] = self.overlap(Op, force_real=True)
        return Nf
    
    def get_nc(self, bl):
        """
        Parameters
        ----------
        bl : str
            Which block in `gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The c-electron density matrix from impurity.
        """
        bl_size = self.gf_struct_dict[bl]
        Nc = np.zeros([bl_size, bl_size])
        for alpha, beta in product(range(bl_size), range(bl_size)):
            Op = c_dag(bl, alpha) * c(bl, beta)
            Nc[alpha,beta] = self.overlap(Op, force_real=True)
        return Nc
    
    def get_mcf(self, bl):
        """
        Parameters
        ----------
        bl : str
            Which block in `gf_struct` to return.

        Returns
        -------
        numpy.ndarray
            The c,f-electron hybridization density matrix from impurity.
        """
        bl_bath = self._bl_loc_to_bath(bl)
        bath_size = self.gf_struct_bath_dict[bl_bath]
        loc_size = self.gf_struct_dict[bl]
        Mcf = np.zeros([loc_size, bath_size], dtype=complex)
        for alpha, a in product(range(loc_size), range(bath_size)):
            Op = c_dag(bl, alpha) * c(bl_bath, a)
            Mcf[alpha,a] = self.overlap(Op, force_real=False)
        return Mcf
    
    def overlap(self, Op, force_real=True):
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
        res = self.gs_vec @ act(Op, self.gs_vec, self.ad)
        if force_real:
            return res.real
        else:
            return res
        
    @property
    def Nf(self):
        """
        dict[numpy.ndarray] : f-electron density matrix.
        """
        Nf = dict()
        for bl, bl_size in self.gf_struct:
            Nf[bl] = self.get_nf(bl)
        return Nf

    @property
    def Nc(self):
        """
        dict[numpy.ndarray] : c-electron density matrix.
        """
        Nc = dict()
        for bl, bl_size in self.gf_struct:
            Nc[bl] = self.get_nc(bl)
        return Nc
    
    @property
    def Mcf(self):
        """
        dict[numpy.ndarray] : Density matrix of hybridization terms (c- and f-electrons).
        """
        Mcf = dict()
        for bl, bl_size in self.gf_struct:
            Mcf[bl] = self.get_mcf(bl)
        return Mcf