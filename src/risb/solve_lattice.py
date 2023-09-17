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
from copy import deepcopy
from numpy.typing import ArrayLike
from typing import Any, Callable, TypeAlias
from risb import helpers
from risb.optimize import DIIS
from .other.from_triqs_hartree import flatten, unflatten

GfStructType : TypeAlias = list[tuple[str,int]]
MFType : TypeAlias = dict[ArrayLike]

# The structure and methods here have been modeled 
# after github.com/TRIQS/hartree_fock
class LatticeSolver:
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with
    a local interaction on each cluster.

    Parameters
    ----------
    h0_k : dict[numpy.ndarray]
        Single-particle dispersion between local clusters. Each key 
        in dictionary must follow `gf_struct`.

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
                 
    embedding : class
        The class that solves the embedding problem. It must already 
        store the local Hamiltonian, ``h_loc``, on a cluster, have a method 
        ``set_h_emb(Lambda_c, D)`` to setup the impurity problem, a method
        ``solve(**embedding_param)`` that solves the impurity problem, and 
        methods ``get_nf(block)`` and ``get_mcf(block)`` for the bath and 
        hybridization density matrices. 
        See class :class:`.EmbeddingAtomDiag`.
        
    update_weights : callable
        The function that gives the integral weights at each k-point on the 
        lattice. It is called as ``update_weights(energies, **kweight_param)``, 
        where the energies are a dictionary with each key a list. 
        See class :class:`.SmearingKWeight`.

    symmetries : list[callable], optional
        Symmetry functions acting on the mean-field matrices.

    force_real : bool, optional
        True if the mean-field matrices are forced to be real

    R : dict[numpy.ndarray], optional
        The unitary matrix from the f-electrons to the c-electrons at the
        mean-field level. Also called renormalization matrix. Each key in 
        dictionary must follow `gf_struct`. Defaults to the identity in each 
        block.

    Lambda : dict[numpy.ndarray], optional
        The correlation potential matrix experienced by the f-electrons.
        Each key in dictionary must follow `gf_struct`. Defaults to the zero 
        matrix in each block.

    root : callable, optional
        The function that drives the self-consistent procedure. It is called
        as ``root(fun, x0, args=, tol=, options=)``, where x0 is the initial 
        guess vector, and fun is the function to minimize, 
        where ``fun = self._target_function``. E.g., to use with `scipy` use 
        ``root(fun, x0, args=(embedding_param, kweight_param, False)``. 
        Defaults to ``solve`` method of :class:`.DIIS`.

    error_fun : str, optional
        At each self-consistent cycle, whether the returned error function is 
        'root' : f1 and f2 root functions
        'recursion' : the difference between consecutive `Lambda` and `R`.
        Defaults to 'root'.

    """
    def __init__(self, 
                 h0_k : MFType,
                 gf_struct : GfStructType,
                 embedding, 
                 update_weights,
                 symmetries : list[Callable[[MFType], dict[MFType]]] | None = [],
                 force_real : bool = True,
                 R : dict[ArrayLike] | None = None,
                 Lambda : dict[ArrayLike] | None = None, 
                 root = None,
                 error_fun : {'root', 'recursion'} = 'root'):
        
        self.h0_k = h0_k
        self.gf_struct = gf_struct
        self.block_names = [bl for bl,_ in self.gf_struct]
        
        self.embedding = embedding
        self._update_weights = update_weights
            
        self._root = root
        if self._root is None:
            self.optimize = DIIS()
            self._root = self.optimize.solve
        
        #: dict[numpy.ndarray] : Renormalization matrix of electrons (unitary matrix 
        #: from c- to f-electrons at the mean-field level).
        self.R = deepcopy(R)
        
        #: dict[numpy.ndarray] : Correlation potential matrix of the quasiparticles.
        self.Lambda = deepcopy(Lambda)
        
        if (self.R is None) or (self.Lambda is None):
            [self.R, self.Lambda] = self._initialize_block_mf_matrices(self.gf_struct)
 
        self.symmetries = symmetries
        self.force_real = force_real
        self.error_fun = error_fun
        
        #: dict[numpy.ndarray] : Bath coupling of impurity.
        self.Lambda_c = dict()

        #: dict[numpy.ndarray] : Hybridization of impurity.
        self.D = dict()

        #: dict[numpy.ndarray] : Density matrix of quasiparticles.
        self.rho_qp = dict()

        #: dict[numpy.ndarray] : Density matrix of cluster.
        self.Nc = dict()

        #: dict[numpy.ndarray] : Density matrix of f-electrons in the impurity.
        self.Nf = dict()

        #: dict[numpy.ndarray] : Hybridization density matrix between the c- 
        #: and f-electrons in the impurity.
        self.Mcf = dict()

        #: dict[numpy.ndarray] : k-space integration weights of the 
        #: quasiparticles in each band.
        self.wks = dict()
        
        #: dict[numpy.ndarray] : Band energy of quasiparticles.
        self.energies_qp = dict()

        #: dict[numpy.ndarray] : Bloch band vectors of quasiparticles.
        self.bloch_vector_qp = dict()

        #: dict[numpy.ndarray] : Lopsided dispersion of quasiparticles between clusters.
        self.lopsided_dispersion_qp = dict()

    def root(self, *args, **kwargs) -> np.ndarray:
        """
        The root function that drives the self-consistent procedure. It 
        is called the same as `scipy.optimize.root`.

        Returns
        -------
        numpy.ndarray
        """
        return self._root(*args, **kwargs)
    
    def update_weights(self, *args, **kwargs) -> dict[np.ndarray]:
        """
        The function that gives the k-space integration weights. It is 
        called as ``update_weights(dict[numpy.ndarray, **params])``.

        Returns
        -------
        numpy.ndarray
        """
        return self._update_weights(*args, **kwargs)

    @staticmethod
    def _initialize_block_mf_matrices(gf_struct : GfStructType) -> tuple[GfStructType, GfStructType]:
        R = dict()
        Lambda = dict()
        for bl, bsize in gf_struct:
            R[bl] = np.zeros((bsize,bsize))
            Lambda[bl] = np.zeros((bsize,bsize))
            np.fill_diagonal(R[bl], 1)
        return (R, Lambda)
    
    def _flatten(self, 
                Lambda : GfStructType, 
                R : GfStructType) -> np.ndarray:
        return flatten(Lambda, R, self.force_real)
    
    def _unflatten(self, x : ArrayLike) -> tuple[MFType, MFType]:
        return unflatten(x, self.gf_struct, self.force_real)
    
    def _target_function(self, 
                        x : ArrayLike, 
                        embedding_param : dict[str, Any], 
                        kweight_param : dict[str, Any], 
                        return_new : bool = True) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        
        self.Lambda, self.R = self._unflatten(x)
        Lambda_new, R_new, f1, f2  = self.one_cycle(embedding_param, kweight_param)
        x_new = self._flatten(Lambda_new, R_new)
        
        if self.error_fun == 'root':
            x_error = self._flatten(f2, f1)
        elif self.error_fun == 'recursion':
            x_error = x - x_new
        else:
            raise ValueError('Unrecognized error functions for root !')
        
        if return_new:
            return x_new, x_error
        else:
            return x_error

    def one_cycle(self, 
                  embedding_param : dict[str, Any] = dict(), 
                  kweight_param : dict[str, Any] = dict()):
                    #-> tuple[MFType, MFType, MFType, MFType]:
        """
        A single iteration of the RISB self-consistent cycle.

        Parameters
        ----------
        embedding_param : dict, optional
            The kwarg arguments to pass to the :meth:`embedding.solve`.
        kweight_param : dict, optional
            The kwarg arguments to pass to :meth:`update_weights`.
        
        Returns
        -------
        Lambda : dict[numpy.ndarray]
            The new guess for the correlation potential matrix.
        R : dict[numpy.ndarray]
            The new guess for the renormalization matrix.
        f1 : dict[numpy.ndarray]
            The return of the fixed-point function that matches the 
            quasiparticle density matrices.
        f2 : dict[numpy.ndarray]
            The return of the fixed-point function that matches the 
            hybridzation density matrices.
        """
    
        for function in self.symmetries:
            self.R = function(self.R)
            self.Lambda = function(self.Lambda)
                
        for bl in self.block_names:
            self.energies_qp[bl], self.bloch_vector_qp[bl] = helpers.get_h_qp(self.R[bl], self.Lambda[bl], self.h0_k[bl])
        
        self.wks = self.update_weights(self.energies_qp, **kweight_param)

        for bl in self.block_names:
            h0_R = helpers.get_h0_R(self.R[bl], self.h0_k[bl], self.bloch_vector_qp[bl])

            self.rho_qp[bl] = helpers.get_pdensity(self.bloch_vector_qp[bl], self.wks[bl])
            self.lopsided_dispersion_qp[bl] = helpers.get_ke(h0_R, self.bloch_vector_qp[bl], self.wks[bl])
        
            self.D[bl] = helpers.get_d(self.rho_qp[bl], self.lopsided_dispersion_qp[bl])
            if self.force_real:
                self.D[bl] = self.D[bl].real
            self.Lambda_c[bl] = helpers.get_lambda_c(self.rho_qp[bl], self.R[bl], self.Lambda[bl], self.D[bl])
        
        for function in self.symmetries:
            self.D = function(self.D)
            self.Lambda_c = function(self.Lambda_c)

        self.embedding.set_h_emb(self.Lambda_c, self.D)
        self.embedding.solve(**embedding_param)

        for bl in self.block_names:
            self.Nf[bl] = self.embedding.get_nf(bl)
            self.Mcf[bl] = self.embedding.get_mcf(bl)
        
        for function in self.symmetries:
            self.Nf = function(self.Nf)
            self.Mcf = function(self.Mcf)

        f1 = dict()
        f2 = dict()
        for bl in self.block_names:
            f1[bl] = helpers.get_f1(self.Mcf[bl], self.rho_qp[bl], self.R[bl])
            f2[bl] = helpers.get_f2(self.Nf[bl], self.rho_qp[bl])
        
        Lambda = dict()
        R = dict()
        for bl in self.block_names:
            Lambda[bl] = helpers.get_lambda(self.R[bl], self.D[bl], self.Lambda_c[bl], self.Nf[bl])
            R[bl] = helpers.get_r(self.Mcf[bl], self.Nf[bl])
            
        for function in self.symmetries:
            Lambda = function(Lambda)
            R = function(R)

        return Lambda, R, f1, f2
    
    def solve(self, 
              one_shot : bool = False, 
              tol : float = 1e-12, 
              root_param : dict[str, Any] = {'maxiter': 1000, 'alpha': 1}, 
              embedding_param : dict[str, Any] = dict(), 
              kweight_param : dict[str, Any] = dict()) -> None:
        """ 
        Solve for the renormalization matrix `R` and correlation potential
        matrix `Lambda`.

        Parameters
        ----------
        one_shot : bool, optional
            True if the calcualtion is just one shot and not self consistent. 
            Default is False.
        tol : float, optional
            Convergence tolerance to pass to :meth:`root`.
        root_param : dict, optional
            kwarg options to pass to :meth:`root`.
        embedding_param : dict, optional
            kwarg options to pass to :meth:`embedding.solve`.
        kweight_param : dict, optional
            kwarg options to pass to :meth:`update_weights`. 

        Returns
        -------
        Sets the self-consistent solutions `Lambda` and `R`.
        """

        if one_shot:
            self.Lambda, self.R, _, _ = self.one_cycle(embedding_param, kweight_param)
        
        else:
            self.root(fun=self._target_function, 
                      x0=self._flatten(self.Lambda, self.R), 
                      args=(embedding_param, kweight_param),
                      tol=tol,
                      options=root_param)
            #from scipy.optimize import root
            #root_finder = root(fun=self._target_function, 
            #                   x0=self._flatten(self.Lambda, self.R), 
            #                   args=(embedding_param, kweight_param, False), 
            #                   method='broyden1')

        
    @property
    def Z(self) -> MFType:
        """
        dict[numpy.ndarray] : Qausiparticle weight.
        """
        Z = dict()
        for bl in self.block_names:
            Z[bl] = self.R[bl] @ self.R[bl].conj().T
        return Z