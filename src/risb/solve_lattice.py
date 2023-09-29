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
from itertools import product
from numpy.typing import ArrayLike
from typing import Any, Callable, TypeAlias
from risb import helpers
from risb.optimize import DIIS

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
        represents a single-particle symmetry.
    gf_struct : list[ list of pairs [ (str,int), ...] ]
        Structure of the matrices. For each cluster, it must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example for a single cluster: ``[ ('up', 3), ('down', 3) ]``.
    embedding : list[class]
        The class that solves the embedding problem for each cluster. It must 
        already store the local Hamiltonian ``h_loc`` on a cluster, have a method 
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
    root : callable, optional
        The function that drives the self-consistent procedure. It is called
        as ``root(fun, x0, args=, tol=, options=)``, where ``x0`` is the initial 
        guess vector, and ``fun`` is the function to minimize, 
        where ``fun = self._target_function``.
        Defaults to :meth:`.DIIS.solve` method of :class:`.DIIS`.
    projectors : list[dict[numpy.ndarray]], optional
        The projectors onto each subspace of an `embedding` cluster.
    gf_struct_mapping : list[dict[str,str]], optional
        The mapping from the symmetry blocks of each cluster in `embedding` 
        to the symmetry blocks of `h0_k`. Default assumes the keys in 
        all clusters are the same as the keys in `h0_k`.
    symmetries : list[callable], optional
        Symmetry functions acting on the mean-field matrices. The argument of 
        the function must take a list of all clusters. 
        E.g., ``[R_cluster1, R_cluster2, ...]``.
    force_real : bool, optional
        True if the mean-field matrices are forced to be real
    error_fun : str, optional
        At each self-consistent cycle, whether the returned error function is 
            - 'root' : f1 and f2 root functions
            - 'recursion' : the difference between consecutive :attr:`Lambda` and :attr:`R`.
        Defaults to 'root'.
    return_x_new : bool, optional
        Whether to return a new guess for ``x`` and the ``error`` at each iteration or 
        only the ``error``. :func:`scipy.optimize.root` should only use the ``error``.

    """
    def __init__(self, 
                 h0_k : MFType,
                 gf_struct : GfStructType,
                 embedding, 
                 update_weights,
                 root = None,
                 projectors = None,
                 gf_struct_mapping : list[dict[str,str]] | None = None,
                 symmetries : list[Callable[[MFType], dict[MFType]]] | None = [],
                 force_real : bool = False,
                 error_fun : {'root', 'recursion'} = 'root',
                 return_x_new : bool = True,
                 ):
        
        self.iteration = 0

        self.h0_k = h0_k
        
        # FIXME is this the best way to make sure gf_struct is a list of gf_struct?
        if isinstance(gf_struct[0][0], str) or isinstance(gf_struct[0][0], int):
            self.gf_struct = [gf_struct]
        else:
            self.gf_struct = gf_struct
        
        #: int : Number of correlated clusters per supercell on the lattice.
        self.n_clusters = len(self.gf_struct)
        
        if isinstance(embedding, list):
            self.embedding = embedding
        else:
            self.embedding = [embedding]

        self._update_weights = update_weights
        
        self._root = root
        if self._root is None:
            self.optimize = DIIS()
            self._root = self.optimize.solve

        if projectors is None:
            self.projectors = [ {bl:np.eye(bl_size) for bl, bl_size in self.gf_struct[i]} for i in range(self.n_clusters)]
        else:
            self.projectors = projectors
                        
        if gf_struct_mapping is None:
            self.gf_struct_mapping = [{bl:bl for bl in h0_k.keys()} for i in range(self.n_clusters)]
        else:
            self.gf_struct_mapping = gf_struct_mapping
 
        self.symmetries = symmetries
        self.force_real = force_real
        self.error_fun = error_fun
        self.return_x_new = return_x_new

        # The Hermitian basis in each block of a cluster
        self._H_basis = [{bl:self._hermitian_basis(bl_size, self.force_real) for bl, bl_size in self.gf_struct[i]} for i in range(self.n_clusters)]
        
        #: list[dict[numpy.ndarray]] : Renormalization matrix
        #: from c- to f-electrons at the mean-field level for each cluster.
        self.R = self._initialize_block_mf_matrix(self.gf_struct, self.force_real)
        
        for i in range(self.n_clusters):
            for bl_sub, _ in self.gf_struct[i]:
                np.fill_diagonal(self.R[i][bl_sub], 1)
        
        #: list[dict[numpy.ndarray]] : Correlation potential matrix of the quasiparticles 
        #: for each cluster.
        self.Lambda = self._initialize_block_mf_matrix(self.gf_struct, self.force_real)
        
        #: list[dict[numpy.ndarray]] : Bath coupling of impurity for each cluster.
        self.Lambda_c = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : Hybridization of impurity for each cluster.
        self.D = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : Density matrix of quasiparticles for each cluster.
        self.rho_qp = [dict() for i in range(self.n_clusters)]
        
        #: list[dict[numpy.ndarray]] : Lopsided kinetic energy of quasiparticles for each cluster.
        self.lopsided_ke_qp = [dict() for i in range(self.n_clusters)]
        
        #: list[dict[numpy.ndarray]] : Kinetic energy of quasiparticles for each cluster.
        self.ke_qp = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : Density matrix of cluster for each cluster.
        self.rho_c = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : Density matrix of f-electrons in the impurity of each cluster.
        self.rho_f = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : Hybridization density matrix between the c- 
        #: and f-electrons in the impurity for each cluster.
        self.rho_cf = [dict() for i in range(self.n_clusters)]
        
        #: list[dict[numpy.ndarray]] : The first root function of the self-consistent loop.
        self.f1 = [dict() for i in range(self.n_clusters)]

        #: list[dict[numpy.ndarray]] : The second root function of th self-consistent loop.
        self.f2 = [dict() for i in range(self.n_clusters)]

        #: dict[numpy.ndarray] : k-space integration weights of the 
        #: quasiparticles in each band.
        self.kweights = dict()
        
        #: dict[numpy.ndarray] : Band energy of quasiparticles.
        self.energies_qp = dict()

        #: dict[numpy.ndarray] : Bloch band vectors of quasiparticles.
        self.bloch_vector_qp = dict()

    def root(self, *args, **kwargs) -> np.ndarray:
        """
        The root function that drives the self-consistent procedure. It 
        is called the same as :func:`scipy.optimize.root`.

        Returns
        -------
        numpy.ndarray
        """
        return self._root(*args, **kwargs)
    
    def update_weights(self, *args, **kwargs) -> dict[np.ndarray]:
        """
        The function that gives the k-space integration weights. It is 
        called as ``update_weights(dict[numpy.ndarray], **params)``.

        Returns
        -------
        numpy.ndarray
        """
        return self._update_weights(*args, **kwargs)
    
    @staticmethod
    def _hermitian_basis(N : int, is_real : bool = False):
        """
        Parameters
        ----------
        N : int
            The dimension of the Hermitian matrics
        is_real : bool
            If True spans only symmetric matrices.

        Returns
        -------
        list[numpy.ndarray]
            A list of matrices that define an orthonormal basis tat spans all
            Hermitian matrices of dimension `N`.
        """
        if is_real:
            n_basis = int(N + N*(N-1)/2)
            H_rs = [np.zeros([N,N]) for _ in range(n_basis)]
        else:
            n_basis = int(N + N*(N-1)/2 + N*(N-1)/2)
            H_rs = [np.zeros([N,N], dtype=complex) for _ in range(n_basis)]

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
            H_rs[i] = H_rs[i] / np.sqrt( np.einsum( 'ij,ji->', H_rs[i].conj().T, H_rs[i] ) )
        return H_rs
    
    @staticmethod
    def _initialize_block_mf_matrix(gf_struct : GfStructType,
                                    is_real : bool) -> MFType:
        n_clusters = len(gf_struct)
        mat = [dict() for i in range(n_clusters)]
        for i in range(n_clusters):
            for bl, bsize in gf_struct[i]:
                if is_real:
                    mat[i][bl] = np.zeros((bsize,bsize))
                else:
                    mat[i][bl] = np.zeros((bsize,bsize), dtype=complex)
        return mat

    def _flatten_mat(self, 
                     A : MFType, 
                     is_coeff_real : bool):
        if len(A) != len(self._H_basis):
            raise ValueError(f'len(A) = {len(A)} and len(H_basis) = {len(self._H_basis)} must have the same number of clusters !')
        x = []
        for i in range(self.n_clusters):
            for bl, bl_size in self.gf_struct[i]:
                for h in self._H_basis[i][bl]:
                    # To extract a coefficient of a basis that spans a vector 
                    # space it is the inner product <basis_vec, vec> = coeff
                    # if the basis_vec is orthonormal. For matrices the inner 
                    # product is <A, B> = Tr(A^+, B).
                    # Note that einsum is order mag faster than tr(a,b) and 
                    # stores no intermediate array
                    # Could also vectorize both matrices as flatten('F') to 
                    # column-major order and then take normal inner product
                    coeff = np.einsum('ij,ji->', h.conj().T, A[i][bl])
                    x.append( coeff.real )
                    if (not is_coeff_real) and (not self.force_real):
                        x.append( coeff.imag )
        return x

    def _unflatten_mat(self,
                       x : ArrayLike, 
                       is_coeff_real : bool, 
                       offset : int = 0) -> tuple[MFType, int]:
        mat = self._initialize_block_mf_matrix(self.gf_struct, self.force_real)
        for i in range(self.n_clusters):
            for bl, bl_size in self.gf_struct[i]:
                for h in self._H_basis[i][bl]:
                    # x stores the coefficients of the basis of Hermitian 
                    # matrices defined in self._H_basis.
                    if is_coeff_real or self.force_real:
                        mat[i][bl] += x[offset] * h
                        offset = offset + 1
                    else:
                        mat[i][bl] += (x[offset] + 1j*x[offset+1]) * h
                        offset = offset + 2
        return mat, offset

    @staticmethod
    # Not really used
    def _make_hermitian(A):
        if isinstance(A, list):
            for i in range(len(A)):
                for bl in A[i].keys():
                    A[i][bl] = 0.5 * (A[i][bl] + A[i][bl].conj().T)
        elif isinstance(A, dict):
            for bl in A.keys():
                A[bl] = 0.5 * (A[bl] + A[bl].conj().T)
        elif isinstance(A, np.ndarray):
            A = 0.5 * (A + A.conj().T)
        else:
            raise ValueError('A is not a list[dict[ndarray]], dict[ndarray], or ndarray !')
        return A
    
    def _target_function(self, 
                        x : ArrayLike, 
                        embedding_param : list[dict[str, Any]], 
                        kweight_param : dict[str, Any], 
                        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        
        self.Lambda, offset = self._unflatten_mat(x, is_coeff_real=True)
        self.R, _ = self._unflatten_mat(x, is_coeff_real=False, offset=offset)
        self.Lambda, self.R, self.f1, self.f2  = self.one_cycle(embedding_param, kweight_param)
        x_new = np.array( self._flatten_mat(self.Lambda, is_coeff_real=True) + self._flatten_mat(self.R, is_coeff_real=False) )
        
        if self.error_fun == 'root':
            x_error = np.array( self._flatten_mat(self.f2, is_coeff_real=True) + self._flatten_mat(self.f1, is_coeff_real=False) )
        elif self.error_fun == 'recursion':
            x_error = x - x_new
        else:
            raise ValueError('Unrecognized error functions for root !')
        
        if self.return_x_new:
            return x_new, x_error
        else:
            return x_error
    
    def one_cycle(self, 
                  embedding_param : list[dict[str, Any]] | None = None, 
                  kweight_param : dict[str, Any] = dict()):
                    #-> tuple[MFType, MFType, MFType, MFType]:
        """
        A single iteration of the RISB self-consistent cycle.

        Parameters
        ----------
        embedding_param : list[dict], optional
            The kwarg arguments to pass to the :meth:`embedding.solve` for each cluster.
        kweight_param : dict, optional
            The kwarg arguments to pass to :meth:`update_weights`.
        
        Returns
        -------
        Lambda : list[dict[numpy.ndarray]]
            The new guess for the correlation potential matrix on each cluster.
        R : list[dict[numpy.ndarray]]
            The new guess for the renormalization matrix on each cluster.
        f1 : list[dict[numpy.ndarray]]
            The return of the fixed-point function that matches the 
            quasiparticle density matrices on each cluster.
        f2 : list[dict[numpy.ndarray]]
            The return of the fixed-point function that matches the 
            hybridzation density matrices on each cluster.
        """

        if embedding_param is None:
            embedding_param = [dict() for i in range(self.n_clusters)]
 
        for function in self.symmetries:
            self.R = function(self.R)
            self.Lambda = function(self.Lambda)
        
        # Make R, Lambda in supercell basis from basis of the clusters
        R_full = {bl:0 for bl in self.h0_k.keys()}
        Lambda_full = {bl:0 for bl in self.h0_k.keys()}
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                bl_full = self.gf_struct_mapping[i][bl]
                R_full[bl_full] += self.projectors[i][bl].conj().T @ self.R[i][bl] @ self.projectors[i][bl]
                Lambda_full[bl_full] += self.projectors[i][bl].conj().T @ self.Lambda[i][bl] @ self.projectors[i][bl]
        
        h0_R = dict()
        #R_h0_R = dict()
        for bl in self.h0_k.keys():
            self.energies_qp[bl], self.bloch_vector_qp[bl] = helpers.get_h_qp(R_full[bl], Lambda_full[bl], self.h0_k[bl])
            h0_R[bl] = helpers.get_h0_R(R_full[bl], self.h0_k[bl], self.bloch_vector_qp[bl])
            #R_h0_R[bl] = helpers.get_R_h0_R(R_full[bl], self.h0_k[bl], self.bloch_vector_qp[bl])
        
        self.kweights = self.update_weights(self.energies_qp, **kweight_param)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                bl_full = self.gf_struct_mapping[i][bl]
                self.rho_qp[i][bl] = helpers.get_pdensity(self.bloch_vector_qp[bl_full], self.kweights[bl_full], self.projectors[i][bl])
                self.lopsided_ke_qp[i][bl] = helpers.get_ke(h0_R[bl_full], self.bloch_vector_qp[bl_full], self.kweights[bl_full], self.projectors[i][bl])
                #self.ke_qp[i][bl] = helpers.get_ke(R_h0_R[bl_full], self.bloch_vector_qp[bl_full], self.kweights[bl_full], self.projectors[i][bl])
        
        # FIXME More expensive to do this than just storing and working with the coefficients, but 
        # nothing compared to solving the embedding problem so this is likely fine
        # Enforce matrix structure from symmetries
        self.rho_qp, _ = self._unflatten_mat( self._flatten_mat( self.rho_qp, is_coeff_real=True ), is_coeff_real=True )

        for function in self.symmetries:
            self.rho_qp = function(self.rho_qp)
            self.lopsided_ke_qp = function(self.lopsided_ke_qp)
            #self.ke_qp = function(self.ke_qp)
        
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.D[i][bl] = helpers.get_d(self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl]).real
                    #self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl]).real
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(self.rho_qp[i][bl], self.R[i][bl], self.Lambda[i][bl], self.D[i][bl]).real
                else:
                    self.D[i][bl] = helpers.get_d(self.rho_qp[i][bl], self.lopsided_ke_qp[i][bl])
                    #self.D[i][bl] = helpers.get_d2(self.rho_qp[i][bl], self.ke_qp[i][bl], self.R[i][bl])
                    self.Lambda_c[i][bl] = helpers.get_lambda_c(self.rho_qp[i][bl], self.R[i][bl], self.Lambda[i][bl], self.D[i][bl])
                    
        # Enforce matrix structure from symmetries
        self.Lambda_c, _ = self._unflatten_mat( self._flatten_mat( self.Lambda_c, is_coeff_real=True ), is_coeff_real=True )
        self.D, _ = self._unflatten_mat( self._flatten_mat( self.D, is_coeff_real=False ), is_coeff_real=False )
        
        for function in self.symmetries:
            self.Lambda_c = function(self.Lambda_c)
            self.D = function(self.D)

        for i in range(self.n_clusters):
            self.embedding[i].set_h_emb(self.Lambda_c[i], self.D[i])
            self.embedding[i].solve(**embedding_param[i])
            for bl, _ in self.gf_struct[i]:
                self.rho_f[i][bl] = self.embedding[i].get_nf(bl)
                self.rho_cf[i][bl] = self.embedding[i].get_mcf(bl)
        
        for function in self.symmetries:
            self.rho_f = function(self.rho_f)
            self.rho_cf = function(self.rho_cf)

        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                self.f1[i][bl] = helpers.get_f1(self.rho_cf[i][bl], self.rho_qp[i][bl], self.R[i][bl])
                self.f2[i][bl] = helpers.get_f2(self.rho_f[i][bl], self.rho_qp[i][bl])
        
        for function in self.symmetries:
            self.f1 = function(self.f1)
            self.f2 = function(self.f2)
        
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                if self.force_real:
                    self.Lambda[i][bl] = helpers.get_lambda(self.R[i][bl], self.D[i][bl], self.Lambda_c[i][bl], self.rho_f[i][bl]).real
                    self.R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl]).real
                else:
                    self.Lambda[i][bl] = helpers.get_lambda(self.R[i][bl], self.D[i][bl], self.Lambda_c[i][bl], self.rho_f[i][bl])
                    self.R[i][bl] = helpers.get_r(self.rho_cf[i][bl], self.rho_f[i][bl])
        
        for function in self.symmetries:
            self.Lambda = function(self.Lambda)
            self.R = function(self.R)

        self.iteration += 1

        return self.Lambda, self.R, self.f1, self.f2
    
    def solve(self, 
              one_shot : bool = False, 
              embedding_param : list[dict[str, Any]] | None = None, 
              kweight_param : dict[str, Any] = dict(),
              **kwargs) -> Any:
        """ 
        Solve for the renormalization matrix :attr:`R` and correlation potential
        matrix :attr:`Lambda`.

        Parameters
        ----------
        one_shot : bool, optional
            True if the calcualtion is just one shot and not self consistent. 
            Default is False.
        embedding_param : list[dict], optional
            kwarg options to pass to :meth:`embedding.solve` for each cluster.
        kweight_param : dict, optional
            kwarg options to pass to :meth:`update_weights`.
        **kwargs
            kwarg options to pass to :meth:`root`.

        Returns
        -------
        x
            The flattened x vector of :attr:`Lambda` and :attr:`R`. If using 
            :func:`scipy.optimize.root` the :class:`scipy.optimize.OptimizeResult` 
            object will be returned.
        Also sets the self-consistent solutions :attr:`Lambda` and :attr:`R`.
        """
        
        if embedding_param is None:
            embedding_param = [dict() for i in range(self.n_clusters)]

        if one_shot:
            self.Lambda, self.R, _, _ = self.one_cycle(embedding_param, kweight_param)
            x = np.array( self._flatten_mat(self.Lambda, is_coeff_real=True) + self._flatten_mat(self.R, is_coeff_real=False) )
        
        else:
            x0 = np.array( self._flatten_mat(self.Lambda, is_coeff_real=True) + self._flatten_mat(self.R, is_coeff_real=False) )
            x = self.root(fun=self._target_function, 
                          x0=x0,
                          args=(embedding_param, kweight_param),
                          **kwargs)
        return x
        
    @property
    def Z(self) -> MFType:
        """
        list[dict[numpy.ndarray]] : Qausiparticle weight of each cluster.
        """
        Z = [dict() for i in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for bl, _ in self.gf_struct[i]:
                Z[i][bl] = self.R[i][bl] @ self.R[i][bl].conj().T
        return Z