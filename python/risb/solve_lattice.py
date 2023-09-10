from risb import sc
from risb.optimize import DIIS2
import numpy as np
from copy import deepcopy

# The structure and methods here have been modeled 
# after github.com/TRIQS/hartree_fock
class LatticeSolver:
    """
    Rotationally invariant slave-bosons (RISB) lattice solver with
    a local interaction on each cluster.

    Parameters
    ----------

    h0_k : dict of arrays
        Single-particle dispersion between local clusters. Each key 
        in dictionary must follow the gf_struct.
        FIXME Make it the full h0_k, and work out the h0_loc and add to
        h_int to make a h_loc = h0_loc + h_int, and then
        h0_kin_k = h0_k - h_0_loc.

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.
                 
    emb_solver : class
        The class that solves the embedding problem. It must already 
        store the local Hamiltonian, h_loc, on a cluster, have a method 
        ``set_h_emb(Lambda_c, D)`` to setup the impurity problem, a method
        ``solve(**emb_parameters)`` that solves the impurity problem, and 
        methods ``get_nf(block)`` and ``get_mcf(block)`` for the bath and 
        hybridization density matrices. See class ``EmbeddingAtomDiag``.
        
    kweight_solver : class
        The class that sets the integral weights at each k-point on the 
        lattice. It must have a method 
        ``update_weights(energies, **kweight_parameters)``, where the energies 
        are a dictionary with each key a list. See class ``SmearingKWeight``.

    symmeties : optional, list of functions
        Symmetry functions acting on the mean-field matrices.

    force_real : optional, bool
        True if the mean-field matrices are forced to be real

    R : optional, dict of arrays
        The unitary matrix from the f-electrons to the c-electrons at the
        mean-field level. Also called renormalization matrix. Each key in 
        dictionary must follow gf_struct. Defaults to the identity in each 
        block.

    Lambda : optional, dict of arrays
        The correlation potential matrix experienced by the f-electrons.
        Each key in dictionary must follow gf_struct. Defaults to the zero 
        matrix in each block.

    optimize_solver : optional, class
        The class that drives the self-consistent procedure. It must have 
        a method ``solve(fun, x0)``, where x0 is the initial guess vector, 
        and fun is the function to minimize, where fun=self.target_function 
        and returns x_new and x_error vectors). E.g., to use with scipy 
        use ``root(fun, x0, args=(False)``. Defaults to ``DIIS2``.

    error_root : optional, string
        At each self-consistent cycle, whether the error is 
        'root' : f1 and f2 root functions
        'recursion' : the difference between consecutive Lambda and R.
        Defaults to 'root'.

    """
    def __init__(self, 
                 h0_k,
                 gf_struct,
                 emb_solver, 
                 kweight_solver,
                 symmetries=[],
                 force_real=True,
                 R=None,
                 Lambda=None, 
                 optimize_solver=None,
                 error_root='root'):
        
        self._h0_k = h0_k
        self._gf_struct = gf_struct
        self._block_names = [bl for bl,_ in self._gf_struct]
        
        self._emb_solver = emb_solver
        self._kweight_solver = kweight_solver
        
        self._R = deepcopy(R)
        self._Lambda = deepcopy(Lambda)
        if (self._R is None) or (self._Lambda is None):
            [self._R, self._Lambda] = self.initialize_block_mf_matrices(self._gf_struct)
 
        self._symmetries = symmetries
        self._force_real = force_real

        self._optimize_solver = optimize_solver
        self._error_root = error_root

    @staticmethod
    def initialize_block_mf_matrices(gf_struct):
        R = dict()
        Lambda = dict()
        for bl, bsize in gf_struct:
            R[bl] = np.zeros((bsize,bsize))
            Lambda[bl] = np.zeros((bsize,bsize))
            np.fill_diagonal(R[bl], 1)
        return (R, Lambda)
    
    # Flatten and unflatten copied from triqs/hartree_fock because 
    # their method is much smarter than what I ever did
    def flatten(self, Lambda, R):
        x = []
        x.append([mat.flatten().real for mat in Lambda.values()])
        if self._force_real:
            x.append([mat.flatten().real for mat in R.values()])
        else:
            x.append([mat.flatten().view(float) for mat in R.values()])
        return np.array(x).flatten()
    
    def unflatten(self, x):
        Lambda = dict()
        R = dict()
        offset = 0
        for bl, bl_size in self._gf_struct:
            Lambda[bl] = x[list(range(offset, offset + bl_size**2))].reshape(bl_size, bl_size)
            offset += bl_size**2
        for bl, bl_size in self._gf_struct:
            if self._force_real:
                R[bl] = x[list(range(offset, offset + bl_size**2))].reshape(bl_size, bl_size)
                offset += bl_size**2
            else:
                R[bl] = x[list(range(offset, offset + bl_size**2))].view(complex).reshape(bl_size, bl_size)
                offset += 2*bl_size**2
        return Lambda, R
    
    def target_function(self, x, return_new=True):
        self._Lambda, self._R = self.unflatten(x)
        Lambda_new, R_new, f1, f2  = self.one_cycle()
        x_new = self.flatten(Lambda_new, R_new)
        
        if self._error_root == 'root':
            x_error = self.flatten(f2, f1)
        elif self._error_root == 'recursion':
            x_error = x - x_new
        else:
            raise ValueError('Unrecognized error functions for root !')
        
        if return_new:
            return x_new, x_error
        else:
            return x_error

    def one_cycle(self, emb_parameters=dict(), kweight_parameters=dict()):

        for function in self._symmetries:
            self._R = function(self.R)
            self._Lambda = function(self.Lambda)
                
        self._Lambda_c = dict()
        self._D = dict()
        self._rho_qp = dict()
        self._Nc = dict()
        self._Nf = dict()
        self._Mcf = dict()
        self._wks = dict()
        
        eig_qp = dict()
        vec_qp = dict()

        for bl in self._block_names:
            eig_qp[bl], vec_qp[bl] = sc.get_h_qp(self._R[bl], self._Lambda[bl], self._h0_k[bl])
        
        self._wks = self._kweight_solver.update_weights(eig_qp, **kweight_parameters)

        for bl in self._block_names:
            h0_R = sc.get_h0_R(self._R[bl], self._h0_k[bl], vec_qp[bl])

            self._rho_qp[bl] = sc.get_pdensity(vec_qp[bl], self._wks[bl])
            ke = sc.get_ke(h0_R, vec_qp[bl], self._wks[bl])
        
            self._D[bl] = sc.get_d(self._rho_qp[bl], ke)
            if self._force_real:
                self._D[bl] = self._D[bl].real
            self._Lambda_c[bl] = sc.get_lambda_c(self._rho_qp[bl], self._R[bl], self._Lambda[bl], self._D[bl])
        
        for function in self._symmetries:
            self._D = function(self._D)
            self._Lambda_c = function(self._Lambda_c)

        self._emb_solver.set_h_emb(self._Lambda_c, self._D)
        self._emb_solver.solve(**emb_parameters)

        for bl in self._block_names:
            self._Nf[bl] = self._emb_solver.get_nf(bl)
            self._Mcf[bl] = self._emb_solver.get_mcf(bl)
        
        for function in self._symmetries:
            self._Nf = function(self._Nf)
            self._Mcf = function(self._Mcf)

        f1 = dict()
        f2 = dict()
        for bl in self._block_names:
            f1[bl] = sc.get_f1(self._Mcf[bl], self._rho_qp[bl], self._R[bl])
            f2[bl] = sc.get_f2(self._Nf[bl], self._rho_qp[bl])
        
        Lambda = dict()
        R = dict()
        for bl in self._block_names:
            Lambda[bl] = sc.get_lambda(self._R[bl], self._D[bl], self._Lambda_c[bl], self._Nf[bl])
            R[bl] = sc.get_r(self._Mcf[bl], self._Nf[bl])
            
        for function in self._symmetries:
            Lambda = function(Lambda)
            R = function(R)

        return Lambda, R, f1, f2
    
    def solve(self, 
              one_shot=False, 
              tol=1e-12, 
              options={'maxiter': 1000}, 
              emb_parameters=dict(), 
              kweight_parameters=dict()):
        """ 
        Solve for the renormalization matrix ``R`` and correlation potential
        matrix ``Lambda``.

        Parameters
        ----------

        one_shot : optional, bool
            True if the calcualtion is just one shot and not self consistent. 
            Default is False.

        tol : optional, float
            Convergence tolerance to pass to ``optimize_root`` class.
            Default is 1e-12.

        options : optional, dict
            Additional options to pass to ``optimize_root`.

        emb_parameters : optional, dict
            Options to pass to ``emb_solver.solve``. Not implemented.

        kweight_parameters : optional, dict
            Options to pass to ``kweight_solver.update_weight``. Not implemented.

        """
        if one_shot:
            self._Lambda, self._R, _, _ = self.one_cycle(emb_parameters, kweight_parameters)
        
        else:
            if self._optimize_solver is None:
                self._optimize_solver = DIIS2()
            self._optimize_solver.solve(fun=self.target_function, 
                                        x0=self.flatten(self._Lambda, self._R), 
                                        tol=tol,
                                        options=options)
            #from scipy.optimize import root
            #root_finder = root(fun=self.target_function, x0=self.flatten(self._Lambda, self._R), args=(False), method='broyden1')

    @property
    def gf_struct(self):
        return self._gf_struct
    
    @property
    def Lambda(self):
        return self._Lambda
    
    @Lambda.setter
    def Lambda(self, value):
        self._Lambda = value
    
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self, value):
        self._R = value
        
    @property
    def rho(self):
        return self._Nc
    
    @property
    def rho_f(self):
        return self._Nf
    
    @property
    def rho_qp(self):
        return self._rho_qp
    
    @property
    def bath_coupling(self):
        return self._Lambda_c

    @property
    def hybrid_coupling(self):
        return self._D
        
    @property
    def Z(self):
        Z = dict()
        for bl in self._block_names:
            Z[bl] = self._R[bl] @ self._R[bl].conj().T
        return Z