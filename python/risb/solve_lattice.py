from risb import sc
import numpy as np
from copy import deepcopy

class LatticeSolver:
    def __init__(self, 
                 h0_k,
                 gf_struct,
                 emb_solver, 
                 kweight_solver,
                 R=None,
                 Lambda=None, 
                 symmetries=[],
                 force_real=True):
        
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


    @staticmethod
    def initialize_block_mf_matrices(gf_struct):
        R = dict()
        Lambda = dict()
        for bl, bsize in gf_struct:
            R[bl] = np.zeros((bsize,bsize))
            Lambda[bl] = np.zeros((bsize,bsize))
            np.fill_diagonal(R[bl], 1)
        return (R, Lambda)

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
        
        Lambda = dict()
        R = dict()
        for bl in self._block_names:
            Lambda[bl] = sc.get_lambda(self._R[bl], self._D[bl], self._Lambda_c[bl], self._Nf[bl])
            R[bl] = sc.get_r(self._Mcf[bl], self._Nf[bl])
        
        for function in self._symmetries:
            Lambda = function(Lambda)
            R = function(R)

        return Lambda, R
    
    def solve(self, one_shot=False, n_cycles=25, tol=1e-6):
        norm = 0
        for iter in range(n_cycles):
            R_old = deepcopy(self._R)
            Lambda_old = deepcopy(self._Lambda)
            
            self._Lambda, self._R = self.one_cycle()
            
            for bl in self._block_names:
                # FIXME use allclose instead?
                norm += np.linalg.norm(self._R[bl] - R_old[bl])
                norm += np.linalg.norm(self._Lambda[bl] - Lambda_old[bl])
            
            if norm < tol:
                break
    
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