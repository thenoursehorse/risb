import numpy
from copy import deepcopy

import time

import risb.sc_cycle as sc

class risbSolver(object):
    def __init__(self, emb_solver, ish_inequiv=0):

        self.emb_solver = emb_solver
        self.ish_inequiv = ish_inequiv

        self.Lambda = {}
        self.R = {}
        self.F1 = {}
        self.F2 = {}
        
        self.pdensity = {}
        self.ke = {}
        self.D = {}
        self.Lambda_c = {}
        self.Ec = {}
        self.Nf = {}
        self.Mcf = {}
        self.Nc = {}

    def one_cycle(self, SK, Lambda=None, R=None):
        if Lambda == None:
            self.Lambda = deepcopy(SK.Lambda[self.ish_inequiv])
        else:
            self.Lambda = deepcopy(Lambda)
        if R == None:
            self.R = deepcopy(SK.R[self.ish_inequiv])
        else:
            self.R = deepcopy(R)
        self.block_names = self.R.keys()

        # the quasiparticle density and kinetic energy from the SumK class
        self.pdensity = deepcopy(SK.get_pdensity(ish_inequiv=self.ish_inequiv))
        self.ke = deepcopy(SK.get_ke(ish_inequiv=self.ish_inequiv))
        
        # FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    self.pdensity[block] = numpy.diag(numpy.diag(self.pdensity[block]))
        #    self.ke[block] = numpy.diag(numpy.diag(self.ke[block]))
            
        # ensure symmetries are enforced to keep solver stable
        SK.symm_deg_mat(self.pdensity, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.ke, ish_inequiv=self.ish_inequiv)
        
        ## FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    self.pdensity[block] = numpy.diag(numpy.diag(self.pdensity[block]))
        #    self.ke[block] = numpy.diag(numpy.diag(self.ke[block]))
        
        # get the values for the impurity problem
        for block in self.block_names:
            self.D[block] = numpy.real(sc.get_d(self.pdensity[block], self.ke[block])) # FIXME assumes real
            self.Lambda_c[block] = numpy.real(sc.get_lambda_c(self.pdensity[block], self.R[block], self.Lambda[block], self.D[block]))
        
        SK.symm_deg_mat(self.D, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.Lambda_c, ish_inequiv=self.ish_inequiv)        
        
        # set h_emb, solve
        psiS_size = self.emb_solver.get_psiS_size()
        self.emb_solver.set_h_emb(self.Lambda_c, self.D) # FIXME currently hard set to embeddingED impurity solver
        #self.Ec = self.emb_solver.solve(ncv = min(30,psiS_size), max_iter = 1000, tolerance=0)
        self.Ec = self.emb_solver.solve(ncv = min(30,psiS_size), max_iter = 10000, tolerance=1e-10)

        for block in self.block_names:
            # get the density matrix of the f-electrons, the 'hybridization', and the c-electrons
            self.Nf[block] = self.emb_solver.get_nf(block)
            self.Mcf[block] = self.emb_solver.get_mcf(block)
            self.Nc[block] = self.emb_solver.get_nc(block)
        
        SK.symm_deg_mat(self.Nf, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.Mcf, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.Nc, ish_inequiv=self.ish_inequiv)
        
        for block in self.block_names:
            # get new lambda and r
            self.Lambda[block] = sc.get_lambda(self.R[block], self.D[block], self.Lambda_c[block], self.Nf[block])
            self.R[block] = sc.get_r(self.Mcf[block], self.Nf[block])
            # the result of the root functions
            self.F1[block] = sc.get_f1(self.Mcf[block], self.pdensity[block], self.R[block])
            self.F2[block] = sc.get_f2(self.Nf[block], self.pdensity[block])
        
        ## FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    out_A[block] = numpy.diag(numpy.diag(out_A[block]))
        #    out_B[block] = numpy.diag(numpy.diag(out_B[block]))
        
        SK.symm_deg_mat(self.Lambda, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.R, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.F1, ish_inequiv=self.ish_inequiv)
        SK.symm_deg_mat(self.F1, ish_inequiv=self.ish_inequiv)

        return self.Lambda, self.R, self.F1, self.F2

    @property
    def Z(self):
        Z = dict()
        for block in self.block_names:
            Z[block] = numpy.dot(self.R[block], self.R[block].conj().transpose())
        return Z
    
    def overlap(self, Op):
        return self.emb_solver.overlap(Op)

    # FIXME this energy is wrong because it assumes there are no projectors
    # Is it just the energy of the impurity?
    def total_energy(self, h_loc):
        energy = self.overlap(h_loc)
        for block in self.block_names:
            energy += numpy.trace( numpy.dot(self.D[block], self.Mcf[block]) )
        return energy

    @property
    def total_density(self):
        out = 0
        for block in self.block_names:
            out += numpy.trace( self.Nc[block] )
        return out