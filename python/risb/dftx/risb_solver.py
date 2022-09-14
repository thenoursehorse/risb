import numpy
import copy

import risb.sc_cycle as sc

class risbSolver(object):
    def __init__(self, emb_solver, ish=0):

        self.emb_solver = emb_solver
        self.ish = ish

        self.Lambda = {}
        self.R = {}
        self.pdensity = {}
        self.ke = {}
        self.D = {}
        self.Lambda_c = {}
        self.Ec = {}
        self.Nf = {}
        self.Mcf = {}
        self.Nc = {}

    def one_cycle(self, SK, Lambda=None, R=None, sc_method='recursion'):
        if Lambda == None:
            self.Lambda = SK.Lambda[self.ish]
        else:
            self.Lambda = Lambda
        if R == None:
            self.R = SK.R[self.ish]
        else:
            self.R = R
        self.block_names = self.R.keys()

        # the quasiparticle density and kinetic energy from the SumK class
        self.pdensity = copy.deepcopy(SK.get_pdensity(ish=self.ish))
        self.ke = copy.deepcopy(SK.get_ke(ish=self.ish))
        
        # FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    self.pdensity[block] = numpy.diag(numpy.diag(self.pdensity[block]))
        #    self.ke[block] = numpy.diag(numpy.diag(self.ke[block]))
            
        # ensure symmetries are enforced to keep solver stable
        SK.symm_deg_mat(self.pdensity, ish=self.ish)
        SK.symm_deg_mat(self.ke, ish=self.ish)
        
        ## FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    self.pdensity[block] = numpy.diag(numpy.diag(self.pdensity[block]))
        #    self.ke[block] = numpy.diag(numpy.diag(self.ke[block]))
        
        # get the values for the impurity problem
        for block in self.block_names:
            self.D[block] = numpy.real(sc.get_d(self.pdensity[block], self.ke[block])) # FIXME assumes real
            self.Lambda_c[block] = numpy.real(sc.get_lambda_c(self.pdensity[block], self.R[block], self.Lambda[block], self.D[block]))
        
        SK.symm_deg_mat(self.D, ish=self.ish)
        SK.symm_deg_mat(self.Lambda_c, ish=self.ish)        
        
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
        
        SK.symm_deg_mat(self.Nf, ish=self.ish)
        SK.symm_deg_mat(self.Mcf, ish=self.ish)
        SK.symm_deg_mat(self.Nc, ish=self.ish)
        
        out_A = {}
        out_B = {}
        for block in self.block_names:
            # get new lambda and r
            if (sc_method == 'recursion') or (sc_method == 'fixed-point'):
                out_A[block] = sc.get_lambda(self.R[block], self.D[block], self.Lambda_c[block], self.Nf[block])
                out_B[block] = sc.get_r(self.Mcf[block], self.Nf[block])
            elif sc_method == "root":
                out_A[block] = sc.get_f1(self.Mcf[block], self.pdensity[block], self.R[block])
                out_B[block] = sc.get_f2(self.Nf[block], self.pdensity[block])
            else:
                raise ValueError("one_cycle of RISB: Implemented only for recursion, root problem, and fixed-point.")
        
        ## FIXME for when Lambda is diagonal but pdensity and ke are not, obv an approximation!
        #for block in self.block_names:
        #    out_A[block] = numpy.diag(numpy.diag(out_A[block]))
        #    out_B[block] = numpy.diag(numpy.diag(out_B[block]))
        
        SK.symm_deg_mat(out_A, ish=self.ish)
        SK.symm_deg_mat(out_B, ish=self.ish)

        return out_A, out_B

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