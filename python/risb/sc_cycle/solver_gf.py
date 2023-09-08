import numpy as np
from triqs.gf import *
import risb.sc_cycle as sc
from risb.embedding_atom_diag import *
import triqs.utility.mpi as mpi
from copy import deepcopy

class Solver():
    """"""

    def __init__(self, beta, gf_struct, n_iw=1025, n_tau=10001, n_l=30, emb_solver=None):
        """
        Initialise the solver.
        Parameters
        ----------
        beta : scalar
               Inverse temperature.
        gf_struct : list of pairs [ (str,[int,...]), ...]
                    Structure of the Green's functions. It must be a
                    list of pairs, each containing the name of the
                    Green's function block as a string and a list of integer
                    indices.
                    For example: ``[ ('up', [0, 1, 2]), ('down', [0, 1, 2]) ]``.
        n_iw : integer, optional
               Number of Matsubara frequencies used for the Green's functions.
        n_tau : integer, optional
               Number of imaginary time points used for the Green's functions.
        n_l : integer, optional
            Number of legendre polynomials to use in accumulations of the Green's functions.
        emb_solver : class, optional
            The solver for the embedding space. Defaults to using AtomDiag within TRIQS.
        """

        if isinstance(gf_struct,dict):
            if mpi.is_master_node(): print("WARNING: RISB: gf_struct should be a list of pairs [ (str,[int,...]), ...], not a dict")
            gf_struct = [ [k, v] for k, v in gf_struct.items() ]

        self.beta = beta
        self.gf_struct = gf_struct
        self.n_iw = n_iw
        self.n_tau = n_tau
        self.n_l = n_l
        if emb_solver is None: 
            self.emb_solver = EmbeddingAtomDiag(self.gf_struct)
        else:
            self.emb_solver = emb_solver

        g_iw_list = []
        self.block_names = [block for block,ind in gf_struct]
        for block,ind in gf_struct:
            g_iw_list.append(GfImFreq(indices = ind, beta = beta, n_points = n_iw))

        self.G0_iw = BlockGf(name_list = self.block_names, block_list = g_iw_list)

        self.Delta_iw = self.G0_iw.copy()
        self.Delta_iw.zero()

        self.Sigma_iw = self.G0_iw.copy()
        self.Sigma_iw.zero()

        self.G_iw = self.G0_iw.copy()
        self.G_iw.zero()

        self.Gqp_iw = self.G0_iw.copy()
        self.Gqp_iw.zero()

        # Mean-field matrices
        self.R = dict()
        for block,ind in self.gf_struct:
            self.R[block] = np.zeros((len(ind),len(ind)))
            np.fill_diagonal(self.R[block], 1)
        
        self.Lambda = dict()
        for block,ind in self.gf_struct:
            self.Lambda[block] = np.zeros((len(ind),len(ind)))

        self.Z = deepcopy(self.R)
        self.D = deepcopy(self.Lambda)
        self.Lambda_c = deepcopy(self.Lambda)
        self.pdensity = deepcopy(self.Lambda)
        self.ke = deepcopy(self.Lambda)
        
        self.density = deepcopy(self.Lambda)

    def solve(self, **params_kw):
        """
        Solve the impurity problem using RISB.

        Parameters
        ----------
        params_kw : dict {'param':value} that is passed to the solver.
                    Required parameter is:
                        * `h_int` (many body operator): The local Hamiltonian of the impurity.
                    Other parameters are:
                        * `mu`  (double): The chemical potential
                        * `e0` (list of matrices): The quadratic terms in h_int to subtract off from Sigma_iw
                        * `R` (list of matrices): The first guess for R
                        * `Lambda` (list of matrices): The first guess for Lambda
        """

        if mpi.is_master_node(): 
            print("╦═╗╦╔═╗┌┐ ┬ ┬")
            print("╠╦╝║╚═╗├┴┐└┬┘")
            print("╩╚═╩╚═╝└─┘ ┴ ")
            #print(" _  ___  __  _    ")
            #print("|_)  |  (_  |_)   ")   
            #print("| \ _|_ __) |_) \/")
            #print("                / ")  
            #print("██████  ██ ███████ ██████ ██    ██ ")
            #print("██   ██ ██ ██      ██   ██ ██  ██  ")
            #print("██████  ██ ███████ ██████   ████  ")
            #print("██   ██ ██      ██ ██   ██   ██  ")
            #print("██   ██ ██ ███████ ██████    ██")
            print()

        h_int = params_kw['h_int']
        
        try:
            mu = params_kw['mu']
        except KeyError:
            mu = 0

        try:
            self.R = params_kw['R']
        except KeyError:
            self.R = self.R
        
        try:
            self.Lambda = params_kw['Lambda']
        except KeyError:
            self.Lambda = self.Lambda
        
        try:
            e0 = params_kw['e0']
        except KeyError:
            e0 = deepcopy(self.R)
            for block in self.block_names:
                e0[block] = 0.0 * e0[block]
        
        print("The chemical potential of the problem:",mu)
        print()

        print("The local Hamiltonian of the problem:",h_int)
        print()

        for block in self.block_names:

            # Hybridization function
            self.Delta_iw = delta(self.G0_iw[block])

            # Quasiparticle local density matrix and kinetic energy matrix
            self.pdensity[block] = sc.get_pdensity_gf(self.G_iw[block], self.R[block]);
            self.ke[block] = np.real( sc.get_ke_gf(self.G_iw[block], self.Delta_iw, self.R[block]) )

            # Impurity hybridization matrix and math energy matrix
            self.D[block] = sc.get_d(self.pdensity[block], self.ke[block])
            self.Lambda_c[block] = sc.get_lambda_c(self.pdensity[block], self.R[block], self.Lambda[block], self.D[block])

        self.emb_solver.set_h_emb(h_int, self.Lambda_c, self.D, mu)
        self.emb_solver.solve()

        # Calculate a new R and Lambda matrix
        for block in self.block_names:
            Nf = self.emb_solver.get_nf(block)
            #Nc = self.emb_solver.get_nc(block)
            Mcf = self.emb_solver.get_mcf(block)

            self.Lambda[block] = sc.get_lambda(self.R[block], self.D[block], self.Lambda_c[block], Nf)
            self.R[block] = sc.get_r(Mcf, Nf)

        for block in self.block_names:
            self.Sigma_iw[block] = sc.get_sigma_z(self.G_iw[block].mesh, self.R[block], self.Lambda[block], mu, e0[block])

        self.G_iw = dyson(G0_iw = self.G0_iw, Sigma_iw = self.Sigma_iw)

        #for block in self.block_names:
        #    self.Gqp_iw[block] = sc.get_gqp_z(self.G_iw[block], self.R[block])

        for block in self.block_names:
            self.Z[block] = np.dot(self.R[block], self.R[block])

        # Calculate new g0_iw: (illustrative)
        #self.G0_iw = dyson(G_iw = self.G_iw, Sigma_iw = self.Sigma_iw)

        for block in self.block_names:
            self.density[block] = self.emb_solver.get_nc(block)


    def density(self):
        return self.density

    def total_density(self):
        density = self.density()
        result = 0
        for block in self.block_names:
            result += np.trace(density[block])
        return result
