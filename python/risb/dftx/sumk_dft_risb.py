
from types import *
import numpy
import copy
from warnings import warn

import triqs.utility.mpi as mpi
from h5 import *
import triqs.utility.dichotomy as dichotomy
from triqs.gf import *

from triqs_dft_tools.sumk_dft_tools import SumkDFT

from scipy.special import factorial
from scipy.special import erfc
from scipy.special import hermite

from scipy.optimize import minimize_scalar

class SumkDFTRISB(SumkDFT):
    """This class extends the SumK method (ab-initio code and triqs) to use RISB"""

    def __init__(self, beta=5, smearing='gaussian', N_mp=1, **kwds):
        super().__init__(**kwds)

        self.beta = beta
        self.smearing = smearing
        self.N_mp = N_mp
        
        # Some set up fo RISB parameters
        self.R = [{} for ish in range(self.n_inequiv_shells)]
        self.Lambda = [{} for ish in range(self.n_inequiv_shells)]
        self.R_sumk = [{} for ish in range(self.n_inequiv_shells)]
        self.Lambda_sumk = [{} for ish in range(self.n_inequiv_shells)]
        self.pdensity = [{} for ish in range(self.n_inequiv_shells)]
        self.pdensity_sumk = [{} for ish in range(self.n_inequiv_shells)]
        self.ke = [{} for ish in range(self.n_inequiv_shells)]
        self.ke_sumk = [{} for ish in range(self.n_inequiv_shells)]

        # The identitity of the correlated subspace in the sumk representation
        self.identity_sumk = [{} for ish in range(self.n_inequiv_shells)]
        
    def fermi_distribution(self, eks, mu=0, beta=5):
        return 1.0 / (numpy.exp(beta * (eks - mu)) + 1.0)

    def gaussian_distribution(self, eks, mu=0, beta=5):
        return 0.5 * erfc( beta * (eks - mu) )
        
    def A_n(self, n):
        return (-1)**n / ( factorial(n) * 4**n * numpy.sqrt(numpy.pi) )
    
    def methfessel_paxton(self, eks, mu=0, beta=5, N=1):
        x = beta * (eks - mu)
        
        S = 0.5 * erfc(x) # S_0
        for n in range(1,N+1):
            H_n = hermite(2*n-1)
            S += self.A_n(n) * H_n(x) * numpy.exp(-x**2)
        return S

    def fweights(self, eks, mu, beta):
        if self.smearing == 'fermi':
            return self.fermi_distribution(eks, mu, beta)
        elif self.smearing == 'gaussian':
            return self.gaussian_distribution(eks, mu, beta)
        elif self.smearing == 'methfessel-paxton':
            return self.methfessel_paxton(eks, mu, beta, self.N_mp)
        else:
            mpi.report('Warning: No smearing specified, defaulting to Gaussian')
            return self.gaussian_distribution(eks, mu, beta)

################
# CORE FUNCTIONS FOR RISB
################

    def downfold_matrix(self, ik, ish, bname, matrix_to_downfold, shells='corr', ir=None):
        # get spin index for proj. matrices
        isp = self.spin_names_to_ind[self.SO][bname]
        n_orb = self.n_orbitals[ik, isp]
        if shells == 'corr':
            dim = self.corr_shells[ish]['dim']
            projmat = self.proj_mat[ik, isp, ish, 0:dim, 0:n_orb]
        elif shells == 'all':
            if ir is None:
                raise ValueError("downfold: provide ir if treating all shells.")
            dim = self.shells[ish]['dim']
            projmat = self.proj_mat_all[ik, isp, ish, ir, 0:dim, 0:n_orb]
        elif shells == 'csc':
            projmat = self.proj_mat_csc[ik, isp, :, 0:n_orb]
        
        matrix_downfolded = numpy.dot(numpy.dot(projmat, matrix_to_downfold), projmat.conjugate().transpose())
        
        return matrix_downfolded

    
    def upfold_matrix(self, ik, ish, bname, matrix_to_upfold, shells='corr', ir=None):
        # get spin index for proj. matrices
        isp = self.spin_names_to_ind[self.SO][bname]
        n_orb = self.n_orbitals[ik, isp]
        if shells == 'corr':
            dim = self.corr_shells[ish]['dim']
            projmat = self.proj_mat[ik, isp, ish, 0:dim, 0:n_orb]
        elif shells == 'all':
            if ir is None:
                raise ValueError("upfold: provide ir if treating all shells.")
            dim = self.shells[ish]['dim']
            projmat = self.proj_mat_all[ik, isp, ish, ir, 0:dim, 0:n_orb]
        elif shells == 'csc':
            projmat = self.proj_mat_csc[ik, isp, 0:n_orb, 0:n_orb]
        
        matrix_upfolded = numpy.dot(numpy.dot(projmat.conjugate().transpose(), matrix_to_upfold), projmat)

        return matrix_upfolded
    
    def rotloc_matrix(self, icrsh, matrix_to_rotate, direction, shells='corr'):
        assert ((direction == 'toLocal') or (direction == 'toGlobal')
                ), "rotloc: Give direction 'toLocal' or 'toGlobal'."
        matrix_rotated = copy.deepcopy(matrix_to_rotate)
        if shells == 'corr':
            rot_mat_time_inv = self.rot_mat_time_inv
            rot_mat = self.rot_mat
        elif shells == 'all':
            rot_mat_time_inv = self.rot_mat_all_time_inv
            rot_mat = self.rot_mat_all

        if direction == 'toGlobal':

            if (rot_mat_time_inv[icrsh] == 1) and self.SO:
                matrix_rotated = numpy.dot(rot_mat[icrsh].conjugate, numpy.dot(matrix_rotated.transpose(), rot_mat[icrsh].transpose() ) )
            else:
                matrix_rotated = numpy.dot(rot_mat[icrsh], numpy.dot(matrix_rotated, rot_mat[icrsh].conjugate().transpose() ) )

        elif direction == 'toLocal':

            if (rot_mat_time_inv[icrsh] == 1) and self.SO:
                matrix_rotated = numpy.dot(rot_mat[icrsh].transpose(), numpy.dot(matrix_rotated.transpose(), rot_mat[icrsh].conjugate() ) )
            else:
                matrix_rotated = numpy.dot(rot_mat[icrsh].conjugate().transpose(), numpy.dot(matrix_rotated, rot_mat[icrsh] ) )
        return matrix_rotated
   
#    def flatten_deg_mat(self, mat_to_flat):
#        vec = []
#        deg = []
#        for ish,mat in enumerate(mat_to_flat): # over inequivalent shells
#            for degsh in self.deg_shells[ish]:
#                key = degsh[0]
#                for i,j in numpy.ndindex(mat[key].shape):
#                    vec.append(mat[key][i,j])
#                    deg.append(len(degsh))
#        return vec, deg
#
#    def flatten_mf_mat(self, Lambda, R):
#        return numpy.append(self.flatten_deg_mat(Lambda)[0],
#                            self.flatten_deg_mat(R)[0])
#
#    def construct_mf_mat(self, vec):
#        counter = 0
#        Lambda = copy.deepcopy(self.Lambda)
#        R = copy.deepcopy(self.R)
#        for ish in range(self.n_inequiv_shells):
#            for degsh in self.deg_shells[ish]:
#                for key in degsh:
#                    Lambda[ish][key] = vec[counter]
#                counter += 1
#        for ish in range(self.n_inequiv_shells):
#            for degsh in self.deg_shells[ish]:
#                for key in degsh:
#                    R[ish][key] = vec[counter]
#                counter += 1
#        return Lambda, R

    def symm_deg_mat(self, mat_to_symm, ish=0):
        for degsh in self.deg_shells[ish]:
            # ss will hold the averaged orbitals in the basis where the
            # blocks are all equal
            ss = None
            n_deg = len(degsh)
            dtype = dict()
            for key in degsh:                
                # In case the matrix is real and the symmetrization is complex
                dtype[key] = mat_to_symm[key].dtype
                mat_to_symm[key] = mat_to_symm[key].astype('complex128')
                
                if ss is None:
                    ss = copy.deepcopy( mat_to_symm[key] )
                    ss.fill(0) # initialize to zero
                    helper = copy.deepcopy( ss )
                # get the transformation matrix
                if isinstance(degsh, dict):
                    v, C = degsh[key]
                else:
                    # for backward compatibility, allow degsh to be a list
                    v = numpy.eye(ss.shape[0])
                    C = False
                # the helper is in the basis where the blocks are all equal
                helper = numpy.dot(numpy.dot(v.conjugate().transpose(), mat_to_symm[key]), v)
                if C:
                    helper = helper.transpose()
                # average over all shells
                ss += helper / (1.0 * n_deg)
            
            # now put back the averaged gf to all shells
            for key in degsh:
                if isinstance(degsh, dict):
                    v, C = degsh[key]
                else:
                    # for backward compatibility, allow degsh to be a list
                    v = numpy.eye(ss.shape[0])
                    C = False
                if C:
                    mat_to_symm[key] = numpy.dot(numpy.dot(v, ss.transpose()), v.conjugate().transpose())
                else:
                    mat_to_symm[key] = numpy.dot(numpy.dot(v, ss), v.conjugate().transpose())
            
                mat_to_symm[key] = mat_to_symm[key].astype(dtype[key])
         
    # FIXME
    # hack to only select the orbitals in the solver space for the sumk space
    # e.g., if used block_structure.pick_gf_struct_solver
    def mat_remove_non_solvers_in_sumk(self, mat, icrsh=0):
        ish = self.corr_to_inequiv[icrsh]
        mat_out = self.block_structure.convert_matrix(G=mat,
                                            ish_from=icrsh,
                                            ish_to=ish, 
                                            space_from='sumk',
                                            space_to='solver',
                                            show_warnings=True)
        
        mat_out = self.block_structure.convert_matrix(G=mat_out,
                                            ish_from=ish,
                                            ish_to=icrsh,
                                            space_from='solver',
                                            space_to='sumk',
                                            show_warnings=True)
        
        return mat_out

    def mat_solver_to_sumk(self, mat, icrsh=0):
        ish = self.corr_to_inequiv[icrsh]
        mat_out = self.block_structure.convert_matrix(
                G=mat,
                ish_from=ish,
                ish_to=icrsh,
                space_from='solver',
                space_to='sumk',
                show_warnings=True)
        return mat_out
    
    def mat_sumk_to_solver(self, mat, icrsh=0):
        ish = self.corr_to_inequiv[icrsh]
        mat_out = self.block_structure.convert_matrix(
                G=mat,
                ish_from=icrsh,
                ish_to=ish,
                space_from='sumk',
                space_to='solver',
                show_warnings=True)
        return mat_out

    # The identity in the truncated solver space
    def set_identity_sumk(self):
        for ish in range(self.n_inequiv_shells):
            icrsh = self.corr_to_inequiv[ish]
            self.identity_sumk[ish] = self.block_structure.create_matrix(ish=ish, space='sumk')
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                self.identity_sumk[ish][sp] = numpy.eye(self.corr_shells[icrsh]['dim'], dtype=numpy.complex)
            self.identity_sumk[ish] = self.mat_remove_non_solvers_in_sumk(mat=self.identity_sumk[ish], icrsh=icrsh)
    
    # From solver structure to sumk structure in local frame
    def set_R_Lambda_sumk(self, Lambda=None, R=None):
        if Lambda is None:
            Lambda = self.Lambda
        if R is None:
            R = self.R 
        
        for ish in range(self.n_inequiv_shells):
            icrsh = self.inequiv_to_corr[ish]
            self.Lambda_sumk[ish] = self.mat_solver_to_sumk(self.Lambda[ish], icrsh=icrsh)
            self.R_sumk[ish] = self.mat_solver_to_sumk(self.R[ish], icrsh=icrsh)
        return self.Lambda_sumk, self.R_sumk
    
    def initialize_R_Lambda(self, h_ksr_loc=None, random=True, zero_Lambda=False):
        if h_ksr_loc is None:
            h_ksr_loc = self.get_h_ksr_loc() # for all correlated shells

        for ish in range(self.n_inequiv_shells):
            icrsh = self.inequiv_to_corr[ish]

            self.R[ish] = self.block_structure.create_matrix(ish=ish, space='solver')
            
            for key in self.R[ish].keys():
                numpy.fill_diagonal(self.R[ish][key], 1)
            
            if random:
                for key in self.R[ish].keys():    
                    self.R[ish][key] *= numpy.random.rand(
                                            self.R[ish][key].shape[0],
                                            self.R[ish][key].shape[1])

            # transform the local terms from sumk blocks to the solver blocks
            self.Lambda[ish] = self.block_structure.convert_matrix(
                    G=h_ksr_loc[icrsh],
                    ish_from=icrsh,
                    ish_to=ish,
                    space_from='sumk',
                    space_to='solver',
                    show_warnings=True)
            
            if random:
                for key in self.Lambda[ish].keys():
                    self.Lambda[ish][key] *= 2. * numpy.random.rand(
                                                    self.Lambda[ish][key].shape[0],
                                                    self.Lambda[ish][key].shape[1])
                        
            # Make sure Lambda is real (discard imaginary)
            for key in self.Lambda[ish].keys():
                self.Lambda[ish][key] = self.Lambda[ish][key].astype(numpy.float_) # FIXME
                if zero_Lambda:
                    self.Lambda[ish][key][:] = 0.0

            # Symmetrize
            self.symm_deg_mat(self.R[ish], ish=ish)
            self.symm_deg_mat(self.Lambda[ish], ish=ish)
        
        # Store in the sumk block space as well
        self.set_R_Lambda_sumk(Lambda=self.Lambda, R=self.R)

        # Set the identity in this space
        self.set_identity_sumk()

        return self.Lambda, self.R
    
    # Kohn-Sham eigenenergies
    def get_h_ks_k(self, ik):
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        h_ks_k = dict()
        for sp in spn:
            n_orb = self.n_orbitals[ik, ntoi[sp]]
            h_ks_k[sp] = copy.deepcopy( self.hopping[ik, ntoi[sp], 0:n_orb, 0:n_orb] )
        return h_ks_k       

    # the W restricted Kohn-Sham Hamiltonian (ksr) at a specified k-pt (in the projected space) in the local frame
    # for each correlated shell (not inequivalent, just all of them)
    def get_h_ksr_k(self, ik):
        h_ksr_k = [{} for icrsh in range(self.n_corr_shells)]
        for icrsh in range(self.n_corr_shells):
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                ind = self.spin_names_to_ind[self.corr_shells[icrsh]['SO']][sp]
                n_orb = self.n_orbitals[ik, ind]
                MMat = self.hopping[ik, ind, 0:n_orb, 0:n_orb]
                h_ksr_k[icrsh][sp] = self.downfold_matrix(ik=ik, ish=icrsh, bname=sp, matrix_to_downfold=MMat)

        # symmetrisation:
        if self.symm_op != 0:
            h_ksr_k = self.symmcorr.symmetrize(h_ksr_k)
        
        # rotate to local frame
        if self.use_rotations:
            for icrsh in range(len(h_ksr_k)):
                for block in h_ksr_k[icrsh].keys():
                    h_ksr_k[icrsh][block] = self.rotloc_matrix(icrsh, h_ksr_k[icrsh][block], direction='toLocal')
    
        for icrsh in range(self.n_corr_shells):
            h_ksr_k[icrsh] = self.mat_remove_non_solvers_in_sumk(mat=h_ksr_k[icrsh], icrsh=icrsh)
        
        return h_ksr_k
    
    # the local parts of the W restricted Kohn-Sham Hamiltonian (ksr) in the local frame
    # for each correlated shell (not inequivalent, just all of them)
    def get_h_ksr_loc(self):
        if not hasattr(self, "h_ksr_loc"):
            self.h_ksr_loc = [{} for icrsh in range(self.n_corr_shells)]
            for icrsh in range(len(self.h_ksr_loc)):
                for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                    self.h_ksr_loc[icrsh][sp] = numpy.zeros([self.corr_shells[icrsh]['dim'],self.corr_shells[icrsh]['dim']], numpy.complex_)

            # do the integral
            ikarray = numpy.array(list(range(self.n_k)))
            for ik in mpi.slice_array(ikarray):
                h_ksr_k = self.get_h_ksr_k(ik=ik)
                for icrsh in range(len(self.h_ksr_loc)):
                    for block in self.h_ksr_loc[icrsh].keys():
                        self.h_ksr_loc[icrsh][block] += self.bz_weights[ik] * h_ksr_k[icrsh][block]

            # collect data from mpi:
            for icrsh in range(len(self.h_ksr_loc)):
                for block in self.h_ksr_loc[icrsh].keys():
                    self.h_ksr_loc[icrsh][block] = mpi.all_reduce(mpi.world, self.h_ksr_loc[icrsh][block], lambda x, y: x + y)
            mpi.barrier()
        
            # already symmetrized from h_ksr_k
        
            # already in local frame
        
        return self.h_ksr_loc

    # Kinetic part of Kohn-Sham Hamiltonian 
    def get_h_ks_kin_k(self, ik):
        h_ks_k = self.get_h_ks_k(ik=ik)
        h_ksr_loc = self.get_h_ksr_loc()
        
        h_ks_kin_k = copy.deepcopy(h_ks_k)
        # H - sum_i H^loc
        for icrsh in range(self.n_corr_shells):
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                h_ks_kin_k[sp] -= self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=h_ksr_loc[icrsh][sp] )
                #h_ks_kin_k[sp] -= self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=self.dc_imp[ish][sp] )

        return h_ks_kin_k
    
    def get_R_bl_uncorr_k(self, ik):
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        R_bl_uncorr_k = dict()
        for sp in spn:
            n_orb = self.n_orbitals[ik, ntoi[sp]]
            R_bl_uncorr_k[sp] = numpy.eye(n_orb, dtype=numpy.complex_)
        
        for icrsh in range(self.n_corr_shells):
            ish = self.corr_to_inequiv[icrsh]
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                R_bl_uncorr_k[sp] -= self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=self.identity_sumk[ish][sp] )
        
        return R_bl_uncorr_k

    # sumk objects are in 'global' frame
    # solver objects are in 'local' frame
    def get_R_bl_k(self, ik, R=None):
        if R is None:
            R = self.R_sumk

        # FIXME does this have to rotate each corr shell individually?
        # rotate to global frame
        #if self.use_rotations:
        #    for ish in range(len(R)):
        #        icrsh = self.inequiv_to_corr[ish]
        #        for block in R[ish].keys():
        #            R[ish][block] = self.rotloc_matrix(icrsh, R[ish][block], direction='toGlobal')

        
        R_bl_k = self.get_R_bl_uncorr_k(ik=ik)

        for icrsh in range(self.n_corr_shells):
            ish = self.corr_to_inequiv[icrsh]
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                R_bl_k[sp] += self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=R[ish][sp] )
        
        return R_bl_k

    def get_Lambda_bl_k(self, ik, Lambda=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        
        # rotate to global frame
        #if self.use_rotations:
        #    for ish in range(len(Lambda)):
        #        icrsh = self.inequiv_to_corr[ish]
        #        for block in Lambda[ish].keys():
        #            Lambda[ish][block] = self.rotloc_matrix(icrsh, Lambda[ish][block], direction='toGlobal')
        
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        Lambda_bl_k = dict()
        for sp in spn:
            n_orb = self.n_orbitals[ik, ntoi[sp]]
            Lambda_bl_k[sp] = numpy.zeros(shape=(n_orb,n_orb), dtype=numpy.complex_)

        for icrsh in range(self.n_corr_shells):
            ish = self.corr_to_inequiv[icrsh]
            for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
                Lambda_bl_k[sp] += self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=Lambda[ish][sp] )
        
        return Lambda_bl_k

    # The quasiparticle Hamiltonian in the Bloch space
    def get_h_bl_qp_k(self, ik, Lambda=None, R=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        
        h_ks_kin_k = self.get_h_ks_kin_k(ik=ik)
        
        h_bl_qp_k = dict()
        R_bl_k = self.get_R_bl_k(ik=ik, R=R)
        for sp in R_bl_k:
            h_bl_qp_k[sp] = numpy.dot( R_bl_k[sp], numpy.dot(h_ks_kin_k[sp], R_bl_k[sp].conj().T) )

        Lambda_bl_k = self.get_Lambda_bl_k(ik=ik, Lambda=Lambda)
        for sp in Lambda_bl_k:
            h_bl_qp_k[sp] += Lambda_bl_k[sp]
        
        # below will be wrong because mu is in quasiparticle space
        # and dc_imp is included in the impurity
        #for icrsh in range(self.n_corr_shells):
        #    for sp in h_bl_qp_k:
        #       h_bl_qp_k[sp] -= self.upfold_matrix( ik=ik, ish=icrsh, bname=sp, matrix_to_upfold=self.dc_imp[icrsh][sp] ) # would be multiplied by R etc
        #       h_bl_qp_k[sp] -= mu * numpy.dot( R_bl_k[sp], R_bl_k[sp].conj().T)

        return h_bl_qp_k
    
    # The lopsided quasiparticle kinetic energy in the Bloch space
    def get_h_bl_qp_kin_k(self, ik, R=None, lopsided=True):
        if R is None:
            R = self.R_sumk
        
        h_ks_kin_k = self.get_h_ks_kin_k(ik=ik)
        
        h_bl_qp_kin_k = dict()
        R_bl_k = self.get_R_bl_k(ik=ik, R=R)
        for sp in R_bl_k:
            if lopsided:
                h_bl_qp_kin_k[sp] = numpy.dot(h_ks_kin_k[sp], R_bl_k[sp].conj().T)
            else:
                h_bl_qp_kin_k[sp] = numpy.dot(R_bl_k[sp], numpy.dot(h_ks_kin_k[sp], R_bl_k[sp].conj().T) )
        
        return h_bl_qp_kin_k

    def get_mesh(self, iw_or_w='iw', beta=None, mesh=None): 
        if (iw_or_w != "iw") and (iw_or_w != "w"):
            raise ValueError("get_G_bl_qp_kw: Implemented only for Re/Im frequency functions.")
        if iw_or_w == "iw":
            if beta is None:
                raise ValueError("get_G_bl_qp_kw: Give the beta for the lattice GfImFreq.")
            # Default number of Matsubara frequencies
            mesh = MeshImFreq(beta=beta, S='Fermion', n_max=1025)
        elif iw_or_w == "w":
            if mesh is None:
                raise ValueError("get_G_bl_qp_kw: Give the mesh=(om_min,om_max,n_points) for the lattice GfReFreq.")
            mesh = MeshReFreq(mesh[0], mesh[1], mesh[2])
        return mesh

    # Construct from h_bl_qp
    def get_G_bl_qp_kw(self, ik, Lambda=None, R=None, mu=None, iw_or_w="iw", beta=None, broadening=None, mesh=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
        
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        
        if broadening is None:
            if mesh is None:
                broadening = 0.01
            else:  # broadening = 2 * \Delta omega, where \Delta omega is the spacing of omega points
                broadening = 2.0 * ((mesh[1] - mesh[0]) / (mesh[2] - 1))
    
        mesh = self.get_mesh(iw_or_w=iw_or_w, beta=beta, mesh=mesh)
         
        # Set up gf
        block_structure = [
            list(range(self.n_orbitals[ik, ntoi[sp]])) for sp in spn]
        gf_struct = [(spn[isp], block_structure[isp])
                     for isp in range(self.n_spin_blocks[self.SO])]
        block_ind_list = [block for block, inner in gf_struct]
        if iw_or_w == "iw":
            glist = lambda: [GfImFreq(indices=inner, mesh=mesh)
                             for block, inner in gf_struct]
        elif iw_or_w == "w":
            glist = lambda: [GfReFreq(indices=inner, mesh=mesh)
                             for block, inner in gf_struct]
        G_bl_qp = BlockGf(name_list=block_ind_list,
                         block_list=glist(), make_copies=False)
        G_bl_qp.zero()

        if iw_or_w == "iw":
            G_bl_qp << iOmega_n
        elif iw_or_w == "w":
            G_bl_qp << Omega + 1j * broadening

        h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)
        for sp in h_bl_qp_k:
            G_bl_qp[sp] -= (h_bl_qp_k[sp] - numpy.identity(h_bl_qp_k[sp].shape[0], numpy.complex_)*mu)

        G_bl_qp.invert()
        return G_bl_qp
   
    # Construct by getting G_bl_qp_kw and multiplying by R_bl
    def get_G_bl_kw(self, ik, Lambda=None, R=None, mu=None, iw_or_w="iw", beta=None, broadening=None, mesh=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta

        G_bl_qp = self.get_G_bl_qp_kw(ik=ik, Lambda=Lambda, R=R, mu=mu, iw_or_w=iw_or_w, beta=beta, broadening=broadening, mesh=mesh)
        G_bl = BlockGf(name_block_generator = G_bl_qp, make_copies = True) # Don't need to zero because will be overridden
        R_bl_k = self.get_R_bl_k(ik=ik, R=R)
        for sp in R_bl_k:
            G_bl[sp].from_L_G_R( R_bl_k[sp].conjugate().T, G_bl[sp], R_bl_k[sp])

        return G_bl
    
    # Construct by summing G_bl_qp_kw and projecting down
    # R and Lambda are in 'global' coordinate system
    # If transform to solver blocks, rotate into 'local' coordinate system
    def get_G_loc_qp(self, Lambda=None, R=None, mu=None, iw_or_w='iw', beta=None, broadening=None, mesh=None,
                      transform_to_solver_blocks=True, show_warnings=True):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
        
        loc_mesh = self.get_mesh(iw_or_w=iw_or_w, beta=beta, mesh=mesh)
        if iw_or_w == "iw":
            G_loc_qp = [BlockGf(name_block_generator=[(block, GfImFreq(indices=inner, mesh=loc_mesh)) for block, inner in self.gf_struct_sumk[ish]],
                                     make_copies=False) for ish in range(self.n_inequiv_shells)]
        elif iw_or_w == "w":
            G_loc_qp = [BlockGf(name_block_generator=[(block, GfReFreq(indices=inner, mesh=loc_mesh)) for block, inner in self.gf_struct_sumk[ish]],
                                     make_copies=False) for ish in range(self.n_inequiv_shells)]
                
        for ish in range(self.n_inequiv_shells):
            G_loc_qp[ish].zero()

        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            G_bl_qp = self.get_G_bl_qp_kw(ik=ik, Lambda=Lambda, R=R, mu=mu, iw_or_w=iw_or_w, beta=beta, broadening=broadening, mesh=mesh)
            G_bl_qp *= self.bz_weights[ik]

            for ish in range(self.n_inequiv_shells):
                tmp = G_loc_qp[ish].copy()
                for bname, gf in tmp:
                    tmp[bname] << self.downfold(ik, ish, bname, G_bl_qp[bname], gf)
                G_loc_qp[ish] += tmp
                
        # Collect data from mpi
        for ish in range(self.n_inequiv_shells):
            G_loc_qp[ish] << mpi.all_reduce(mpi.world, G_loc_qp[ish], lambda x, y: x + y)
        mpi.barrier()

        ## FIXME symmcorr.symmetrize is over all correlated shells, so this needs to be fixed
        ## Symmetrize
        #if self.symm_op != 0:
        #    G_loc_qp = self.symmcorr.symmetrize(G_loc_qp)
        
        #if self.use_rotations:
        #    for ish in range(self.n_inequiv_shells):
        #        icrsh = self.inequiv_to_corr[ish]
        #        for bname, gf in G_loc_qp[ish]:
        #            G_loc_qp[ish][bname] << self.rotloc(ish, gf, direction='toLocal')
        
        if transform_to_solver_blocks:        
            for ish in range(self.n_inequiv_shells):
                G_loc_qp[ish] = self.block_structure.convert_gf(
                        G=G_loc_qp[ish],
                        ish_from=self.inequiv_to_corr[ish],
                        ish_to=ish,
                        space_from='sumk',
                        space_to='solver',
                        show_warnings = show_warnings)
        
        return G_loc_qp

    # Construct by calculating G_loc_qp_w and multiplying by local R
    def get_G_loc(self, Lambda=None, R=None, mu=None, iw_or_w='iw', beta=None, broadening=None, mesh=None,
                      transform_to_solver_blocks=True, show_warnings=True):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
    
        G_loc_qp = self.get_G_loc_qp(Lambda=Lambda, R=R, mu=mu, iw_or_w=iw_or_w, beta=beta, broadening=broadening, mesh=mesh,
                                       transform_to_solver_blocks=False, show_warnings=show_warnings)
       
        G_loc = copy.deepcopy(G_loc_qp)

        for ish in range(self.n_inequiv_shells):
            for block in R[ish]:
                G_loc[ish][block].from_L_G_R(R[ish][block].conjugate().transpose(), G_loc_qp[ish][block], R[ish][block])

        #if self.use_rotations:
        #    for ish in range(self.n_inequiv_shells):
        #        icrsh = self.inequiv_to_corr[ish]
        #        for bname, gf in G_loc_qp[ish]:
        #            G_loc[ish][bname] << self.rotloc(ish, gf, direction='toLocal')
        
        if transform_to_solver_blocks:        
            for ish in range(self.n_inequiv_shells):
                G_loc[ish] = self.block_structure.convert_gf(
                        G=G_loc[ish],
                        ish_from=self.inequiv_to_corr[ish],
                        ish_to=ish,
                        space_from='sumk',
                        space_to='solver',
                        show_warnings = show_warnings)
        
        return G_loc
    
    # FIXME
    def get_Sigma(self): # Can I use dyson's equation to get this? Or will doing inverses be fine?
        return None

    def get_dens_uncorr_k(self, mu=None, beta=None):
        return None

    # quasiparticle density matrix calculated from mean-field
    def get_pdensity(self, ish=0, Lambda=None, R=None, mu=None, beta=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
         
        icrsh = self.inequiv_to_corr[ish]
        
        for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
            self.pdensity_sumk[ish][sp] = numpy.zeros([self.corr_shells[icrsh]['dim'],self.corr_shells[icrsh]['dim']], numpy.complex_)
        
        # do the integral
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)
            for block in self.pdensity_sumk[ish].keys():
                eig, vec = numpy.linalg.eigh(h_bl_qp_k[block])
                f_bl_qp = numpy.dot(vec, numpy.dot( numpy.diag(self.fweights(eks=eig, mu=mu, beta=beta)), vec.conjugate().transpose() ) )
                f_bl_qp *= self.bz_weights[ik]
                f_bl_qp = self.downfold_matrix(ik=ik, ish=icrsh, bname=block, matrix_to_downfold=f_bl_qp)
                self.pdensity_sumk[ish][block] += f_bl_qp
            
        # collect data from mpi:
        for block in self.pdensity_sumk[ish].keys():
            self.pdensity_sumk[ish][block] = mpi.all_reduce(mpi.world, self.pdensity_sumk[ish][block], lambda x, y: x + y)
        mpi.barrier()

        # pdensity is transposed
        for block in self.pdensity_sumk[ish].keys():
            self.pdensity_sumk[ish][block] = self.pdensity_sumk[ish][block].transpose()

        ## FIXME symmcorr.symmetrize is over all correlated shells, so this needs to be fixed
        ## symmetrisation:
        #if self.symm_op != 0:
        #    self.pdensity_sumk[ish] = self.symmcorr.symmetrize(self.pdensity_sumk[ish])
        
        # FIXME
        # report on how large the imaginary components are and discard
        err = 0
        for block in self.pdensity_sumk[ish].keys():
            err += numpy.sum(numpy.abs(self.pdensity_sumk[ish][block].imag))
            self.pdensity_sumk[ish][block] = self.pdensity_sumk[ish][block].astype(numpy.float_)
        if abs(err) > 1e-20:
            mpi.report("Warning: Imaginary part in pdensity will be ignored. The sum is ({})".format(str(abs(err))))
        
        # transform from sumk blocks to the solver blocks
        # Note that sumk in this convert_matrix function is always indexed for all correlated shells
        # And solver is always indexed for inequivalent shells
        self.pdensity[ish] = self.block_structure.convert_matrix(
                G=self.pdensity_sumk[ish],
                G_struct=None,
                ish_from=icrsh,
                ish_to=ish,
                space_from='sumk',
                space_to='solver',
                show_warnings=True)

        # FIXME
        # report on how large the imaginary components are and discard
        #err = 0
        #for block in self.pdensity[ish].keys():
        #    err += numpy.sum(numpy.abs(self.pdensity[ish][block].imag))
        #    self.pdensity[ish][block] = self.pdensity[ish][block].astype(numpy.float_)
        #if abs(err) > 1e-20:
        #    mpi.report("Warning: Imaginary part in pdensity will be ignored. The sum is ({})".format(str(abs(err))))
            
        return self.pdensity[ish]
   
    # lopsided quasiparticle kinetic energy calculated from mean-field
    def get_ke(self, ish=0, Lambda=None, R=None, mu=None, beta=None, lopsided=True):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
        
        icrsh = self.inequiv_to_corr[ish]
         
        for sp in self.spin_block_names[self.corr_shells[icrsh]['SO']]:
            self.ke_sumk[ish][sp] = numpy.zeros([self.corr_shells[icrsh]['dim'],self.corr_shells[icrsh]['dim']], numpy.complex_)

        # Do the integral
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)
            h_bl_qp_kin_k = self.get_h_bl_qp_kin_k(ik=ik, R=R, lopsided=lopsided)
            for block in self.ke_sumk[ish].keys():
                eig, vec = numpy.linalg.eigh(h_bl_qp_k[block])
                f_bl_qp = numpy.dot(vec, numpy.dot( numpy.diag(self.fweights(eks=eig, mu=mu, beta=beta)), vec.conjugate().transpose() ) )
                f_bl_qp *= self.bz_weights[ik]
                ke_bl = numpy.dot(h_bl_qp_kin_k[block], f_bl_qp)
                self.ke_sumk[ish][block] += self.downfold_matrix(ik=ik, ish=icrsh, bname=block, matrix_to_downfold=ke_bl)

        # collect data from mpi:
        for block in self.ke_sumk[ish].keys():
            self.ke_sumk[ish][block] = mpi.all_reduce(mpi.world, self.ke_sumk[ish][block], lambda x, y: x + y)
        mpi.barrier()

        ## FIXME symmcorr.symmetrize is over all correlated shells, so this needs to be fixed
        ## symmetrisation:
        #if self.symm_op != 0:
        #    self.ke_sumk[ish] = self.symmcorr.symmetrize(self.ke_sumk[ish])
        
        # FIXME
        # report on how large the imaginary components are and discard
        err = 0
        for block in self.ke_sumk[ish].keys():
            err += numpy.sum(numpy.abs(self.ke_sumk[ish][block].imag))
            self.ke_sumk[ish][block] = self.ke_sumk[ish][block].astype(numpy.float_)
        if abs(err) > 1e-20:
            mpi.report("Warning: Imaginary part in ke will be ignored. The sum is ({})".format(str(abs(err))))
         
        # transform from sumk blocks to the solver blocks
        self.ke[ish] = self.block_structure.convert_matrix(
                G=self.ke_sumk[ish],
                G_struct=None,
                ish_from=icrsh,
                ish_to=ish,
                space_from='sumk',
                space_to='solver',
                show_warnings=True)
        
        return self.ke[ish]
    
    # Total kinetic energy in all of the bands (not lopsided)
    def get_ke_total(self, ish=0, Lambda=None, R=None, mu=None, beta=None, lopsided=False):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta

        ke_total = 0

        # Do the integral
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)
            h_bl_qp_kin_k = self.get_h_bl_qp_kin_k(ik=ik, R=R, lopsided=lopsided)
            for block in h_bl_qp_k.keys():
                eig, vec = numpy.linalg.eigh(h_bl_qp_k[block])
                f_bl_qp = numpy.dot(vec, numpy.dot( numpy.diag(self.fweights(eks=eig, mu=mu, beta=beta)), vec.conjugate().transpose() ) )
                f_bl_qp *= self.bz_weights[ik]
                ke_bl = numpy.dot(h_bl_qp_kin_k[block], f_bl_qp)
                ke_total += numpy.real(numpy.trace(ke_bl))
        
        return ke_total

    def get_density_k_risb(self, ik, Lambda=None, R=None, mu=None, beta=None, dm=None, for_rho=False):
        
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
            
        if for_rho:
            R_bl_k = self.get_R_bl_k(ik)
        
        h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)
        
        dens_k = {}
        for bname in h_bl_qp_k.keys():
            eig, vec = numpy.linalg.eigh(h_bl_qp_k[bname])
            dens_k[bname] = numpy.dot(vec, numpy.dot( numpy.diag(self.fweights(eks=eig, mu=mu, beta=beta)), vec.conjugate().transpose() ) )
            
            if for_rho:
                dens_k[bname] = numpy.dot( R_bl_k[bname].conjugate().transpose(), numpy.dot(dens_k[bname], R_bl_k[bname]) )                

            if dm != None:
                for icrsh in range(self.n_corr_shells):
                    ish = self.corr_to_inequiv[icrsh]
    
                    # density of imp in large space for block subtracted
                    dens_bl_corr_k = self.downfold_matrix(ik=ik, ish=icrsh, bname=bname, matrix_to_downfold=dens_k[bname])
                    dens_k[bname] -= self.upfold_matrix( ik=ik, ish=icrsh, bname=bname, \
                                                        matrix_to_upfold=dens_bl_corr_k )
                    
                    # density of imp from impurity problem added back in
                    dens_k[bname] += self.upfold_matrix( ik=ik, ish=icrsh, bname=bname, \
                                                        matrix_to_upfold=dm[ish][bname] )
        return dens_k

    def total_density_risb(self, Lambda=None, R=None, mu=None, beta=None, dm=None):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta

        # do the integral
        dens = 0.0
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            dens_k = self.get_density_k_risb(ik=ik, Lambda=Lambda, R=R, mu=mu, beta=beta, dm=dm)
            for bname in dens_k.keys():
                dens += self.bz_weights[ik] * dens_k[bname].trace()
        
        # collect data from mpi:
        dens = mpi.all_reduce(mpi.world, dens, lambda x, y: x + y)
        mpi.barrier()

        if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part in density will be ignored ({})".format(str(abs(dens.imag))))
        return dens.real

    def calc_mu_risb(self, Lambda=None, R=None, beta=None, dm=None, 
                           precision=0.01, delta=0.5, mu_max_iter=100, method='dichotomy',
                           offset=0):
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if beta is None:
            beta = self.beta
            
        density = self.density_required - self.charge_below + offset

        dichotomy_fail = False
        if method == 'dichotomy':
            F = lambda mu: self.total_density_risb(Lambda=Lambda, R=R, mu=mu, beta=beta, dm=dm)
            res = dichotomy.dichotomy(function=F,
                                      x_init=self.chemical_potential, y_value=density,
                                      precision_on_y=precision, delta_x=delta, max_loops=mu_max_iter,
                                      x_name="Chemical Potential", y_name="Total Density",
                                      verbosity=3)[0]
            
            # If dichotomy fails, fall back to brent
            if res is None:
                dichotomy_fail = True
            else:
                self.chemical_potential = res
        
        # Slow but 'more robust' method
        if (method != 'dichotomy') or dichotomy_fail:
            F = lambda mu: numpy.abs(self.total_density_risb(Lambda=Lambda, R=R, mu=mu, beta=beta, dm=dm) - density)
            res = minimize_scalar(F, method='brent', tol=precision, bracket=(self.chemical_potential-delta, self.chemical_potential+delta),
                                  options={'maxiter':mu_max_iter})
            self.chemical_potential = res.x
        
        return self.chemical_potential
        
    def save_deltaN(self, deltaN, filename, dm_type, band_window):
        if dm_type == 'wien2k':
            if mpi.is_master_node():
                if self.SP == 0:
                    f = open(filename, 'w')
                else:
                    f = open(filename + 'up', 'w')
                    f1 = open(filename + 'dn', 'w')
                # write chemical potential (in Rydberg):
                f.write("%.14f\n" % (self.chemical_potential / self.energy_unit))
                if self.SP != 0:
                    f1.write("%.14f\n" %
                             (self.chemical_potential / self.energy_unit))
                # write beta in rydberg-1
                #f.write("%.14f\n" % (G_latt_iw.mesh.beta * self.energy_unit))
                f.write("%.14f\n" % (self.beta * self.energy_unit))
                if self.SP != 0:
                    #f1.write("%.14f\n" % (G_latt_iw.mesh.beta * self.energy_unit))
                    f1.write("%.14f\n" % (self.beta * self.energy_unit))

                if self.SP == 0:  # no spin-polarization

                    for ik in range(self.n_k):
                        f.write("%s\n" % self.n_orbitals[ik, 0])
                        for inu in range(self.n_orbitals[ik, 0]):
                            for imu in range(self.n_orbitals[ik, 0]):
                                valre = (deltaN['up'][ik][
                                         inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                                valim = (deltaN['up'][ik][
                                         inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                                f.write("%.14f  %.14f " % (valre, valim))
                            f.write("\n")
                        f.write("\n")
                    f.close()

                elif self.SP == 1:  # with spin-polarization

                    # dict of filename: (spin index, block_name)
                    if self.SO == 0:
                        to_write = {f: (0, 'up'), f1: (1, 'down')}
                    if self.SO == 1:
                        to_write = {f: (0, 'ud'), f1: (0, 'ud')}
                    for fout in to_write.keys():
                        isp, sp = to_write[fout]
                        for ik in range(self.n_k):
                            fout.write("%s\n" % self.n_orbitals[ik, isp])
                            for inu in range(self.n_orbitals[ik, isp]):
                                for imu in range(self.n_orbitals[ik, isp]):
                                    fout.write("%.14f  %.14f " % (deltaN[sp][ik][
                                               inu, imu].real, deltaN[sp][ik][inu, imu].imag))
                                fout.write("\n")
                            fout.write("\n")
                        fout.close()
        
        elif dm_type == 'vasp':
            assert self.SP == 0, "Spin-polarized density matrix is not implemented"

            if mpi.is_master_node():
                with open(filename, 'w') as f:
                    f.write(" %i  -1  ! Number of k-points, default number of bands\n"%(self.n_k))
                    for ik in range(self.n_k):
                        ib1 = band_window[0][ik, 0]
                        ib2 = band_window[0][ik, 1]
                        f.write(" %i  %i  %i\n"%(ik + 1, ib1, ib2))
                        for inu in range(self.n_orbitals[ik, 0]):
                            for imu in range(self.n_orbitals[ik, 0]):
                                valre = (deltaN['up'][ik][inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                                valim = (deltaN['up'][ik][inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                                f.write(" %.14f  %.14f"%(valre, valim))
                            f.write("\n")
        else:
            raise NotImplementedError("Unknown density matrix type: '%s'"%(dm_type))
   
    # deltaN = N_RISB - N_ks
    def deltaN_risb(self, dm, filename=None, dm_type='wien2k', R=None, Lambda=None, mu=None, beta=None):
        assert dm_type in ('vasp', 'wien2k'), "'dm_type' must be either 'vasp' or 'wienk'"

        if filename is None:
            if dm_type == 'wien2k':
                filename = 'dens_mat.dat'
            elif dm_type == 'vasp':
                filename = 'GAMMA'

        assert isinstance(filename, str), ("deltaN_risb: "
                                              "filename has to be a string!")
        
        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta
        
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        dens = {sp: 0.0 for sp in spn}
        band_en_correction = 0.0
        
        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        dens = {sp: 0.0 for sp in spn}
        band_en_correction = 0.0

        # Fetch Fermi weights and energy window band indices
        if dm_type == 'vasp':
            fermi_weights = 0
            band_window = 0
            if mpi.is_master_node():
                with HDFArchive(self.hdf_file,'r') as ar:
                    fermi_weights = ar['dft_misc_input']['dft_fermi_weights']
                    band_window = ar['dft_misc_input']['band_window']
            fermi_weights = mpi.bcast(fermi_weights)
            band_window = mpi.bcast(band_window)

            # Convert Fermi weights to a density matrix
            dens_mat_dft = {}
            for sp in spn:
                dens_mat_dft[sp] = [fermi_weights[ik, ntoi[sp], :].astype(numpy.complex_) for ik in range(self.n_k)]

        # Set up deltaN:
        deltaN = {}
        for sp in spn:
            deltaN[sp] = [numpy.zeros([self.n_orbitals[ik, ntoi[sp]], self.n_orbitals[
                                      ik, ntoi[sp]]], numpy.complex_) for ik in range(self.n_k)]
          
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            dens_k = self.get_density_k_risb(ik=ik, Lambda=Lambda, R=R, mu=mu, beta=beta, dm=dm, for_rho=True)
            
            for bname in dens_k.keys():
                # rotate into the DFT band basis
                if dm_type == 'vasp' and self.proj_or_hk == 'hk':
                    dens_k[bname] = self.upfold_matrix(ik=ik, ish=0, bname=bname, matrix_to_upfold=dens_k[bname], shells='csc')
                
                deltaN[bname][ik] = copy.deepcopy(dens_k[bname])
                dens[bname] += self.bz_weights[ik] * dens_k[bname].trace().real # trace because total density is basis independent
                
                # In 'vasp'-mode subtract the DFT density matrix
                if dm_type == 'vasp':
                    nb = self.n_orbitals[ik, ntoi[bname]]
                    diag_inds = numpy.diag_indices(nb)
                    deltaN[bname][ik][diag_inds] -= dens_mat_dft[bname][ik][:nb]

                    if self.charge_mixing and self.deltaNOld is not None:
                        G2 = numpy.sum(self.kpts_cart[ik,:]**2)
                        # Kerker mixing
                        mix_fac = self.charge_mixing_alpha * G2 / (G2 + self.charge_mixing_gamma**2)
                        deltaN[bname][ik][diag_inds] = (1.0 - mix_fac) * self.deltaNOld[bname][ik][diag_inds] + mix_fac * deltaN[bname][ik][diag_inds]
                    
                    dens[bname] -= self.bz_weights[ik] * dens_mat_dft[bname][ik].sum().real
                    isp = ntoi[bname]
                    b1, b2 = band_window[isp][ik, :2]
                    nb = b2 - b1 + 1
                    assert nb == self.n_orbitals[ik, ntoi[bname]], "Number of bands is inconsistent at ik = %s"%(ik)
                    band_en_correction += numpy.dot(deltaN[bname][ik], self.hopping[ik, isp, :nb, :nb]).trace().real * self.bz_weights[ik]
        
        # mpi reduce:
        for bname in deltaN:
            for ik in range(self.n_k):
                deltaN[bname][ik] = mpi.all_reduce(
                    mpi.world, deltaN[bname][ik], lambda x, y: x + y)
            dens[bname] = mpi.all_reduce(
                mpi.world, dens[bname], lambda x, y: x + y)
        self.deltaNOld = copy.deepcopy(deltaN)
        mpi.barrier()

        band_en_correction = mpi.all_reduce(mpi.world, band_en_correction, lambda x,y : x+y)
        
        # now save to file:
        self.save_deltaN(deltaN=deltaN, filename=filename, dm_type=dm_type, band_window=band_window)

        return deltaN, dens, band_en_correction
    
    # FIXME below is technically not correct (tried to adapt from DMFT)
    # It does not take the correlated density for the correction, and instead takes the pdensity
    def calc_density_correction_risb(self, filename=None, dm_type='wien2k', Lambda=None, R=None, mu=None, beta=None):
        assert dm_type in ('vasp', 'wien2k'), "'dm_type' must be either 'vasp' or 'wienk'"

        if filename is None:
            if dm_type == 'wien2k':
                filename = 'dens_mat.dat'
            elif dm_type == 'vasp':
                filename = 'GAMMA'

        assert isinstance(filename, str), ("calc_density_correction: "
                                              "filename has to be a string!")

        if Lambda is None:
            Lambda = self.Lambda_sumk
        if R is None:
            R = self.R_sumk
        if mu is None:
            mu = self.chemical_potential
        if beta is None:
            beta = self.beta

        ntoi = self.spin_names_to_ind[self.SO]
        spn = self.spin_block_names[self.SO]
        dens = {sp: 0.0 for sp in spn}
        band_en_correction = 0.0

        # Fetch Fermi weights and energy window band indices
        if dm_type == 'vasp':
            fermi_weights = 0
            band_window = 0
            if mpi.is_master_node():
                with HDFArchive(self.hdf_file,'r') as ar:
                    fermi_weights = ar['dft_misc_input']['dft_fermi_weights']
                    band_window = ar['dft_misc_input']['band_window']
            fermi_weights = mpi.bcast(fermi_weights)
            band_window = mpi.bcast(band_window)

            # Convert Fermi weights to a density matrix
            dens_mat_dft = {}
            for sp in spn:
                dens_mat_dft[sp] = [fermi_weights[ik, ntoi[sp], :].astype(numpy.complex_) for ik in range(self.n_k)]

        # Set up deltaN:
        deltaN = {}
        for sp in spn:
            deltaN[sp] = [numpy.zeros([self.n_orbitals[ik, ntoi[sp]], self.n_orbitals[
                                      ik, ntoi[sp]]], numpy.complex_) for ik in range(self.n_k)]


        # Calculate (change in) density matrix
        ikarray = numpy.array(list(range(self.n_k)))
        for ik in mpi.slice_array(ikarray):
            #h_bl_k = self.get_h_bl_k(ik=ik, Lambda=Lambda, R=R)
            h_bl_qp_k = self.get_h_bl_qp_k(ik=ik, Lambda=Lambda, R=R)

            if dm_type == 'vasp' and self.proj_or_hk == 'hk':
                # rotate into the DFT band basis
                for bname in h_bl_qp_k.keys():
                    h_bl_qp_k[bname] = self.upfold_matrix(ik=ik, ish=0, bname=bname, matrix_to_upfold=h_bl_qp_k[bname], shells='csc')

            for bname in h_bl_qp_k.keys():
                eig, vec = numpy.linalg.eigh(h_bl_qp_k[bname])
                dens_matrix = numpy.dot(vec, numpy.dot( numpy.diag(self.fweights(eks=eig, mu=mu, beta=beta)), vec.conjugate().transpose() ) )
                
                deltaN[bname][ik] = copy.deepcopy(dens_matrix)
                dens[bname] += self.bz_weights[ik] * dens_matrix.trace().real # trace because total density is basis independent

                # In 'vasp'-mode subtract the DFT density matrix
                if dm_type == 'vasp':
                    nb = self.n_orbitals[ik, ntoi[bname]]
                    diag_inds = numpy.diag_indices(nb)
                    deltaN[bname][ik][diag_inds] -= dens_mat_dft[bname][ik][:nb]

                    if self.charge_mixing and self.deltaNOld is not None:
                        G2 = numpy.sum(self.kpts_cart[ik,:]**2)
                        # Kerker mixing
                        mix_fac = self.charge_mixing_alpha * G2 / (G2 + self.charge_mixing_gamma**2)
                        deltaN[bname][ik][diag_inds] = (1.0 - mix_fac) * self.deltaNOld[bname][ik][diag_inds] + mix_fac * deltaN[bname][ik][diag_inds]
                    dens[bname] -= self.bz_weights[ik] * dens_mat_dft[bname][ik].sum().real
                    isp = ntoi[bname]
                    b1, b2 = band_window[isp][ik, :2]
                    nb = b2 - b1 + 1
                    assert nb == self.n_orbitals[ik, ntoi[bname]], "Number of bands is inconsistent at ik = %s"%(ik)
                    band_en_correction += numpy.dot(deltaN[bname][ik], self.hopping[ik, isp, :nb, :nb]).trace().real * self.bz_weights[ik]
        
        # mpi reduce:
        for bname in deltaN:
            for ik in range(self.n_k):
                deltaN[bname][ik] = mpi.all_reduce(
                    mpi.world, deltaN[bname][ik], lambda x, y: x + y)
            dens[bname] = mpi.all_reduce(
                mpi.world, dens[bname], lambda x, y: x + y)
        self.deltaNOld = copy.deepcopy(deltaN)
        mpi.barrier()

        band_en_correction = mpi.all_reduce(mpi.world, band_en_correction, lambda x,y : x+y)

        # now save to file:
        if dm_type == 'wien2k':
            if mpi.is_master_node():
                if self.SP == 0:
                    f = open(filename, 'w')
                else:
                    f = open(filename + 'up', 'w')
                    f1 = open(filename + 'dn', 'w')
                # write chemical potential (in Rydberg):
                f.write("%.14f\n" % (self.chemical_potential / self.energy_unit))
                if self.SP != 0:
                    f1.write("%.14f\n" %
                             (self.chemical_potential / self.energy_unit))
                # write beta in rydberg-1
                #f.write("%.14f\n" % (G_latt_iw.mesh.beta * self.energy_unit))
                f.write("%.14f\n" % (self.beta * self.energy_unit))
                if self.SP != 0:
                    #f1.write("%.14f\n" % (G_latt_iw.mesh.beta * self.energy_unit))
                    f1.write("%.14f\n" % (self.beta * self.energy_unit))

                if self.SP == 0:  # no spin-polarization

                    for ik in range(self.n_k):
                        f.write("%s\n" % self.n_orbitals[ik, 0])
                        for inu in range(self.n_orbitals[ik, 0]):
                            for imu in range(self.n_orbitals[ik, 0]):
                                valre = (deltaN['up'][ik][
                                         inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                                valim = (deltaN['up'][ik][
                                         inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                                f.write("%.14f  %.14f " % (valre, valim))
                            f.write("\n")
                        f.write("\n")
                    f.close()

                elif self.SP == 1:  # with spin-polarization

                    # dict of filename: (spin index, block_name)
                    if self.SO == 0:
                        to_write = {f: (0, 'up'), f1: (1, 'down')}
                    if self.SO == 1:
                        to_write = {f: (0, 'ud'), f1: (0, 'ud')}
                    for fout in to_write.keys():
                        isp, sp = to_write[fout]
                        for ik in range(self.n_k):
                            fout.write("%s\n" % self.n_orbitals[ik, isp])
                            for inu in range(self.n_orbitals[ik, isp]):
                                for imu in range(self.n_orbitals[ik, isp]):
                                    fout.write("%.14f  %.14f " % (deltaN[sp][ik][
                                               inu, imu].real, deltaN[sp][ik][inu, imu].imag))
                                fout.write("\n")
                            fout.write("\n")
                        fout.close()
        
        elif dm_type == 'vasp':
            assert self.SP == 0, "Spin-polarized density matrix is not implemented"

            if mpi.is_master_node():
                with open(filename, 'w') as f:
                    f.write(" %i  -1  ! Number of k-points, default number of bands\n"%(self.n_k))
                    for ik in range(self.n_k):
                        ib1 = band_window[0][ik, 0]
                        ib2 = band_window[0][ik, 1]
                        f.write(" %i  %i  %i\n"%(ik + 1, ib1, ib2))
                        for inu in range(self.n_orbitals[ik, 0]):
                            for imu in range(self.n_orbitals[ik, 0]):
                                valre = (deltaN['up'][ik][inu, imu].real + deltaN['down'][ik][inu, imu].real) / 2.0
                                valim = (deltaN['up'][ik][inu, imu].imag + deltaN['down'][ik][inu, imu].imag) / 2.0
                                f.write(" %.14f  %.14f"%(valre, valim))
                            f.write("\n")
        else:
            raise NotImplementedError("Unknown density matrix type: '%s'"%(dm_type))

        res = deltaN, dens

        if dm_type == 'vasp':
            res += (band_en_correction,)

        return res