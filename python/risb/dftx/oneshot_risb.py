import os
import sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import copy
import numpy as np
from scipy import optimize

import triqs.utility.mpi as mpi
from h5 import *
import triqs.version as triqs_version
from triqs.operators.util.hamiltonians import *
from triqs.operators.util.U_matrix import *
import triqs_dft_tools.version as dft_tools_version

import triqs_dft_tools.converters.vasp as vaspconv
import triqs_dft_tools.converters.wien2k as wien2kconv
import triqs_dft_tools.converters.plovasp.converter as plo_converter

import risb.sc_cycle as sc
from embedding_ed import EmbeddingEd

from risb_solver import risbSolver
from sumk_dft_risb import SumkDFTRISB
  
class oneshotRISB(object):
    def __init__(self, filename, 
                       dft_solver='vasp', root_solver='diis', sc_method='recursion', prnt=True, 
                       plo_cfg_filename='plo.cfg', analyze_block_threshold=1e-2, use_rotations=False,
                       beta=5.0, smearing='gaussian', linear_mix_factor=1,
                       mu_tol=1e-6, mu_delta=0.5, mu_max_iter=100, mu_method='dichotomy', mu_offset=0, mu_mix_factor=1,
                       root_tol=1e-6, maxfev=500, maxiter=500,
                       U=0, J=0, DC_type=0, DC_value=None):
    
        self.filename = filename
        self.dft_solver = dft_solver
        self.root_solver = root_solver
        self.sc_method = sc_method
        self.prnt = prnt
        self.plo_cfg_filename = plo_cfg_filename

        self.beta = beta
        self.smearing = smearing # gaussian, fermi, methfessel-paxton
        self.analyze_block_threshold = analyze_block_threshold
        self.mu_tol = mu_tol
        self.mu_delta = mu_delta
        self.mu_max_iter = mu_max_iter
        self.mu_method = mu_method
        self.mu_offset = mu_offset
        self.mu_mix_factor = mu_mix_factor
        self.linear_mix_factor = linear_mix_factor
        
        self.root_tol = root_tol
        self.maxfev = maxfev
        self.maxiter = maxiter

        self.U = U
        self.J = J
        
        self.DC_type = DC_type # 0 = FLL
        self.DC_value = DC_value # != None uses a fixed number

        self.use_rotations = use_rotations
        
        self.SK = None
        self.h0_loc = None
        self.h_int = None
        self.h_dc = None

        self.S = None

        self.dens = None
        self.band_en_correction = None
        self.dft_energy = None
        self.dc_energy = None
        self.corr_energy = None
        self.total_dc_energy = None
        self.total_corr_energy = None
        self.total_energy = None
        self.energy_correction = None
       
    def disablePrint(self):
        sys.stdout = open(os.devnull, 'w')

    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def get_dft_energy(self):
        """
        Reads energy from the last line of OSZICAR.
        """
        with open('OSZICAR', 'r') as f:
            nextline = f.readline()
            while nextline.strip():
                line = nextline
                nextline = f.readline()
    #            print "OSZICAR: ", line[:-1]

        try:
            dft_energy = float(line.split()[2])
        except ValueError:
            mpi.report("Cannot read energy from OSZICAR, setting it to zero")
            dft_energy = 0.0

        return dft_energy

    def get_h_dc(self, dm=None):
    
        # Calculate double counting
        h_dc = []
        Vdc = []
        for ish in range(self.SK.n_inequiv_shells):
            self.disablePrint()
            if dm == None:
                dm = self.SK.get_pdensity(ish=ish)
            self.SK.calc_dc(dm, U_interact=self.U, J_hund=self.J, orb=ish, use_dc_formula=self.DC_type, use_dc_value=self.DC_value)
            self.enablePrint()
            # Below is to ensure Vdc is rotated into solver space
            Vdc.append( next(iter( self.SK.block_structure.convert_matrix(self.SK.dc_imp[ish],space_from='sumk',space_to='solver').items() ))[1][0,0].real )
            h_dc.append( Operator() )
            for index, (key, value) in enumerate(self.SK.sumk_to_solver[ish].items()):
                if (value[0] == None) or (value[1] == None):
                    continue
                s = key[0]
                o = key[1]
                h_dc[ish] += Vdc[ish] * n(value[0],value[1])
        
        self.SK.dc_imp = mpi.bcast(self.SK.dc_imp)
        self.SK.dc_energ = mpi.bcast(self.SK.dc_energ)
    
        return h_dc

    def one_dmft_cycle(self, Lambda, R):
 
        # Ensure symmetries are enforced #FIXME inequiv or corr?
        for ish in range(self.SK.n_inequiv_shells):
            self.SK.symm_deg_mat(Lambda[ish], ish=ish)
            self.SK.symm_deg_mat(R[ish], ish=ish)

        # Update the sumk calss
        self.SK.Lambda = Lambda
        self.SK.R = R
        self.SK.set_R_Lambda_sumk()
        
        # Calculate mu and density
        self.disablePrint()
        old_mu = self.SK.chemical_potential
        if np.abs(self.mu_offset) > self.mu_tol:
            mu_below = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method, offset=-self.mu_offset)
            mu_above = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method, offset=self.mu_offset)
            mu = 0.5*(mu_above + mu_below)
        else:
            mu = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method)
        mu = self.mu_mix_factor * mu + (1.0-self.mu_mix_factor) * old_mu
        self.SK.chemical_potential = mpi.bcast(mu)
        self.enablePrint()
        
        # Get new h_dc
        self.h_dc = self.get_h_dc()

        mpi.barrier()
        
        # Solve one cycle of RISB
        out_A = [{} for ish in range(self.SK.n_inequiv_shells)]
        out_B = [{} for ish in range(self.SK.n_inequiv_shells)]
        self.S = []
        for ish in range(self.SK.n_inequiv_shells):
            h_loc = self.h0_loc[ish] + self.h_int[ish] - self.h_dc[ish]

            if mpi.is_master_node():
                print("Solving U =", self.U, "J =", self.J, "impurity for shell", ish, "using", self.root_solver, "with", self.sc_method, "mu =", self.SK.chemical_potential)
            
            emb_solver = EmbeddingEd(h_loc, self.SK.block_structure.gf_struct_solver_list[ish])
            self.S.append( risbSolver(emb_solver=emb_solver, ish=ish) )
            self.disablePrint()
            out_A[ish], out_B[ish] = self.S[ish].one_cycle(SK=self.SK, sc_method=self.sc_method)
            self.enablePrint()
        
        if self.sc_method == 'recursion':
            Lambda_diff = [{} for ish in range(self.SK.n_inequiv_shells)]
            R_diff = [{} for ish in range(self.SK.n_inequiv_shells)]
            for ish in range(self.SK.n_inequiv_shells):
                for block in Lambda[ish].keys():
                    Lambda_diff[ish][block] = out_A[ish][block] - Lambda[ish][block]
            
            for ish in range(self.SK.n_inequiv_shells):
                for block in R[ish].keys():
                    R_diff[ish][block] = out_B[ish][block] - R[ish][block]

            return Lambda_diff, R_diff
        
        elif (self.sc_method == 'root') or (self.sc_method == 'fixed-point'):
            return out_A, out_B

        else:
            raise ValueError("one_dmft_cycle: Implemented only for recursion, root problem, and fixed-point.")

    def root_flatten(self, A, B):
        x = []

    #    # All solver or SK blocks
    #    for ish in range(SK.n_inequiv_shells):
    #        for block in A[ish].keys():
    #            for i,j in np.ndindex(A[ish][block].shape):
    #                x.append( A[ish][block][i,j].real )
    #     
    #    for ish in range(SK.n_inequiv_shells):
    #        for block in B[ish].keys():
    #            for i,j in np.ndindex(B[ish][block].shape):
    #                x.append( B[ish][block][i,j].real )

        # Only 1 of each deg orbital
        for ish in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish]:
                block_0 = degsh[0]
                for i,j in np.ndindex(A[ish][block_0].shape):
                    x.append( A[ish][block_0][i,j].real )
        
        for ish in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish]:
                block_0 = degsh[0]
                for i,j in np.ndindex(A[ish][block_0].shape):
                    x.append( B[ish][block_0][i,j].real )

        return np.array(x, dtype=np.float_)

    def root_construct(self, x):
        Lambda = [{} for ish in range(self.SK.n_inequiv_shells)]
        R = [{} for ish in range(self.SK.n_inequiv_shells)]
        counter = 0
        
    #    # All solver or sumk blocks
    #    for ish in range(SK.n_inequiv_shells):
    #        SK_Lambda = SK.Lambda[ish]
    #        #SK_Lambda = SK.Lambda_sumk[ish] # FIXME test for doing root in sumk structure
    #        for block in SK_Lambda.keys():
    #            Lambda[ish][block] = np.zeros(SK_Lambda[block].shape, dtype=np.float_)
    #            for i,j in np.ndindex(Lambda[ish][block].shape):
    #                Lambda[ish][block][i,j] = x[counter]
    #                counter += 1
    #    
    #    for ish in range(SK.n_inequiv_shells):
    #        SK_R = SK.R[ish]
    #        #SK_R = SK.R_sumk[ish] # FIXME test for doing root in sumk structure
    #        for block in SK_R.keys():
    #            R[ish][block] = np.zeros(SK_R[block].shape, dtype=np.float_)
    #            for i,j in np.ndindex(R[ish][block].shape):
    #                R[ish][block][i,j] = x[counter]
    #                counter += 1

        # Only 1 of each deg orbital
        for ish in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish]:
                block_0 = degsh[0]
                Lambda[ish][block_0] = np.zeros(self.SK.Lambda[ish][block_0].shape, dtype=np.float_)
                for i,j in np.ndindex(Lambda[ish][block_0].shape):
                    Lambda[ish][block_0][i,j] = x[counter]
                    counter += 1
                for block in degsh:
                    Lambda[ish][block] = Lambda[ish][block_0]
         
        for ish in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish]:
                block_0 = degsh[0]
                R[ish][block_0] = np.zeros(self.SK.R[ish][block_0].shape, dtype=np.float_)
                for i,j in np.ndindex(R[ish][block_0].shape):
                    R[ish][block_0][i,j] = x[counter]
                    counter += 1
                for block in degsh:
                    R[ish][block] = R[ish][block_0]
        
        return Lambda, R

    def stop_check(self):
        if mpi.is_master_node:
            if os.path.isfile('STOPRISB'):
                print('\nStopping RISB.\n', flush=True)
                mpi.world.Abort(1)
    
    def root_fun(self, x):
        self.stop_check()

        Lambda, R = self.root_construct(x=x)
        
        ## FIXME test for setting up root solvers as sumk structures
        #for ish in range(SK.n_inequiv_shells):
        #    icrsh = SK.inequiv_to_corr[ish]
        #    Lambda[ish] = SK.mat_sumk_to_solver(Lambda[ish], icrsh=icrsh)
        #    R[ish] = SK.mat_sumk_to_solver(R[ish], icrsh=icrsh)
        
        out_A, out_B = self.one_dmft_cycle(Lambda=Lambda, R=R)
        
        ## FIXME test for setting up root solvers as sumk structures
        #for ish in range(SK.n_inequiv_shells):
        #    icrsh = SK.inequiv_to_corr[ish]
        #    out_A[ish] = SK.mat_solver_to_sumk(out_A[ish], icrsh=icrsh)
        #    out_B[ish] = SK.mat_solver_to_sumk(out_B[ish], icrsh=icrsh)
        
        return self.root_flatten(A=out_A, B=out_B)

    def solve_hybr(self, Lambda, R):
        mpi.report("hybr:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='hybr', options={'maxfev':self.maxfev})
        self.enablePrint()
        if sol.success:
            mpi.report(f"sol.success: {sol.success}, sol.message: {sol.message}, sol.nfev: {sol.nfev}")
        else:
            mpi.report(f"sol.success: {sol.success}, sol.message: {sol.message}, sol.nfev: {sol.nfev}")
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_lm(self, Lambda, R):
        mpi.report("lm:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='lm', options={'maxiter':self.maxiter})
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nfev:", sol.nfev)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_broyden(self, Lambda, R):
        mpi.report("broyden:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='broyden1', \
                             options={'maxiter':self.maxiter, 'line_search':'armijo', \
                             'jac_options':{'reduction_method':'restart', 'max_rank':5} })
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nit:", sol.nit)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_anderson(self, Lambda, R):
        mpi.report("anderson:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='anderson', options={'maxiter':self.maxiter})
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nit:", sol.nit)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_linearmixing(self, Lambda, R):
        mpi.report("linearmixing:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='linearmixing', options={'maxiter':self.maxiter})
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nit:", sol.nit)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_krylov(self, Lambda, R):
        mpi.report("krylov:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='krylov', \
                             options={ 'maxiter':self.maxiter}) #, 'line_search':'armijo', \
                             #         'jac_options':{'method':'cgs'} })
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nit:", sol.nit)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    def solve_df_sane(self, Lambda, R):
        mpi.report("df-sane:")
        sol = optimize.root( fun=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                             tol=self.root_tol, method='df-sane', options={'maxfev':self.maxfev})
        self.enablePrint()
        if sol.success:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message, "sol.nit:", sol.nit)
        else:
            mpi.report("sol.success:", sol.success, "sol.message:", sol.message)
        self.disablePrint()
        Lambda, R = self.root_construct(x=sol.x)
        return Lambda, R, sol

    # Forward-recursion (fixed-point method, this should be the same as linearmixing with alpha=1 and no line search?)
    def solve_recursion(self, Lambda, R, alpha=1.):
        assert self.sc_method == 'recursion', "sc_method must be recursion if root_finder is recursion"
        mpi.report("recursion:")
        
        root_fun = self.root_fun
        root_flatten = self.root_flatten
        tol = self.root_tol
        maxiter = self.maxiter

        success = False
        x = root_flatten(A=Lambda, B=R)   # initial guess for x
        for n in range(maxiter):
            # search direction (xnew = xold + alpha * dx, where dx = xnew - xold)
            dx = root_fun(x=x) 
            x += alpha * dx
            #x += alpha * (1. - 0.01*n) * dx (to try mix in more as we go along, but this works out poorly)
            norm = np.linalg.norm(dx)
            
            self.enablePrint()
            if mpi.is_master_node():
                print(f'n: {n+1}, rms(risb): {norm}')
            self.disablePrint()
            
            if norm < tol:
                success = True
                break

        self.enablePrint()
        if success:
            mpi.report(f"The solution converged. nit: {n+1}, tol: {norm}")
        else:
            mpi.report(f"The solution did NOT converge. nit: {n+1} tol: {norm}")
        self.disablePrint()
        Lambda, R = self.root_construct(x=x)
        return Lambda, R, success

    def solve_pulay(self, Lambda, R, alpha=0.4):
        assert sc_method == 'recursion', "sc_method must be recursion if root_finder is pulay"
        mpi.report("pulay:")
        
        root_fun = self.root_fun
        root_flatten = self.root_flatten
        tol = self.root_tol
        maxiter = self.maxiter

        success = False
        x_in = root_flatten(A=Lambda, B=R)   # initial guess for x
        for n in range(maxiter):
            x_in_old = copy.copy(x_in)
            
            if n > 1:
                x_out_old = copy.copy(x_out)
                dx_old = copy.copy(dx)
            
            dx = root_fun(x=x_in) 
            x_out = x_in + dx

            if n > 1:
                beta =  np.dot(dx, dx-dx_old) / np.dot(dx,dx_old)
                x_in_tilde = (1.-beta)*x_in + beta*x_in_old
                x_out_tilde = (1.-beta)*x_out + beta*x_out_old
                x_in = (1.-alpha)*x_in_tilde + alpha*x_out_tilde

                norm = np.linalg.norm(x_in - x_in_old)
            
                self.enablePrint()
                if mpi.is_master_node():
                    print(f'n: {n+1}, rms(risb): {norm}')
                self.disablePrint()
             
            else:
                x_in = (1.-alpha)*x_in + alpha*x_out
                norm = 1.
                
            if norm < tol:
                success = True
                break
        
        self.enablePrint()
        if success:
            mpi.report(f"The solution converged. nit: {n+1}, tol: {norm}")
        else:
            mpi.report(f"The solution did NOT converge. nit: {n+1} tol: {norm}")
        self.disablePrint()
        Lambda, R = self.root_construct(x=x_in)
        return Lambda, R, success

    # Note that the crop algorithm should use N=3 and needs a C_inv conditioner to remove most of the non-linearities 
    # from the Jacobian, otherwise it will not converge.
    # alpha = 0.4 can sometimes be useful too
    def solve_diis(self, Lambda, R, alpha=1, N=5, C_inv=None, crop=False):
        if crop:
            mpi.report(f"diis({N})_crop alpha={alpha}:")
            root_solver = f"diis({N})_crop alpha={alpha}"
        else:
            mpi.report(f"diis({N}) alpha={alpha}:")
            root_solver = f"diis({N}) alpha={alpha}"
        
        root_fun = self.root_fun
        root_flatten = self.root_flatten
        tol = self.root_tol
        maxiter = self.maxiter

        x0 = root_flatten(A=Lambda, B=R)
        success = False
        
        if C_inv == None:
            C_inv = np.eye(len(x0))

        x = [np.empty(shape=len(x0)) for n in range(N)]
        r = [np.empty(shape=len(x0)) for n in range(N)]
        x[0] = x0

        for n in range(maxiter):
            r[n%N] = alpha * root_fun(x=x[n%N])
           
            norm = np.linalg.norm(r[n%N])
            self.enablePrint()
            if mpi.is_master_node():
                print(f'n: {n}, rms(risb): {norm}')
            self.disablePrint()
            
            if (norm < tol) or (n > maxiter):
                x = x[n%N] + np.dot(C_inv, r[n%N])
                success = True
                break
          
            if n > N-1:
                m = N
            else:
                m = n+1
            B = np.empty(shape=(m,m))
            
            # Construct the B matrix
            for i in range(m):
                for j in range(m):
                    B[i,j] = np.dot(r[i], np.dot(C_inv, r[j]) )

            # Add the constraint lambda
            B = np.column_stack( ( B, -np.ones(B.shape[0]) ) )
            B = np.vstack( ( B, -np.ones(B.shape[1]) ) )
            B[m,m] = 0.
            
            # Solve for the c coefficients (last element in c gives lambda constraint)
            rhs = np.zeros(B.shape[0])
            rhs[-1] = -1.
            c = np.dot(np.linalg.pinv(B), rhs)
            
            # Calculate optimial r(n)
            r_opt = np.zeros(x[0].shape)
            for i in range(m):
                r_opt += c[i] * r[i]

            # Calculate optimal x(n)
            x_opt = np.zeros(x[0].shape)
            for i in range(m):
                x_opt += c[i] * x[i]
            
            # CROP algorithm updates subspace with optimized vectors
            if crop:
                r[n%N] = copy.deepcopy(r_opt)
                x[n%N] = copy.deepcopy(x_opt)

            # Calculate optimal guess for the next iteration
            x[(n+1)%N] = x_opt + np.dot(C_inv, r_opt)

        self.enablePrint()
        if success:
            mpi.report(f"The solution converged. nit: {n+1}, tol: {norm}")
        else:
            x = x[(n+1)%N]
            mpi.report(f"The solution did NOT converge. nit: {n+1} tol: {norm}")
        self.disablePrint()
        Lambda, R = self.root_construct(x=x)
        return Lambda, R, success

    def solve_fixed_point(self, Lambda, R):
        assert sc_method == 'fixed-point', "sc_method must be fixed-point if root_finder is fixed-point"
        mpi.report("fixed-point:")
        x = optimize.fixed_point( func=self.root_fun, x0=self.root_flatten(A=Lambda, B=R), \
                                  xtol=self.root_tol, method='del2', maxiter=self.maxiter)
        Lambda, R = self.root_construct(x=x)
        return Lambda, R, x

    def convert_vasp_proj(self):
        if mpi.is_master_node():
            if not os.path.exists(self.plo_cfg_filename):
                raise FileNotFoundError("Need a PLO config file.")
                kill_all()
            plo_converter.generate_and_output_as_text(self.plo_cfg_filename, vasp_dir='./')
        mpi.barrier()

    def sumk_setup(self, it=0, recycle_mf=False, recycle_structure=False, folder='data/'):
        if not self.prnt:
            self.disablePrint()

        hdf_filename = folder + self.filename + '-converter' + f'-U{self.U:.3f}_J{self.J:.3f}'

        mpi.report('')
        # Convert the projectors
        if self.dft_solver == 'vasp':
            Converter = vaspconv.VaspConverter(filename = self.filename, hdf_filename = hdf_filename+'.h5')
        elif self.dft_solver == 'vasp-hk':
            Converter = vaspconv.VaspConverter(filename = self.filename, hdf_filename = hdf_filename+'.h5', proj_or_hk = 'hk')
        elif self.dft_solver == 'wien2k':
            Converter = wien2kconv.Wien2kConverter(filename = self.filename, hdf_filename = hdf_filename+'.h5')
        else:
            raise ValueError("dft_solver must be vasp or vasp-hk or wien2k")

        # Get the projectors # FIXME add for wien2k
        if self.dft_solver == 'vasp':
            self.convert_vasp_proj()
        else:
            raise ValueError("dft_solver must be vasp or vasp-hk or wien2k")
        mpi.report('')
        
        # Put projectors in hdf5 sumk format
        Converter.convert_dft_input()

        self.SK = SumkDFTRISB(hdf_file = hdf_filename+'.h5', use_dft_blocks = False,
                              beta=self.beta, smearing=self.smearing)
        mpi.report('')

        # Analyze the structure to get the blocks for the mean field matrices
        if recycle_structure:
            mpi.report('Loading block structure.')
            self.load_structure(it=it, folder=folder)
        else:
            mpi.report('Analysing block structure.')
            self.SK.analyse_block_structure(threshold = self.analyze_block_threshold)        
            # Analyze from the Green's function
            #Sigma = SK.block_structure.create_gf(beta=beta)
            #SK.put_Sigma([Sigma])
            #G = SK.extract_G_loc()
            #SK.analyse_block_structure_from_gf(G, threshold = self.analyze_block_threshold)
        
        for ish in range(len(self.SK.deg_shells)):
            num_block_deg_orbs = len(self.SK.deg_shells[ish])
            mpi.report('Found {0:d} blocks of degenerate orbitals in shell {1:d}'.format(num_block_deg_orbs, ish))
            for block in range(num_block_deg_orbs):
                mpi.report('block {0:d} consists of orbitals {1}'.format(block, self.SK.deg_shells[ish][block]))
        mpi.report('')
            
        # Find diagonal local basis set:
        if self.use_rotations:
            self.rot_mat = SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)

        # FIXME add options to remove orbitals from the projectors
        # Remove the completely filled orbitals
        #SK.block_structure.pick_gf_struct_solver([{'up_1': [0],'up_3': [0],'down_1': [0],'down_3': [0]}])

        # Get the local quadratic terms from H(k)
        h0_loc_mat = []
        for ish in range(self.SK.n_inequiv_shells):
            h0_loc_mat.append( self.SK.get_h_ksr_loc()[ish] )
       
        # Transform from sumk blocks to the solver blocks
        #ish = SK.corr_to_inequiv[0] # the index of the inequivalent shell corresponding to icrsh
        for ish in range(self.SK.n_inequiv_shells):
            icrsh = self.SK.inequiv_to_corr[ish] # FIXME
            h0_loc_mat[ish] = self.SK.mat_sumk_to_solver(h0_loc_mat[ish], icrsh=icrsh)
            self.SK.symm_deg_mat(h0_loc_mat[ish], ish=ish)

        # Construct it as a Hamiltonian
        # In this case it is always diagonal (and real)
        h0_loc = []
        for ish in range(self.SK.n_inequiv_shells):
            h0_loc.append( Operator() )
            for block,orbs in h0_loc_mat[ish].items():
                for idx,val in np.ndenumerate(orbs):
                    h0_loc[ish] += val.real * c_dag(block,idx[0]) * c(block,idx[1])
        self.h0_loc = h0_loc

        self.SK.smearing = self.smearing
     
        # Initialize Lambda and R to the non-interacting parameters
        _, _ = self.SK.initialize_R_Lambda(random=False, zero_Lambda=False)
        
        # FIXME if R is close to zero I should make it slightly non-zero so it does
        # not get stuck there
        if recycle_mf:
            mu, Lambda, R = self.load_mf(it=it, folder=folder)
            self.SK.chemical_potential = mu
            self.SK.Lambda = Lambda
            
            # Make sure R is not numerically zero # FIXME what to do here?
            for icrsh in range(self.SK.n_inequiv_shells):
                for key, value in R[icrsh].items():
                    if np.linalg.norm(value) < 1e-4:
                        value = 1 * np.eye(value.shape[0])
            self.SK.R = R
            self.SK.set_R_Lambda_sumk()

    def get_h_int(self):
        h_int = []
        
        for ish in range(self.SK.n_inequiv_shells):
            h_int.append( Operator() )
            icrsh = self.SK.inequiv_to_corr[ish]

            # The orbitals in the correlated shells
            n_orb = self.SK.corr_shells[icrsh]['dim']
            spin_names = ['up','down']
            orb_names = [i for i in range(0,n_orb)]
        

            # FIXME what if want to use kanamori or density density or whatever
            #U_sph = U_matrix(l=2, U_int=U, J_hund=J)
            #U_cubic = transform_U_matrix(U_sph, spherical_to_cubic(l=2, convention=''))
            #Umat, Upmat = reduce_4index_to_2index(U_cubic)
            #h_int[ish] = h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[ish], U=Umat, Uprime=Upmat)
            #h_int[ish] = h_int_kanamori(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[ish], U=Umat, Uprime=Upmat, J_hund=J, off_diag=True)

            # Construct the interacting hamiltonian in the slater parameterization
            Umat = U_matrix(l=2, U_int=self.U, J_hund=self.J, basis='other', T=self.SK.T[ish].conjugate())
            h_sumk = h_int_slater(spin_names=spin_names, orb_names=orb_names, map_operator_structure=self.SK.sumk_to_solver[ish], U_matrix=Umat, off_diag=True)
            if self.use_rotations:
                h_int[ish] = SK.block_structure.convert_operator(h_sumk, ish=icrsh)
            else:
                h_int[ish] = h_sumk
            #h_sumk = h_int_slater(spin_names=spin_names, orb_names=orb_names, U_matrix=U_mat, off_diag=True)
            
            h_int[ish] = h_int[ish].real # FIXME allow to be not real
        
        return h_int

    def dmft_cycle(self, reset_sumk=False, it=0, mix_mf=False, recycle_mf=False, recycle_structure=False, folder='data/'):
        # Remove the stopfile if exists
        if mpi.is_master_node():
            if os.path.isfile('STOPRISB'):
                os.remove('STOPRISB')

        if reset_sumk:
            self.sumk_setup(it=it, recycle_mf=recycle_mf, recycle_structure=recycle_structure, folder=folder)

        self.h_int = self.get_h_int()

        sol = None
        R = self.SK.R
        Lambda = self.SK.Lambda

        if not self.prnt:
            self.disablePrint()

        # FIXME use this for diis with some iteratison
        # Start off with a few recursive iterations to kick us away from metastable fixed point, or diverging points
        old_sc_method = copy.deepcopy(self.sc_method)
        self.sc_method = 'recursion'
        old_root_solver = copy.deepcopy(self.root_solver)
        self.root_solver = 'recursion'
        old_tol = copy.deepcopy(self.root_tol)
        self.root_tol = 5e-2
        Lambda, R, _ = self.solve_recursion(Lambda=Lambda, R=R)
        self.sc_method = copy.deepcopy(old_sc_method)
        self.root_solver = copy.deepcopy(old_root_solver)
        self.root_tol = copy.deepcopy(old_tol)
        
        # hybr seems to only converge to the correct solution (with lm recursion, and lm root), but sometimes does not work
        if self.root_solver == 'hybr':
            Lambda, R, sol = self.solve_hybr(Lambda=Lambda, R=R)
        # lm always converges to something, but can be the wrong solution if start off far away (with lm recursion)
        # (doing some recursion recursion can help with this)
        # Seems better to use lm root
        elif self.root_solver == 'lm':
            Lambda, R, sol = self.solve_lm(Lambda=Lambda, R=R)
        # Never used to work, but it sometimes works now. When it does work it converges to correct solution, else diverges.
        # (only for broyden recursion)
        elif self.root_solver == 'broyden':
            Lambda, R, sol = self.solve_broyden(Lambda=Lambda, R=R)
        # Never works
        elif self.root_solver == 'anderson':
            Lambda, R, sol = self.solve_anderson(Lambda=Lambda, R=R)
        # Only works with linearmixing recursion, and is very slow to converge
        # Slow convergence is likely due to the linesearch it uses
        elif self.root_solver == 'linearmixing':
            Lambda, R, sol = self.solve_linearmixing(Lambda=Lambda, R=R)
        # Mostly never converges (krylov recursion, or krylov root). When it does, it is very slow and can take a long time
        elif self.root_solver == 'krylov':
            Lambda, R, sol = self.solve_krylov(Lambda=Lambda, R=R)
        # Never works
        elif self.root_solver == 'df-sane':
            Lambda, R, sol = self.solve_df_sane(Lambda=Lambda, R=R)
        # Always works well. It is fastest with alpha=1
        elif self.root_solver == 'recursion':
            Lambda, R, success = self.solve_recursion(Lambda=Lambda, R=R)
        # very slow convergence, but does work
        elif self.root_solver == 'pulay':
            Lambda, R, success = self.solve_pulay(Lambda=Lambda, R=R)
        # works very well, very fast convergence. _very_ rarely goes to wrong fixed point
        elif self.root_solver == 'diis':
            Lambda, R, success = self.solve_diis(Lambda=Lambda, R=R)
        elif self.root_solver == 'crop':
            Lambda, R, success = self.solve_diis(Lambda=Lambda, R=R)
        # Works very well, but maybe slow? (note method=iteration is the same as recursion recursion with alpha=1)
        elif self.root_solver == 'fixed-point':
            Lambda, R, x = self.solve_fixed_point(Lambda=Lambda, R=R)
        else:
            raise ValueError("dmft_cycle: Implemented only for root finders hybr, lm, broyden, anderson, linearmixing, krylov, df-sane, recursion, and fixed-point.")
            
        if mix_mf:
            _, Lambda_old, R_old = self.load_mf(it=it, folder=folder)
            for icrsh in range(len(Lambda_old)):
                for block in self.SK.gf_struct_solver[icrsh].keys():
                    R[icrsh][block] = self.linear_mix_factor * R[icrsh][block] + (1.0-self.linear_mix_factor) * R_old[icrsh][block]
                    Lambda[icrsh][block] = self.linear_mix_factor * Lambda[icrsh][block] + (1.0-self.linear_mix_factor) * Lambda_old[icrsh][block]

        ## If is true solution, it should be a self-consistent one, so check
        old_sc_method = copy.deepcopy(self.sc_method)
        self.sc_method = 'recursion'
        old_root_solver = copy.deepcopy(self.root_solver)
        self.root_solver = 'recursion'
        old_maxiter = copy.deepcopy(self.maxiter)
        self.maxiter = 1
        _, _, _ = self.solve_recursion(Lambda=Lambda, R=R)
        self.sc_method = copy.deepcopy(old_sc_method)
        self.root_solver = copy.deepcopy(old_root_solver)
        self.maxiter = copy.deepcopy(old_maxiter)
        
        # FIXME test
        #risb_ke = self.SK.get_ke_total() 
        #risb_le = []
        #for ish in range(self.SK.n_inequiv_shells):
        #    h_loc = self.h0_loc[ish] + self.h_int[ish] - self.h_dc[ish]
        #    risb_le.append( self.S[ish].overlap(h_loc) )
        #risb_energy = copy.copy(risb_ke)
        #for ish in range(self.SK.n_inequiv_shells):
        #    risb_energy += risb_le[ish]
        
        # Get local density from impuirity in sumk space
        Nc_sumk = []
        for ish in range(self.SK.n_inequiv_shells):
            icrsh = self.SK.inequiv_to_corr[ish]
            Nc_sumk.append( self.SK.mat_solver_to_sumk(self.S[ish].Nc, icrsh=icrsh) )
        
        # Calculate the changes in densities
        _, self.dens, self.band_en_correction = self.SK.deltaN_risb(dm=Nc_sumk, dm_type='vasp') # band_en_correction only for vasp
        #deltaN, dens, band_en_correction = self.SK.calc_density_correction_risb(dm_type='vasp')

        # Calculate some energies
        self.dc_energy = self.SK.dc_energ
        self.corr_energy = []
        for ish in range(self.SK.n_inequiv_shells):
            self.corr_energy.append( self.S[ish].overlap(self.h_int[ish]) / self.SK.energy_unit )
        
        self.total_corr_energy = 0
        self.total_dc_energy = 0
        for icrsh in range(self.SK.n_inequiv_shells):
            self.total_corr_energy += self.corr_energy[icrsh]
            self.total_dc_energy = self.dc_energy[icrsh] / self.SK.energy_unit
                
        self.dft_energy = self.get_dft_energy()
        self.total_energy = self.dft_energy + self.band_en_correction / self.SK.energy_unit
        for icrsh in range(self.SK.n_corr_shells):
            ish = self.SK.corr_to_inequiv[icrsh]
            self.total_energy += self.corr_energy[ish] - self.dc_energy[ish]
        
        self.energy_correction = self.total_corr_energy - self.total_dc_energy
        
        self.enablePrint()
        
        mpi.report('')
        mpi.report(f'dens: {self.dens}')
        mpi.report(f'band_en_correction: {self.band_en_correction}')
        
        # print some observables
        mpi.report(f'mu: {self.SK.chemical_potential}')
        for ish in range(self.SK.n_inequiv_shells):
            mpi.report(f'Z: {self.S[ish].Z}')
            mpi.report(f'Lambda: {self.S[ish].Lambda}')
            mpi.report(f'Nc: {self.S[ish].Nc}')
            mpi.report(f'Total charge of correlated space: {self.S[ish].total_density}')
        
        #mpi.report(f'dens: {dens}')
        #mpi.report(f'dft_energy: {dft_energy}')
        #mpi.report(f'band_en_correction: {band_en_correction}')
        #mpi.report(f'correnerg: {correnerg}')
        #mpi.report(f'dc_energy: {self.SK.dc_energ}')
        #mpi.report(f'total_energy: {total_energy}')
        #mpi.report(f'risb_ke: {risb_ke}')
        #mpi.report(f'risb_le: {risb_le}')
        #mpi.report(f'risb_energy: {risb_energy}')
        mpi.report('')
 
        return self.total_corr_energy, self.total_dc_energy

    def save(self, it=0, folder='data/'):
        filename_out = folder + self.filename + f'-U{self.U:.3f}_J{self.J:.3f}'
        if mpi.is_master_node():
            ar = HDFArchive(filename_out + '.h5','a')
            if not 'risb' in ar: ar.create_group('risb')
            if not 'iterations' in ar['risb']: ar['risb'].create_group('iterations')
            
            ar['risb']['total_iterations'] = it
            for ish in range(self.SK.n_inequiv_shells):
                ar['risb']['iterations']['Lambda_ish'+str(ish)+'_it'+str(it)] = self.S[ish].Lambda
                ar['risb']['iterations']['R_ish'+str(ish)+'_it'+str(it)] = self.S[ish].R
                ar['risb']['iterations']['Z_ish'+str(ish)+'_it'+str(it)] = self.S[ish].Z
                ar['risb']['iterations']['Nc_ish'+str(ish)+'_it'+str(it)] = self.S[ish].Nc
                ar['risb']['iterations']['Nf_ish'+str(ish)+'_it'+str(it)] = self.S[ish].Nf
                ar['risb']['iterations']['total_density_ish'+str(ish)+'_it'+str(it)] = self.S[ish].total_density
                
                ar['risb']['iterations']['Vdc_up_ish'+str(ish)+'_it'+str(it)] = self.SK.dc_imp[ish]['up'][0,0].real
                ar['risb']['iterations']['Vdc_down_ish'+str(ish)+'_it'+str(it)] = self.SK.dc_imp[ish]['down'][0,0].real
                ar['risb']['iterations']['dc_energy_ish'+str(ish)+'_it'+str(it)] = self.dc_energy[ish]
                ar['risb']['iterations']['corr_energy_ish'+str(ish)+'_it'+str(it)] = self.corr_energy[ish]
            
            ar['risb']['iterations']['mu_it'+str(it)] = self.SK.chemical_potential
            ar['risb']['iterations']['dens_it'+str(it)] = self.dens
            ar['risb']['iterations']['band_en_correction_it'+str(it)] = self.band_en_correction

            ar['risb']['iterations']['dft_energy_it'+str(it)] = self.dft_energy
            ar['risb']['iterations']['total_dc_energy_it'+str(it)] = self.total_dc_energy
            ar['risb']['iterations']['total_corr_energy_it'+str(it)] = self.total_corr_energy
            ar['risb']['iterations']['total_energy_it'+str(it)] = self.total_energy
            
            ar['risb']['iterations']['beta_it'+str(it)] = self.beta
            
            ar['risb']['gf_struct_solver'] = self.SK.gf_struct_solver
            ar['risb']['sumk_to_solver'] = self.SK.sumk_to_solver
            ar['risb']['solver_to_sumk'] = self.SK.solver_to_sumk
            ar['risb']['solver_to_sumk_block'] = self.SK.solver_to_sumk_block
            ar['risb']['deg_shells'] = self.SK.deg_shells

        if mpi.is_master_node(): del ar 
    
    def load_mf(self, it=0, folder='data/'):
        
        filename_out = folder + self.filename + f'-U{self.U:.3f}_J{self.J:.3f}'
        if mpi.is_master_node():
            ar = HDFArchive(filename_out + '.h5','a')
 
            mu = ar['risb']['iterations']['mu_it'+str(it)]
            Lambda = copy.deepcopy(self.SK.Lambda) # FIXME
            R = copy.deepcopy(self.SK.R)
            for ish in range(self.SK.n_inequiv_shells):
                Lambda[ish] = ar['risb']['iterations']['Lambda_ish'+str(ish)+'_it'+str(it)]
                R[ish] = ar['risb']['iterations']['R_ish'+str(ish)+'_it'+str(it)]

        if mpi.is_master_node(): del ar
        mu = mpi.bcast(mu)
        Lambda = mpi.bcast(Lambda)
        R = mpi.bcast(R)

        return mu, Lambda, R

    def load_structure(self, it=0, folder='data/'):
        
        def string_to_tuple(string):
            string = string.replace("(", "")
            string = string.replace(")", "")
            string = string.replace("'", "")
            string = string.replace(",", " ")
            lst = string.split()
            lst[1] = int(lst[1])
            return tuple(lst)

        def fix_keys(struct):
            new_struct = [dict() for ish in range(len(struct))]
            for ish in range(len(struct)):
                for key, value in struct[ish].items():
                    if isinstance(key, str):
                        new_key = string_to_tuple(key)
                        new_struct[ish][new_key] = value
            return new_struct
        
        filename_out = folder + self.filename + f'-U{self.U:.3f}_J{self.J:.3f}'
        
        if mpi.is_master_node():
            ar = HDFArchive(filename_out + '.h5','a')
            
            self.SK.gf_struct_solver = ar['risb']['gf_struct_solver']
            # For some reason the next two get pickled weirdly to hdf5 structure so fix it
            self.SK.sumk_to_solver = fix_keys(ar['risb']['sumk_to_solver'])
            self.SK.solver_to_sumk = fix_keys(ar['risb']['solver_to_sumk'])
            self.SK.solver_to_sumk_block = ar['risb']['solver_to_sumk_block']
            self.SK.deg_shells = ar['risb']['deg_shells']

            self.SK.set_identity_sumk()
        
        if mpi.is_master_node(): del ar
        
        self.SK.gf_struct_solver = mpi.bcast(self.SK.gf_struct_solver)
        self.SK.sumk_to_solver = mpi.bcast(self.SK.sumk_to_solver)
        self.SK.solver_to_sumk = mpi.bcast(self.SK.solver_to_sumk)
        self.SK.solver_to_sumk_block = mpi.bcast(self.SK.solver_to_sumk_block)
        self.SK.deg_shells = mpi.bcast(self.SK.deg_shells)