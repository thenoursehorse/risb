import os
import sys
import time

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

from embedding_ed import EmbeddingEd

from risb_solver import risbSolver
from sumk_dft_risb import SumkDFTRISB
  
class oneshotRISB(object):
    def __init__(self, filename,
                       U,
                       J,
                       dft_solver='vasp',
                       sc_method='recursion',
                       prnt=True, 
                       plo_cfg_filename='plo.cfg',
                       analyze_block_tol=1e-2, 
                       use_rotations=False,
                       beta=100,
                       smearing='fermi',
                       calc_mu = True,
                       mu_tol=1e-6,
                       mu_delta=0.5,
                       mu_max_iter=100,
                       mu_method='dichotomy',
                       mu_offset=0,
                       mu_mix_factor=1,
                       DC_type=0,
                       DC_value=None):
    
        self.filename = filename
        self.dft_solver = dft_solver
        self.sc_method = sc_method
        self.prnt = prnt
        self.plo_cfg_filename = plo_cfg_filename
        
        self.U = U
        self.J = J

        self.beta = beta
        self.smearing = smearing # gaussian, fermi, methfessel-paxton
        self.analyze_block_tol = analyze_block_tol
        self.calc_mu = calc_mu
        self.mu_tol = mu_tol
        self.mu_delta = mu_delta
        self.mu_max_iter = mu_max_iter
        self.mu_method = mu_method
        self.mu_offset = mu_offset
        self.mu_mix_factor = mu_mix_factor
                
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

    # FIXME Fix up embedding solver so can just add h_dc as a constant energy shift
    # Why can't I just add dc to the mean-field matrices?
    def get_h_dc(self, dm=None):
        h_dc = [Operator() for _ in range(self.SK.n_inequiv_shells)]
    
        # Calculate double counting
        Vdc = []
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            ish_corr = self.SK.inequiv_to_corr[ish_inequiv]
            if dm is None:
                dm_in = self.SK.get_pdensity(ish_inequiv=ish_inequiv)
            else:
                dm_in = dm
            self.disablePrint() # FIXME what does orb take?
            self.SK.calc_dc(dm_in, U_interact=self.U, J_hund=self.J, orb=ish_corr, use_dc_formula=self.DC_type, use_dc_value=self.DC_value)
            self.enablePrint()
            # Below is to ensure Vdc is rotated into solver space
            Vdc.append( next(iter( self.SK.block_structure.convert_matrix(G=self.SK.dc_imp[ish_corr],
                                                                ish_from=ish_corr,
                                                                ish_to=ish_inequiv,
                                                                space_from='sumk',
                                                                space_to='solver',
                                                                show_warnings=True
                                                                ).items() ))[1][0,0].real )
            for index, (key, value) in enumerate(self.SK.sumk_to_solver[ish_inequiv].items()):
                if (value[0] == None) or (value[1] == None):
                    continue
                s = key[0]
                o = key[1]
                h_dc[ish_inequiv] += Vdc[ish_inequiv] * n(value[0],value[1])
        
        self.SK.dc_imp = mpi.bcast(self.SK.dc_imp)
        self.SK.dc_energ = mpi.bcast(self.SK.dc_energ)
    
        return h_dc
    
    def get_h_int(self):
        h_int = [Operator() for _ in range(self.SK.n_inequiv_shells)]
        
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            ish_corr = self.SK.inequiv_to_corr[ish_inequiv]

            # The orbitals in the correlated shells
            n_orb = self.SK.corr_shells[ish_corr]['dim']
            spin_names = ['up','down']
            orb_names = [i for i in range(0,n_orb)]
        

            # FIXME what if want to use kanamori or density density or whatever
            #U_sph = U_matrix(l=2, U_int=U, J_hund=J)
            #U_cubic = transform_U_matrix(U_sph, spherical_to_cubic(l=2, convention=''))
            #Umat, Upmat = reduce_4index_to_2index(U_cubic)
            #h_int[ish] = h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[ish], U=Umat, Uprime=Upmat)
            #h_int[ish] = h_int_kanamori(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[ish], U=Umat, Uprime=Upmat, J_hund=J, off_diag=True)

            # Construct the interacting hamiltonian in the slater parameterization
            Umat = U_matrix(l=2, U_int=self.U, J_hund=self.J, basis='other', T=self.SK.T[ish_inequiv].conjugate())
            h_sumk = h_int_slater(spin_names=spin_names, orb_names=orb_names, map_operator_structure=self.SK.sumk_to_solver[ish_inequiv], U_matrix=Umat, off_diag=True)
            if self.use_rotations:
                h_int[ish_inequiv] = SK.block_structure.convert_operator(h_sumk, ish=ish_corr)
            else:
                h_int[ish_inequiv] = h_sumk
            #h_sumk = h_int_slater(spin_names=spin_names, orb_names=orb_names, U_matrix=U_mat, off_diag=True)
            
            h_int[ish_inequiv] = h_int[ish_inequiv].real # FIXME allow to be not real
        
        return h_int

    def update_SK(self, Lambda, R, mu=None):
        # Ensure symmetries are enforced #FIXME inequiv or corr?
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            self.SK.symm_deg_mat(Lambda[ish_inequiv], ish_inequiv=ish_inequiv)
            self.SK.symm_deg_mat(R[ish_inequiv], ish_inequiv=ish_inequiv)

        # Update the sumk calss
        self.SK.Lambda = copy.deepcopy(Lambda)
        self.SK.R = copy.deepcopy(R)
        self.SK.set_R_Lambda_sumk()
        if mu is not None:
            self.SK.chemical_potential = mpi.bcast(mu)

    def risb_one_cycle(self, Lambda, R, mu=None):
        self.update_SK(Lambda=Lambda, R=R)
        
        start = time.time()
        # Calculate mu and density
        if self.calc_mu:
            self.disablePrint()
            old_mu = self.SK.chemical_potential
            if np.abs(self.mu_offset) > self.mu_tol:
                mu_below = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method, offset=-self.mu_offset)
                mu_above = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method, offset=self.mu_offset)
                mu = 0.5*(mu_above + mu_below)
            else:
                mu = self.SK.calc_mu_risb(precision = self.mu_tol, delta=self.mu_delta, mu_max_iter=self.mu_max_iter, method=self.mu_method)
            mu = self.mu_mix_factor * mu + (1.0-self.mu_mix_factor) * old_mu
            self.enablePrint()
            self.SK.chemical_potential = mpi.bcast(mu)
        end = time.time()
        print("Time to calc mu: ", end-start)

        # Get new h_dc
        self.h_dc = self.get_h_dc()

        mpi.barrier()
        
        # Solve one cycle of RISB
        Lambda_new = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
        R_new = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
        F1 = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
        F2 = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]

        self.S = []
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            h_loc = self.h0_loc[ish_inequiv] + self.h_int[ish_inequiv] - self.h_dc[ish_inequiv]

            if mpi.is_master_node():
                print("Solving U =", self.U, "J =", self.J, "impurity for shell", ish_inequiv, "with", self.sc_method, "mu =", self.SK.chemical_potential)
            
            start = time.time()
            emb_solver = EmbeddingEd(h_loc, self.SK.block_structure.gf_struct_solver_list[ish_inequiv])
            end = time.time()
            print(f"Time to setup embedding {ish_inequiv}: ", end-start)
            self.S.append( risbSolver(emb_solver=emb_solver, ish_inequiv=ish_inequiv) )
            
            start = time.time()
            Lambda_new[ish_inequiv], R_new[ish_inequiv], F1[ish_inequiv], F2[ish_inequiv] = self.S[ish_inequiv].one_cycle(SK=self.SK)
            end = time.time()
            print(f"Time to solve embedding {ish_inequiv}: ", end-start)
        
        if not self.calc_mu:
            density_new = self.SK.total_density_risb()
            density_error = density_new - self.SK.density_required
        else:
            density_new = None
            density_error = None

        # If S.one_cycle returns new Lambda and R, error function is the
        # difference to previous iteration (error from fixed point)
        if self.sc_method == 'recursion':
            Lambda_error = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
            R_error = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
            for ish_inequiv in range(self.SK.n_inequiv_shells):
                for block in Lambda[ish_inequiv].keys():
                    Lambda_error[ish_inequiv][block] = Lambda_new[ish_inequiv][block] - Lambda[ish_inequiv][block]
            
            for ish_inequiv in range(self.SK.n_inequiv_shells):
                for block in R[ish_inequiv].keys():
                    R_error[ish_inequiv][block] = R_new[ish_inequiv][block] - R[ish_inequiv][block]

        # If S.one_cycle returns the two root equations f1 and f2, the error
        # is how far it is from a root
        elif (self.sc_method == 'root'):
            Lambda_error = F1
            R_error = F2
        
        else:
            raise ValueError("risb_error: Implemented only for recursion and root problem !")

        output = {}
        output['Lambda'] = Lambda_new
        output['R'] = R_new
        output['density'] = density_new
        output['Lambda_error'] = Lambda_error
        output['R_error'] = R_error
        output['density_error'] = density_error

        return output

    def flatten(self, Lambda, R, mu=None):
        x = []
        
        # Only 1 of each deg orbital
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish_inequiv]:
                block_0 = degsh[0]
                for i,j in np.ndindex(Lambda[ish_inequiv][block_0].shape):
                    x.append( Lambda[ish_inequiv][block_0][i,j].real )
        
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish_inequiv]:
                block_0 = degsh[0]
                for i,j in np.ndindex(R[ish_inequiv][block_0].shape):
                    x.append( R[ish_inequiv][block_0][i,j].real )

        if mu is not None:
            x.append(mu)

        return np.array(x, dtype=np.float_)

    def construct(self, x):
        Lambda = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
        R = [{} for ish_inequiv in range(self.SK.n_inequiv_shells)]
        counter = 0
        
        # Only 1 of each deg orbital
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish_inequiv]:
                block_0 = degsh[0]
                Lambda[ish_inequiv][block_0] = np.zeros(self.SK.Lambda[ish_inequiv][block_0].shape, dtype=np.float_)
                for i,j in np.ndindex(Lambda[ish_inequiv][block_0].shape):
                    Lambda[ish_inequiv][block_0][i,j] = x[counter]
                    counter += 1
                for block in degsh:
                    Lambda[ish_inequiv][block] = Lambda[ish_inequiv][block_0]
         
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            for degsh in self.SK.deg_shells[ish_inequiv]:
                block_0 = degsh[0]
                R[ish_inequiv][block_0] = np.zeros(self.SK.R[ish_inequiv][block_0].shape, dtype=np.float_)
                for i,j in np.ndindex(R[ish_inequiv][block_0].shape):
                    R[ish_inequiv][block_0][i,j] = x[counter]
                    counter += 1
                for block in degsh:
                    R[ish_inequiv][block] = R[ish_inequiv][block_0]

        if counter < len(x):
            mu = x[-1]
            return Lambda, R, mu
        
        return Lambda, R, None

    def stop_check(self):
        if mpi.is_master_node:
            if os.path.isfile('STOPRISB'):
                print('\nStopping RISB.\n', flush=True)
                mpi.world.Abort(1)
    
    def function(self, x):
        self.stop_check()
        Lambda, R, mu = self.construct(x=x)
        output = self.risb_one_cycle(Lambda=Lambda, R=R, mu=mu)
        x_new = self.flatten(Lambda=output['Lambda'], R=output['R'], mu=output['density'])
        x_error = self.flatten(Lambda=output['Lambda_error'], R=output['R_error'], mu=output['density_error'])
        return x_new, x_error

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

        deg_shells_empty = copy.deepcopy(self.SK.deg_shells)

        # Analyze the structure to get the blocks for the mean field matrices
        if recycle_structure:
            mpi.report('Loading block structure.')
            self.load_structure(it=it, folder=folder)
        else:
            mpi.report('Analysing block structure.')
            self.SK.analyse_block_structure(threshold=self.analyze_block_tol)
            # Analyze from the Green's function
            #Sigma = SK.block_structure.create_gf(beta=beta)
            #SK.put_Sigma([Sigma])
            #G = SK.extract_G_loc()
            #SK.analyse_block_structure_from_gf(G, threshold = self.analyze_block_tol)
        
        #self.SK.deg_shells = copy.deepcopy(deg_shells_empty)

        # FIXME to specifically set the gf_struct
        t2g = [0,1,3]
        eg = [2,4]
        dm = np.zeros([5,5])
        for i in t2g:
            for j in t2g:
                dm[i,j] = 1
        for i in eg:
            for j in eg:
                dm[i,j] = 2
        dm_full = [dict()]
        dm_full[0]['up'] = dm
        dm_full[0]['down'] = dm
        #self.SK.analyse_block_structure(threshold = self.analyze_block_threshold, dm=dm_full)
        
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            num_block_deg_orbs = len(self.SK.deg_shells[ish_inequiv])
            mpi.report('Found {0:d} blocks of degenerate orbitals in shell {1:d}'.format(num_block_deg_orbs, ish_inequiv))
            for block in range(num_block_deg_orbs):
                mpi.report('block {0:d} consists of orbitals {1}'.format(block, self.SK.deg_shells[ish_inequiv][block]))
        mpi.report('')
            
        # Find diagonal local basis set:
        if self.use_rotations:
            self.rot_mat = SK.calculate_diagonalization_matrix(prop_to_be_diagonal='eal', calc_in_solver_blocks=True)

        # FIXME add options to remove orbitals from the projectors
        # Remove the completely filled orbitals
        #SK.block_structure.pick_gf_struct_solver([{'up_1': [0],'up_3': [0],'down_1': [0],'down_3': [0]}])

        # Get the local quadratic terms from H(k)
        self.h0_sumk_loc = self.SK.get_h0_sumk_loc()
        
        # Transform from sumk blocks to the solver blocks
        self.h0_solver_loc = []
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            ish_corr = self.SK.inequiv_to_corr[ish_inequiv]
            self.h0_solver_loc.append( self.SK.mat_sumk_to_solver(self.h0_sumk_loc[ish_corr], ish_corr=ish_corr) )
            self.SK.symm_deg_mat(self.h0_solver_loc[ish_inequiv], ish_inequiv=ish_inequiv)

        # Construct it as a Hamiltonian
        # In this case it is always diagonal (and real)
        # FIXME Surely this can't be only diagonal! (and always real)
        self.h0_loc = [Operator() for _ in range(self.SK.n_inequiv_shells)]
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            for block,h0_s in self.h0_solver_loc[ish_inequiv].items():
                for idx,val in np.ndenumerate(h0_s):
                    self.h0_loc[ish_inequiv] += val.real * c_dag(block,idx[0]) * c(block,idx[1])

        self.SK.smearing = self.smearing
     
        # Initialize Lambda and R
        _, _ = self.SK.initialize_R_Lambda(random=False, zero_Lambda=False)
        
        # FIXME if R is close to zero I should make it slightly non-zero so it does
        # not get stuck there
        if recycle_mf:
            mu, Lambda, R = self.load_mf(it=it, folder=folder)

            # Make sure R is not numerically zero # FIXME what to do here?
            for ish_inequiv in range(self.SK.n_inequiv_shells):
                for key, value in R[ish_inequiv].items():
                    if np.linalg.norm(value) < 1e-4:
                        value = 1 * np.eye(value.shape[0])
            self.update_SK(Lambda=Lambda, R=R, mu=mu)

    def dmft_cycle(self, solver, reset_sumk=False, it=0, mix_mf=False, recycle_mf=False, recycle_structure=False, folder='data/'):
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

        if self.calc_mu:
            x0 = self.flatten(Lambda=Lambda, R=R, mu=None)
        else:
            x0 = self.flatten(Lambda=Lambda, R=R, mu=self.SK.chemical_potential)
        x = solver.solve(x0=x0, function=self.function)
        Lambda, R, mu = self.construct(x)

        # Get local density from impuirity in sumk space
        Nc_sumk = []
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            ish_corr = self.SK.inequiv_to_corr[ish_inequiv]
            Nc_sumk.append( self.SK.mat_solver_to_sumk(self.S[ish_inequiv].Nc, ish_corr=ish_corr) )
        
        # Calculate the changes in densities
        _, self.dens, self.band_en_correction = self.SK.deltaN_risb(dm=Nc_sumk, dm_type='vasp') # band_en_correction only for vasp
        #deltaN, dens, band_en_correction = self.SK.calc_density_correction_risb(dm_type='vasp')

        # Calculate some energies
        self.dc_energy = self.SK.dc_energ
        self.corr_energy = []
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            self.corr_energy.append( self.S[ish_inequiv].overlap(self.h_int[ish_inequiv]) / self.SK.energy_unit )
        
        self.total_corr_energy = 0
        self.total_dc_energy = 0
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            self.total_corr_energy += self.corr_energy[ish_inequiv]
            self.total_dc_energy = self.dc_energy[ish_inequiv] / self.SK.energy_unit
                
        self.dft_energy = self.get_dft_energy()
        self.total_energy = self.dft_energy + self.band_en_correction / self.SK.energy_unit
        for ish_corr in range(self.SK.n_corr_shells):
            ish_inequiv = self.SK.corr_to_inequiv[ish_corr]
            self.total_energy += self.corr_energy[ish_inequiv] - self.dc_energy[ish_inequiv]
        
        self.energy_correction = self.total_corr_energy - self.total_dc_energy
        
        self.enablePrint()
        
        mpi.report('')
        mpi.report(f'dens: {self.dens}')
        mpi.report(f'band_en_correction: {self.band_en_correction}')
        mpi.report(f'Total_energy: {self.total_energy}')
        
        # print some observables
        mpi.report(f'mu: {self.SK.chemical_potential}')
        for ish_inequiv in range(self.SK.n_inequiv_shells):
            mpi.report(f'Z: {self.S[ish_inequiv].Z}')
            mpi.report(f'Lambda: {self.S[ish_inequiv].Lambda}')
            mpi.report(f'Nc: {self.S[ish_inequiv].Nc}')
            mpi.report(f'Total charge of correlated space: {self.S[ish_inequiv].total_density}')
        
        #mpi.report(f'dens: {dens}')
        #mpi.report(f'dft_energy: {dft_energy}')
        #mpi.report(f'band_en_correction: {band_en_correction}')
        #mpi.report(f'correnerg: {correnerg}')
        #mpi.report(f'dc_energy: {self.SK.dc_energ}')
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