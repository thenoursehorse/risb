import os
import signal
import subprocess
import time
from shutil import copyfile

import triqs.utility.mpi as mpi

class cscVASP(object):
    def __init__(self, vasp_cmd='vasp_std', vasp_name='vasp_std', write_wavecar=True, width=10, oneshot_save=True):
        self.vasp_cmd = vasp_cmd
        self.vasp_name = vasp_name
        self.write_wavecar = write_wavecar
        self.width = width
        self.oneshot_save = oneshot_save

        self.mpi_spawn = mpi.MPI.COMM_SELF.Spawn
        self.mpi_size = mpi.size
            
        self.output_variables = {}

    # FIXME assumes that all running vasp is associated with this run
    # also, what if another user is running vasp?
    def vasp_pids(self):
        pids = [int(x) for x in (subprocess.check_output(["pidof",self.vasp_name]).split())]
        return pids
    
    # FIXME I need a way to catch an error in python, and then kill the vasp pids
    # because sometimes a bad input will kill python but vasp will keep going
    def kill_vasp(self,pids):
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
    
    def kill_all(self):
        if mpi.is_master_node():
            print('\nKilling all.\n', flush=True)
        mpi.world.Abort(1) # FIXME do I need to do more than this to kill the whole application?
    
    def start_vasp(self):
        if mpi.is_master_node():
            if os.path.isfile('STOPCAR'):
                os.remove('STOPCAR')
    
        if mpi.is_master_node():
            print('\nStarting VASP\n')
            self.mpi_spawn(self.vasp_cmd, \
                           args="", \
                           maxprocs=self.mpi_size)
        mpi.barrier()
    
    def resume_vasp(self):
        if mpi.is_master_node():
            open('./vasp.lock', 'a').close()
        mpi.barrier()
    
    def stop_vasp(self):
        # FIXME need to wait for vasp to properly stop before kill all if write_wavecar is true
        if self.write_wavecar:
            if mpi.is_master_node():
                print('\nStopping VASP.\n')
                with open('STOPCAR', 'wt') as f:
                    f.write('LABORT = .TRUE.\n')
        mpi.barrier()
        #self.kill_vasp()
        self.kill_all()
    
    def is_vasp_lock_present(self):
        res_bool = False
        if mpi.is_master_node():
            res_bool = os.path.isfile('./vasp.lock')
        mpi.barrier()
        res_bool = mpi.bcast(res_bool)
        return res_bool
    
    def remove_gamma(self):
        if mpi.is_master_node():
            if os.path.isfile('./GAMMA'):
                os.remove('./GAMMA')
        mpi.barrier()
    
    def print_csc(self):
        width = self.width
        if mpi.is_master_node():
            print()
            print("="*80)
            print(f"{'iter': <6} \
                    {'dE': <{width}.{width}} \
                    {'Total Energy': <{width}.{width}} \
                    {'DFT Energy': <{width}.{width}} \
                    {'Corr. Energy': <{width}.{width}} \
                    {'DFT DC': <{width}.{width}} \
                    {'E Correction': <{width}.{width}}" \
                 )
            print("{iteration: <6} \
                   {dE: <{width}.{width}G} \
                   {total_energy: <{width}.{width}G} \
                   {dft_energy: <{width}.{width}G} \
                   {corr_energy: <{width}.{width}G} \
                   {dc_energy: <{width}.{width}G} \
                   {energy_correction: <{width}.{width}G}".format(iteration = self.output_variables['iteration'], \
                                                                  dE = self.output_variables['dE'], \
                                                                  total_energy = self.output_variables['total_energy'], \
                                                                  dft_energy = self.output_variables['dft_energy'], \
                                                                  corr_energy = self.output_variables['corr_energy'], \
                                                                  dc_energy = self.output_variables['dc_energy'], \
                                                                  energy_correction = self.output_variables['energy_correction'], \
                                                                  width = width) \
                 )
            print("="*80)
            print()
        mpi.barrier()
    
    def save_csc(self):
        width = self.width
        if mpi.is_master_node():
            if self.output_variables['iteration'] == 0:
                with open('CSCOUT', 'w') as f:
                    f.write(f"{'iter': <6} \
                            {'dE': <{width}.{width}} \
                            {'Total Energy': <{width}.{width}} \
                            {'DFT Energy': <{width}.{width}} \
                            {'Corr. Energy': <{width}.{width}} \
                            {'DFT DC': <{width}.{width}} \
                            {'E Correction': <{width}.{width}}\n" \
                        )
            with open('CSCOUT', 'a') as f:
                f.write("{iteration: <6} \
                        {dE: <{width}.{width}G} \
                        {total_energy: <{width}.{width}G} \
                        {dft_energy: <{width}.{width}G} \
                        {corr_energy: <{width}.{width}G} \
                        {dc_energy: <{width}.{width}G} \
                        {energy_correction: <{width}.{width}G}\n".format(iteration = self.output_variables['iteration'], \
                                                            dE = self.output_variables['dE'], \
                                                            total_energy = self.output_variables['total_energy'], \
                                                            dft_energy = self.output_variables['dft_energy'], \
                                                            corr_energy = self.output_variables['corr_energy'], \
                                                            dc_energy = self.output_variables['dc_energy'], \
                                                            energy_correction = self.output_variables['energy_correction'], \
                                                            width = width) \
                    )
        mpi.barrier()
    
    def run_csc(self, oneshot_obj, n_iter, n_iter_dft=1, tol=1e-4, min_iterations=5, mix_mf=False, recycle_mf=False, recycle_structure=False, remove_gamma=True):
        # Requires to have done a single oneshot iteration first
        if remove_gamma:
            copyfile(src='GAMMA',dst='GAMMA_recent')
        old_total_energy = oneshot_obj.total_energy
        
        self.start_vasp()
        # Wait for vasp to start (creates a lock file)
        #while not self.is_vasp_lock_present():
        #    time.sleep(1)
                     
        iteration = 1
        dE = 10
        while (iteration < n_iter) and (abs(dE) > abs(tol)):
            # Remove gamma if vasp wrote it
            #self.remove_gamma()
            
            # Run the dft steps
            iter_dft = 0
            if remove_gamma:
                copyfile(src='GAMMA',dst='GAMMA_recent')
            while iter_dft < n_iter_dft:
                self.resume_vasp()
                while self.is_vasp_lock_present():
                    time.sleep(1)
                iter_dft += 1
                if remove_gamma:
                    copyfile(src='GAMMA_recent',dst='GAMMA')
    
            # Run the dmft cycle and write gamma for next dft
            _, _ = oneshot_obj.dmft_cycle(reset_sumk=True, it=iteration-1, mix_mf=mix_mf, recycle_mf=recycle_mf, recycle_structure=recycle_structure)
            if self.oneshot_save:
                oneshot_obj.save(it=iteration)
            dE = oneshot_obj.total_energy - old_total_energy
            mpi.barrier()
            
            # store old energy
            old_total_energy = oneshot_obj.total_energy
    
            # Print and save energies
            self.output_variables['iteration'] = iteration
            self.output_variables['dE'] = dE
            self.output_variables['dft_energy'] = oneshot_obj.dft_energy
            self.output_variables['corr_energy'] = oneshot_obj.total_corr_energy
            self.output_variables['dc_energy'] = oneshot_obj.total_dc_energy
            self.output_variables['total_energy'] = oneshot_obj.total_energy
            self.output_variables['energy_correction'] = oneshot_obj.energy_correction
            self.print_csc()
            self.save_csc()
                            
            iteration += 1
    
            if iteration >= n_iter:
                if mpi.is_master_node():
                    print("\nMaximum number of iterations reached.\n")
            
            if abs(dE) < abs(tol):
                if mpi.is_master_node():
                    print("\nTolerance reached.\n")

            if iteration < min_iterations:
                dE = 10
    
        # Stop vasp and exit
        self.stop_vasp()