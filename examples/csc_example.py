import sys
import os

from risb.dftx import cscVASP
from risb.dftx import oneshotRISB

import triqs.utility.mpi as mpi
import sys
from fractions import Fraction
import numpy as np

if __name__ == '__main__':
    # The parameters
    mpi.report("Usage: python3 csc_example.py <filename> <dft_solver> <root_solver> <sc_method> <U> <J> <print> <save> <n_iter_dft>")
    if len(sys.argv) != 10:
        sys.exit(1)
    mpi.report('Argument List:', str(sys.argv))
    filename = str(sys.argv[1])
    dft_solver = str(sys.argv[2])
    root_solver = str(sys.argv[3])
    sc_method = str(sys.argv[4])
    U = float(sys.argv[5])
    J = float(sys.argv[6])
    prnt = str(sys.argv[7])
    if prnt in ['True','true','1']:
        prnt = True
    else:
        prnt = False
    save = str(sys.argv[8])
    if save in ['True','true','1']:
        save = True
    else:
        save = False
    n_iter_dft = int(sys.argv[9])

    # Setup oneshot risb container
    oneshot = oneshotRISB(filename=filename, dft_solver=dft_solver, \
                          root_solver=root_solver, sc_method=sc_method, \
                          prnt=prnt, plo_cfg_filename='plo.cfg',
                          beta=10.0,root_tol=1e-6,mu_tol=1e-6,
                          U=U, J=J)

    # Setup csc container for vasp
    vasp_dir = "$HOME/vasp/bin/vasp_std"
    vasp_cmd = f'{vasp_dir}'
    csc = cscVASP(vasp_cmd=vasp_cmd, oneshot_save=save)

    # Do a single dmft cycle and output GAMMA
    # because vasp requires a GAMMA on the first iteration
    res = oneshot.dmft_cycle(reset_sumk=True)

    # Run the csc calculation now
    csc.run_csc(oneshot, n_iter_dft)