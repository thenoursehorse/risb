import sys
import os
import argparse
from distutils.util import strtobool
import numpy as np

from pathlib import Path
home = str(Path.home())
# FIXME
sys.path.insert(1, os.path.join(sys.path[0], home + '/Documents/GitHub/risb/python/risb/dftx/'))
sys.path.insert(1, os.path.join(sys.path[0], home + '/Documents/GitHub/risb/python/risb/sc_cycle/'))

#from solver import SolverNewton, Annealing
#from solver import LinearMixing, DIIS, DIIS2
from solver import *
from oneshot_risb_new import oneshotRISB

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str, required=True)
    parser.add_argument('-U', type=float, required=True)
    parser.add_argument('-J', type=float, required=True)
    parser.add_argument('-dft_solver', type=str, default='vasp')
    parser.add_argument('-sc_method', type=str, default='recursion')
    parser.add_argument('-plo_cfg', type=str, default='plo.cfg')
    parser.add_argument('-analyze_block_tol', type=float, default=1e-2)
    parser.add_argument('-beta', type=float, default=100)
    parser.add_argument('-alpha', type=float, default=0.05)
    parser.add_argument('-maxiter', type=int, default=500)
    parser.add_argument('-tol', type=float, default=1e-5)
    parser.add_argument('-history_size', type=int, default=5)
    parser.add_argument('-full_restart', type=int, default=0)
    parser.add_argument('-calc_mu', type=lambda x: bool(strtobool(x)), default='True')
    parser.add_argument('-mu_method', type=str, default='dichotomy')
    parser.add_argument('-mu_tol', type=float, default=1e-8)
    parser.add_argument('-print', type=lambda x: bool(strtobool(x)), default='True')
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')
    parser.add_argument('-lm_tol', type=float, default=1e-2)
    parser.add_argument('-lm_maxiter', type=int, default=50)
    args = parser.parse_args()
    print(args)
    
    # Setup oneshot risb container
    oneshot = oneshotRISB(filename=args.filename,
                          dft_solver=args.dft_solver,
                          sc_method=args.sc_method,
                          analyze_block_tol=args.analyze_block_tol,
                          prnt=args.print,
                          plo_cfg_filename=args.plo_cfg,
                          beta=args.beta,
                          calc_mu=args.calc_mu,
                          mu_method=args.mu_method,
                          mu_tol=args.mu_tol,
                          mu_offset=0.01,
                          mu_mix_factor=1,
                          U=args.U,
                          J=args.J)

    # Do some linear mixing to hopefully get close to a solution
    #oneshot.sc_method = 'recursion'
    #optimizer = LinearMixing()
    #solver = SolverNewton(update_x=optimizer.update_x, tol=args.lm_tol, maxiter=args.lm_maxiter, history_size=args.history_size)
    #res = oneshot.dmft_cycle(solver=solver, reset_sumk=True)
    
    ## Use diis to converge
    oneshot.sc_method = args.sc_method
     
    annealer = Annealing(alpha=args.alpha)
    #annealer = Annealing(anneal_type='step', alpha=args.alpha, reset_iter=10)
    optimizer = DIIS2(history_size=args.history_size, full_restart=args.full_restart)
    #optimizer = AdDIIS(history_size=args.history_size)
    
    #optimizer.load_history(x=solver.x, error=solver.error)
    solver = SolverNewton(update_x=optimizer.update_x, annealer=annealer, maxiter=args.maxiter, tol=args.tol)
    
    #res = oneshot.dmft_cycle(solver=solver, reset_sumk=False)
    res = oneshot.dmft_cycle(solver=solver, reset_sumk=True)
    
    if args.save:
        oneshot.save(it=0)