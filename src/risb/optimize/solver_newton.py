# Copyright (c) 2021 H. L. Nourse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https:#www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: H. L. Nourse

from copy import deepcopy
import numpy as np
from .common import load_history, insert_vector

class Annealing:
    '''
    Scheduler for optimization to alter the step size alpha.

    Parameters
    ----------

    alpha : optional, float
        Initial alpha. Default is 1.

    method : optional, string
        Scheduling method. Optionals are:
        'flat' : Do not change initial alpha.
        'step' : Multiplicatively change alpha.
        Default is 'flat'.

    t_reset : optional, int
        Reset alpha to initial value after this many iterations. Default is 5.

    step_scaling : optional, float
        Scaling factor for changing alpha at each time step. Default is 0.5.

    '''
    def __init__(self, alpha=1.0, method='flat', t_reset=5, step_scaling=0.5):
        self.alpha = alpha
        self.alpha0 = deepcopy(self.alpha)
        self.method = method

        # for step
        self.t_reset = t_reset
        self.step_scaling = step_scaling
        
        if method == 'flat':
            self.get_alpha = self._get_flat
        elif method == 'step':
            self.get_alpha = self._step
        else:
            raise ValueError("Unrecognized method !")

        self.t = 0

    def _step(self):
        if self.t > self.t_reset:
            self.alpha *= self.step_scaling
        else:
            self.t += 1
        if self.alpha >= 1.0:
            self.alpha = deepcopy(self.alpha0)
        return self.alpha
        
    def _get_flat(self):
        return self.alpha
        
class NewtonSolver:
    '''
    A general driving class for quasi-Newton methods to minimize a function.

    Parameters
    ----------
        
    history_size : optional, int
        Maximum size of subspace. Defaults to 5.

    t_restart : optional, int
        Fully reset subspace after this many iterations. Defaults infinity.

    verbose : optional, bool
        Whether to report information during optimization. Default False.

    annealer : optional, class
        A scheduling class to update the step size alpha. Must have a method
        ``get_alpha()`` that returns alpha at the current step. Defaults to 
        alpha = 1 at each timestep.

    todo : fix up verbose messages
    '''
    def __init__(self, history_size=5, t_restart=np.inf, verbose=False, annealer=None):
        self.history_size = history_size
        self.t_restart = t_restart
        self.verbose = verbose
        
        if not annealer:
            self.annealer = Annealing()
        else:
            self.annealer = annealer

        self.n = None
        self.norm = None
        self.success = None
        
        self.x = []
        self.g_x = []
        self.error = []
        
        self.initialized = False
        self.t = 0

    def load_history(self, x, error):
        self.x, self.error = load_history(x, error, self.history_size)
        self.initialized = True

    def insert_vector(self, vec, vec_new, max_size):
        insert_vector(vec, vec_new, max_size)

    def solve(self, fun, x0, args=(), tol=1e-12, options={'maxiter':1000}):
        self.success = False
        x = deepcopy(x0)

        if self.history_size > 0:
            self.insert_vector(self.x, x, self.history_size)

        for self.n in range(options['maxiter']):

            g_x, error = fun(x, *args)

            self.norm = np.linalg.norm(error)
            if self.verbose:
                print(f"n: {self.n}, rms(risb): {self.norm}")
            if self.norm < tol:
                self.success = True
                break

            # Scheduling
            alpha = self.annealer.get_alpha()
            
            x = self.update_x(x, g_x, error, alpha)
            
            if self.history_size > 0:
                self.insert_vector(self.x, x, self.history_size)
                self.insert_vector(self.g_x, g_x, self.history_size)
                self.insert_vector(self.error, error, self.history_size)
        
        if self.verbose:
            if self.success:
                print(f"The solution converged. nit: {self.n}, tol: {self.norm}")
            else:
                print(f"The solution did NOT converge. nit: {self.n} tol: {self.norm}")
        
        return x