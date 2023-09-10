from copy import deepcopy
import numpy as np
from .common import load_history, insert_vector

class Annealing:
    def __init__(self, alpha=1.0, anneal_type='flat', reset_iter=5, step_scaling=2):
        self.alpha = alpha
        self.anneal_type = anneal_type

        # for step
        self.reset_iter = reset_iter
        self.step_scaling = step_scaling
        
        if anneal_type == 'flat':
            self.get_alpha = self._get_flat
        elif anneal_type == 'step':
            self.get_alpha = self._step
        else:
            raise ValueError("unrecognized annealing type !")

        self.t = 0

    def _step(self):
        if self.t > self.reset_iter:
            self.alpha *= self.step_scaling
        else:
            self.t += 1
        if self.alpha >= 1.0:
            self.alpha = 1.0
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

    '''
    def __init__(self, history_size=5, t_restart=np.inf, verbose=True, annealer=None):
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

    def solve(self, fun, x0, tol=1e-12, options={'maxiter':1000}):
        self.success = False
        x = deepcopy(x0)

        if self.history_size > 0:
            self.insert_vector(self.x, x, self.history_size)

        for self.n in range(options['maxiter']):

            g_x, error = fun(x=x)

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