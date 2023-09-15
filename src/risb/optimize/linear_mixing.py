from numpy.typing import ArrayLike
from . import NewtonSolver

# error function f(x_i) = x_i - x_i-1
# x_i+1 = x_t - [J_i]^-1 f(x_i) = x_i + dx_i
# dx_i = - [J_i]^-1 F(x_i)
# Linear mixing is J_i = - 1/alpha
# dx_i = alpha * F(x_i)
# g(x_i) is the fixed-point function that gives a new x_i+1
class LinearMixing(NewtonSolver):
    '''
    Linear mixing a new vector with an old vector.

    Parameters
    ----------
        
    verbose : optional, bool
        Whether to report information during optimization. Default False.

    '''
    def __init__(self, /,
                  history_size : int = 0, 
                  **kwargs):
        super().__init__(history_size=history_size, **kwargs)
    
    def update_x(self, 
                 x : ArrayLike, 
                 g_x : ArrayLike, 
                 error : None = None, 
                 alpha : float = 1.0):
        self.t += 1
        return x + alpha * (g_x - x)