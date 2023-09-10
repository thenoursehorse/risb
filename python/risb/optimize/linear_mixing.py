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

    annealer : optional, class
        A scheduling class to update the step size alpha. Must have a method
        ``get_alpha()`` that returns alpha at the current step. Defaults to 
        alpha = 1 at each timestep.

    '''
    def __init__(self, *args, history_size=0, **kwargs):
        super().__init__(*args, history_size=history_size, **kwargs)
    
    def update_x(self, x, g_x, error=None, alpha=1.0):
        self.t += 1
        return x + alpha * (g_x - x)