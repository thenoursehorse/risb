# noqa: D100

from numpy.typing import ArrayLike

from .solver_newton import NewtonSolver


# error function f(x_i) = x_i - x_i-1
# x_i+1 = x_t - [J_i]^-1 f(x_i) = x_i + dx_i
# dx_i = - [J_i]^-1 F(x_i)
# Linear mixing is J_i = - 1/alpha
# dx_i = alpha * F(x_i)
# g(x_i) is the fixed-point function that gives a new x_i+1
class LinearMixing(NewtonSolver):
    """Linear mixing a new vector with an old vector."""

    def __init__(self, /, **kwargs):
        super().__init__(history_size=0, **kwargs)

    def update_x(self, alpha: float = 1.0) -> ArrayLike:
        """Return a new guess for the vector x."""
        return self.x[0] + alpha * (self.g_x[0] - self.x[0])
