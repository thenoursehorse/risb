from .solver_newton import NewtonSolver
from .linear_mixing import LinearMixing
from .diis import DIIS

# FIXME sort out doc inheritance from base class
#DIIS.__doc__ = NewtonSolver.__doc__
#LinearMixing.__doc__ = NewtonSolver.__doc__