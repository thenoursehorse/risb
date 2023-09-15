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
from typing import Any, Callable
from numpy.typing import ArrayLike

class NewtonSolver:
    '''
    Base class for quasi-Newton methods to find the root of a function.

    Parameters
    ----------
        
    history_size : int, optional
        Maximum size of subspace. Defaults to 5.

    t_restart : int, optional
        Fully reset subspace after this many iterations. Defaults infinity.

    verbose : bool, optional
        Whether to report information during optimization. Default False.

    Note: The method self.update_x must be defined in the child class, and it 
    is called as self.update_x(x, g_x, error, options['alpha']).

    todo: fix up verbose messages
    '''
    def __init__(self, 
                 history_size : int = 5, 
                 t_restart : float = np.inf, 
                 verbose : bool = False) -> None:

        self.history_size = history_size
        self.t_restart = t_restart
        self.verbose = verbose
        
        self.x : list[ArrayLike] = []
        self.g_x : list[ArrayLike] = [] 
        self.error : list[ArrayLike] = []
        
        self.initialized = False
        self.t = 0

    @staticmethod
    def load_history(x : list[ArrayLike], 
                     error : list[ArrayLike], 
                     max_size : int) -> tuple[ list[ArrayLike], list[ArrayLike] ]:

        if (len(x) != len(error)) and (len(x) != (len(error) + 1)):
            raise ValueError('x and error are the wrong lengths !')

        x_out = deepcopy(x)
        error_out = deepcopy(error)

        while len(x_out) >= max_size:
            x_out.pop()
        while len(error_out) >= max_size:
            error_out.pop()

        return x_out, error_out

    @staticmethod
    def insert_vector(vec : list[ArrayLike], 
                      vec_new : ArrayLike, 
                      max_size : int | None = None) -> None:

        # Note these operations are mutable on input list
        vec.insert(0, vec_new)
        if max_size is not None:
            if len(vec) >= max_size:
                vec.pop()

    def solve(self, 
              fun : Callable[..., ArrayLike],
              x0 : ArrayLike, 
              args : tuple[Any, ...] = (), 
              tol : float = 1e-12, 
              options : dict[str, Any] = {'maxiter': 1000, 'alpha': 1}) -> ArrayLike:
        """
        Find the root of a function. It is called similarly to 
        scipy.optimize.root

        Parameters
        ----------

        fun : callable
            The function to find the root of. It must be callable as 
            fun(x, *args).

        x0 : array_like
            Initial guess of the parameters. This does not neccessarily have to 
            be flattened, but it usually is.

        args : tuple, optional
            Additional arguments to pass to the function. Default none.

        tol : float, optional
            The tolerance. When the 2-norm difference of the return of the 
            function is less than this, the solver stops. Default 1e-12.

        options : dict, optional
            Additional options. 
                maxiter : Maximum number of iterations. Default 1000.
                alpha : Step size. Default 1.
        
        Returns
        -------

        x : array_like
            The x that is the root of the function.
        
        """
        if 'maxiter' not in options:
            options['maxiter'] = 1000
        if 'alpha' not in options:
            options['alpha'] = 1

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

            x = self.update_x(x, g_x, error, options['alpha'])
            
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