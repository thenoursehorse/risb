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

import numpy as np
import scipy
from . import NewtonSolver

class DIIS(NewtonSolver):
    '''
    Direct inversion in the iterative subspace to minimize a function.
    
    Algorithm 2 (Anderson type) in 10.1051/m2an/2021069,  hal-02492983v5.
    
    Parameters
    ----------
        
    history_size : optional, int
        Maximum size of subspace. Defaults to 5.

    t_restart : optional, int
        Fully reset subspace after this many iterations. Defaults infinity.

    verbose : optional, bool
        Whether to report information during optimization. Default False.

    t_period : optional, int
        Take a linear mixing step afer this many iterations. Defaults to
        round(history_size / 2).
    
    '''
    def __init__(self, *args, history_size=5, t_period=0, **kwargs):
        super().__init__(*args, history_size=history_size, **kwargs)
        if t_period == 0:
            self.t_period = int(np.round(self.history_size/2.0))
        else:
            self.t_period = t_period

    def extrapolate(self):
        # Construct the B matrix
        m = len(self.error)
        B = np.empty(shape=(m,m))
        for i in range(m):
            for j in range(m):
                B[i,j] = np.dot(self.error[i], self.error[j])

        # Add the constraint lambda
        B = np.column_stack( ( B, -np.ones(B.shape[0]) ) )
        B = np.vstack( ( B, -np.ones(B.shape[1]) ) )
        B[m,m] = 0.
            
        # Solve for the c coefficients (last element in c gives lambda constraint)
        rhs = np.zeros(B.shape[0])
        rhs[-1] = -1.
        
        c = np.dot(scipy.linalg.pinv(B), rhs)
        
        # Calculate optimal x(n)
        x_opt = np.zeros(self.x[0].shape)
        for i in range(m):
            x_opt += c[i] * self.g_x[i]
        return x_opt

    # x_i, error(x_i), g(x_i) where g(x_i) is the fixed-point function 
    # that gives a new x_i
    def update_x(self, x, g_x, error, alpha=1.0):
        
        if (self.t % self.t_restart) == 0:
            self.x = []
            self.g_x = []
            self.error = []
        
        # Collect history
        self.insert_vector(self.x, x, self.history_size)
        self.insert_vector(self.g_x, g_x, self.history_size)
        self.insert_vector(self.error, alpha*error, self.history_size)

        if ((self.t+1) % self.t_period == 0):
            # Do DIIS
            x_opt = self.extrapolate()
        else:
            # Do linear mixing
            x_opt = x + alpha*(g_x-x)

        self.t += 1
        return x_opt