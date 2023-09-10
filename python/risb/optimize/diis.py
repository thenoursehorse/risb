import numpy as np
import scipy
from copy import deepcopy
from . import NewtonSolver
    
# Fig 2 in J Math Chem (2011) 49:1889–1914
# Note that the crop algorithm should use N=3 and needs a C_inv conditioner to remove most of the non-linearities 
# from the Jacobian, otherwise it will not converge.
class DIIS(NewtonSolver):
    '''
    WARNING: This class is somehow broken now, but DIIS2 works fine.
    
    Direct inversion in the iterative subspace to minimize a function.
    
    Parameters
    ----------
        
    history_size : optional, int
        Maximum size of subspace. Defaults to 5.

    t_restart : optional, int
        Fully reset subspace after this many iterations. Defaults infinity.

    verbose : optional, bool
        Whether to report information during optimization. Default False.

    C_inv : optional, array
        Inverse of preconditioner matrix. Default None.

    use_crop : optional, bool
        Whether to use the CROP algorithm to update the subspace with the 
        optimized residual and x vectors. Default False.

    ''' 
    def __init__(self, *args, history_size=5, C_inv=None, use_crop=False, **kwargs):
        super().__init__(*args, history_size=history_size, **kwargs)
        self.C_inv = C_inv
        self.use_crop = use_crop

    def update_x(self, x, g_x, error, alpha=1.0):
        if not self.initialized:
            self.insert_vector(self.x, x, self.history_size)
            self.initialized = True
            
        if self.C_inv is None:
            self.C_inv = np.eye(len(x))

        # Collect history of residuals
        self.insert_vector(self.error, alpha*error, self.history_size)
            
        # Construct the B matrix
        m = len(self.error)
        B = np.empty(shape=(m,m))
        for i in range(m):
            for j in range(m):
                B[i,j] = np.dot(self.error[i], np.dot(self.C_inv, self.error[j]) )

        # Add the constraint lambda
        B = np.column_stack( ( B, -np.ones(B.shape[0]) ) )
        B = np.vstack( ( B, -np.ones(B.shape[1]) ) )
        B[m,m] = 0.
            
        # Solve for the c coefficients (last element in c gives lambda constraint)
        rhs = np.zeros(B.shape[0])
        rhs[-1] = -1.
        
        # NOTE Near critical points DIIS has many linear dependencies in its 
        # space. Below is what pyscf does to remove them. Doesn't pinv do this 
        # anyway by setting these coefficients to zero?
        e, v = scipy.linalg.eigh(B)
        if np.any(abs(e) < 1e-14):
            print('DIIS has linear dependence in vectors !')
            idx = abs(e)>1e-14
            c = np.dot(v[:,idx]*(1./e[idx]), np.dot(v[:,idx].T.conj(), rhs))
        else:
            try:
                c = scipy.linalg.solve(B, rhs)
            except scipy.linalg.LinAlgrror as e:
                print('DIIS is singular with eigenvalues {0} !'.format(e))
                raise e
        
        #c = np.dot(scipy.linalg.pinv(B), rhs)
        
        # Calculate optimal x(n)
        x_opt = np.zeros(self.x[0].shape)
        for i in range(m):
            x_opt += c[i] * self.x[i]

        # Calculate optimial error(n)
        error_opt = np.zeros(self.x[0].shape)
        for i in range(m):
            error_opt += c[i] * self.error[i]
        
        # CROP algorithm updates subspace with optimized vectors
        if self.use_crop: 
            self.error[0] = deepcopy(error_opt)

        # Update direction to x
        dx = np.dot(self.C_inv, error_opt)
                    
        # Collect history of x
        self.insert_vector(self.x, x_opt + dx, self.history_size)

        self.t += 1
        return self.x[0]

class DIIS2(NewtonSolver):
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

class AdDIIS(NewtonSolver):
    '''
    WARNING: This class does not work.

    Direct inversion in the iterative subspace to minimize a function.

    Algorithm 4 in (Anderson type) 10.1051/m2an/2021069,  hal-02492983v5.
    
    Parameters
    ----------
        
    history_size : optional, int
        Maximum size of subspace. Defaults to 5.

    t_restart : optional, int
        Fully reset subspace after this many iterations. Defaults infinity.

    verbose : optional, bool
        Whether to report information during optimization. Default False.

    delta : optional, float
        I forget what this does. Default 1e-5.

    '''
    def __init__(self, *args, history_size=5, delta=1e-5, **kwargs):
        super().__init__(*args, history_size=history_size, **kwargs)
        self.delta = delta

        self.s = []
        self.E = []
        self.m_t = 0
        
    def extrapolate(self):
        # Error difference
        s = self.error[0] - self.error[1]
        self.insert_vector(self.s, s, self.history_size-1)

        #if (self.m_t == 1):
        #    self.B = np.reshape(s,(-1,1))
        #else:
        #    # Add new error difference to beginning
        #    self.B = np.hstack((np.reshape(s,(-1,1)), self.B))
        #
        #if len(self.B) > self.history_size:
        #    # Remove last column
        #    self.B = np.delete(arr=self.B, obj=-1, axis=1)
        #    #self.B = self.B[:,:-1] 
 
        #Q, R = scipy.linalg.qr(self.B, mode="economic")
        #rhs = -np.dot(Q.T, np.reshape(self.error[0],(-1,1)))
        #gamma = scipy.linalg.solve_triangular(R[:self.m_t,:self.m_t], rhs[:self.m_t], lower=False)
        ##gamma = scipy.linalg.solve(R, Q.T @ self.error[0])

        self.s = []
        for i in range(len(self.g_x)-1):
            # FIXME size of this?
            self.insert_vector(self.s, self.error[i+1] - self.error[0], self.history_size-1)
        self.B = np.asarray(self.s).T
        print("TEST", self.B.shape)
        Q, R = scipy.linalg.qr(self.B, mode="economic")
        #Q, R = scipy.linalg.qr(self.B)
        b = Q.T @ self.error[0]
        #c_tilde = scipy.linalg.solve_triangular(R, -b, lower=False)
        c_tilde = scipy.linalg.solve(R, -b, lower=False)
        cn = 1 - np.sum(c_tilde)
        x_opt = cn * self.g_x[0]
        for i in range(len(c_tilde)):
            x_opt += c_tilde[i] + self.g_x[i+1]

        #rhs = deepcopy(self.error[0])
        ##gamma = scipy.linalg.pinv(self.B)[:self.m_t,:self.m_t] @ rhs[:self.m_t]
        #gamma = scipy.linalg.pinv(self.B) @ rhs
        #
        #print("TEST1", self.B.shape, rhs.shape, gamma.shape)
        #
        #print("TEST2", len(self.g_x), len(gamma), len(self.s))
        #
        #x_opt = deepcopy(self.g_x[0])
        #for i in range(len(gamma)):
        #    x_opt = gamma[i] * (self.g_x[i] - self.g_x[i+1])
        
            
        ## compute c_i coefficients from unconstrained gamma coefficients
        #self.m_t = len(self.B)
        #c = np.zeros(self.m_t+1)
        #c[0] = -gamma[0]
        #for i in range(1,m_t):
        #    c[i] = gamma[i-1] - gamma[i]
        #c[m_t] = 1.0 - np.sum(c[:m_t])
        #
        ## Calculate optimal x(n)
        #x_opt = np.zeros(self.x[0].shape)
        #for i in range(self.m_t+1):
        #    x_opt += c[i] * self.g_x[i]
        
        return x_opt

    def update_x(self, x, g_x, error, alpha=1.0):
        
        # Collect history
        self.insert_vector(self.x, x, self.history_size)
        self.insert_vector(self.g_x, g_x, self.history_size)
        self.insert_vector(self.error, alpha*error, self.history_size)

        if self.m_t > 0:
            x_opt = self.extrapolate()
        else:
            x_opt = deepcopy(g_x)
        
        if len(self.x) < self.history_size-1:
            self.m_t += 1

        self.t += 1
        return x_opt
            