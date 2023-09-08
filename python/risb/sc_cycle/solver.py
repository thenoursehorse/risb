import numpy as np
import scipy
from copy import deepcopy

# error function f(x_i) = x_i - x_i-1
# x_i+1 = x_t - [J_i]^-1 f(x_i) = x_i + dx_i
# dx_i = - [J_i]^-1 F(x_i)
# Linear mixing is J_i = - 1/alpha
# dx_i = alpha * F(x_i)
# g(x_i) is the fixed-point function that gives a new x_i+1
class LinearMixing(object):
    '''
    Linear mix an input vector x.
    '''
    def __init__(self):
        self.t = 0
    
    def update_x(self, x, g_x, error=None, alpha=1.0):
        self.t += 1
        return x + alpha * (g_x - x)
    
def load_history(x, error, max_size):
    if (len(x) != len(error)) and (len(x) != (len(error) + 1)):
        raise ValueError('x and error are the wrong lengths !')

    x_out = deepcopy(x)
    error_out = deepcopy(error)

    while len(x_out) >= max_size:
        x_out.pop()
    while len(error_out) >= max_size:
        error_out.pop()

    return x_out, error_out

def insert_vector(vec, vec_new, max_size=None):
    # Note these operations are mutable on input list
    vec.insert(0, vec_new)
    if max_size is not None:
        if len(vec) >= max_size:
            vec.pop()

# Fig 2 in J Math Chem (2011) 49:1889–1914
# Note that the crop algorithm should use N=3 and needs a C_inv conditioner to remove most of the non-linearities 
# from the Jacobian, otherwise it will not converge.
class DIIS(object):
    '''
    Direct inversion in the iterative subspace.

    Args:
        history_size : (Default 5) Maximum size of subspace.
        
        C_inv : (Default None) Preconditioner matrix.

        use_crop : (Default False) Whether to use the CROP algorithm to update 
            the subspace with the optimized residual and x vectors.
    '''
    def __init__(self, history_size=5, restart_size=np.inf, C_inv=None, use_crop=False):
        self.history_size = history_size
        self.restart_size = restart_size
        self.C_inv = C_inv
        self.use_crop = use_crop

        self.x = []
        self.error = []
        
        self.initialized = False
        self.t = 0
    
    def load_history(self, x, error):
        self.x, self.error = load_history(x, error, self.history_size)
        self.initialized = True

    def insert_vector(self, vec, vec_new, max_size):
        insert_vector(vec, vec_new, max_size)

    def update_x(self, x, error, alpha=1.0):
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

        if self.m >= self.restart_size:
            None
        
        self.t += 1
        return self.x[0]

# Algorithm 2 (Anderson type) in 10.1051/m2an/2021069,  hal-02492983v5
class DIIS2(object):
    '''
    Direct inversion in the iterative subspace.

    Args:
        history_size : (Default 5) Maximum size of subspace.
    '''
    def __init__(self, history_size=5, period=0, full_restart=0):
        self.history_size = history_size
        if period == 0:
            self.period = int(np.round(self.history_size/2.0))
        else:
            self.period = period
        self.full_restart = full_restart

        self.x = []
        self.g_x = []
        self.error = []
        
        self.t = 0
        
    def load_history(self, x, error):
        self.x, self.error = load_history(x, error, self.history_size)

    def insert_vector(self, vec, vec_new, max_size):
        insert_vector(vec, vec_new, max_size)
    
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
        
        if self.full_restart != 0:
            if (self.t % self.full_restart) == 0:
                self.x = []
                self.g_x = []
                self.error = []
        
        # Collect history
        self.insert_vector(self.x, x, self.history_size)
        self.insert_vector(self.g_x, g_x, self.history_size)
        self.insert_vector(self.error, alpha*error, self.history_size)

        if ((self.t+1) % self.period == 0):
            # Do DIIS
            x_opt = self.extrapolate()
        else:
            # Do linear mixing
            x_opt = x + alpha*(g_x-x)

        self.t += 1
        return x_opt

# Algorithm 4 in (Anderson type) 10.1051/m2an/2021069,  hal-02492983v5
class AdDIIS(object):
    '''
    Direct inversion in the iterative subspace.

    Args:
        history_size : (Default 5) Maximum size of subspace.
    '''
    def __init__(self, history_size=5, delta=1e-5):
        self.history_size = history_size
        self.delta = delta

        self.history_size = history_size

        self.x = []
        self.g_x = []
        self.error = []

        self.s = []
        self.E = []

        self.t = 0
        self.m_t = 0
        
    def load_history(self, x, error):
        self.x, self.error = load_history(x, error, self.history_size)

    def insert_vector(self, vec, vec_new, max_size=None):
        insert_vector(vec, vec_new, max_size)
    
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
            self.insert_vector(self.s, self.error[i+1] - self.error[0])
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
            
class Annealing(object):
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
        
class SolverNewton(object):
    def __init__(self, update_x, tol=1e-6, maxiter=1000, stdout=True, history_size=0, annealer=None):
        self.update_x = update_x
        self.tol = tol
        self.maxiter = maxiter
        self.stdout = stdout
        self.history_size = history_size
        
        if not annealer:
            self.annealer = Annealing()
        else:
            self.annealer = annealer

        self.x = []
        self.g_x = []
        self.error = []

        self.n = None
        self.norm = None
        self.success = None

    def load_history(self, x, error):
        self.x, self.error = load_history(x, error, self.history_size)
    
    def insert_vector(self, vec, vec_new, max_size):
        insert_vector(vec, vec_new, max_size)

    def solve(self, x0, function):
        self.success = False
        x = deepcopy(x0)

        if self.history_size > 0:
            self.insert_vector(self.x, x, self.history_size)

        for self.n in range(self.maxiter):

            g_x, error = function(x=x)

            self.norm = np.linalg.norm(error)
            if self.stdout:
                print(f"n: {self.n}, rms(risb): {self.norm}")
            if self.norm < self.tol:
                self.success = True
                break

            # Scheduling
            alpha = self.annealer.get_alpha()
            
            x = self.update_x(x, g_x, error, alpha)
            
            if self.history_size > 0:
                self.insert_vector(self.x, x, self.history_size)
                self.insert_vector(self.g_x, g_x, self.history_size)
                self.insert_vector(self.error, error, self.history_size)
        
        if self.stdout:
            if self.success:
                print(f"The solution converged. nit: {self.n}, tol: {self.norm}")
            else:
                print(f"The solution did NOT converge. nit: {self.n} tol: {self.norm}")
        
        return x