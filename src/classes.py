import numpy as np
from numba import boolean, int32, int64, float64, njit, objmode
from numba.typed import List
from numba.experimental import jitclass
import scipy as sp
from src.helpers import *
from src.model_functions import *

#from numba.typed import List

spec = [
    ('tau', float64),
    ('int_steps', int32),
    ('h', float64),
    ('sigma', float64),
    ('beta', float64),
    ('rho', float64),
    ('state', float64[:]),
    ('standardized', boolean)
]

#@jitclass(spec)
class LorenzModel():
    def __init__(self, tau=0.1, int_steps=10, sigma=10.,
                 beta=8 / 3, rho=28., ic=np.array([]), ic_seed=0,standardized=False, mean=0.,std=0.,dyn_noise=0.,noise_seed=10):
        self.D = 3
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.params = np.array([self.sigma, self.rho, self.beta])
        self.matparams = np.zeros((1,1,1), dtype = np.float64)
        self.dxdt = lorenz63_dxdt
        self.standardized = standardized
        self.mean = mean
        self.std = std
        self.dyn_noise=dyn_noise
        self.rng = np.random.default_rng(noise_seed)
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)
    def run_with_noise(self, T, discard_len=0):
        model_output = model_run_with_noise(self.state, model_forward_with_noise, self.dxdt, self.int_steps, self.h,
                                 self.params, T, discard_len,dyn_noise=self.dyn_noise, rng=self.rng)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run(self, T, discard_len=0):
        model_output = model_run(self.state, model_forward, self.dxdt, self.int_steps, self.h,
                                 self.params, T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run_array(self, ic_array):

        if self.standardized:
            model_output = model_run_array(self.unstandardize(ic_array), model_forward, self.dxdt,
                                           self.int_steps, self.h, self.params)
            return self.standardize(model_output)
        else:
            model_output = model_run_array(ic_array, model_forward, self.dxdt, self.int_steps, self.h,
                                           self.params)
            return model_output

    def forward(self):
        self.state = model_forward(self.state, self.dxdt, self.int_steps, self.h, self.params)
        return self.state
class DoubleScrollModel():
    def __init__(self, tau=0.7, int_steps=70, R1=1.2,
                 R2=3.44, R4=0.193, beta=11.6,Ir=2.25*10**(-5),ic=np.array([]), ic_seed=0,standardized=False, mean=0.,std=0.):
        self.D = 3
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.R1 = R1
        self.R2 = R2
        self.R4 = R4
        self.beta = beta
        self.Ir = Ir
        self.params = np.array([self.R1, self.R2, self.R4, self.beta,self.Ir])
        self.matparams = np.zeros((1,1,1), dtype = np.float64)
        self.dxdt = doublescoll_dxdt
        self.standardized = standardized
        self.mean = mean
        self.std = std
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1)
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        model_output = model_run(self.state, model_forward, self.dxdt, self.int_steps, self.h,
                                 self.params, T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run_array(self, ic_array):

        if self.standardized:
            model_output = model_run_array(self.unstandardize(ic_array), model_forward, self.dxdt,
                                           self.int_steps, self.h, self.params)
            return self.standardize(model_output)
        else:
            model_output = model_run_array(ic_array, model_forward, self.dxdt, self.int_steps, self.h,
                                           self.params)
            return model_output

    def forward(self):
        self.state = model_forward(self.state, self.dxdt, self.int_steps, self.h, self.params)
        return self.state
class Orstein_Uhlenbeck():
      def __init__(self, D=3, tau=0.1, sigma=0.1, theta=0.7, mu=0, ic=np.array([]), ic_seed=0,noise_seed=10,discard_len=1000):
          self.D=3
          self.h=tau
          self.theta=theta
          self.sigma=sigma
          self.mu=mu
          self.params = np.array([self.theta, self.sigma, self.mu])
          self.rng = np.random.default_rng(noise_seed)
          self.discard_len=discard_len
          if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
          elif ic.size == D:
            self.state = ic.flatten()
          else:
                raise ValueError
      def run(self, T,discard_len):
          model_output = model_forward_Orstein_Uhlenbeck(self.state,T, discard_len=discard_len, h=self.h,params=self.params,rng=self.rng)
          self.state = model_output[-1]
          return model_output[discard_len:]          
class RosslerModel():
    def __init__(self, tau=0.1, int_steps=10, alpha=0.1,
                 beta=0.1, gamma=14., ic=np.array([]), ic_seed=0, standardized=False, mean=0., std=0., dyn_noise=0,
                 noise_seed=10):
        self.D = 3
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.params = np.array([self.alpha, self.beta, self.gamma])
        self.matparams = np.zeros((1, 1, 1), dtype=np.float64)
        self.dxdt = rossler_dxdt
        self.standardized = standardized
        self.mean = mean
        self.std = std
        self.dyn_noise = dyn_noise
        self.rng = np.random.default_rng(noise_seed)
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = (np.random.rand(3) * 2 - 1) * np.array([1., 1., 30.])
        elif ic.size == 3:
            self.state = ic.flatten()
        else:
            raise ValueError

    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        model_output = model_run(self.state, model_forward, self.dxdt, self.int_steps, self.h,
                                 self.params, T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run_with_noise(self, T, discard_len=0):
        model_output = model_run_with_noise(self.state, model_forward_with_noise, self.dxdt, self.int_steps, self.h,
                                            self.params, T, discard_len, dyn_noise=self.dyn_noise, rng=self.rng)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run_array(self, ic_array):

        if self.standardized:
            model_output = model_run_array(self.unstandardize(ic_array), model_forward, self.dxdt,
                                           self.int_steps, self.h, self.params)
            return self.standardize(model_output)
        else:
            model_output = model_run_array(ic_array, model_forward, self.dxdt, self.int_steps, self.h,
                                           self.params)
            return model_output

    def forward(self):
        self.state = model_forward(self.state, self.dxdt, self.int_steps, self.h, self.params)
        return self.state
#@jitclass(spec)
class Kursiv():
    def __init__(self, N=64, tau=0.25, int_steps=10, u0=np.array([]), ic_seed=0,d=2*np.pi,standardized=False, mean=0.,std=0.,params = np.array([[],[]], dtype = np.complex128),dyn_noise = np.zeros((1,1), dtype = np.double),const=0,noise_seed=10):
        self.D = N
        self.tau = tau
        self.int_steps = int_steps
        self.d=d
        self.standardized = standardized
        self.mean = mean
        self.std = std
        if params.size == 0:
            self.params = precompute_KS_params(N, d, tau, const=const)
        else:
            self.params=params
        self.dyn_noise=dyn_noise
        self.rng = np.random.default_rng(noise_seed)
        self.const=const
        if u0.size == 0:
            np.random.seed(ic_seed)
            self.state = np.random.rand(N)  # u0 is initialized to be random number between 0 and 1 #
            self.state= self.state-np.mean(self.state) # mean is subtracted #
        elif u0.size == N:
            self.state = u0.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        if self.dyn_noise.size == 1 and self.dyn_noise[0,0] == 0.:
            noise=self.dyn_noise
        else:
            noise=self.dyn_noise*self.rng.normal(size=(self.D,(T+1)*int_steps))
        model_output = kursiv_run(self.state, tau=self.tau, N=self.D, d=self.d, T=T, params=self.params,
                       int_steps=self.int_steps, noise=noise,const=self.const)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output[discard_len:]

    def run_array(self, u0_array):
        if self.standardized:
            model_output = kursiv_run_array(self.unstandarize(u0_array), tau = self.tau, N = self.D, d = self.d, params = self.params,
                           const=self.const)
            return self.standardize(model_output)
        else:
            model_output = kursiv_run_array(u0_array, tau = self.tau, N = self.D, d = self.d,const=self.const,params = self.params)
            return model_output

    def forward(self):
        self.state = kursiv_model_forward(self.state,self.int_steps,self.params)
        return self.state

spec = [
    ('tau', float64),
    ('int_steps', int32),
    ('h', float64),
    ('D', int32),
    ('F', float64),
    ('state', float64[:]),
    ('standardized', boolean),
    ('mean', float64),
    ('std', float64)
]


#@jitclass(spec)
class LorenzModel1():
    def __init__(self, tau=0.05, int_steps=5, D=30, F=8., ic=np.array([]), ic_seed=0, standardized=False, mean=0.,
                 std=0.):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.D = D
        self.F = F
        self.standardized = standardized
        self.mean = mean
        self.std  = std
        if isinstance(self.F, np.ndarray):
            self.params = self.F
        else:
            self.params = np.array([self.F])
        self.dxdt = lorenzmodel1_dxdt
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = np.random.rand(D) * 2 - 1
        elif ic.size == D:
            self.state = ic.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        model_output = model_run(self.state, model_forward, self.dxdt, self.int_steps, self.h,
                                 self.params, T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output

    def run_array(self, ic_array):
        if self.standardized:
            model_output = model_run_array(self.unstandardize(ic_array), model_forward, self.dxdt,
                                           self.int_steps, self.h, self.params)
            return self.standardize(model_output)
        else:
            model_output = model_run_array(ic_array, model_forward, self.dxdt,
                                           self.int_steps, self.h, self.params)
            return model_output

    def forward(self):
        self.state = model_forward(self.state, self.dxdt, self.int_steps, self.h, self.params)
        return self.state


spec = [
    ('tau', float64),
    ('int_steps', int32),
    ('h', float64),
    ('D', int32),
    ('K', int32),
    ('smat', float64[:, :]),
    ('F', float64),
    ('state', float64[:]),
    ('standardized', boolean),
    ('mean', float64),
    ('std', float64)
]


#@jitclass(spec)
class LorenzModel2():
    def __init__(self, tau=0.1, int_steps=10, D=30, K=6, F=8, ic=np.array([]), ic_seed=0, standardized=False,
                 mean=0., std=0.):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.D = D
        self.K = K
        self.F = F
        self.standardized = standardized
        self.mean = mean
        self.std = std
        self.params = np.array([self.F, self.K], dtype = np.float64)
        self.s_mat_data, self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape = getsmat(self.D, self.K)
        self.dxdt = lorenzmodel2_dxdt
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = np.random.rand(D) * 2 - 1
        elif ic.size == D:
            self.state = ic.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        model_output = lorenzmodel2_run(self.state, self.dxdt, self.int_steps, self.h,
                                 self.params, self.s_mat_data, self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape,
                                 T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output

    def run_array(self, ic_array):
        if self.standardized:
            model_output = lorenzmodel2_run_array(self.unstandardize(ic_array), self.dxdt,
                                           self.int_steps, self.h, self.s_mat_data, self.s_mat_indices,
                                           self.s_mat_indptr, self.s_mat_shape,self.params)
            return self.standardize(model_output)
        else:
            model_output = lorenzmodel2_run_array(ic_array, self.dxdt,
                                           self.int_steps, self.h, self.s_mat_data, self.s_mat_indices,
                                           self.s_mat_indptr, self.s_mat_shape,self.params)
            return model_output

    def forward(self):
        self.state = lorenzmodel2_forward(self.state, lorenzmodel2_dxdt, self.int_steps, self.h, self.s_mat_data,
                                   self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape,self.params)
        return self.state

class LorenzModel3():
    def __init__(self, tau=0.005, int_steps=1, D=30, K=6, b=10, c = 2.5, I=12, F=8, ic=np.array([]), ic_seed=0,
                 standardized=False, mean=0., std=0.):
        self.tau = tau
        self.int_steps = int_steps
        self.h = tau / int_steps
        self.D = D
        self.K = K
        self.b = b
        self.c = c
        self.I = I
        self.alpha = (3*self.I**2 + 3)/(2*self.I**3 + 4*self.I)
        self.beta = (2*self.I**2 + 1)/(self.I**4 + 2*self.I**2)
        self.F = F
        self.standardized = standardized
        self.mean = mean
        self.std = std
        self.params = np.array([self.F, self.K, self.b, self.c, self.I, self.alpha, self.beta], dtype = np.float64)
        self.s_mat_data, self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape = getsmat(self.D, self.K)
        self.z2x_data, self.z2x_indices, self.z2x_indptr, self.z2x_shape = getz2x(self.D, self.I, self.alpha, self.beta)
        self.dxdt = lorenzmodel2_dxdt
        if ic.size == 0:
            np.random.seed(ic_seed)
            self.state = np.random.rand(D) * 2 - 1
        elif ic.size == D:
            self.state = ic.flatten()
        else:
            raise ValueError
    def unstandardize(self, x):
        return unstandardize(x, self.mean, self.std)

    def standardize(self, x):
        return standardize(x, self.mean, self.std)

    def run(self, T, discard_len=0):
        model_output = lorenzmodel3_run(self.state, self.dxdt, self.int_steps, self.h, self.z2x_data, self.z2x_indices,
                                        self.z2x_indptr, self.z2x_shape, self.params, self.s_mat_data,
                                        self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape, T, discard_len)
        self.state = model_output[-1]
        if self.standardized:
            model_output = self.standardize(model_output)
        return model_output

    def run_array(self, ic_array):
        if self.standardized:
            model_output = lorenzmodel3_run_array(self.unstandardize(ic_array), self.dxdt,
                                           self.int_steps, self.h, self.s_mat_data, self.s_mat_indices,
                                           self.s_mat_indptr, self.s_mat_shape,self.params)
            return self.standardize(model_output)
        else:
            model_output = lorenzmodel3_run_array(ic_array, self.dxdt,
                                           self.int_steps, self.h, self.s_mat_data, self.s_mat_indices,
                                           self.s_mat_indptr, self.s_mat_shape,self.params)
            return model_output

    def forward(self):
        self.state = lorenzmodel2_forward(self.state, lorenzmodel2_dxdt, self.int_steps, self.h, self.s_mat_data,
                                   self.s_mat_indices, self.s_mat_indptr, self.s_mat_shape,self.params)
        return self.state


@njit(fastmath=False)
def gen_adjacency(N, avg_degree, spectral_radius, seed=0, transpose = False):
    np.random.seed(seed)
    v0 = np.random.rand(N)
    with objmode(A_data='float64[:]', A_indices='int32[:]', A_indptr='int32[:]', A_shape='int32[:]'):
        A = sp.sparse.random(N, N, avg_degree / N, format='csc', random_state=seed)
        A.data = A.data * 2 - 1.0
        A = A * spectral_radius / np.abs(sp.sparse.linalg.eigs(A, k=1, return_eigenvectors=False, v0 = v0,
                                                             maxiter=10 ** 4)[0])
        if transpose:
            A_dense = A.todense()
            A_dense = np.transpose(A_dense)
            A = sp.sparse.csc_matrix(A_dense)
        A_data, A_indices, A_indptr, A_shape = A.data, A.indices, A.indptr, \
            np.array([A.shape[0], A.shape[1]], dtype = np.int32)
    return A_data, A_indices, A_indptr, A_shape


@njit(fastmath=False)
def gen_input_weights(N, D, input_weight, seed=0, transpose = False):
    B = np.zeros((N, D))
    q = N // D
    np.random.seed(seed)
    for i in range(D):
        B[i * q:(i + 1) * q, i] = 2 * np.random.rand(q) - 1
    leftover_node_inputs = np.random.choice(D, N - D * q, replace=False)
    for i, idx in enumerate(leftover_node_inputs):
        B[q * D + i, idx] = 2 * np.random.rand() - 1
    B = B * input_weight
    if transpose:
        B = np.ascontiguousarray(B.T)
    with objmode(B_data='float64[:]', B_indices='int32[:]', B_indptr='int32[:]', B_shape='int32[:]'):
        B_sp = sp.sparse.csc_matrix(B)
        B_data, B_indices, B_indptr, B_shape = B_sp.data, B_sp.indices, B_sp.indptr, \
                                               np.array([B_sp.shape[0], B_sp.shape[1]], dtype = np.int32)
    return B_data, B_indices, B_indptr, B_shape

@njit(fastmath=False)
def gen_input_weights_fully_dense(N, D, input_weight, seed=0, transpose = False):
    np.random.seed(seed)
    B = 2 * np.random.rand(N,D) - 1
    B = B * input_weight/np.sqrt(D)
    if transpose:
        B = np.ascontiguousarray(B.T)
    with objmode(B_data='float64[:]', B_indices='int32[:]', B_indptr='int32[:]', B_shape='int32[:]'):
        B_sp = sp.sparse.csc_matrix(B)
        B_data, B_indices, B_indptr, B_shape = B_sp.data, B_sp.indices, B_sp.indptr, \
                                               np.array([B_sp.shape[0], B_sp.shape[1]], dtype = np.int32)
    return B_data, B_indices, B_indptr, B_shape

@njit(fastmath=False)
def gen_input_weights_dense(N, D, input_weight, num_non_zero=2, seed=0, transpose = False):
    B = np.zeros((N, D))
    q = N // D
    np.random.seed(seed)
    for i in range(D-num_non_zero):
        B[i * q:(i + 1) * q, i:i+num_non_zero] = 2 * np.random.rand(q,num_non_zero) - 1
    index = np.arange(D)
    for i in range(D-num_non_zero,D):
        index = np.roll(index,1)
        B[i * q:(i + 1) * q, index[:num_non_zero]] = 2 * np.random.rand(q,num_non_zero) - 1
    leftover_node_inputs=np.array((D,N-D*q,num_non_zero))
    for i, idx in enumerate(leftover_node_inputs):
        index = np.arange(D)
        index = np.roll(index,-idx)
        B[q * D + i, index[:num_non_zero]] = 2 * np.random.rand(num_non_zero) - 1
    B = B * input_weight
    if transpose:
        B = np.ascontiguousarray(B.T)
    with objmode(B_data='float64[:]', B_indices='int32[:]', B_indptr='int32[:]', B_shape='int32[:]'):
        B_sp = sp.sparse.csc_matrix(B)
        B_data, B_indices, B_indptr, B_shape = B_sp.data, B_sp.indices, B_sp.indptr, \
                                               np.array([B_sp.shape[0], B_sp.shape[1]], dtype = np.int32)
    return B_data, B_indices, B_indptr, B_shape
@njit(fastmath=False)
def gen_bias(N, input_bias, seed=0):
    np.random.seed(seed)
    return input_bias * (2 * np.random.rand(N) - 1)

@njit(fastmath=False)
def gen_constant_bias(N, input_bias, seed=0):
    return np.ones(N)*input_bias

def csc_to_csr(data, indices, indptr, shape):
    mat = sp.sparse.csc_matrix((data, indices, indptr), shape = [shape[0], shape[1]]).tocsr()
    return mat.data, mat.indices, mat.indptr, np.array([mat.shape[0], mat.shape[1]], dtype = np.int32)


spec = [
    ('reservoir_size', int32),
    ('input_size', int32),
    ('feature_size', int32),
    ('seed', int32),
    ('spectral_radius', float64),
    ('input_weight', float64),
    ('random_bias', boolean),
    ('bias_weight', float64),
    ('leakage', float64),
    ('avg_degree', float64),
    ('A_data', float64[:]),
    ('A_indices', int32[:]),
    ('A_indptr', int32[:]),
    ('A_shape', int32[:]),
    ('B_data', float64[:]),
    ('B_indices', int32[:]),
    ('B_indptr', int32[:]),
    ('B_shape', int32[:]),
    ('C', float64[:]),
    ('regularization', float64),
    ('train_noise', float64),
    ('W', float64[:, :]),
    ('squarenodes', boolean),
    ('output_bias', boolean),
    ('input_pass', boolean),
    ('r', float64[:]),
    ('feature', float64[:]),
    ('standardized', boolean),
    ('mean', float64),
    ('std', float64)
]

#@jitclass(spec)
class Reservoir():
    def __init__(self, reservoir_size=300, input_size=30, spectral_radius=0.6, avg_degree=3,
                 input_weight=1.0, bias_weight=1.0, leakage=1.0, regularization = 0., train_noise = 0.,
                 random_bias = True, squarenodes=True, output_bias=True, input_pass=True, seed=0,
                 standardized=False, mean=0., std=0., noise_seed = 10, Uhlenbeck_noise=False,Uhlenbeck_process=None,num_non_zero=1,fully_dense=False):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.spectral_radius = spectral_radius
        self.avg_degree = avg_degree
        self.input_weight = input_weight
        self.random_bias = random_bias
        self.bias_weight = bias_weight
        self.leakage = leakage
        self.regularization = regularization
        self.train_noise = train_noise
        self.squarenodes = squarenodes
        self.output_bias = output_bias
        self.input_pass = input_pass
        self.seed = seed
        self.noise_seed=noise_seed
        self.standardized = standardized
        self.rng = np.random.default_rng(noise_seed)
        self.Uhlenbeck_noise=Uhlenbeck_noise
        self.Uhlenbeck_process=Uhlenbeck_process
        self.A_data, self.A_indices, self.A_indptr, self.A_shape = gen_adjacency(self.reservoir_size,
                                                                                 self.avg_degree,
                                                                                 self.spectral_radius, self.seed)
        if fully_dense:
            self.B_data, self.B_indices, self.B_indptr, self.B_shape = gen_input_weights_fully_dense(self.reservoir_size,
                                                                                 self.input_size,
                                                                                 self.input_weight, self.seed)
        elif num_non_zero == 1:
            self.B_data, self.B_indices, self.B_indptr, self.B_shape = gen_input_weights(self.reservoir_size,
                                                                                 self.input_size,
                                                                                 self.input_weight, self.seed)
        elif num_non_zero > 1:
            self.B_data, self.B_indices, self.B_indptr, self.B_shape = gen_input_weights_dense(self.reservoir_size,
                                                                                         self.input_size,
                                                                                         self.input_weight, num_non_zero=num_non_zero, seed=self.seed)
        else:
            raise ValueError('num_non_zero has to be >=1')
        if random_bias:
            self.C = gen_bias(self.reservoir_size, self.bias_weight, self.seed)
        else:
            self.C = gen_constant_bias(self.reservoir_size, self.bias_weight, self.seed)

        self.feature_size = self.reservoir_size
        if self.squarenodes:
            self.feature_size *= 2
        if self.input_pass:
            self.feature_size += self.input_size
        if self.output_bias:
            self.feature_size += 1

        self.r = np.zeros(self.reservoir_size)
        self.feature = np.zeros(self.feature_size)

        if standardized:
            if mean:
                self.mean = mean
            if std:
                self.std = std

    def reset(self):
        self.r = np.zeros(self.reservoir_size)
        self.feature = np.zeros(self.feature_size)
        self.rng = np.random.default_rng(self.noise_seed)

    def train(self, data, T, discard_len=0, return_error=False):
        features = \
            self.get_features(data, T, discard_len, train=True)
        if return_error:
            info_mat, target_mat, output = self.get_train_mats(data, features, T, discard_len, return_error)
            train_preds, train_error = self.train_from_mats(info_mat, target_mat, return_error, features, output)
            return train_preds, train_error
        else:
            info_mat, target_mat = self.get_train_mats(data, features, T, discard_len, return_error)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, features, T, discard_len, return_error=False):
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat

    def batch_get_train_mats_fast(self, data, T, num_batch=1,discard_len=0):
        assert (T % num_batch == 0)
        batch_size = T // num_batch
        info_mat = np.zeros((self.feature_size, self.feature_size))
        target_mat = np.zeros((self.feature_size, self.input_size))
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T + batch_size, batch_size)):
            if batch_begin == 0:
                features = self.get_features(data[:batch_end + discard_len], batch_size, discard_len, train=True)
            else:
                features = self.get_features(data[batch_begin + discard_len:batch_end + discard_len], batch_size,0, train=True)
            output = data[batch_begin + discard_len+1:batch_end + discard_len+1]
            info_mat += (features.T @ features) / T
            target_mat += (features.T @ output) / T
        return info_mat,target_mat

    def batch_get_train_mats_fast_with_model(self,model, model_discard_len, T, num_batch=1,discard_len=0):
        assert (T % num_batch == 0)
        batch_size = T // num_batch
        info_mat = np.zeros((self.feature_size, self.feature_size))
        target_mat = np.zeros((self.feature_size, self.input_size))
        #model.run(2*model_discard_len,0)
        model.run(model_discard_len, 0)
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T + batch_size, batch_size)):
            if batch_begin == 0:
                data_batch = model.run(batch_end + discard_len, 0)
                features = self.get_features(data_batch[:-1], batch_size, discard_len, train=True)
                output = data_batch[batch_begin + discard_len+1:batch_end + discard_len+1]
            else:
                data_batch = model.run(batch_end- batch_begin)
                features = self.get_features(data_batch[:-1], batch_size,0, train=True)
                output = data_batch[1:]
            #output = data[batch_begin + discard_len+1:batch_end + discard_len+1]
            info_mat += (features.T @ features) / T
            target_mat += (features.T @ output) / T
        return info_mat,target_mat
    def batch_get_train_mats_fast_with_model_noise_in_target(self,model, model_discard_len, T, num_batch=1,discard_len=0):
        assert (T % num_batch == 0)
        batch_size = T // num_batch
        info_mat = np.zeros((self.feature_size, self.feature_size))
        target_mat = np.zeros((self.feature_size, self.input_size))
        #model.run(2*model_discard_len,0)
        model.run(model_discard_len, 0)
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T + batch_size, batch_size)):
            if batch_begin == 0:
                data_batch = model.run(batch_end + discard_len, 0)
                data_batch = data_batch+self.rng.normal(size=data_batch.shape) * self.train_noise
                features = self.get_features(data_batch[:-1], batch_size, discard_len, train=False)
                output = data_batch[batch_begin + discard_len+1:batch_end + discard_len+1]
            else:
                data_batch = model.run(batch_end- batch_begin)
                data_batch = data_batch+self.rng.normal(size=data_batch.shape) * self.train_noise
                features = self.get_features(data_batch[:-1], batch_size,0, train=False)
                output = data_batch[1:]
            #output = data[batch_begin + discard_len+1:batch_end + discard_len+1]
            info_mat += (features.T @ features) / T
            target_mat += (features.T @ output) / T
        return info_mat,target_mat

    def get_train_mats_out(self, data, T, discard_len, return_error=False):
        features = \
            self.get_features(data, T, discard_len, train=True)
        output = data[discard_len + 1:discard_len + T + 1]
        if discard_len == 0:
            output = data[:T]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output,features
        else:
            return info_mat, target_mat
    def get_train_mats_out_dyn_noise_in_input(self, data, data_noise, T, discard_len, return_error=False):
        features = \
            self.get_features(data_noise, T, discard_len, train=False)
        output = data[discard_len + 1:discard_len + T + 1]
        if discard_len == 0:
            output = data[:T]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat
    def get_train_mats_out_noise_in_target(self, data, T, discard_len, return_error=False):
        data = data + self.rng.normal(size=data.shape) * self.train_noise
        features = \
            self.get_features(data, T, discard_len, train=False)
        output = data[discard_len + 1:discard_len + T + 1]
        if discard_len == 0:
            output = data[:T]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output,features
        else:
            return info_mat, target_mat
    def get_train_mats_out_random_sample(self, data, T, T_all, discard_len, return_error=False,input_features=False,features_all_input=None,output_all_input=None,seed=1):
        if input_features:
            features_all = features_all_input
            output_all = output_all_input
        else:
            features_all = \
                self.get_features(data, T_all, discard_len, train=True)
            output_all = data[discard_len + 1:discard_len + T_all + 1]
            if discard_len == 0:
                output_all = data[:T_all]
        rng = np.random.default_rng(seed)
        random_sample_index = rng.choice(T_all,T,replace=False)
        features = features_all[random_sample_index]
        output = output_all[random_sample_index]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat,features,output,features_all,output_all
        else:
            return info_mat, target_mat
    def get_train_mats_out_random_sample_noise_only_in_target(self, data, T, T_all, discard_len, return_error=False,input_features=False,features_all_input=None,output_all_input=None,seed=1):
        if input_features:
            features_all = features_all_input
            output_all = output_all_input
        else:
            features_all = \
                self.get_features(data, T_all, discard_len, train=False)
            output_all = data[discard_len + 1:discard_len + T_all + 1]
            output_all = output_all + self.rng.normal(size=output_all.shape) * self.train_noise
            if discard_len == 0:
                output_all = data[:T_all]
        rng = np.random.default_rng(seed)
        random_sample_index = rng.choice(T_all,T,replace=False)
        features = features_all[random_sample_index]
        output = output_all[random_sample_index]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat,features_all,output_all
        else:
            return info_mat, target_mat
    def get_train_mats_out_noise_only_in_target(self, data, T, discard_len, return_error=False):
        features = \
            self.get_features(data, T, discard_len, train=False)
        output = data[discard_len + 1:discard_len + T + 1]
        output = output + self.rng.normal(size=output.shape) * self.train_noise 
        if discard_len == 0:
            output = data[:T]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output,features
        else:
            return info_mat, target_mat
    def train_from_mats(self, info_mat, target_mat, return_error=False, features=None, output=None):
        self.W = np.linalg.solve(info_mat + np.diag(
            np.ones(self.feature.size) * self.regularization), target_mat)
        if return_error:
            train_preds = features @ self.W
            train_error = mean_numba_axis1((train_preds - output) ** 2.0)
            return train_error

    def train_from_mats_svd(self,P,D,Q_t,features,output, return_error=False):
        coeff=D/(D**2/features.shape[0]+self.regularization)
        intermed=P.T @ output / features.shape[0]
        self.W = Q_t.T @ np.diag(coeff) @ intermed
        if return_error:
            train_preds = features @ self.W
            train_error = mean_numba_axis1((train_preds - output) ** 2.0)
            return train_error
    def train_from_mats_svd_transpose_feature(self,P,D,Q_t,features,output, return_error=False):
        coeff=D/(D**2/features.shape[0]+self.regularization)
        self.W =(output.T/features.shape[0])@Q_t.T@np.diag(coeff)@P.T
        if return_error:
            train_preds =  self.W @ features.T
            train_error = mean_numba_axis1((train_preds - output.T) ** 2.0)
            return train_error  
    def get_features(self, data, T, discard_len=0, train=False):
        if train:
            train_data = data + self.rng.normal(size=data.shape) * self.train_noise
        else:
            train_data = data
        if self.Uhlenbeck_noise:
            train_data = data + self.Uhlenbeck_process.run(data.shape[0]-1)
        
        features, self.r, self.feature = \
            get_reservoir_features(train_data, self.r, discard_len, T, self.leakage, self.A_data,
                                         self.A_indices,
                                         self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                         self.B_shape, self.C, self.feature_size, self.squarenodes, self.input_pass,
                                         self.output_bias)
        return features

    def sync_and_predict(self, data, T, sync_len=0):
        self.r = np.zeros(self.r.size)
        prediction, self.r, self.feature = sync_and_predict_reservoir(data, self.r, sync_len, T, self.leakage,
                                                                      self.A_data, self.A_indices,
                                                self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                                self.B_shape, self.C, self.W, self.feature_size, self.squarenodes,
                                                self.input_pass, self.output_bias)
        return prediction

    def sync_and_predict_open_loop(self, data, T, sync_len=0):
        self.r = np.zeros(self.r.size)
        prediction, self.r, self.feature, features = sync_and_predict_reservoir_open_loop(data, self.r, sync_len, T, self.leakage,
                                                                      self.A_data, self.A_indices,
                                                self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                                self.B_shape, self.C, self.W, self.feature_size, self.squarenodes,
                                                self.input_pass, self.output_bias)
        return prediction,features

    def predict(self, T):
        prediction, self.r, self.feature = predict_reservoir(self.r, self.feature, T, self.leakage, self.A_data,
                                                             self.A_indices,
                                       self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                       self.B_shape, self.C, self.W, self.feature_size, self.squarenodes,
                                       self.input_pass, self.output_bias)
        return prediction

    def synchronize(self, data, sync_len):
        self.r = np.zeros(self.r.size)
        self.r, self.feature = synchronize_reservoir(data, self.r, sync_len, self.leakage, self.A_data, self.A_indices,
                                                     self.A_indptr, self.A_shape, self.B_data, self.B_indices,
                                                     self.B_indptr, self.B_shape, self.C, self.feature_size,
                                                     self.squarenodes, self.input_pass, self.output_bias)

    def forward(self, data):
        self.r, self.feature = forward_reservoir(data, self.r, self.leakage, self.A_data, self.A_indices,
                                                 self.A_indptr, self.A_shape, self.B_data, self.B_indices,
                                                 self.B_indptr, self.B_shape, self.C, self.feature_size,
                                                 self.squarenodes, self.input_pass, self.output_bias)


class ReservoirLocal():
    def __init__(self, reservoir_size=100, input_size = 30, num_regions = 0, local_overlap = 3, spectral_radius=0.6,
                 avg_degree=3, input_weight=1.0, bias_weight=1.0, leakage=1.0, regularization = 0., train_noise = 0.,
                 random_bias = True, squarenodes=True, output_bias=True, input_pass = True, seed=0,
                 standardized=False, mean=0., std=0., transposeA = True, noise_seed = 10):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.local_overlap = local_overlap
        self.spectral_radius = spectral_radius
        self.avg_degree = avg_degree
        self.input_weight = input_weight
        self.random_bias = random_bias
        self.bias_weight = bias_weight
        self.leakage = leakage
        self.regularization = regularization
        self.train_noise = train_noise
        self.squarenodes = squarenodes
        self.output_bias = output_bias
        self.input_pass = input_pass
        self.seed = seed
        self.standardized = standardized
        self.rng = np.random.default_rng(noise_seed)
        if num_regions == 0:
            self.num_regions = self.input_size
        else:
            self.num_regions = num_regions
        assert((self.input_size / self.num_regions) % 1 == 0 )
        self.local_output_size = self.input_size // self.num_regions
        self.local_input_size  = self.local_output_size + 2*self.local_overlap


        self.A_data, self.A_indices, self.A_indptr, self.A_shape = gen_adjacency(self.reservoir_size,
                                                                                 self.avg_degree,
                                                                                 self.spectral_radius, self.seed,
                                                                                 transpose = transposeA)

        self.B_data, self.B_indices, self.B_indptr, self.B_shape = gen_input_weights(self.reservoir_size,
                                                                                 self.local_input_size,
                                                                                 self.input_weight, self.seed,
                                                                                 transpose = True)
        self.A_data, self.A_indices, self.A_indptr, self.A_shape = csc_to_csr(self.A_data, self.A_indices,
                                                                              self.A_indptr, self.A_shape)

        self.B_data, self.B_indices, self.B_indptr, self.B_shape = csc_to_csr(self.B_data, self.B_indices,
                                                                              self.B_indptr, self.B_shape )
        if random_bias:
            self.C = gen_bias(self.reservoir_size, self.bias_weight, self.seed)
        else:
            self.C = gen_constant_bias(self.reservoir_size, self.bias_weight, self.seed)

        self.feature_size = self.reservoir_size
        if self.squarenodes:
            self.feature_size *= 2
        if self.input_pass:
            self.feature_size += self.local_input_size
        if self.output_bias:
            self.feature_size += 1

        self.r = np.zeros((self.num_regions, self.reservoir_size))
        self.feature = np.zeros((self.num_regions, self.feature_size))

        self.mean = mean
        self.std = std

    def train(self, data, T, discard_len=0, return_error = False):
        features = \
            self.get_features(data, T, discard_len, train = True)
        if return_error:
            info_mat, target_mat, output = self.get_train_mats(data, features, T, discard_len, return_error)
            train_preds, train_error = self.train_from_mats(info_mat, target_mat, return_error, features, output)
            return train_preds, train_error
        else:
            info_mat, target_mat = self.get_train_mats(data, features, T, discard_len, return_error)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, features, T, discard_len, return_error = False):
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (features.T @ features) / T
        target_mat = (features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error = False, features = None, output = None):
        self.W = np.linalg.solve(info_mat + np.diag(
            np.ones(self.feature.size) * self.regularization), target_mat)
        if return_error:
            train_preds = features @ self.W
            train_error = np.sqrt(mean_numba_axis1((train_preds - output) ** 2.0))
            return train_preds, train_error

    def get_features(self, data, T, discard_len=0, train = False):
        if train:
            train_data = data + self.rng.normal(size = data.shape)*self.train_noise
        else:
            train_data = data
        features, self.r, self.feature = \
            get_reservoir_features_local(train_data, self.r, self.feature, discard_len, T, self.leakage, self.A_data,
                                         self.A_indices,
                                         self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                         self.B_shape, self.C, self.feature_size, self.squarenodes, self.input_pass,
                                         self.output_bias, self.num_regions, self.local_output_size,
                                         self.local_input_size, self.local_overlap)
        return features

    def sync_and_predict(self, data, T, sync_len=0):
        self.r = np.zeros(self.r.shape)
        data_in = roll_data_2d(data, self.num_regions, self.local_output_size, self.local_input_size,
                               self.local_overlap)
        prediction, self.r, self.feature =\
            sync_and_predict_reservoir_local(data_in, self.r, self.feature, sync_len, T, self.leakage, self.A_data, self.A_indices,
                                             self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                             self.B_shape, self.C, self.W, self.feature_size, self.squarenodes,
                                             self.input_pass, self.output_bias, self.num_regions,
                                             self.local_output_size, self.local_input_size, self.local_overlap)
        return prediction

    def predict(self, T):
        prediction, self.r, self.feature = \
            predict_reservoir_local(self.r, self.feature, T, self.leakage, self.A_data, self.A_indices, self.A_indptr,
                                    self.A_shape, self.B_data, self.B_indices, self.B_indptr, self.B_shape, self.C,
                                    self.W, self.feature_size, self.squarenodes,self.input_pass, self.output_bias,
                                    self.num_regions, self.local_output_size, self.local_input_size, self.local_overlap)
        return prediction

    def synchronize(self, data, sync_len):
        self.r = np.zeros(self.r.shape)
        data_in = roll_data_2d(data, self.num_regions, self.local_output_size, self.local_input_size,
                               self.local_overlap)
        self.r, self.feature = \
            synchronize_reservoir_local(data_in, self.r, sync_len, self.leakage, self.A_data, self.A_indices,
                                        self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                        self.B_shape, self.C, self.feature_size, self.squarenodes, self.input_pass,
                                        self.output_bias, self.num_regions, self.local_output_size,
                                        self.local_input_size, self.local_overlap)

    def forward(self, data):
        rolled_data = roll_data_1d(data, self.num_regions, self.local_output_size, self.local_input_size,
                                   self.local_overlap)
        self.r, self.feature = \
            forward_reservoir_local(data, self.r, self.leakage, self.A_data, self.A_indices, self.A_indptr,
                                    self.A_shape, self.B_data, self.B_indices, self.B_indptr, self.B_shape, self.C,
                                    self.feature_size, self.squarenodes, self.input_pass, self.output_bias)

@njit(fastmath=False)
def RBFK(X,Y,sigma): # let X be an N by 1 vector, Y be an M by 1 vector
    X_Norm=np.sum(X**2,axis=-1)
    Y_Norm=np.sum(Y**2,axis=-1)
    return np.exp(-1/(2*sigma**2)*(X_Norm[:,None]+Y_Norm[None,:]-2*np.dot(X,Y.T)))

class Kernel_Reservoir():
    def __init__(self, reservoir_size=300, input_size=30, spectral_radius=0.6, avg_degree=3,
                 input_weight=1.0, bias_weight=1.0, leakage=1.0, regularization=0., train_noise=0.,
                 random_bias=True, squarenodes=True, output_bias=True, input_pass=True, seed=0,
                 standardized=False, mean=0., std=0., noise_seed=10, Uhlenbeck_noise=False, Uhlenbeck_process=None,sigma=1):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.spectral_radius = spectral_radius
        self.avg_degree = avg_degree
        self.input_weight = input_weight
        self.random_bias = random_bias
        self.bias_weight = bias_weight
        self.leakage = leakage
        self.regularization = regularization
        self.train_noise = train_noise
        self.squarenodes = squarenodes
        self.output_bias = output_bias
        self.input_pass = input_pass
        self.seed = seed
        self.noise_seed = noise_seed
        self.standardized = standardized
        self.rng = np.random.default_rng(noise_seed)
        self.Uhlenbeck_noise = Uhlenbeck_noise
        self.Uhlenbeck_process = Uhlenbeck_process
        self.sigma=sigma

        self.A_data, self.A_indices, self.A_indptr, self.A_shape = gen_adjacency(self.reservoir_size,
                                                                                 self.avg_degree,
                                                                                 self.spectral_radius, self.seed)

        self.B_data, self.B_indices, self.B_indptr, self.B_shape = gen_input_weights(self.reservoir_size,
                                                                                     self.input_size,
                                                                                     self.input_weight, self.seed)
        if random_bias:
            self.C = gen_bias(self.reservoir_size, self.bias_weight, self.seed)
        else:
            self.C = gen_constant_bias(self.reservoir_size, self.bias_weight, self.seed)

        self.feature_size = self.reservoir_size
        if self.squarenodes:
            self.feature_size *= 2
        if self.input_pass:
            self.feature_size += self.input_size
        if self.output_bias:
            self.feature_size += 1

        self.r = np.zeros(self.reservoir_size)
        self.feature = np.zeros(self.feature_size)

        if standardized:
            if mean:
                self.mean = mean
            if std:
                self.std = std
        self.K_inv = None
        self.output = None
        self.train_features = None

    def get_train_mats(self, data, T, discard_len=0, train=False, return_error=False):
        if train:
            train_data = data + self.rng.normal(size=data.shape) * self.train_noise
        else:
            train_data = data
        if self.Uhlenbeck_noise:
            train_data = data + self.Uhlenbeck_process.run(data.shape[0] - 1)

        features, self.r, self.feature = \
            get_reservoir_features(train_data, self.r, discard_len, T, self.leakage, self.A_data,
                                   self.A_indices,
                                   self.A_indptr, self.A_shape, self.B_data, self.B_indices, self.B_indptr,
                                   self.B_shape, self.C, self.feature_size, self.squarenodes, self.input_pass,
                                   self.output_bias)

        self.train_features = features
        self.K = RBFK(features,features,sigma=self.sigma)
        self.output = data[discard_len + 1:discard_len + T + 1]
    
    def train_from_mats(self,T, return_error=False):
        self.K_inv = np.linalg.solve(self.K+self.regularization*np.identity(T),np.identity(T))
        if return_error:
            train_preds = self.output.T @ self.K_inv @ K
            train_error = mean_numba_axis1((train_preds.T - self.output) ** 2.0)
            return train_preds, train_error, self.output

    
    def sync_and_predict(self, data, T, sync_len=0):
        self.r = np.zeros(self.r.size)
        prediction, self.r, self.feature = sync_and_predict_kernel_reservoir(data, self.r, self.train_features, self.K_inv,self.output, self.sigma, sync_len, T, self.leakage,
                                                                      self.A_data, self.A_indices,
                                                                      self.A_indptr, self.A_shape, self.B_data,
                                                                      self.B_indices, self.B_indptr,
                                                                      self.B_shape, self.C, self.feature_size,
                                                                      self.squarenodes,
                                                                      self.input_pass, self.output_bias)
        return prediction

    def predict(self, T):
        prediction, self.r, self.feature = predict_kernel_reservoir(self.r, self.feature, self.train_features, self.K_inv,self.output, self.sigma, T, self.leakage, self.A_data,
                                                             self.A_indices,
                                                             self.A_indptr, self.A_shape, self.B_data, self.B_indices,
                                                             self.B_indptr,
                                                             self.B_shape, self.C, self.feature_size,
                                                             self.squarenodes,
                                                             self.input_pass, self.output_bias)
        return prediction

    def synchronize(self, data, sync_len):
        self.r = np.zeros(self.r.size)
        self.r, self.feature = synchronize_reservoir(data, self.r, sync_len, self.leakage, self.A_data, self.A_indices,
                                                     self.A_indptr, self.A_shape, self.B_data, self.B_indices,
                                                     self.B_indptr, self.B_shape, self.C, self.feature_size,
                                                     self.squarenodes, self.input_pass, self.output_bias)





#@jitclass(spec)
class LorenzModel1Hybrid():
    model: LorenzModel1
    reservoir: Reservoir
    def __init__(self, model, reservoir, H = None, regularization = 0.,model_difference = False,prior= False,
                 reservoir_train_noise = 0.):
        self.model = model
        self.reservoir = reservoir
        self.bias_feature = np.zeros(self.model.D + self.reservoir.feature_size)
        self.prior = prior
        if self.reservoir.input_pass:
            self.model_difference = model_difference
        else:
            self.model_difference = False
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if type(regularization) == float:

            if regularization == 0.:
                self.regularization = self.reservoir.regularization
            else:
                self.regularization = regularization
        else:
            self.regularization = regularization
        if reservoir_train_noise == 0.:
            self.reservoir_train_noise = self.reservoir.train_noise
        else:
            self.reservoir_train_noise = reservoir_train_noise
            self.reservoir.train_noise = reservoir_train_noise

    def observe(self):
        return self.H(self.state)

    def train(self, data, T, discard_len=0, return_error = True):
        bias_features = \
            self.get_features(data, T, discard_len, train = True)
        if return_error:
            info_mat, target_mat, output = self.get_train_mats(data, bias_features, T, discard_len, return_error)
            train_preds, train_error,bias_features,self.W = self.train_from_mats(info_mat, target_mat, return_error, bias_features, output)
            return train_preds, train_error,bias_features,self.W,info_mat,target_mat,output
        else:
            info_mat, target_mat = self.get_train_mats(data, bias_features, T, discard_len, return_error)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, bias_features, T, discard_len, return_error = False):
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (bias_features.T @ bias_features) / T
        target_mat = (bias_features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat
    def get_train_mats_out(self, data, T, discard_len, return_error = False):
        bias_features = \
            self.get_features(data, T, discard_len, train = True)
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (bias_features.T @ bias_features) / T
        target_mat = (bias_features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error = False, bias_features = None, output = None):
        if self.prior:
            if self.model_difference:
                W0 = np.zeros((self.bias_feature.size,self.model.D))
                W0[np.arange(self.model.D), np.arange(self.model.D)] = 1.0
                if self.reservoir.output_bias:
                    W0[-(np.arange(self.model.D)+2), -(np.arange(self.model.D)+1)] = 1.0
                else:
                    W0[-(np.arange(self.model.D)+1), -(np.arange(self.model.D)+1)] = 1.0
            else:
                W0 = np.zeros((self.bias_feature.size, self.model.D))
                W0[np.arange(self.model.D), np.arange(self.model.D)] = 1.0
            self.W = np.linalg.solve(info_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization), target_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization) @ W0)
            if np.isnan(np.sum(self.W)):
                print("NAN in W")
        else:
            self.W = np.linalg.solve(info_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization), target_mat)
            if np.isnan(np.sum(self.W)):
                print("NAN in W")
        if return_error:
            train_preds = bias_features @ self.W
            train_error = np.sqrt(mean_numba_axis1((train_preds - output) ** 2.0))
            if np.isnan(np.sum(train_preds)):
                print("NAN in train_preds")
            return train_preds, train_error,bias_features,self.W

    def get_features(self, data, T, discard_len=0, train = False):
        if self.model_difference: #change this please!
            bias_features = np.zeros((T, self.bias_feature.size))
            bias_features[:, self.model.D:] = self.reservoir.get_features(data, T, discard_len, train=train)
            if self.reservoir.output_bias:
                bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len + T])-bias_features[:, -(self.model.D+1):-1]
            else:
                bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len + T]) - bias_features[:, -self.model.D:]
        else:
            bias_features = np.zeros((T, self.bias_feature.size))
            bias_features[:, self.model.D:] = self.reservoir.get_features(data, T, discard_len, train = train)
            bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len+T])
        self.bias_feature = bias_features[-1]
        return bias_features

    def sync_and_predict(self, data, T, sync_len=0):
        self.synchronize(data, sync_len)
        return self.predict(T)
    def predict(self, T):
        output, self.bias_feature, self.reservoir.r = \
            predict_hybrid(self.bias_feature, self.model.D, self.W, T, self.model.dxdt,
                           self.model.int_steps, self.model.h,
                           self.model.params,
                           self.model.standardized, self.model.mean, self.model.std,
                           self.reservoir.r, self.reservoir.leakage,
                           self.reservoir.A_data, self.reservoir.A_indices, self.reservoir.A_indptr,
                           self.reservoir.A_shape, self.reservoir.B_data, self.reservoir.B_indices,
                           self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                           self.reservoir.feature_size, self.reservoir.squarenodes, self.reservoir.input_pass,
                           self.reservoir.output_bias,self.model_difference)
        self.state = output[-1]
        self.reservoir.feature = self.bias_feature[self.model.D:]
        if self.model_difference:
            if self.reservoir.output_bias:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1])
                else:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1]
            else:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-self.model.D:])
                else:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-self.model.D:]

        else:
            if self.model.standardized:
                self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D])
            else:
                self.model.state = self.bias_feature[:self.model.D]
        return output

    def synchronize(self, data, sync_len):
        self.reservoir.r = np.zeros(self.reservoir.reservoir_size)
        self.reservoir.synchronize(data, sync_len)
        self.bias_feature[self.model.D:] = self.reservoir.feature
        if self.model_difference:
            if self.reservoir.output_bias:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(data[sync_len - 1])
                    self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())-self.bias_feature[-(self.model.D+1):-1]
                else:
                    self.model.state = data[sync_len - 1]
                    self.bias_feature[:self.model.D] = self.model.forward()-self.bias_feature[-(self.model.D+1):-1]
            else:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(data[sync_len - 1])
                    self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())-self.bias_feature[-self.model.D:]
                else:
                    self.model.state = data[sync_len - 1]
                    self.bias_feature[:self.model.D] = self.model.forward()-self.bias_feature[-self.model.D:]
        else:
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1])
                self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())
            else:
                self.model.state = data[sync_len-1]
                self.bias_feature[:self.model.D] = self.model.forward()

    def forward(self, data):
        if self.model.standardized:
            self.bias_feature, self.r = \
                get_feature_hybrid_standardized(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                                     self.model.params, self.model.mean,
                                                     self.model.std, self.reservoir.r, self.reservoir.leakage,
                                                     self.reservoir.A_data, self.reservoir.A_indices,
                                                     self.reservoir.A_indptr, self.reservoir.A_shape,
                                                     self.reservoir.B_data, self.reservoir.B_indices,
                                                     self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                                     self.reservoir.feature_size, self.reservoir.squarenodes,
                                                     self.reservoir.input_pass, self.reservoir.output_bias,self.model_difference)
            self.reservoir.feature = self.bias_feature[self.model.D:]
            if self.model_difference:
                if self.reservoir.output_bias:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1])
                else:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D] + self.bias_feature[-self.model.D:])
            else:
                self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D])
            self.state = self.bias_feature @ self.W
        else:
            self.bias_feature, self.r = \
                get_feature_hybrid(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                        self.model.params, self.reservoir.r,
                                        self.reservoir.leakage,
                                        self.reservoir.A_data, self.reservoir.A_indices,
                                        self.reservoir.A_indptr, self.reservoir.A_shape,
                                        self.reservoir.B_data, self.reservoir.B_indices,
                                        self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                        self.reservoir.feature_size, self.reservoir.squarenodes,
                                        self.reservoir.input_pass, self.reservoir.output_bias,self.model_difference)
            self.reservoir.feature = self.bias_feature[self.model.D:]
            if self.model_difference:
                if self.reservoir.output_bias:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1]
                else:
                    self.model.state = self.bias_feature[:self.model.D] + self.bias_feature[-self.model.D:]
            else:
                self.model.state = self.bias_feature[:self.model.D]
            self.model.state = self.bias_feature[:self.model.D]
            self.state = self.bias_feature @ self.W

    def get_state(self):
        self.state = self.bias_feature @ self.W


class KursivHybrid():
    model: Kursiv
    reservoir: Reservoir
    def __init__(self, model, reservoir, H = None, regularization = 0.,model_difference = False,prior= False,
                 reservoir_train_noise = 0.):
        self.model = model
        self.reservoir = reservoir
        self.bias_feature = np.zeros(self.model.D + self.reservoir.feature_size)
        self.prior = prior
        if self.reservoir.input_pass:
            self.model_difference = model_difference
        else:
            self.model_difference = False
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if type(regularization) == float:

            if regularization == 0.:
                self.regularization = self.reservoir.regularization
            else:
                self.regularization = regularization
        else:
            self.regularization = regularization
        if reservoir_train_noise == 0.:
            self.reservoir_train_noise = self.reservoir.train_noise
        else:
            self.reservoir_train_noise = reservoir_train_noise
            self.reservoir.train_noise = reservoir_train_noise

    def observe(self):
        return self.H(self.state)

    def train(self, data, T, discard_len=0, return_error = True):
        bias_features = \
            self.get_features(data, T, discard_len, train = True)
        if return_error:
            info_mat, target_mat, output = self.get_train_mats(data, bias_features, T, discard_len, return_error)
            train_preds, train_error,bias_features,self.W = self.train_from_mats(info_mat, target_mat, return_error, bias_features, output)
            return train_preds, train_error,bias_features,self.W,info_mat,target_mat,output
        else:
            info_mat, target_mat = self.get_train_mats(data, bias_features, T, discard_len, return_error)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, bias_features, T, discard_len, return_error = False):
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (bias_features.T @ bias_features) / T
        target_mat = (bias_features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat
    def get_train_mats_out(self, data, T, discard_len, return_error = False):
        bias_features = \
            self.get_features(data, T, discard_len, train = True)
        output = data[discard_len + 1:discard_len + T + 1]
        info_mat = (bias_features.T @ bias_features) / T
        target_mat = (bias_features.T @ output) / T
        if return_error:
            return info_mat, target_mat, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error = False, bias_features = None, output = None):
        if self.prior:
            if self.model_difference:
                W0 = np.zeros((self.bias_feature.size,self.model.D))
                W0[np.arange(self.model.D), np.arange(self.model.D)] = 1.0
                if self.reservoir.output_bias:
                    W0[-(np.arange(self.model.D)+2), -(np.arange(self.model.D)+1)] = 1.0
                else:
                    W0[-(np.arange(self.model.D)+1), -(np.arange(self.model.D)+1)] = 1.0
            else:
                W0 = np.zeros((self.bias_feature.size, self.model.D))
                W0[np.arange(self.model.D), np.arange(self.model.D)] = 1.0
            print(W0[np.arange(self.model.D), :])
            self.W = np.linalg.solve(info_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization), target_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization) @ W0)
            if np.isnan(np.sum(self.W)):
                print("NAN in W")
        else:
            self.W = np.linalg.solve(info_mat + np.diag(
                np.ones(self.bias_feature.size) * self.regularization), target_mat)
            if np.isnan(np.sum(self.W)):
                print("NAN in W")
        if return_error:
            train_preds = bias_features @ self.W
            if np.isnan(np.sum(train_preds)):
                print("NAN in train_preds")
            return train_preds, train_error
    def get_features(self, data, T, discard_len=0, train = False):
        if self.model_difference: #change this please!
            bias_features = np.zeros((T, self.bias_feature.size))
            bias_features[:, self.model.D:] = self.reservoir.get_features(data, T, discard_len, train=train)
            if self.reservoir.output_bias:
                bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len + T])-bias_features[:, -(self.model.D+1):-1]
            else:
                bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len + T]) - bias_features[:, -self.model.D:]
        else:
            bias_features = np.zeros((T, self.bias_feature.size))
            bias_features[:, self.model.D:] = self.reservoir.get_features(data, T, discard_len, train = train)
            bias_features[:, :self.model.D] = self.model.run_array(data[discard_len:discard_len+T])
        self.bias_feature = bias_features[-1]
        return bias_features

    def sync_and_predict(self, data, T, sync_len=0):
        self.synchronize(data, sync_len)
        return self.predict(T)
    def predict(self, T):
        output, self.bias_feature, self.reservoir.r = \
        predict_hybrid_kursiv(self.bias_feature, self.model.D, self.W, T,
                           self.model.int_steps,
                           self.model.params,
                           self.model.standardized, self.model.mean, self.model.std,
                           self.reservoir.r, self.reservoir.leakage,
                           self.reservoir.A_data, self.reservoir.A_indices, self.reservoir.A_indptr,
                           self.reservoir.A_shape, self.reservoir.B_data, self.reservoir.B_indices,
                           self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                           self.reservoir.feature_size, self.reservoir.squarenodes, self.reservoir.input_pass,
                           self.reservoir.output_bias,self.model_difference)
        self.state = output[-1]
        self.reservoir.feature = self.bias_feature[self.model.D:]
        if self.model_difference:
            if self.reservoir.output_bias:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1])
                else:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1]
            else:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-self.model.D:])
                else:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-self.model.D:]

        else:
            if self.model.standardized:
                self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D])
            else:
                self.model.state = self.bias_feature[:self.model.D]
        return output

    def synchronize(self, data, sync_len):
        self.reservoir.r = np.zeros(self.reservoir.reservoir_size)
        self.reservoir.synchronize(data, sync_len)
        self.bias_feature[self.model.D:] = self.reservoir.feature
        if self.model_difference:
            if self.reservoir.output_bias:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(data[sync_len - 1])
                    self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())-self.bias_feature[-(self.model.D+1):-1]
                else:
                    self.model.state = data[sync_len - 1]
                    self.bias_feature[:self.model.D] = self.model.forward()-self.bias_feature[-(self.model.D+1):-1]
            else:
                if self.model.standardized:
                    self.model.state = self.model.unstandardize(data[sync_len - 1])
                    self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())-self.bias_feature[-self.model.D:]
                else:
                    self.model.state = data[sync_len - 1]
                    self.bias_feature[:self.model.D] = self.model.forward()-self.bias_feature[-self.model.D:]
        else:
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1])
                self.bias_feature[:self.model.D] = self.model.standardize(self.model.forward())
            else:
                self.model.state = data[sync_len-1]
                self.bias_feature[:self.model.D] = self.model.forward()

    def forward(self, data):
        if self.model.standardized:
            self.bias_feature, self.r = \
                get_feature_hybrid_standardized_kursiv(data, self.model.int_steps,
                                                     self.model.params, self.model.mean,
                                                     self.model.std, self.reservoir.r, self.reservoir.leakage,
                                                     self.reservoir.A_data, self.reservoir.A_indices,
                                                     self.reservoir.A_indptr, self.reservoir.A_shape,
                                                     self.reservoir.B_data, self.reservoir.B_indices,
                                                     self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                                     self.reservoir.feature_size, self.reservoir.squarenodes,
                                                     self.reservoir.input_pass, self.reservoir.output_bias,self.model_difference)
            self.reservoir.feature = self.bias_feature[self.model.D:]
            if self.model_difference:
                if self.reservoir.output_bias:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1])
                else:
                    self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D] + self.bias_feature[-self.model.D:])
            else:
                self.model.state = self.model.unstandardize(self.bias_feature[:self.model.D])
            self.state = self.bias_feature @ self.W
        else:
            self.bias_feature, self.r = \
                get_feature_hybrid_kursiv(data, self.model.int_steps,
                                        self.model.params, self.reservoir.r,
                                        self.reservoir.leakage,
                                        self.reservoir.A_data, self.reservoir.A_indices,
                                        self.reservoir.A_indptr, self.reservoir.A_shape,
                                        self.reservoir.B_data, self.reservoir.B_indices,
                                        self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                        self.reservoir.feature_size, self.reservoir.squarenodes,
                                        self.reservoir.input_pass, self.reservoir.output_bias,self.model_difference)
            self.reservoir.feature = self.bias_feature[self.model.D:]
            if self.model_difference:
                if self.reservoir.output_bias:
                    self.model.state = self.bias_feature[:self.model.D]+self.bias_feature[-(self.model.D+1):-1]
                else:
                    self.model.state = self.bias_feature[:self.model.D] + self.bias_feature[-self.model.D:]
            else:
                self.model.state = self.bias_feature[:self.model.D]
            self.model.state = self.bias_feature[:self.model.D]
            self.state = self.bias_feature @ self.W

    def get_state(self):
        self.state = self.bias_feature @ self.W
class LorenzModel1HybridLocal():
    model: LorenzModel1
    reservoir: ReservoirLocal
    def __init__(self, model, reservoir, H = None, regularization = 0.,
                 reservoir_train_noise = 0.):
        self.model = model
        self.reservoir = reservoir
        self.bias_feature = np.zeros((self.reservoir.num_regions, self.reservoir.local_output_size + \
                                      self.reservoir.feature_size))
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if regularization == 0.:
            self.regularization = self.reservoir.regularization
        else:
            self.regularization = regularization
        if reservoir_train_noise == 0.:
            self.reservoir_train_noise = self.reservoir.train_noise
        else:
            self.reservoir_train_noise = reservoir_train_noise

    def observe(self):
        return self.H(self.state)

    def train(self, data, T, discard_len=0, return_error=True, num_batches = 1):
        if return_error:
            info_mat, target_mat, bias_features, output = self.get_train_mats(data, T, discard_len, return_error,
                                                               num_batches = 1)
            train_preds, train_error = self.train_from_mats(info_mat, target_mat, return_error, bias_features, output)
            return train_preds, train_error
        else:
            info_mat, target_mat = self.get_train_mats(data, T, discard_len, return_error, num_batches = num_batches)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, T, discard_len, return_error=False, num_batches = 1):
        assert(T % num_batches == 0)
        batch_size = T // num_batches
        info_mat = np.zeros((self.bias_feature.shape[1], self.bias_feature.shape[1]))
        target_mat = np.zeros((self.bias_feature.shape[1], self.reservoir.local_output_size))
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T+batch_size, batch_size)):
            if batch_begin == 0:
                bias_features = self.get_features(data[:batch_end+discard_len], batch_size, discard_len, train = True)
            else:
                bias_features = self.get_features(data[batch_begin+discard_len:batch_end + discard_len], batch_size,
                                                  0, train=True)
            output = roll_data_2d(data[discard_len + batch_begin + 1:discard_len + batch_end + 1],
                                  self.reservoir.num_regions, self.reservoir.local_output_size,
                                  self.reservoir.local_output_size, 0).reshape(bias_features.shape[0] *
                                                                               self.reservoir.num_regions, -1)
            train_features = bias_features.reshape(bias_features.shape[0] * self.reservoir.num_regions, -1)
            info_mat += (train_features.T @ train_features) / (T * self.reservoir.num_regions)
            target_mat += (train_features.T @ output) / (T * self.reservoir.num_regions)
        if return_error:
            return info_mat, target_mat, bias_features, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error=False, bias_features=None, output=None):
        self.W = np.linalg.solve(info_mat + np.diag(
            np.ones(self.bias_feature.shape[1]) * self.regularization), target_mat)
        if return_error:
            train_preds = (bias_features @ self.W).reshape(bias_features.shape[0], -1)
            output_unrolled = output.reshape(bias_features.shape[0], self.model.D)
            train_error = np.sqrt(mean_numba_axis1((train_preds - output_unrolled) ** 2.0))
            return train_preds, train_error

    def get_features(self, data, T, discard_len=0, train = False):
        rolled_data = roll_data_2d(data, self.reservoir.num_regions, self.reservoir.local_output_size,
                                   self.reservoir.local_input_size, self.reservoir.local_overlap)
        bias_features = np.zeros((T, self.bias_feature.shape[0], self.bias_feature.shape[1]))
        bias_features[..., self.reservoir.local_output_size:] = self.reservoir.get_features(rolled_data, T, discard_len,
                                                                                            train = train)
        bias_features[..., :self.reservoir.local_output_size] = \
            roll_data_2d(self.model.run_array(data[discard_len:discard_len+T, ::self.model_grid_ratio]),
                         self.reservoir.num_regions, self.reservoir.local_output_size,
                         self.reservoir.local_output_size, 0)
        self.bias_feature = bias_features[-1]
        return bias_features

    def sync_and_predict(self, data, T, sync_len=0):
        self.synchronize(data, sync_len)
        return self.predict(T)

    def predict(self, T):
        output, self.bias_feature, self.reservoir.r = \
            predict_hybrid_local(self.bias_feature, self.model.D, self.W, T, self.model.dxdt,
                                 self.model.int_steps, self.model.h,
                                 self.model.params,
                                 self.model.standardized, self.model.mean, self.model.std,
                                 self.reservoir.r, self.reservoir.leakage,
                                 self.reservoir.A_data, self.reservoir.A_indices, self.reservoir.A_indptr,
                                 self.reservoir.A_shape, self.reservoir.B_data, self.reservoir.B_indices,
                                 self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                 self.reservoir.feature_size, self.reservoir.squarenodes, self.reservoir.input_pass,
                                 self.reservoir.output_bias, self.reservoir.num_regions,
                                 self.reservoir.local_output_size, self.reservoir.local_input_size,
                                 self.reservoir.local_overlap)
        self.state = output[-1]
        self.reservoir.feature = self.bias_feature[:, self.reservoir.local_output_size:]
        if self.model.standardized:
            self.model.state = \
                self.model.unstandardize(self.bias_feature[:, :self.reservoir.local_output_size].flatten())
        else:
            self.model.state = self.bias_feature[:, :self.reservoir.local_output_size].flatten()
        return output

    def synchronize(self, data, sync_len):
        self.reservoir.r = np.zeros((self.reservoir.num_regions, self.reservoir.reservoir_size))
        self.reservoir.synchronize(data, sync_len)
        self.bias_feature[:, self.reservoir.local_output_size:] = self.reservoir.feature
        if self.model.standardized:
            self.model.state = self.model.unstandardize(data[sync_len - 1])
            self.bias_feature[:, :self.reservoir.local_output_size] = \
                roll_data_1d(self.model.standardize(self.model.forward()), self.reservoir.num_regions,
                             self.reservoir.local_output_size, self.reservoir.local_output_size, 0)
        else:
            self.model.state = data[sync_len-1]
            self.bias_feature[:, :self.reservoir.local_output_size] = \
                roll_data_1d(self.model.forward(), self.reservoir.num_regions, self.reservoir.local_output_size,
                             self.reservoir.local_output_size, 0)

    def forward(self, data):
        if self.model.standardized:
            self.bias_feature, self.r = \
                get_feature_hybrid_local_standardized(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                                           self.model.params, self.model.mean,
                                                           self.model.std, self.reservoir.r, self.reservoir.leakage,
                                                           self.reservoir.A_data, self.reservoir.A_indices,
                                                           self.reservoir.A_indptr, self.reservoir.A_shape,
                                                           self.reservoir.B_data, self.reservoir.B_indices,
                                                           self.reservoir.B_indptr, self.reservoir.B_shape,
                                                           self.reservoir.C,
                                                           self.reservoir.feature_size, self.reservoir.squarenodes,
                                                           self.reservoir.input_pass, self.reservoir.output_bias,
                                                           self.reservoir.num_regions, self.reservoir.local_output_size,
                                                           self.reservoir.local_input_size,
                                                           self.reservoir.local_overlap)
            self.reservoir.feature = self.bias_feature[:, self.reservoir.local_output_size:]
            self.model.state = \
                self.model.unstandardize(self.bias_feature[:, :self.reservoir.local_output_size].flatten())
            self.state = (self.bias_feature @ self.W).flatten()
        else:
            self.bias_feature, self.r = \
                get_feature_hybrid_local(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                              self.model.params, self.reservoir.r,
                                              self.reservoir.leakage,
                                              self.reservoir.A_data, self.reservoir.A_indices,
                                              self.reservoir.A_indptr, self.reservoir.A_shape,
                                              self.reservoir.B_data, self.reservoir.B_indices,
                                              self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                              self.reservoir.feature_size, self.reservoir.squarenodes,
                                              self.reservoir.input_pass, self.reservoir.output_bias,
                                              self.reservoir.num_regions, self.reservoir.local_output_size,
                                              self.reservoir.local_input_size, self.reservoir.local_overlap)
            self.reservoir.feature = self.bias_feature[:, self.reservoir.local_output_size:]
            self.model.state = self.bias_feature[:, :self.reservoir.local_output_size].flatten()
            self.state = (self.bias_feature @ self.W).flatten()

    def get_state(self):
        self.state = (self.bias_feature @ self.W).flatten()

class LorenzModel1BiasHybridLocal():
    model: LorenzModel1
    reservoir: ReservoirLocal
    def __init__(self, model, reservoir, H = None, regularization = 0.,
                 reservoir_train_noise = 0., model_in_feature = False, model_interp_fun = None):
        self.model = model
        self.reservoir = reservoir
        self.model_in_feature = model_in_feature
        self.D = self.reservoir.input_size
        self.model_grid_ratio = self.D // self.model.D
        if isinstance(model_interp_fun, type(None)):
            self.model_interp_fun = numba_lin_interp_periodic
        else:
            self.model_interp_fun = model_interp_fun
        if self.model_in_feature:
            self.bias_feature = np.zeros((self.reservoir.num_regions, self.reservoir.local_output_size + \
                                          self.reservoir.feature_size))
        else:
            self.bias_feature = np.zeros((self.reservoir.num_regions, self.reservoir.feature_size))
        self.obs_feature  = np.zeros(self.bias_feature.shape)
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if regularization == 0.:
            self.regularization = self.reservoir.regularization
        else:
            self.regularization = regularization
        if reservoir_train_noise == 0.:
            self.reservoir_train_noise = self.reservoir.train_noise
        else:
            self.reservoir_train_noise = reservoir_train_noise

    def observe(self):
        return self.H(self.state + (self.obs_feature @ self.W_H).flatten())

    def train(self, data, T, discard_len=0, return_error=False, num_batches = 1):
        if return_error:
            info_mat, target_mat, bias_features, model_forecasts, output = \
                self.get_train_mats(data, T, discard_len, return_error, num_batches = 1)
            train_preds, train_error = self.train_from_mats(info_mat, target_mat, return_error, bias_features, output,
                                                            model_forecasts)
            return train_preds, train_error
        else:
            info_mat, target_mat = self.get_train_mats(data, T, discard_len, return_error, num_batches = num_batches)
            self.train_from_mats(info_mat, target_mat)

    def get_train_mats(self, data, T, discard_len, return_error=False, num_batches = 1):
        assert (T % num_batches == 0)
        batch_size = T // num_batches
        info_mat = np.zeros((self.bias_feature.shape[1], self.bias_feature.shape[1]))
        target_mat = np.zeros((self.bias_feature.shape[1], self.reservoir.local_output_size))
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T + batch_size, batch_size)):
            if batch_begin == 0:
                bias_features, model_forecasts = self.get_features(data[:batch_end + discard_len], batch_size,
                                                                   discard_len, train=True)
            else:
                bias_features, model_forecasts = self.get_features(
                    data[batch_begin + discard_len:batch_end + discard_len], batch_size, 0, train=True)
            output = roll_data_2d(data[discard_len + batch_begin + 1:discard_len + batch_end + 1],
                                  self.reservoir.num_regions, self.reservoir.local_output_size,
                                  self.reservoir.local_output_size, 0).reshape(bias_features.shape[0] *
                                                                               self.reservoir.num_regions, -1)
            train_features = bias_features.reshape(bias_features.shape[0] * self.reservoir.num_regions, -1)
            #grid_model_forecasts = rolled_output_to_grid(model_forecasts, self.reservoir.num_regions,
            #                                             self.model_interp_fun, self.model_grid_ratio,
            #                                             self.reservoir.local_output_size)
            info_mat += (train_features.T @ train_features) / (T * self.reservoir.num_regions)
            target_mat += (train_features.T @ (output - model_forecasts.reshape(bias_features.shape[0] *
                                                                           self.reservoir.num_regions, -1)))\
                     / (T * self.reservoir.num_regions)
        if return_error:
            return info_mat, target_mat, bias_features, model_forecasts, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error=False, bias_features=None, output=None,
                        model_forecasts = None):
        self.W_M = np.linalg.solve(info_mat + np.diag(
            np.ones(self.bias_feature.shape[1]) * self.regularization), target_mat)
        if return_error:
            train_preds = get_hybrid_bias_output_2D(bias_features, model_forecasts, self.W_M)
            output_unrolled = output.reshape(bias_features.shape[0], self.D)
            train_error = np.sqrt(mean_numba_axis1((train_preds - output_unrolled) ** 2.0))
            return train_preds, train_error

    def get_features(self, data, T, discard_len=0, train = False):
        rolled_data = roll_data_2d(data, self.reservoir.num_regions, self.reservoir.local_output_size,
                                   self.reservoir.local_input_size, self.reservoir.local_overlap)
        unrolled_model_forecasts = self.model.run_array(data[discard_len:discard_len + T, ::self.model_grid_ratio])
        model_forecasts = roll_interp_data_2d(
            unrolled_model_forecasts,
            self.D, self.reservoir.num_regions, self.model_interp_fun, self.model_grid_ratio,
            self.reservoir.local_output_size)
        if self.model_in_feature:
            bias_features = np.zeros((T, self.bias_feature.shape[0], self.bias_feature.shape[1]))
            bias_features[..., self.reservoir.local_output_size:] = self.reservoir.get_features(rolled_data, T,
                                                                                                discard_len,
                                                                                                train = train)
            bias_features[..., :self.reservoir.local_output_size] = model_forecasts
        else:
            bias_features = self.reservoir.get_features(rolled_data, T, discard_len, train = train)
        self.bias_feature = bias_features[-1]
        return bias_features, model_forecasts

    def sync_and_predict(self, data, T, sync_len=0):
        self.synchronize(data, sync_len)
        return self.predict(T)

    def predict(self, T):
        if self.model.standardized:
            model_in = self.model.standardize(self.model.state)
        else:
            model_in = self.model.state
        output, self.bias_feature, self.reservoir.r, model_state = \
            predict_bias_hybrid_local(self.bias_feature, model_in, self.D, self.W_M, T, self.model.dxdt,
                                      self.model.int_steps, self.model.h,
                                      self.model.params,
                                      self.model.standardized, self.model.mean, self.model.std,
                                      self.reservoir.r, self.reservoir.leakage,
                                      self.reservoir.A_data, self.reservoir.A_indices, self.reservoir.A_indptr,
                                      self.reservoir.A_shape, self.reservoir.B_data, self.reservoir.B_indices,
                                      self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                      self.reservoir.feature_size, self.reservoir.squarenodes,
                                      self.reservoir.input_pass,
                                      self.reservoir.output_bias, self.reservoir.num_regions,
                                      self.reservoir.local_output_size, self.reservoir.local_input_size,
                                      self.reservoir.local_overlap, self.model_in_feature,
                                      self.model_interp_fun, self.model_grid_ratio)
        self.state = output[-1]
        if self.model_in_feature:
            self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
        else:
            self.reservoir.feature = self.bias_feature
        if self.model.standardized:
            self.model.state = self.model.unstandardize(model_state)
        else:
            self.model.state = model_state
        return output

    def synchronize(self, data, sync_len):
        self.reservoir.r = np.zeros((self.reservoir.num_regions, self.reservoir.reservoir_size))
        self.reservoir.synchronize(data, sync_len)
        if self.model_in_feature:
            self.bias_feature[:, self.reservoir.local_output_size:] = self.reservoir.feature
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1, ::self.model_grid_ratio])
                self.bias_feature[:, :self.reservoir.local_output_size] = \
                    roll_interp_data_1d(self.model.standardize(self.model.forward()), self.D,
                                        self.reservoir.num_regions, self.model_interp_fun, self.model_grid_ratio,
                                        self.reservoir.local_output_size)
            else:
                self.model.state = data[sync_len-1, ::self.model_grid_ratio]
                self.bias_feature[:, :self.reservoir.local_output_size] = \
                    roll_interp_data_1d(self.model.forward(), self.D, self.reservoir.num_regions, self.model_interp_fun,
                                        self.model_grid_ratio, self.reservoir.local_output_size)
        else:
            self.bias_feature = self.reservoir.feature
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1, ::self.model_grid_ratio])
                tmp = self.model.forward()
            else:
                self.model.state = data[sync_len - 1, ::self.model_grid_ratio]
                tmp = self.model.forward()


    def forward(self, data):
        if self.model.standardized:
            self.bias_feature, self.r, model_state = \
                get_feature_bias_hybrid_local_standardized(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                                           self.model.params, self.model.mean,
                                                           self.model.std, self.reservoir.r, self.reservoir.leakage,
                                                           self.reservoir.A_data, self.reservoir.A_indices,
                                                           self.reservoir.A_indptr, self.reservoir.A_shape,
                                                           self.reservoir.B_data, self.reservoir.B_indices,
                                                           self.reservoir.B_indptr, self.reservoir.B_shape,
                                                           self.reservoir.C,
                                                           self.reservoir.feature_size, self.reservoir.squarenodes,
                                                           self.reservoir.input_pass, self.reservoir.output_bias,
                                                           self.reservoir.num_regions, self.reservoir.local_output_size,
                                                           self.reservoir.local_input_size,
                                                           self.reservoir.local_overlap, self.model_in_feature,
                                                           self.model_interp_fun, self.model_grid_ratio)
            if self.model_in_feature:
                self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
            else:
                self.reservoir.feature = self.bias_feature
            self.model.state = self.model.unstandardize(model_state)
            self.state = get_hybrid_bias_output(self.bias_feature, model_state, self.W_M, self.D,
                                                self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)
        else:
            self.bias_feature, self.r, model_state = \
                get_feature_bias_hybrid_local(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                              self.model.params, self.reservoir.r,
                                              self.reservoir.leakage,
                                              self.reservoir.A_data, self.reservoir.A_indices,
                                              self.reservoir.A_indptr, self.reservoir.A_shape,
                                              self.reservoir.B_data, self.reservoir.B_indices,
                                              self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                              self.reservoir.feature_size, self.reservoir.squarenodes,
                                              self.reservoir.input_pass, self.reservoir.output_bias,
                                              self.reservoir.num_regions, self.reservoir.local_output_size,
                                              self.reservoir.local_input_size, self.reservoir.local_overlap,
                                              self.model_in_feature, self.model_interp_fun,
                                              self.model_grid_ratio)
            if self.model_in_feature:
                self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
            else:
                self.reservoir.feature = self.bias_feature
            self.model.state = model_state
            self.state = get_hybrid_bias_output(self.bias_feature, model_state, self.W_M, self.D,
                                                self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)

    def get_state(self):
        if self.model.standardized:
            self.state = get_hybrid_bias_output(self.bias_feature, self.model.standardize(self.model.state),
                                                self.W_M, self.D, self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)
        else:
            self.state = get_hybrid_bias_output(self.bias_feature, self.model.state,
                                                self.W_M, self.D, self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)

class LorenzModel1BiasHybridLocal():
    model: LorenzModel1
    reservoir: ReservoirLocal
    def __init__(self, model, reservoir, H = None, regularization = 0.,
                 reservoir_train_noise = 0., model_in_feature = False, model_interp_fun = None):
        self.model = model
        self.reservoir = reservoir
        self.model_in_feature = model_in_feature
        self.D = self.reservoir.input_size
        self.model_grid_ratio = self.D // self.model.D
        if isinstance(model_interp_fun, type(None)):
            self.model_interp_fun = numba_lin_interp_periodic
        else:
            self.model_interp_fun = model_interp_fun
        if self.model_in_feature:
            self.bias_feature = np.zeros((self.reservoir.num_regions, self.reservoir.local_output_size + \
                                          self.reservoir.feature_size))
        else:
            self.bias_feature = np.zeros((self.reservoir.num_regions, self.reservoir.feature_size))
        self.obs_feature  = np.zeros(self.bias_feature.shape)
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if regularization == 0.:
            self.regularization = self.reservoir.regularization
        else:
            self.regularization = regularization
        if reservoir_train_noise == 0.:
            self.reservoir_train_noise = self.reservoir.train_noise
        else:
            self.reservoir_train_noise = reservoir_train_noise

    def observe(self):
        return self.H(self.state + (self.obs_feature @ self.W_H).flatten())

    def train(self, data, T, discard_len=0, return_error=False, num_batches = 1, train_w_kf = False):
        if train_w_kf:
            if return_error:
                self.W_M, bias_features, model_forecasts, output = \
                    self.train_with_kf(data, T, discard_len, return_error)
                train_preds, train_error = self.get_train_error(bias_features, model_forecasts, output)
                return train_preds, train_error
            else:
                self.W_M = self.train_with_kf(data, T, discard_len, return_error)
        else:
            if return_error:
                info_mat, target_mat, bias_features, model_forecasts, output = \
                    self.get_train_mats(data, T, discard_len, return_error, num_batches = 1)
                train_preds, train_error = self.train_from_mats(info_mat, target_mat, return_error, bias_features,
                                                                output, model_forecasts)
                return train_preds, train_error
            else:
                info_mat, target_mat = self.get_train_mats(data, T, discard_len, return_error, num_batches = num_batches)
                self.train_from_mats(info_mat, target_mat)

    def train_with_kf(self, data, T, discard_len, return_error = False, Q = np.zeros((1,1), dtype = np.float64)):
        P_W_M = np.eye(self.W_M.size) * self.regularization
        R = np.eye(self.reservoir.local_output_size * self.reservoir.num_regions) * self.reservoir.train_noise ** 2
        self.synchronize(data, discard_len)
        W_M = np.zeros((self.bias_feature.shape[1], self.reservoir.local_output_size))
        if return_error:
            bias_features = np.zeros((T, self.reservoir.num_regions, self.bias_feature.shape[1]))
            model_forecasts = np.zeros((T, self.reservoir.num_regions, self.reservoir.local_output_size))
            outputs = np.zeros((T, self.reservoir.num_regions * self.reservoir.local_output_size))
        for i in range(discard_len, T + discard_len):
            bias_feature, model_forecast = self.get_features(data[i].reshape(1, -1), 1, 0, train = False)
            output = data[i+1] + np.random.randn(data.shape[1]) * self.reservoir.train_noise
            W_M, P_W_M, ill_conditioned = kf_cycle(output, model_forecast, bias_feature, W_M, P_W_M, R, self.D,
                                                   self.reservoir.local_output_size, Q = Q)
            if return_error:
                bias_features[i - discard_len] = bias_feature
                model_forecasts[i - discard_len] = model_forecast
                outputs[i - discard_len] = output
        if return_error:
            return W_M, bias_features, model_forecasts, outputs
        else:
            return W_M



    def get_train_mats(self, data, T, discard_len, return_error=False, num_batches = 1):
        assert (T % num_batches == 0)
        batch_size = T // num_batches
        info_mat = np.zeros((self.bias_feature.shape[1], self.bias_feature.shape[1]))
        target_mat = np.zeros((self.bias_feature.shape[1], self.reservoir.local_output_size))
        for batch_begin, batch_end in zip(range(0, T, batch_size), range(batch_size, T + batch_size, batch_size)):
            if batch_begin == 0:
                bias_features, model_forecasts = self.get_features(data[:batch_end + discard_len], batch_size,
                                                                   discard_len, train=True)
            else:
                bias_features, model_forecasts = self.get_features(
                    data[batch_begin + discard_len:batch_end + discard_len], batch_size, 0, train=True)
            output = roll_data_2d(data[discard_len + batch_begin + 1:discard_len + batch_end + 1],
                                  self.reservoir.num_regions, self.reservoir.local_output_size,
                                  self.reservoir.local_output_size, 0).reshape(bias_features.shape[0] *
                                                                               self.reservoir.num_regions, -1)
            train_features = bias_features.reshape(bias_features.shape[0] * self.reservoir.num_regions, -1)
            #grid_model_forecasts = rolled_output_to_grid(model_forecasts, self.reservoir.num_regions,
            #                                             self.model_interp_fun, self.model_grid_ratio,
            #                                             self.reservoir.local_output_size)
            info_mat += (train_features.T @ train_features) / (T * self.reservoir.num_regions)
            target_mat += (train_features.T @ (output - model_forecasts.reshape(bias_features.shape[0] *
                                                                           self.reservoir.num_regions, -1)))\
                     / (T * self.reservoir.num_regions)
        if return_error:
            return info_mat, target_mat, bias_features, model_forecasts, output
        else:
            return info_mat, target_mat

    def train_from_mats(self, info_mat, target_mat, return_error=False, bias_features=None, output=None,
                        model_forecasts = None):
        self.W_M = np.linalg.solve(info_mat + np.diag(
            np.ones(self.bias_feature.shape[1]) * self.regularization), target_mat)
        if return_error:
            train_preds, train_error = self.get_train_error(bias_features, model_forecasts, output)
            return train_preds, train_error

    def get_train_error(self, bias_features, model_forecasts, output):
            train_preds = get_hybrid_bias_output_2D(bias_features, model_forecasts, self.W_M)
            output_unrolled = output.reshape(bias_features.shape[0], self.D)
            train_error = np.sqrt(mean_numba_axis1((train_preds - output_unrolled) ** 2.0))
            return train_preds, train_error

    def get_features(self, data, T, discard_len=0, train = False):
        rolled_data = roll_data_2d(data, self.reservoir.num_regions, self.reservoir.local_output_size,
                                   self.reservoir.local_input_size, self.reservoir.local_overlap)
        unrolled_model_forecasts = self.model.run_array(data[discard_len:discard_len + T, ::self.model_grid_ratio])
        model_forecasts = roll_interp_data_2d(
            unrolled_model_forecasts,
            self.D, self.reservoir.num_regions, self.model_interp_fun, self.model_grid_ratio,
            self.reservoir.local_output_size)
        if self.model_in_feature:
            bias_features = np.zeros((T, self.bias_feature.shape[0], self.bias_feature.shape[1]))
            bias_features[..., self.reservoir.local_output_size:] = self.reservoir.get_features(rolled_data, T,
                                                                                                discard_len,
                                                                                                train = train)
            bias_features[..., :self.reservoir.local_output_size] = model_forecasts
        else:
            bias_features = self.reservoir.get_features(rolled_data, T, discard_len, train = train)
        self.bias_feature = bias_features[-1]
        return bias_features, model_forecasts

    def sync_and_predict(self, data, T, sync_len=0):
        self.synchronize(data, sync_len)
        return self.predict(T)

    def predict(self, T):
        if self.model.standardized:
            model_in = self.model.standardize(self.model.state)
        else:
            model_in = self.model.state
        output, self.bias_feature, self.reservoir.r, model_state = \
            predict_bias_hybrid_local(self.bias_feature, model_in, self.D, self.W_M, T, self.model.dxdt,
                                      self.model.int_steps, self.model.h,
                                      self.model.params,
                                      self.model.standardized, self.model.mean, self.model.std,
                                      self.reservoir.r, self.reservoir.leakage,
                                      self.reservoir.A_data, self.reservoir.A_indices, self.reservoir.A_indptr,
                                      self.reservoir.A_shape, self.reservoir.B_data, self.reservoir.B_indices,
                                      self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                      self.reservoir.feature_size, self.reservoir.squarenodes,
                                      self.reservoir.input_pass,
                                      self.reservoir.output_bias, self.reservoir.num_regions,
                                      self.reservoir.local_output_size, self.reservoir.local_input_size,
                                      self.reservoir.local_overlap, self.model_in_feature,
                                      self.model_interp_fun, self.model_grid_ratio)
        self.state = output[-1]
        if self.model_in_feature:
            self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
        else:
            self.reservoir.feature = self.bias_feature
        if self.model.standardized:
            self.model.state = self.model.unstandardize(model_state)
        else:
            self.model.state = model_state
        return output

    def synchronize(self, data, sync_len):
        self.reservoir.r = np.zeros((self.reservoir.num_regions, self.reservoir.reservoir_size))
        self.reservoir.synchronize(data, sync_len)
        if self.model_in_feature:
            self.bias_feature[:, self.reservoir.local_output_size:] = self.reservoir.feature
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1, ::self.model_grid_ratio])
                self.bias_feature[:, :self.reservoir.local_output_size] = \
                    roll_interp_data_1d(self.model.standardize(self.model.forward()), self.D,
                                        self.reservoir.num_regions, self.model_interp_fun, self.model_grid_ratio,
                                        self.reservoir.local_output_size)
            else:
                self.model.state = data[sync_len-1, ::self.model_grid_ratio]
                self.bias_feature[:, :self.reservoir.local_output_size] = \
                    roll_interp_data_1d(self.model.forward(), self.D, self.reservoir.num_regions, self.model_interp_fun,
                                        self.model_grid_ratio, self.reservoir.local_output_size)
        else:
            self.bias_feature = self.reservoir.feature
            if self.model.standardized:
                self.model.state = self.model.unstandardize(data[sync_len - 1, ::self.model_grid_ratio])
                tmp = self.model.forward()
            else:
                self.model.state = data[sync_len - 1, ::self.model_grid_ratio]
                tmp = self.model.forward()


    def forward(self, data):
        if self.model.standardized:
            self.bias_feature, self.r, model_state = \
                get_feature_bias_hybrid_local_standardized(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                                           self.model.params, self.model.mean,
                                                           self.model.std, self.reservoir.r, self.reservoir.leakage,
                                                           self.reservoir.A_data, self.reservoir.A_indices,
                                                           self.reservoir.A_indptr, self.reservoir.A_shape,
                                                           self.reservoir.B_data, self.reservoir.B_indices,
                                                           self.reservoir.B_indptr, self.reservoir.B_shape,
                                                           self.reservoir.C,
                                                           self.reservoir.feature_size, self.reservoir.squarenodes,
                                                           self.reservoir.input_pass, self.reservoir.output_bias,
                                                           self.reservoir.num_regions, self.reservoir.local_output_size,
                                                           self.reservoir.local_input_size,
                                                           self.reservoir.local_overlap, self.model_in_feature,
                                                           self.model_interp_fun, self.model_grid_ratio)
            if self.model_in_feature:
                self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
            else:
                self.reservoir.feature = self.bias_feature
            self.model.state = self.model.unstandardize(model_state)
            self.state = get_hybrid_bias_output(self.bias_feature, model_state, self.W_M, self.D,
                                                self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)
        else:
            self.bias_feature, self.r, model_state = \
                get_feature_bias_hybrid_local(data, self.model.dxdt, self.model.int_steps, self.model.h,
                                              self.model.params, self.reservoir.r,
                                              self.reservoir.leakage,
                                              self.reservoir.A_data, self.reservoir.A_indices,
                                              self.reservoir.A_indptr, self.reservoir.A_shape,
                                              self.reservoir.B_data, self.reservoir.B_indices,
                                              self.reservoir.B_indptr, self.reservoir.B_shape, self.reservoir.C,
                                              self.reservoir.feature_size, self.reservoir.squarenodes,
                                              self.reservoir.input_pass, self.reservoir.output_bias,
                                              self.reservoir.num_regions, self.reservoir.local_output_size,
                                              self.reservoir.local_input_size, self.reservoir.local_overlap,
                                              self.model_in_feature, self.model_interp_fun,
                                              self.model_grid_ratio)
            if self.model_in_feature:
                self.reservoir.feature = self.bias_feature[:, self.model_grid_ratio:]
            else:
                self.reservoir.feature = self.bias_feature
            self.model.state = model_state
            self.state = get_hybrid_bias_output(self.bias_feature, model_state, self.W_M, self.D,
                                                self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)

    def get_state(self):
        if self.model.standardized:
            self.state = get_hybrid_bias_output(self.bias_feature, self.model.standardize(self.model.state),
                                                self.W_M, self.D, self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)
        else:
            self.state = get_hybrid_bias_output(self.bias_feature, self.model.state,
                                                self.W_M, self.D, self.reservoir.num_regions, self.model_grid_ratio,
                                                self.reservoir.local_output_size)

class ETKF():
    def __init__(self, model, ensemble_members = 5, mult_covariance_inflation = 0.3,
                 hybrid_int_steps = 1, assimilate_model = False, assimilate_obs = False,
                 H = None, init_ensemble_state = np.array([], dtype = np.float64),
                 init_ensemble_var = 0., seed = 0, init_W_M_ensemble = None, r_in_ensemble = False,
                 W_M_kf = False, P_W_M = np.ones((1,1), dtype = np.float64), Q = np.zeros((1,1), dtype = np.float64),
                 method = '1b', H_mat = np.zeros((1,1), dtype = np.float64)):
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.ensemble_members = ensemble_members
        self.hybrid_int_steps = hybrid_int_steps
        self.W_M_kf = W_M_kf
        self.P_W_M = P_W_M
        self.Q = Q
        self.method = method
        self.H_mat = H_mat
        if 'Hybrid' in self.model_name:
            self.assimilate_model = assimilate_model
            self.assimilate_obs = assimilate_obs
            self.reservoir_background = np.ones((self.ensemble_members, 1, 1)) * self.model.reservoir.r
            self.r_in_ensemble = r_in_ensemble
        else:
            self.assimilate_model = False
            self.assimilate_obs = False
        if isinstance(H, type(None)):
            self.H = default_H
        else:
            self.H = H
        if isinstance(mult_covariance_inflation, np.ndarray):
            self.inflation_vec = (1 + mult_covariance_inflation)
        else:
            self.inflation_vec = (1 + mult_covariance_inflation) * np.ones(init_ensemble_state.size)
        np.random.seed(seed)
        if isinstance(init_ensemble_var, np.ndarray) and init_W_M_ensemble is not None:
            self.ensemble = np.ascontiguousarray(np.concatenate((
                np.random.multivariate_normal(init_ensemble_state[:-init_W_M_ensemble.shape[1]],
                                              np.diag(init_ensemble_var[:-init_W_M_ensemble.shape[1]]),
                                              size=self.ensemble_members),
                init_W_M_ensemble * init_ensemble_var[-init_W_M_ensemble.shape[1]:]
            ), axis = 1))
        elif isinstance(init_ensemble_var, np.ndarray):
            self.ensemble = \
                np.ascontiguousarray(np.random.multivariate_normal(init_ensemble_state, np.diag(init_ensemble_var),
                                                                   size=self.ensemble_members))
        else:
            self.ensemble = init_ensemble_state + \
                                 np.random.randn(self.ensemble_members, init_ensemble_state.size)*np.sqrt(init_ensemble_var)
        self.analysis_size = self.inflation_vec.size

    def analysis(self):
        return mean_numba_axis0(self.ensemble)
    def run_cycle(self, observations, obs_indices, R_inverse, T, spin_up = True, total_spin_up_len = 500,
                  model_spin_up_len = 250, sync_len = 50, model_H = None, model_ensemble_members = 0,
                  return_ensemble = True, return_background = False, fourD = False):
        assert(total_spin_up_len >= sync_len + model_spin_up_len)
        if isinstance(model_H, type(None)):
            model_H = self.H
        if model_ensemble_members == 0:
            model_ensemble_members = self.ensemble_members
        if fourD:
            len_mult = self.hybrid_int_steps
        else:
            len_mult = 1
        if 'Hybrid' in self.model_name:
            hybrid_spin_up_len = total_spin_up_len - sync_len - model_spin_up_len
            if spin_up:
                model_ensembles = run_etkf_cycle_lorenz(observations[:(model_spin_up_len+sync_len)*len_mult],
                                                        obs_indices[:(model_spin_up_len+sync_len)*len_mult],
                                                        R_inverse, model_H,
                                                        np.expand_dims(self.ensemble[:model_ensemble_members,
                                                                       :self.model.D:self.model.model_grid_ratio], 0),
                                                        self.inflation_vec[:self.model.D:self.model.model_grid_ratio],
                                                        self.model.model.dxdt,
                                                        self.model.model.int_steps, self.model.model.h,
                                                        self.model.model.params, self.hybrid_int_steps, fourD=fourD)
                #model_ensembles = np.ascontiguousarray(model_ensembles[::self.hybrid_int_steps])
                self.ensemble[:model_ensemble_members, :self.model.D] = interp_model_2d(model_ensembles[-1], self.model.D,
                                                                  self.model.model_grid_ratio)
                if model_ensemble_members != self.ensemble_members:
                    for ensemble_begin, ensemble_end in zip(range(model_ensemble_members,
                                                                  self.ensemble_members,
                                                                  model_ensemble_members),
                                                            range(2*model_ensemble_members,
                                                                  self.ensemble_members+model_ensemble_members,
                                                                  model_ensemble_members)):
                        self.ensemble[ensemble_begin:min(self.ensemble_members, ensemble_end), :self.model.D] = \
                            self.ensemble[:min(model_ensemble_members, self.ensemble_members - ensemble_begin),
                                                  :self.model.D]
                #print('Model Ensemble')
                #print(self.ensemble[1,:5])
                bias_features, self.reservoir_background, model_states = \
                    synchronize_background_bias_hybrid_local(
                        interp_model_3d(model_ensembles[-sync_len*self.hybrid_int_steps:-1],
                                        self.model.D,
                                        self.model.model_grid_ratio),
                        self.model.model.dxdt,
                        self.model.model.int_steps, self.model.model.h,
                        self.model.model.params,
                        self.reservoir_background, self.model.reservoir.leakage,
                        self.model.reservoir.A_data, self.model.reservoir.A_indices,
                        self.model.reservoir.A_indptr,self.model.reservoir.A_shape,
                        self.model.reservoir.B_data, self.model.reservoir.B_indices,
                        self.model.reservoir.B_indptr, self.model.reservoir.B_shape,
                        self.model.reservoir.C,
                        self.model.W_M.reshape(1, -1, self.model.reservoir.local_output_size),
                        self.model.reservoir.feature_size,
                        self.model.reservoir.squarenodes,
                        self.model.reservoir.input_pass,
                        self.model.reservoir.output_bias,
                        self.model.reservoir.num_regions,
                        self.model.reservoir.local_output_size,
                        self.model.reservoir.local_input_size,
                        self.model.reservoir.local_overlap,
                        self.model.model_in_feature,
                        self.model.model_interp_fun,
                        self.model.model_grid_ratio,
                        self.hybrid_int_steps)
            if self.r_in_ensemble:
                self.ensemble[:, self.model.D:self.model.D + self.model.reservoir.reservoir_size * self.model.reservoir.num_regions] = \
                    self.reservoir_background[:,-1]

                #print('Reservoir after synchronization')
                #print(self.reservoir_background[1,1,:5])
                #print('Model States after Synchronization')
                #print(model_states[1,:5])
            ensembles, bias_features, self.reservoir_background, model_states, W_M, P_W_M = \
                run_etkf_cycle_bias_hybrid_local(observations[(model_spin_up_len+sync_len)*len_mult: \
                                                              (total_spin_up_len+T)*len_mult],
                                                 obs_indices[(model_spin_up_len+sync_len)*len_mult: \
                                                             (total_spin_up_len+T)*len_mult],
                                                 R_inverse, self.H, np.expand_dims(self.ensemble,0),
                                                 self.inflation_vec, self.model.model.dxdt,
                                                 self.model.model.int_steps, self.model.model.h,
                                                 self.model.model.params,
                                                 self.reservoir_background, self.model.reservoir.leakage,
                                                 self.model.reservoir.A_data, self.model.reservoir.A_indices,
                                                 self.model.reservoir.A_indptr,self.model.reservoir.A_shape,
                                                 self.model.reservoir.B_data, self.model.reservoir.B_indices,
                                                 self.model.reservoir.B_indptr, self.model.reservoir.B_shape,
                                                 self.model.reservoir.C, 
                                                 self.model.W_M.reshape(1, -1, self.model.reservoir.local_output_size),
                                                 self.model.W_H.reshape(1, -1, self.model.reservoir.local_output_size),
                                                 self.model.reservoir.feature_size,
                                                 self.model.reservoir.squarenodes,
                                                 self.model.reservoir.input_pass,
                                                 self.model.reservoir.output_bias,
                                                 self.model.reservoir.num_regions,
                                                 self.model.reservoir.local_output_size,
                                                 self.model.reservoir.local_input_size,
                                                 self.model.reservoir.local_overlap,
                                                 self.model.model_in_feature, self.model.D,
                                                 self.model.model_interp_fun, self.model.model_grid_ratio,
                                                 self.hybrid_int_steps,
                                                 self.assimilate_model,
                                                 self.assimilate_obs,
                                                 return_ensemble, fourD = fourD, r_in_ensemble = self.r_in_ensemble,
                                                 W_M_kf = self.W_M_kf, P_W_M = self.P_W_M, Q = self.Q,
                                                 method = self.method, H_mat = self.H_mat)
            if self.W_M_kf:
                self.model.W_M = W_M[0]
                self.P_W_M = P_W_M

            if return_ensemble and return_background:
                analyses = np.mean(ensembles[hybrid_spin_up_len*self.hybrid_int_steps:\
                                             (hybrid_spin_up_len+T)*self.hybrid_int_steps], axis = 1)
                return analyses, ensembles[hybrid_spin_up_len*self.hybrid_int_steps:\
                                           (hybrid_spin_up_len+T)*self.hybrid_int_steps]
            elif return_ensemble:
                analyses = np.ascontiguousarray(np.mean(ensembles[hybrid_spin_up_len*self.hybrid_int_steps: \
                                             (hybrid_spin_up_len + T)*self.hybrid_int_steps:\
                                             self.hybrid_int_steps], axis=1))
                return analyses, np.ascontiguousarray(ensembles[hybrid_spin_up_len*self.hybrid_int_steps: \
                                           (hybrid_spin_up_len+T)*self.hybrid_int_steps:\
                                           self.hybrid_int_steps])
            elif return_background:
                return ensembles[hybrid_spin_up_len*self.hybrid_int_steps:\
                                 (hybrid_spin_up_len+T)*self.hybrid_int_steps, 0]
            else:
                return np.ascontiguousarray(ensembles[hybrid_spin_up_len*self.hybrid_int_steps:\
                                 (hybrid_spin_up_len+T)*self.hybrid_int_steps:\
                                 self.hybrid_int_steps, 0])
        elif 'Model1' in self.model_name or self.model_name == 'LorenzModel':
            print(self.ensemble.shape)
            ensembles = run_etkf_cycle_lorenz(observations[:(total_spin_up_len+T)*len_mult],
                                              obs_indices[:(total_spin_up_len+T)*len_mult],
                                              R_inverse, self.H, np.expand_dims(self.ensemble, 0),
                                              self.inflation_vec, self.model.dxdt,
                                              self.model.int_steps, self.model.h,
                                              self.model.params, self.hybrid_int_steps, return_ensemble,
                                              fourD=fourD)

            if return_ensemble and return_background:
                analyses = np.mean(ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                             (total_spin_up_len + T) * self.hybrid_int_steps], axis=1)
                return analyses, ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                           (total_spin_up_len + T) * self.hybrid_int_steps]
            elif return_ensemble:
                analyses = np.mean(ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                             (total_spin_up_len + T) * self.hybrid_int_steps: \
                                             self.hybrid_int_steps], axis=1)
                return analyses, ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                           (total_spin_up_len + T) * self.hybrid_int_steps: \
                                           self.hybrid_int_steps]
            elif return_background:
                return ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                 (total_spin_up_len + T) * self.hybrid_int_steps, 0]
            else:
                return ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                 (total_spin_up_len + T) * self.hybrid_int_steps: \
                                 self.hybrid_int_steps, 0]
        elif 'Model2' in self.model_name:
            ensembles = run_etkf_cycle_lorenz_2(observations[:(total_spin_up_len + T)*len_mult],
                                                obs_indices[:(total_spin_up_len + T)*len_mult],
                                                R_inverse, self.H, self.ensemble,
                                                self.inflation_vec, self.model.dxdt,
                                                self.model.int_steps, self.model.h,
                                                self.model.s_mat_data, self.model.s_mat_indices,
                                                self.model.s_mat_indptr, self.model.s_mat_shape,
                                                self.model.params, self.hybrid_int_steps, return_ensemble,
                                                return_background, fourD = fourD)
            if return_ensemble and return_background:
                analyses = np.mean(ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                             (total_spin_up_len + T) * self.hybrid_int_steps], axis=1)
                return analyses, ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                           (total_spin_up_len + T) * self.hybrid_int_steps]
            elif return_ensemble:
                analyses = np.mean(ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                             (total_spin_up_len + T) * self.hybrid_int_steps: \
                                             self.hybrid_int_steps], axis=1)
                return analyses, ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                           (total_spin_up_len + T) * self.hybrid_int_steps: \
                                           self.hybrid_int_steps]
            elif return_background:
                return ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                 (total_spin_up_len + T) * self.hybrid_int_steps, 0]
            else:
                return ensembles[total_spin_up_len * self.hybrid_int_steps: \
                                 (total_spin_up_len + T) * self.hybrid_int_steps: \
                                 self.hybrid_int_steps, 0]



