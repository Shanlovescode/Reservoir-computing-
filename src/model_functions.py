from numba import njit, objmode, prange,jit
from scipy import special
from numpy.fft import fft, ifft
from numba.typed import List
import numpy as np
from src.helpers import *
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from numba_progress import ProgressBar
from scipy.sparse import coo_matrix, csr_matrix


@njit()
def standardize(x, mean, std):
    return (x - mean)/std

@njit()
def unstandardize(x, mean, std):
    return x*std + mean

@njit()
def rk4(x, dxdt, h, params = np.zeros(1, dtype = np.float64)):
    k1 = dxdt(x, params)
    k2 = dxdt(x + k1 / 2 * h, params)
    k3 = dxdt(x + k2 / 2 * h, params)
    k4 = dxdt(x + h * k3, params)

    xnext = x + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    return xnext

@njit()
def rk4_model2(x, dxdt, h, s_data, s_indices, s_indptr, s_shape, params = np.zeros(1, dtype = np.float64)):
    k1 = dxdt(x, params, s_data, s_indices, s_indptr, s_shape)
    k2 = dxdt(x + k1 / 2 * h, params, s_data, s_indices, s_indptr, s_shape)
    k3 = dxdt(x + k2 / 2 * h, params, s_data, s_indices, s_indptr, s_shape)
    k4 = dxdt(x + h * k3, params, s_data, s_indices, s_indptr, s_shape)

    xnext = x + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    return xnext

@njit()
def rk4_model3(x, dxdt, h, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr, s_shape,
               params = np.zeros(1, dtype = np.float64)):
    k1 = dxdt(x, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr, s_shape)
    k2 = dxdt(x + k1 / 2 * h, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr,
              s_shape)
    k3 = dxdt(x + k2 / 2 * h, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr,
              s_shape)
    k4 = dxdt(x + h * k3, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr, s_shape)

    xnext = x + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    return xnext
@njit()
def rossler_dxdt(x, params):
    return np.array([(- x[1] - x[2]),
                     x[0] + params[0] * x[1],
                     params[1] + x[0] * x[2] - params[2] * x[2]])
@njit()
def lorenz63_dxdt(x, params):
    return np.array([params[0] * (- x[0] + x[1]),
                     params[1] * x[0] - x[1] - x[0] * x[2],
                     x[0] * x[1] - params[2] * x[2]])
@njit()
def doublescoll_dxdt(x, params):
    return np.array([x[0]/params[0] - (x[0]-x[1])/params[1]-2 * params[4] * np.sinh(params[3]*(x[0]-x[1])),
                     (x[0]-x[1])/params[1] + 2 * params[4] * np.sinh(params[3]*(x[0]-x[1])) - x[2],
                     x[1] - params[2] * x[2]])
@njit()
def lorenzmodel1_dxdt(x, params):
    z = np.roll(x, -1)
    p = np.roll(x, -2) * z
    q = z * np.roll(x, 1)
    return - x - p + q + params

@njit()
def lorenzmodel2_dxdt(x, params, s_data, s_indices, s_indptr, s_shape):
    """
    m2 - differential equation for the Lorenz model 2
    Inputs:
    Z - values of Z at each grid point

    ModelParams - struct containing model parameters
    Output: dZ - time derivative at each grid point
    """
    x = np.ascontiguousarray(x)
    xx = XYquadratic_sparse(x, x, int(params[1]), s_data, s_indices, s_indptr, s_shape)
    return xx - x + params[0]

@njit()
def lorenzmodel3_dxdt(z, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_data, s_indices, s_indptr, s_shape):
    """
    m2 - differential equation for the Lorenz model 2
    Inputs:
    Z - values of Z at each grid point

    ModelParams - struct containing model parameters
    Output: dZ - time derivative at each grid point
    """
    x = mult_vec_csr(z2x_data, z2x_indices, z2x_indptr, z2x_shape, z)
    y = z - x
    xx = XYquadratic_sparse(x, x, int(params[0]), s_data, s_indices, s_indptr, s_shape)
    y_1 = np.roll(y, -1)
    y_2 = np.roll(y, -2)
    yy = -y_2 * y_1 + y_1 * np.roll(y, 1)
    yx = -y_2 * np.roll(x, -1) + y_1 * np.roll(x, 1)
    return xx + params[2]*(params[2]*yy - y) + params[3]*yx - x + params[0]



@njit()
def model_forward(x, dxdt, int_steps, h, params):
    for i in range(int_steps):
        x = rk4(x, dxdt, h, params)
    return x

@njit()
def model_forward_with_noise(x, dxdt, int_steps, h, params,dyn_noise = 0,rng = None):
    for i in range(int_steps):
        x = x + dyn_noise * rng.normal(size=x.shape) 
        x = rk4(x, dxdt, h, params)
        if np.isnan(np.sum(x)):
            raise ValueError('nan in noise of x')
    return x
@njit()
def model_forward_Orstein_Uhlenbeck(x, T, discard_len, h, params,rng):
    model_output = np.zeros((T + discard_len + 1, x.size))
    model_output[0] = x
    for i in range(T + discard_len):
        model_output[i + 1] =model_output[i]+params[0]*(params[2] - model_output[i])*h + params[1]* np.sqrt(h) * rng.normal(size=x.shape)
    return model_output[discard_len:]
@njit()
def kursiv_model_forward(u, int_steps, params):
    for i in range(int_steps):
        u = kursiv_forecast(u, params)
    return u

@njit()
def lorenzmodel2_forward(x, dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params):
    for i in range(int_steps):
        x = rk4_model2(x, dxdt, h, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params)
    return x

@njit()
def lorenzmodel3_forward(x, dxdt, int_steps, h, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_mat_data,
                         s_mat_indices, s_mat_indptr, s_mat_shape, params):
    for i in range(int_steps):
        x = rk4_model3(x, dxdt, h, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_mat_data, s_mat_indices,
                       s_mat_indptr, s_mat_shape, params)
    return x


@njit()
def model_run(x, forward, dxdt, int_steps, h, params, T, discard_len = 0):
    model_output = np.zeros((T + discard_len + 1, x.size))
    model_output[0] = x
    for i in range(T + discard_len):
        model_output[i + 1] = forward(model_output[i], dxdt, int_steps, h, params)
    return model_output[discard_len:]

@njit()
def model_run_with_noise(x, forward, dxdt, int_steps, h, params, T, discard_len = 0,dyn_noise = 0,rng = None):
    model_output = np.zeros((T + discard_len + 1, x.size))
    model_output[0] = x
    for i in range(T + discard_len):
        model_output[i + 1] = forward(model_output[i], dxdt, int_steps, h, params,dyn_noise,rng)
    return model_output[discard_len:]

@jit(nopython = True, fastmath = True)
def precompute_KS_params(N, d, tau, M = 16, const = 0):
    k = np.concatenate((np.arange(int(N/2)), np.arange(-int(N/2), 0)))*2*np.pi/d
    L = (1+const)*k**2.0 - k**4.0
    E = np.exp(tau*L)
    E2 = np.exp(tau/2*L)
    r = np.exp(1j * np.pi * (np.arange(1, M+1)-0.5)/M)
    LR = tau*(np.zeros((1,M)) + L.reshape(-1,1)) + (np.zeros((N,1)) + r)
    Q  = tau*mean_numba_axis1(np.real((np.exp(LR/2)-1)/LR))
    f1 = tau*mean_numba_axis1(np.real((-4-LR+np.exp(LR)*(4-3*LR+LR**2.0))/(LR**3.0)))
    f2 = tau*mean_numba_axis1(np.real((2+LR+np.exp(LR)*(-2+LR))/(LR**3.0)))
    f3 = tau*mean_numba_axis1(np.real((-4-3*LR-LR**2.0+np.exp(LR)*(4-LR))/(LR**3.0)))
    g  = -0.5*1j*k
    params = np.zeros((7,N), dtype = np.complex128)
    params[0] = E
    params[1] = E2
    params[2] = Q
    params[3] = f1
    params[4] = f2
    params[5] = f3
    params[6] = g

    return params

@jit(nopython = True, fastmath = True)
def kursiv_forecast(u, params, noise = np.zeros(1, dtype = np.double)):

    with objmode(unext = 'double[:]'):
        v  = fft(u + noise)
        Nv = params[6]*fft(np.real(ifft(v))**2.0)
        a  = params[1]*v + params[2]*Nv
        Na = params[6]*fft(np.real(ifft(a))**2.0)
        b  = params[1]*v + params[2]*Na
        Nb = params[6]*fft(np.real(ifft(b))**2.0)
        c  = params[1]*a + params[2]*(2*Nb - Nv)
        Nc = params[6]*fft(np.real(ifft(c))**2.0)
        vnext  = params[0]*v + Nv*params[3] + 2*(Na+Nb)*params[4] + Nc*params[5]
        unext = np.real(ifft(vnext))
    return unext
@jit(nopython = True, fastmath = True)
def kursiv_run(u0, tau = 0.25, N = 64, d = 22, T = 100, params = np.array([[],[]], dtype = np.complex128),
                   int_steps = 1, noise = np.zeros((1,1), dtype = np.double),const=0):
    if params.size == 0:
        new_params = precompute_KS_params(N, d, tau,const=const)
    else:
        new_params = params
    steps = T*int_steps
    u_arr = np.zeros((steps+int_steps,N))
    u_arr[0,:] = u0
    if noise.size == 1 and noise[0,0] == 0.:
        for i in range(steps+int_steps-1):
            u_arr[i+1,:] = kursiv_forecast(u_arr[i,:], new_params)
    else:
        #noise_arr=noise[0,0]*rng_noise.normal(size=u_arr.shape)
        for i in range(steps+int_steps-1):
            u_arr[i+1,:] = kursiv_forecast(u_arr[i,:], new_params, noise[i,:])
    return np.ascontiguousarray(u_arr[::int_steps,:])

@njit(parallel = False)
def model_run_array(ic_array, forward, dxdt, int_steps, h, params):
    model_output = np.zeros(ic_array.shape)
    for i in range(ic_array.shape[0]):
        model_output[i] = forward(ic_array[i], dxdt, int_steps, h, params)

    return model_output
@jit(nopython = True, fastmath = True)
def kursiv_forecast_pred(u, params, noise = np.zeros((1,1), dtype = np.double)):
    with objmode(unext = 'double[:,:]'):
        v  = fft(u + noise,axis = 1)
        Nv = params[6]*fft(np.real(ifft(v, axis = 1))**2.0, axis = 1)
        a  = params[1]*v + params[2]*Nv
        Na = params[6]*fft(np.real(ifft(a, axis = 1))**2.0, axis = 1)
        b  = params[1]*v + params[2]*Na
        Nb = params[6]*fft(np.real(ifft(b, axis = 1))**2.0, axis = 1)
        c  = params[1]*a + params[2]*(2*Nb - Nv)
        Nc = params[6]*fft(np.real(ifft(c, axis = 1))**2.0, axis = 1)
        v  = params[0]*v + Nv*params[3] + 2*(Na+Nb)*params[4] + Nc*params[5]
        unext = np.real(ifft(v, axis = 1))
    return unext

@jit(nopython = True, fastmath = True)
def kursiv_run_array(u0_array, tau = 0.25, N = 64, d = 22, T = 100, params = np.array([[],[]], dtype = np.complex128),
                        noise = np.zeros((1,1), dtype = np.double),const=0):
    if params.size == 0:
        new_params = precompute_KS_params(N, d, tau,const=const)
    else:
        new_params = params
    if noise.size == 1 and noise[0, 0] == 0.:
        u_arr = kursiv_forecast_pred(u0_array, new_params)
    else:
        u_arr = kursiv_forecast_pred(u0_array, new_params, noise)
    return u_arr

@njit(parallel = False)
def lorenz_model_1_run_array(ic_array, dxdt, int_steps, h, params):
    model_output = np.zeros(ic_array.shape)
    for i in range(ic_array.shape[0]):
        model_output[i] = model_forward(ic_array[i], dxdt, int_steps, h, params)

    return model_output

@njit()
def lorenzmodel2_run(x, dxdt, int_steps, h, params, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape,
                     T, discard_len = 0):
    model_output = np.zeros((T + discard_len + 1, x.size))
    model_output[0] = x
    for i in range(T + discard_len):
        model_output[i + 1] = lorenzmodel2_forward(model_output[i], dxdt, int_steps, h, s_mat_data, s_mat_indices,
                                                   s_mat_indptr, s_mat_shape, params)
    return model_output[discard_len:]

@njit(parallel = False)
def lorenzmodel2_run_array(ic_array, dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape,
                           params):
    model_output = np.zeros(ic_array.shape)
    for i in range(ic_array.shape[0]):
        model_output[i] = lorenzmodel2_forward(ic_array[i], dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape,
                                  params)

    return model_output

@njit()
def lorenzmodel3_run(x, dxdt, int_steps, h, params, z2x_data, z2x_indices, z2x_indptr, z2x_shape, s_mat_data,
                         s_mat_indices, s_mat_indptr, s_mat_shape, T, discard_len = 0):
    model_output = np.zeros((T + discard_len + 1, x.size))
    model_output[0] = x
    for i in range(T + discard_len):
        model_output[i + 1] = lorenzmodel3_forward(model_output[i], dxdt, int_steps, h, z2x_data, z2x_indices,
                                                   z2x_indptr, z2x_shape, s_mat_data, s_mat_indices, s_mat_indptr,
                                                   s_mat_shape, params)
    return model_output[discard_len:]

@njit(parallel = False)
def lorenzmodel3_run_array(ic_array, dxdt, int_steps, h, z2x_data, z2x_indices, z2x_indptr, z2x_shape,
                           s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params):
    model_output = np.zeros(ic_array.shape)
    for i in range(ic_array.shape[0]):
        model_output[i] = lorenzmodel3_forward(ic_array[i], dxdt, int_steps, h, z2x_data, z2x_indices, z2x_indptr,
                                               z2x_shape, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params)

    return model_output


@njit()
def get_feature(data, r, feature_size, squarenodes = False, input_pass = False, output_bias = False):
    feature = np.zeros(feature_size)
    feature[:r.size] = r
    current_size = r.size
    if squarenodes:
        next_size = current_size + r.size
        feature[current_size:next_size] = r ** 2.0
        current_size += r.size
    if input_pass:
        next_size = current_size + data.size
        feature[current_size:next_size] = data
        current_size += data.size
    if output_bias:
        next_size = current_size + 1
        feature[current_size:next_size] = 1.0
    return feature

@njit()
def numba_lin_interp_periodic(x, xp, fp):
    return np.interp(x, np.append(xp, xp[-1] + xp[1]), np.append(fp, fp[0]))
@njit()
def numba_cubic_spline(x, xp, fp):
    with objmode(f = 'double[:]'):
        cs = CubicSpline(np.append(xp, xp[-1] + xp[1]), np.append(fp, fp[0]), bc_type = 'periodic')
        f = cs(x)
    return f

@njit()
def roll_data_1d(data, num_regions, stride, overlap_size, local_overlap):
    rolled_data = np.zeros((num_regions, overlap_size))
    data_extend = np.concatenate((data[-local_overlap:], data, data[:local_overlap]))
    for i, region_begin in enumerate(range(0, data.size, stride)):
        rolled_data[i] = data_extend[region_begin:region_begin+overlap_size]
    return np.ascontiguousarray(rolled_data)

@njit()
def roll_interp_data_1d(data, output_size, num_regions, interp_fun, model_grid_ratio, stride):
    interp_data = interp_fun(np.arange(output_size), np.arange(0, output_size, model_grid_ratio), data)
    return roll_data_1d(interp_data, num_regions, stride, stride, 0)

@njit()
def roll_data_2d(data, num_regions, stride, overlap_size, local_overlap):
    rolled_data = np.zeros((data.shape[0], num_regions, overlap_size))
    data_extend = np.concatenate((data[:, -local_overlap:], data, data[:, :local_overlap]), axis = 1)
    for i, region_begin in enumerate(range(0, data.shape[1], stride)):
        rolled_data[:, i] = data_extend[:,region_begin:region_begin+overlap_size]
    return np.ascontiguousarray(rolled_data)

@njit()
def roll_data_3d(data, num_regions, stride, overlap_size, local_overlap):
    rolled_data = np.zeros((data.shape[0], data.shape[1], num_regions, overlap_size))
    data_extend = np.concatenate((data[:, :, -local_overlap:], data, data[:, :, :local_overlap]), axis = 2)
    for i, region_begin in enumerate(range(0, data.shape[2], stride)):
        rolled_data[:, :, i] = data_extend[:, :, region_begin:region_begin+overlap_size]
    return np.ascontiguousarray(rolled_data)

@njit()
def roll_interp_data_2d(data, output_size, num_regions, interp_fun, model_grid_ratio, stride):
    interp_data = np.zeros((data.shape[0], output_size))
    x = np.arange(output_size)
    xp = np.arange(0, output_size, model_grid_ratio)
    for i in range(data.shape[0]):
        interp_data[i] = interp_fun(x, xp, data[i])
    return roll_data_2d(interp_data, num_regions, stride, stride, 0)

@njit()
def roll_interp_data_3d(data, output_size, num_regions, interp_fun, model_grid_ratio, stride):
    interp_data = np.zeros((data.shape[0], data.shape[1], output_size))
    x = np.arange(output_size)
    xp = np.arange(0, output_size, model_grid_ratio)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            interp_data[i, j] = interp_fun(x, xp, data[i, j])
    return roll_data_3d(interp_data, num_regions, stride, stride, 0)

@njit()
def interp_model_1d(model_output, output_size, model_grid_ratio):
    interp_model_output = np.interp(np.arange(output_size + 1),
                                    np.arange(0, output_size + model_grid_ratio, model_grid_ratio),
                                    np.append(model_output, model_output[0]))
    return interp_model_output[:-1]

@njit()
def interp_model_2d(model_output, output_size, model_grid_ratio):
    interp_model_output = np.zeros((model_output.shape[0], output_size))
    interp_x = np.arange(output_size + 1)
    data_x = np.arange(0, output_size + model_grid_ratio, model_grid_ratio)
    data_periodic = np.concatenate((model_output, np.ascontiguousarray(model_output[:, -1]).reshape(-1, 1)), axis=1)
    for i in range(model_output.shape[0]):
        interp_model_output[i] = np.interp(interp_x, data_x, data_periodic[i])[:-1]
    return interp_model_output

@njit()
def interp_model_3d(model_output, output_size, model_grid_ratio):
    interp_model_output = np.zeros((model_output.shape[0], model_output.shape[1], output_size))
    interp_x = np.arange(output_size + 1)
    data_x = np.arange(0, output_size + model_grid_ratio, model_grid_ratio)
    data_periodic = np.concatenate((model_output,
                                    np.ascontiguousarray(np.expand_dims(model_output[:,:, -1], axis = -1))), axis=2)
    for i in range(model_output.shape[0]):
        for j in range(model_output.shape[1]):
            interp_model_output[i,j] = np.interp(interp_x, data_x, data_periodic[i,j])[:-1]
    return interp_model_output
@njit()
def roll_model_output_1d(model_output, num_regions, output_size, model_grid_ratio, stride):
    #stretched_model_output = stretch_model_1d(model_output, output_size, model_grid_ratio)
    interp_model_output = interp_model_1d(model_output, output_size, model_grid_ratio)
    return roll_data_1d(interp_model_output, num_regions, stride, stride, 0)

@njit()
def roll_model_output_2d(model_output, num_regions, output_size, model_grid_ratio, stride):
    #stretched_model_output = stretch_model_2d(model_output, output_size, model_grid_ratio)
    interp_model_output = interp_model_2d(model_output, output_size, model_grid_ratio)
    return roll_data_2d(interp_model_output, num_regions, stride, stride, 0)

@njit()
def rolled_output_to_grid(model_output, num_regions, model_region_ratio, model_grid_ratio, stride):
    unrolled_model_output = model_output.reshape(-1, num_regions*model_region_ratio)
    unrolled_model_output = interp_model_2d(unrolled_model_output, stride*num_regions, model_grid_ratio)
    return unrolled_model_output.reshape(-1, num_regions, stride)

@njit()
def get_local_feature(rolled_data, r, feature_size, squarenodes = False, input_pass = False, output_bias = False):
    feature = np.zeros((r.shape[0], feature_size))
    feature[:,:r.shape[1]] = r
    current_size = r.shape[1]
    if squarenodes:
        next_size = current_size + r.shape[1]
        feature[:,current_size:next_size] = r ** 2.0
        current_size += r.shape[1]
    if input_pass:
        next_size = current_size + rolled_data.shape[1]
        feature[:,current_size:next_size] = rolled_data
        current_size += rolled_data.shape[1]
    if output_bias:
        next_size = current_size + 1
        feature[:,current_size:next_size] = 1.0
    return np.ascontiguousarray(feature)
@njit()
def get_local_feature_3D(rolled_data, r, feature_size, squarenodes = False, input_pass = False, output_bias = False):
    feature = np.zeros((r.shape[0], r.shape[1], feature_size))
    feature[:,:,:r.shape[2]] = r
    current_size = r.shape[2]
    if squarenodes:
        next_size = current_size + r.shape[2]
        feature[:,:,current_size:next_size] = r ** 2.0
        current_size += r.shape[2]
    if input_pass:
        next_size = current_size + rolled_data.shape[2]
        feature[:,:,current_size:next_size] = rolled_data
        current_size += rolled_data.shape[2]
    if output_bias:
        next_size = current_size + 1
        feature[:,:,current_size:next_size] = 1.0
    return np.ascontiguousarray(feature)
@njit()
def forward_reservoir(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape,
                      C, feature_size, squarenodes, input_pass, output_bias):
    r = (1 - leakage) * r + leakage * np.tanh(mult_vec(A_data, A_indices, A_indptr, A_shape, r) +
                                              mult_vec(B_data, B_indices, B_indptr, B_shape, data) + C)
    feature = get_feature(data, r, feature_size, squarenodes, input_pass, output_bias)
    return r, feature
@njit()
def synchronize_reservoir(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                          B_shape, C, feature_size, squarenodes, input_pass, output_bias):
    for i in range(sync_len):
        r, feature = forward_reservoir(data[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
    return r, feature

@njit()
def predict_kernel_reservoir(r, feature,train_features, K_inv,train_output, sigma,T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                          B_shape, C, feature_size, squarenodes, input_pass, output_bias):
    output = np.zeros((T + 1, B_shape[1]))
    feature = feature.reshape(1,-1)
    output[0] = (train_output.T @ K_inv @ RBFK(train_features, feature, sigma=sigma)).T
    for i in range(T):
        r, feature = forward_reservoir(output[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
        feature = feature.reshape(1, -1)
        output[i+1] = (train_output.T @ K_inv @ RBFK(train_features, feature, sigma=sigma)).T
    return output, r, feature
@njit()
def sync_and_predict_kernel_reservoir(data, r, train_features, K_inv,train_output, sigma, sync_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                               B_indptr, B_shape, C,  feature_size, squarenodes, input_pass, output_bias):
    r, feature = synchronize_reservoir(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                       B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                       output_bias)

    output, r, feature = predict_kernel_reservoir(r, feature,train_features, K_inv,train_output, sigma, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                           B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                           output_bias)
    return output, r, feature
@njit()
def synchronize_reservoir_local(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias, num_regions,
                                stride, overlap_size, local_overlap):
    for i in range(sync_len):
        r, feature = forward_reservoir_local(data[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                             B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                             output_bias, num_regions, stride, overlap_size, local_overlap)
    return r, feature

@njit()
def predict_reservoir(r, feature, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                          B_shape, C, W, feature_size, squarenodes, input_pass, output_bias):
    output = np.zeros((T + 1, B_shape[1]))
    output[0] = feature @ W
    for i in range(T):
        r, feature = forward_reservoir(output[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
        output[i+1] = feature @ W
    return output, r, feature
@njit()
def predict_reservoir_open_loop(r, feature, data, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                          B_shape, C, W, feature_size, squarenodes, input_pass, output_bias):
    output = np.zeros((T + 1, B_shape[1]))
    features = np.zeros((T + 1, len(feature)))
    output[0] = feature @ W
    features[0] = feature
    for i in range(T):
        r, feature = forward_reservoir(data[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
        output[i+1] = feature @ W
        features[i+1]=feature
    return output, r, feature,features


@njit()
def predict_kernel_reservoir(r, feature,train_features, K_inv,train_output, sigma,T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                          B_shape, C, feature_size, squarenodes, input_pass, output_bias):
    output = np.zeros((T + 1, B_shape[1]))
    feature = feature.reshape(1,-1)
    output[0] = (train_output.T @ K_inv @ RBFK(train_features, feature, sigma=sigma)).T
    print('here')
    for i in range(T):
        r, feature = forward_reservoir(output[i], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
        feature = feature.reshape(1, -1)
        output[i+1] = (train_output.T @ K_inv @ RBFK(train_features, feature, sigma=sigma)).T
    return output, r, feature
@njit()
def get_reservoir_output_local_flattened(W, feature):
    return (feature @ W).flatten()

@njit()
def predict_reservoir_local(r, feature, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr,
                            B_shape, C, W, feature_size, squarenodes, input_pass, output_bias, num_regions, stride,
                            overlap_size, local_overlap):
    output = np.zeros((T + 1, num_regions*stride))
    output[0] = get_reservoir_output_local_flattened(W, feature)
    for i in range(T):
        rolled_output = roll_data_1d(output[i], num_regions, stride, overlap_size, local_overlap)
        r, feature = forward_reservoir_local(rolled_output, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                             B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                             output_bias, num_regions, stride, overlap_size, local_overlap)
        output[i+1] = get_reservoir_output_local_flattened(W, feature)
    return output, r, feature

@njit()
def sync_and_predict_reservoir(data, r, sync_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                               B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass, output_bias):
    r, feature = synchronize_reservoir(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                       B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                       output_bias)

    output, r, feature = predict_reservoir(r, feature, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                           B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                           output_bias)
    return output, r, feature
@njit()
def sync_and_predict_reservoir_open_loop(data, r, sync_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                               B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass, output_bias):
    r, feature = synchronize_reservoir(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                       B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                       output_bias)

    output, r, feature, features = predict_reservoir_open_loop(r, feature, data[sync_len:],T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                           B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                           output_bias)
    return output, r, feature, features

@njit()
def sync_and_predict_kernel_reservoir(data, r, train_features, K_inv,train_output, sigma, sync_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                               B_indptr, B_shape, C,  feature_size, squarenodes, input_pass, output_bias):
    r, feature = synchronize_reservoir(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                       B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                       output_bias)

    output, r, feature = predict_kernel_reservoir(r, feature,train_features, K_inv,train_output, sigma, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                           B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                           output_bias)
    return output, r, feature
@njit()
def sync_and_predict_reservoir_local(data, r, feature, sync_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                     B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                     output_bias, num_regions, stride, overlap_size, local_overlap):
    if sync_len != 0:
        r, feature = synchronize_reservoir_local(data, r, sync_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                 B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                                 output_bias, num_regions, stride, overlap_size, local_overlap)

    output, r, feature = predict_reservoir_local(r, feature, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                 B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes,
                                                 input_pass, output_bias, num_regions, stride, overlap_size,
                                                 local_overlap)
    return output, r, feature

@njit()
def get_reservoir_features(data, r, discard_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                           B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias):
    output = np.zeros((T, feature_size))
    if discard_len != 0:
        r, feature = synchronize_reservoir(data, r, discard_len, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                           B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                           output_bias)
    for i, idx in enumerate(range(discard_len, T+discard_len)):
        r, feature = forward_reservoir(data[idx], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                       B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
        output[i] = feature
    return output, r, feature

@njit()
def get_reservoir_features_local(data, r, feature, discard_len, T, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                 B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias,
                                 num_regions, stride, overlap_size, local_overlap):
    output = np.zeros((T, num_regions, feature_size))
    if discard_len != 0:
        r, feature = synchronize_reservoir_local(data, r, discard_len, leakage, A_data, A_indices, A_indptr, A_shape,
                                             B_data, B_indices, B_indptr, B_shape, C, feature_size, squarenodes,
                                             input_pass, output_bias, num_regions, stride, overlap_size, local_overlap)
    for i, idx in enumerate(range(discard_len, T+discard_len)):
        r, feature = forward_reservoir_local(data[idx], r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                             B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                             output_bias, num_regions, stride, overlap_size, local_overlap)
        output[i] = feature
    return output, r, feature

@njit()
def train_reservoir(data, r, discard_len, T, regularization, train_noise, leakage, A_data, A_indices, A_indptr,
                    A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                    output_bias, seed = 10):
    np.random.seed(seed)
    noisy_data = data + np.random.randn(data.shape[0], data.shape[1])*train_noise
    features, r, feature = get_reservoir_features(noisy_data, r, discard_len, T, leakage, A_data, A_indices, A_indptr,
                                                  A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                                  squarenodes, input_pass, output_bias)
    output = data[discard_len + 1:discard_len + T + 1]
    info_mat = (features.T @ features) / T + np.diag(np.ones(feature_size) * regularization)
    target_mat = (features.T @ output) / T
    W = np.linalg.solve(info_mat, target_mat)
    train_preds = features @ W
    train_error = np.sqrt(mean_numba_axis1((train_preds - output) ** 2.0))
    return W, r, feature, train_preds, train_error

@njit()
def train_reservoir_local(data, r, feature, discard_len, T, regularization, train_noise, leakage, A_data, A_indices, A_indptr,
                          A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                          output_bias, num_regions, stride, overlap_size, local_overlap, seed = 10):
    np.random.seed(seed)
    noisy_data = data + np.random.randn(data.shape[0], data.shape[1])*train_noise
    rolled_data = roll_data_2d(noisy_data, num_regions, stride, overlap_size, local_overlap)
    features, r, feature = get_reservoir_features_local(rolled_data, r, feature, discard_len, T, leakage, A_data, A_indices,
                                                        A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C,
                                                        feature_size, squarenodes, input_pass, output_bias, num_regions,
                                                        stride, overlap_size, local_overlap)
    output = roll_data_2d(data[discard_len + 1:discard_len + T + 1],
                          num_regions, stride, stride, 0).reshape(features.shape[0]*num_regions, -1)
    train_features = features.reshape(features.shape[0]*num_regions, -1)
    info_mat = (train_features.T @ train_features) / (T*num_regions) + np.diag(np.ones(feature_size) * regularization)
    target_mat = (train_features.T @ output) / (T*num_regions)
    W = np.linalg.solve(info_mat, target_mat)
    with objmode(train_preds = 'float64[:,:]'):
        train_preds = (features @ W).reshape(T, -1)
    train_error = np.sqrt(mean_numba_axis1((train_preds - data[discard_len + 1:discard_len + T + 1]) ** 2.0))
    return W, r, feature, train_preds, train_error

@njit()
def get_feature_hybrid(data, dxdt, int_steps, h, params, r,
                            leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                            B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                            output_bias,model_difference):
    model_out = model_forward(data, dxdt, int_steps, h, params)
    r, reservoir_feature = forward_reservoir(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                             B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
    if model_difference:
        if output_bias:
            bias_feature = np.append(model_out-reservoir_feature[-(len(model_out)+1):-1], reservoir_feature)
        else:
            bias_feature = np.append(model_out-reservoir_feature[-len(model_out):], reservoir_feature)
    else:
        bias_feature = np.append(model_out, reservoir_feature)
    return bias_feature, r

@njit()
def get_feature_hybrid_kursiv(data, int_steps, params, r,
                            leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                            B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                            output_bias,model_difference):
    model_out = kursiv_model_forward(data, int_steps, params)
    r, reservoir_feature = forward_reservoir(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                             B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
    if model_difference:
        if output_bias:
            bias_feature = np.append(model_out-reservoir_feature[-(len(model_out)+1):-1], reservoir_feature)
        else:
            bias_feature = np.append(model_out-reservoir_feature[-len(model_out):], reservoir_feature)
    else:
        bias_feature = np.append(model_out, reservoir_feature)
    return bias_feature, r

@njit()
def get_feature_hybrid_local(data, dxdt, int_steps, h, params, r,
                                  leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                  B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                  output_bias, num_regions, stride, overlap_size, local_overlap):
    model_input = data[:, local_overlap:local_overlap + stride].flatten()
    model_out = model_forward(model_input, dxdt, int_steps, h, params)
    rolled_model_out = roll_data_1d(model_out, num_regions, stride, stride, 0)
    r, reservoir_feature = forward_reservoir_local(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                   B_indices,B_indptr, B_shape, C, feature_size, squarenodes,
                                                   input_pass, output_bias, num_regions, stride, overlap_size,
                                                   local_overlap)
    bias_feature = np.ascontiguousarray(np.concatenate((rolled_model_out, reservoir_feature), axis = 1))
    return bias_feature, r

@njit()
def get_feature_bias_hybrid_local(data, dxdt, int_steps, h, params, r,
                                  leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                  B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                  output_bias, num_regions, stride, overlap_size, local_overlap, model_in_feature,
                                  model_interp_fun, model_grid_ratio):
    model_input = data[:, local_overlap:local_overlap + stride].flatten()[::model_grid_ratio]
    model_out = model_forward(model_input, dxdt, int_steps, h, params)
    r, reservoir_feature = forward_reservoir_local(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                   B_indices,B_indptr, B_shape, C, feature_size, squarenodes,
                                                   input_pass, output_bias, num_regions, stride, overlap_size,
                                                   local_overlap)
    if model_in_feature:
        rolled_model_out = roll_interp_data_1d(model_out, num_regions*stride, num_regions, model_interp_fun,
                                               model_grid_ratio, stride)
        bias_feature = np.ascontiguousarray(np.concatenate((rolled_model_out, reservoir_feature), axis = 1))
    else:
        bias_feature = reservoir_feature
    return bias_feature, r, model_out

@njit()
def get_feature_hybrid_standardized(data, dxdt, int_steps, h, params, mean, std, r,
                                         leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                         B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                         output_bias,model_difference):
    model_out = standardize(model_forward(unstandardize(data, mean, std), dxdt, int_steps, h, params),
                            mean, std)
    r, reservoir_feature = forward_reservoir(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                             B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
    if model_difference:
        if output_bias:
            bias_feature = np.append(model_out-reservoir_feature[-(len(model_out)+1):-1], reservoir_feature)
        else:
            bias_feature = np.append(model_out-reservoir_feature[-len(model_out):], reservoir_feature)
    else:
        bias_feature = np.append(model_out, reservoir_feature)
    return bias_feature, r

@njit()
def get_feature_hybrid_standardized_kursiv(data, int_steps, params, mean, std, r,
                                         leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                         B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                         output_bias,model_difference):
    model_out = standardize(kursiv_model_forward(unstandardize(data, mean, std), int_steps,params),
                            mean, std)
    r, reservoir_feature = forward_reservoir(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                             B_indptr, B_shape, C, feature_size, squarenodes, input_pass, output_bias)
    if model_difference:
        if output_bias:
            bias_feature = np.append(model_out-reservoir_feature[-(len(model_out)+1):-1], reservoir_feature)
        else:
            bias_feature = np.append(model_out-reservoir_feature[-len(model_out):], reservoir_feature)
    else:
        bias_feature = np.append(model_out, reservoir_feature)
    return bias_feature, r

@njit()
def get_feature_hybrid_local_standardized(data, dxdt, int_steps, h, params, mean, std, r,
                                               leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                               B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                               output_bias, num_regions, stride, overlap_size, local_overlap):
    model_input = data[:, local_overlap:local_overlap + stride].flatten()
    model_out = standardize(model_forward(unstandardize(model_input, mean, std), dxdt, int_steps, h, params),
                            mean, std)
    rolled_model_out = roll_data_1d(model_out, num_regions, stride, stride, 0)
    r, reservoir_feature = forward_reservoir_local(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                   B_indices,B_indptr, B_shape, C, feature_size, squarenodes,
                                                   input_pass, output_bias, num_regions, stride, overlap_size,
                                                   local_overlap)
    bias_feature = np.ascontiguousarray(np.concatenate((rolled_model_out, reservoir_feature), axis = 1))
    return bias_feature, r

@njit()
def get_feature_bias_hybrid_local_standardized(data, dxdt, int_steps, h, params, mean, std, r,
                                               leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                               B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                               output_bias, num_regions, stride, overlap_size, local_overlap,
                                               model_in_feature, model_interp_fun, model_grid_ratio):
    model_input = data[:, local_overlap:local_overlap + stride].flatten()[::model_grid_ratio]
    model_out = standardize(model_forward(unstandardize(model_input, mean, std), dxdt, int_steps, h, params),
                            mean, std)
    r, reservoir_feature = forward_reservoir_local(data, r, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                   B_indices,B_indptr, B_shape, C, feature_size, squarenodes,
                                                   input_pass, output_bias, num_regions, stride, overlap_size,
                                                   local_overlap)
    if model_in_feature:
        rolled_model_out = roll_interp_data_1d(model_out, num_regions*stride, num_regions, model_interp_fun,
                                               model_grid_ratio, stride)
        bias_feature = np.ascontiguousarray(np.concatenate((rolled_model_out, reservoir_feature), axis = 1))
    else:
        bias_feature = reservoir_feature
    return bias_feature, r, model_out

@njit()
def get_hybrid_bias_output(bias_feature, model_state, W, output_size, num_regions, model_interp_fun, model_grid_ratio,
                           stride):
    rolled_model = roll_interp_data_1d(model_state, output_size, num_regions, model_interp_fun, model_grid_ratio,
                                       stride)
    return (bias_feature @ W + rolled_model).flatten()

##@njit()
def get_hybrid_bias_output_2D(bias_feature, model_state, W):
    return (bias_feature @ np.expand_dims(W, axis = 0) + model_state).reshape(bias_feature.shape[0], -1)

@njit()
def get_hybrid_bias_obs_output_3D(bias_features, model_state, W, output_size, num_regions, model_interp_fun,
                                  model_grid_ratio, model_in_feature, stride):
    rolled_model = roll_interp_data_3d(model_state, output_size, num_regions, model_interp_fun, model_grid_ratio,
                                       stride)
    if not model_in_feature:
        bias_features = np.concatenate((rolled_model, bias_features), axis = 2)
    output = np.zeros((bias_features.shape[0], bias_features.shape[1], output_size))
    if W.shape[0] == 1:
        for i in range(bias_features.shape[0]):
            for j in range(bias_features.shape[1]):
                output[i, j] = (bias_features[i, j] @ W[0]).flatten()
    else:
        for i in range(bias_features.shape[0]):
            for j in range(bias_features.shape[1]):
                output[i, j] = (bias_features[i, j] @ W[j]).flatten()
    return output


@njit()
def predict_hybrid(bias_feature, output_size, W, T, dxdt, int_steps, h, params, standardized, mean, std,
                   r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C,
                   feature_size, squarenodes, input_pass, output_bias,model_difference):
    output = np.zeros((T + 1, output_size))
    output[0] = bias_feature @ W
    if standardized:
        for i in range(T):
            bias_feature, r = get_feature_hybrid_standardized(output[i], dxdt, int_steps, h, params,
                                                                   mean, std, r, leakage, A_data, A_indices, A_indptr,
                                                                   A_shape, B_data, B_indices, B_indptr, B_shape, C,
                                                                   feature_size, squarenodes, input_pass,
                                                                   output_bias,model_difference)
            output[i+1] = bias_feature @ W
    else:
        for i in range(T):
            bias_feature, r = get_feature_hybrid(output[i], dxdt, int_steps, h, params, r,
                                                      leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                                      B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                                      output_bias,model_difference)
            output[i+1] = bias_feature @ W
    return output, bias_feature, r

@njit()
def predict_hybrid_kursiv(bias_feature, output_size, W, T, int_steps, params, standardized, mean, std,
                   r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C,
                   feature_size, squarenodes, input_pass, output_bias,model_difference):
    output = np.zeros((T + 1, output_size))
    output[0] = bias_feature @ W
    if standardized:
        for i in range(T):
            bias_feature, r = get_feature_hybrid_standardized_kursiv(output[i], int_steps, params,
                                                                   mean, std, r, leakage, A_data, A_indices, A_indptr,
                                                                   A_shape, B_data, B_indices, B_indptr, B_shape, C,
                                                                   feature_size, squarenodes, input_pass,
                                                                   output_bias,model_difference)
            output[i+1] = bias_feature @ W
    else:
        for i in range(T):
            bias_feature, r = get_feature_hybrid_kursiv(output[i], int_steps, params, r,
                                                      leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                                      B_indptr, B_shape, C, feature_size, squarenodes, input_pass,
                                                      output_bias,model_difference)
            output[i+1] = bias_feature @ W
    return output, bias_feature, r

@njit()
def predict_hybrid_local(bias_feature, output_size, W, T, dxdt, int_steps, h, params, standardized, mean, std,
                         r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C,
                         feature_size, squarenodes, input_pass, output_bias, num_regions, stride, overlap_size,
                         local_overlap):
    output = np.zeros((T + 1, output_size))
    output[0] = (bias_feature @ W).flatten()
    if standardized:
        for i in range(T):
            rolled_data = roll_data_1d(output[i], num_regions, stride, overlap_size, local_overlap)
            bias_feature, r = get_feature_hybrid_local_standardized(rolled_data, dxdt, int_steps, h, params,
                                                                         mean, std, r, leakage, A_data,
                                                                         A_indices, A_indptr, A_shape, B_data,
                                                                         B_indices, B_indptr, B_shape, C, feature_size,
                                                                         squarenodes, input_pass, output_bias,
                                                                         num_regions, stride, overlap_size,
                                                                         local_overlap)
            output[i+1] = (bias_feature @ W).flatten()
    else:
        for i in range(T):
            rolled_data = roll_data_1d(output[i], num_regions, stride, overlap_size, local_overlap)
            bias_feature, r = get_feature_hybrid_local(rolled_data, dxdt, int_steps, h, params, r,
                                                            leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                            B_indices, B_indptr, B_shape, C, feature_size, squarenodes,
                                                            input_pass, output_bias, num_regions, stride, overlap_size,
                                                            local_overlap)
            output[i+1] = (bias_feature @ W).flatten()
    return output, bias_feature, r

@njit()
def predict_bias_hybrid_local(bias_feature, model_state, output_size, W, T, dxdt, int_steps, h, params, standardized, mean, std,
                                r, leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape,
                                C, feature_size, squarenodes, input_pass, output_bias, num_regions, stride,
                                overlap_size, local_overlap, model_in_feature, model_interp_fun, model_grid_ratio):
    output = np.zeros((T + 1, output_size))
    if standardized:
        output[0] = get_hybrid_bias_output(bias_feature, model_state, W, output_size, num_regions, model_interp_fun,
                                           model_grid_ratio, stride)
        for i in range(T):
            rolled_data = roll_data_1d(output[i], num_regions, stride, overlap_size, local_overlap)
            bias_feature, r, model_state = \
                get_feature_bias_hybrid_local_standardized(rolled_data, dxdt, int_steps, h, params, mean, std, r,
                                                           leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                           B_indices, B_indptr, B_shape, C, feature_size, squarenodes,
                                                           input_pass, output_bias, num_regions, stride, overlap_size,
                                                           local_overlap, model_in_feature, model_interp_fun,
                                                           model_grid_ratio)
            output[i+1] = get_hybrid_bias_output(bias_feature, model_state, W, output_size, num_regions,
                                                 model_interp_fun, model_grid_ratio, stride)
    else:
        output[0] = get_hybrid_bias_output(bias_feature, model_state, W, output_size, num_regions,
                                           model_interp_fun, model_grid_ratio, stride)
        for i in range(T):
            rolled_data = roll_data_1d(output[i], num_regions, stride, overlap_size, local_overlap)
            bias_feature, r, model_state = get_feature_bias_hybrid_local(rolled_data, dxdt, int_steps, h, params, r,
                                                            leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                                            B_indices, B_indptr, B_shape, C, feature_size, squarenodes,
                                                            input_pass, output_bias, num_regions, stride, overlap_size,
                                                            local_overlap, model_in_feature, model_interp_fun,
                                                                         model_grid_ratio)
            output[i+1] = get_hybrid_bias_output(bias_feature, model_state, W, output_size, num_regions,
                                                 model_interp_fun, model_grid_ratio, stride)
    return output, bias_feature, r, model_state

@njit()
def run_etkf_cycle_lorenz(observations, obs_indices, R_inverse, H, analysis, inflation_vec, dxdt, int_steps, h, params,
                          hybrid_int_steps, return_ensemble = True, fourD = False):
    #print(observations.shape)
    #print(analysis.shape)
    if return_ensemble:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, analysis.shape[1], analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0]*hybrid_int_steps + 1, analysis.shape[1], analysis.shape[2]))
        analyses[0] = analysis[-1]
    else:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, 1, analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0] * hybrid_int_steps + 1, 1, analysis.shape[2]))
        analyses[0, 0] = mean_numba_axis0(analysis[-1])
    for i in range(num_cycles):
        if return_ensemble:
            mean_analysis = mean_numba_axis0(analysis[-1])
        else:
            mean_analysis = analyses[i*hybrid_int_steps, 0]
        background_init = (analysis[-1] - mean_analysis) * inflation_vec + mean_analysis
        background = background_lorenz(background_init, dxdt, int_steps, h, params, hybrid_int_steps)
        if hybrid_int_steps > 1 and not fourD:
            if return_ensemble:
                analyses[i*hybrid_int_steps+1:(i+1)*hybrid_int_steps] = background[:-1]
            else:
                for j in range(hybrid_int_steps-1):
                    analyses[i*hybrid_int_steps+1+j, 0] = mean_numba_axis0(background[j])
        if fourD:
            analysis, illConditioned = etkf_cycle_from_background(observations[i*hybrid_int_steps:(i+1)*hybrid_int_steps],
                                                                  obs_indices[i*hybrid_int_steps:(i+1)*hybrid_int_steps],
                                                                  R_inverse, background, analysis.shape[2], H,
                                                                  inflation_vec=inflation_vec)
        else:
            analysis, illConditioned = etkf_cycle_from_background(np.ascontiguousarray(observations[i])\
                                                                                       .reshape(1, observations.shape[1]),
                                                                  np.array([obs_indices[i]]),
                                                                  R_inverse,
                                                                  np.ascontiguousarray(background[-1])\
                                                                                       .reshape(1, background.shape[1],
                                                                                         background.shape[2]),
                                                                  analysis.shape[2], H,
                                                                  inflation_vec = inflation_vec)
        if illConditioned:
            return analyses[:i * hybrid_int_steps + 1]
        if return_ensemble:
            if fourD:
                analyses[i*hybrid_int_steps+1:(i + 1) * hybrid_int_steps+1] = analysis
            else:
                analyses[(i + 1) * hybrid_int_steps] = analysis[-1]
        else:
            if fourD:
                for j in range(hybrid_int_steps):
                    analyses[i*hybrid_int_steps+1+j, 0] = mean_numba_axis0(analysis[j])
            else:
                analyses[(i+1)*hybrid_int_steps, 0] = mean_numba_axis0(analysis[-1])
    return analyses

@njit()
def run_etkf_cycle_lorenz_2(observations, obs_indices, R_inverse, H, analysis, inflation_vec, dxdt, int_steps, h,
                            s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params, hybrid_int_steps,
                            return_ensemble = True, fourD = False):
    if return_ensemble:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, analysis.shape[1], analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0] * hybrid_int_steps + 1, analysis.shape[1], analysis.shape[2]))
        analyses[0] = analysis[-1]
    else:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, 1, analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0] * hybrid_int_steps + 1, 1, analysis.shape[2]))
        analyses[0, 0] = mean_numba_axis0(analysis[-1])
    for i in range(num_cycles):
        if return_ensemble:
            mean_analysis = mean_numba_axis0(analysis[-1])
        else:
            mean_analysis = analyses[i*hybrid_int_steps, 0]
        background_init = (analysis[-1] - mean_analysis) * inflation_vec + mean_analysis
        #background_init = analyses[i]
        background = background_lorenz_2(background_init, dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr,
                                         s_mat_shape, params, hybrid_int_steps)
        if hybrid_int_steps > 1 and not fourD:
            if return_ensemble:
                analyses[i*hybrid_int_steps+1:(i+1)*hybrid_int_steps] = background[:-1]
            else:
                for j in range(hybrid_int_steps-1):
                    analyses[i*hybrid_int_steps+1+j, 0] = mean_numba_axis0(background[j])
        if fourD:
            analysis, illConditioned = etkf_cycle_from_background(
                observations[i * hybrid_int_steps:(i + 1) * hybrid_int_steps],
                obs_indices[i * hybrid_int_steps:(i + 1) * hybrid_int_steps],
                R_inverse, background, analysis.shape[2], H,
                inflation_vec=inflation_vec)
        else:
            analysis, illConditioned = etkf_cycle_from_background(observations[i].reshape(1, observations.shape[1]),
                                                                  np.array(obs_indices[i]),
                                                                  R_inverse, background[-1].reshape(1, background.shape[1],
                                                                                         background.shape[2]),
                                                                  analysis.shape[2], H,
                                                                  inflation_vec=inflation_vec)
        if illConditioned:
            return analyses[:i * hybrid_int_steps + 1]
        if return_ensemble:
            if fourD:
                analyses[i * hybrid_int_steps + 1:(i + 1) * hybrid_int_steps + 1] = analysis
            else:
                analyses[(i + 1) * hybrid_int_steps] = analysis[-1]
        else:
            if fourD:
                for j in range(hybrid_int_steps):
                    analyses[i * hybrid_int_steps + 1 + j, 0] = mean_numba_axis0(analysis[j])
            else:
                analyses[(i + 1) * hybrid_int_steps, 0] = mean_numba_axis0(analysis[-1])
        if illConditioned:
            break
    return analyses

def run_etkf_cycle_bias_hybrid_local(observations, obs_indices, R_inverse, H, analysis, inflation_vec, dxdt, int_steps,
                                     h, params,
                                     reservoir_prev_background, leakage, A_data, A_indices, A_indptr, A_shape, B_data,
                                     B_indices, B_indptr, B_shape, C, W_M, W_H, feature_size, squarenodes, input_pass,
                                     output_bias, num_regions, stride, overlap_size, local_overlap, model_in_feature,
                                     output_size, model_interp_fun, model_grid_ratio, hybrid_int_steps,
                                     assimilate_model = False, assimilate_obs = False, return_ensemble = True,
                                     fourD = False, r_in_ensemble = False, W_M_kf = False,
                                     P_W_M = np.ones((1,1), dtype = np.float64),
                                     Q = np.zeros((1,1), dtype = np.float64), method = '1b', 
                                     H_mat = np.zeros((1,1), dtype = np.float64)):
    H_mat_T = np.ascontiguousarray(H_mat.T)
    if return_ensemble:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, analysis.shape[1], analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0] * hybrid_int_steps + 1, analysis.shape[1], analysis.shape[2]))
        analyses[0] = analysis[-1]
    else:
        if fourD:
            num_cycles = observations.shape[0] // hybrid_int_steps
            analyses = np.zeros((observations.shape[0] + 1, 1, analysis.shape[2]))
        else:
            num_cycles = observations.shape[0]
            analyses = np.zeros((observations.shape[0] * hybrid_int_steps + 1, 1, analysis.shape[2]))
        analyses[0, 0] = mean_numba_axis0(analysis[-1])
    D = num_regions * stride
    if model_in_feature:
        N = (feature_size + stride) * stride
    else:
        N = feature_size * stride
    with ProgressBar(total = num_cycles) as progress:
        analyses, prev_bias_features, reservoir_prev_background, prev_model_states, W_M, P_W_M = \
            etkf_cycle_bias_hybrid_local(num_cycles, analyses, D, N, observations, obs_indices,
                                         R_inverse, H, analysis, inflation_vec, dxdt,
                                         int_steps,
                                         h, params, reservoir_prev_background, leakage, A_data, A_indices, A_indptr,
                                         A_shape,
                                         B_data, B_indices, B_indptr, B_shape, C, W_M, W_H, feature_size, squarenodes,
                                         input_pass, output_bias, num_regions, stride, overlap_size, local_overlap,
                                         model_in_feature, output_size, model_interp_fun, model_grid_ratio,
                                         hybrid_int_steps, assimilate_model, assimilate_obs, return_ensemble, fourD,
                                         progress, r_in_ensemble, W_M_kf, P_W_M, Q, method, H_mat, H_mat_T)
    return analyses, prev_bias_features, reservoir_prev_background, prev_model_states, W_M, P_W_M

@njit()
def etkf_cycle_bias_hybrid_local(num_cycles, analyses, D, N, observations, obs_indices, R_inverse, H, analysis,
                                 inflation_vec, dxdt, int_steps,
                                 h, params, reservoir_prev_background, leakage, A_data, A_indices, A_indptr, A_shape,
                                 B_data, B_indices, B_indptr, B_shape, C, W_M, W_H, feature_size, squarenodes,
                                 input_pass, output_bias, num_regions, stride, overlap_size, local_overlap,
                                 model_in_feature, output_size, model_interp_fun, model_grid_ratio, hybrid_int_steps,
                                 assimilate_model, assimilate_obs, return_ensemble, fourD, progress, r_in_ensemble,
                                 W_M_kf, P_W_M, Q, method, H_mat, H_mat_T):
    reservoir_vec_size = A_shape[0] * num_regions
    for i in range(num_cycles):
        if return_ensemble:
            mean_analysis = mean_numba_axis0(analysis[-1])
        else:
            mean_analysis = analyses[i*hybrid_int_steps, 0]
        #print('Mean Analysis at iter %d' % i)
        #print(mean_analysis[:5])
        background     = np.zeros((hybrid_int_steps, analysis.shape[1], analysis.shape[2]))
        background[-1] = (analysis[-1] - mean_analysis) * inflation_vec + mean_analysis
        for j in range(hybrid_int_steps-1):
            background[j] = background[-1]
        if r_in_ensemble:
            if assimilate_model and assimilate_obs:
                if model_in_feature:
                    W_M = np.ascontiguousarray(background[-1, :, D + reservoir_vec_size:D + N + reservoir_vec_size]).\
                        reshape(analysis.shape[1], feature_size + stride, stride)
                else:
                    W_M = np.ascontiguousarray(background[-1, :, D + reservoir_vec_size:D + N + reservoir_vec_size]).\
                        reshape(analysis.shape[1], feature_size, stride)
                W_H = np.ascontiguousarray(background[-1, :, D + N + reservoir_vec_size:]).\
                    reshape(analysis.shape[1], feature_size + stride, stride)
            elif assimilate_model:
                if model_in_feature:
                    W_M = np.ascontiguousarray(background[-1, :, D + reservoir_vec_size:]).\
                        reshape(analysis.shape[1], feature_size + stride, stride)
                else:
                    W_M = np.ascontiguousarray(background[-1, :, D + reservoir_vec_size:]).\
                        reshape(analysis.shape[1], feature_size, stride)
            elif assimilate_obs:
                W_H = np.ascontiguousarray(background[-1, :, D + reservoir_vec_size:]).\
                    reshape(analysis.shape[1], feature_size + stride, stride)
        else:
            if assimilate_model and assimilate_obs:
                if model_in_feature:
                    W_M = np.ascontiguousarray(background[-1, :, D:D + N]).reshape(analysis.shape[1], feature_size + stride,
                                                                               stride)
                else:
                    W_M = np.ascontiguousarray(background[-1, :, D:D + N]).reshape(analysis.shape[1], feature_size, stride)
                W_H = np.ascontiguousarray(background[-1, :, D + N:]).reshape(analysis.shape[1], feature_size + stride, stride)
            elif assimilate_model:
                if model_in_feature:
                    W_M = np.ascontiguousarray(background[-1, :, D:]).reshape(analysis.shape[1], feature_size + stride, stride)
                else:
                    W_M = np.ascontiguousarray(background[-1, :, D:]).reshape(analysis.shape[1], feature_size, stride)
            elif assimilate_obs:
                W_H = np.ascontiguousarray(background[-1, :, D:]).reshape(analysis.shape[1], feature_size + stride, stride)
        background_in = background[-1, :, :D]
        #print('Background in at iter %d' % i)
        #print(background_in[1,:5])
        background[:, :, :D], bias_features, reservoir_background, model_states = \
            background_bias_hybrid_local(background_in, dxdt, int_steps, h, params,
                                         reservoir_prev_background, leakage, A_data, A_indices,
                                         A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C, W_M,
                                         feature_size, squarenodes, input_pass, output_bias,
                                         num_regions, stride, overlap_size, local_overlap, model_in_feature,
                                         assimilate_model, output_size, model_interp_fun, model_grid_ratio,
                                         hybrid_int_steps)
        #for j in range(hybrid_int_steps):
        #    print('Background at step %d' % j)
        #    print(background[j,1,:5])
        if r_in_ensemble:
            background[:, :, D:D + reservoir_vec_size] = reservoir_background.reshape(background.shape[0],
                                                                                      background.shape[1],
                                                                                      reservoir_vec_size)

        if hybrid_int_steps > 1 and not fourD:
            if return_ensemble:
                analyses[i*hybrid_int_steps+1:(i+1)*hybrid_int_steps] = background[:-1]
            else:
                for j in range(hybrid_int_steps-1):
                    analyses[i*hybrid_int_steps+1+j, 0] = mean_numba_axis0(background[j])
                    
        if W_M_kf and method in ['1a', '1b', '1c', '2'] and not fourD:
            S_data_all, S_indices, S_indptr, S_shape = get_S_kf(
                np.expand_dims(mean_numba_axis0_3D(bias_features[-1]), 0), D, stride)
            S_data = S_data_all[0]
        if W_M_kf and method in ['1c', '2'] and not fourD:
            HS_data, HS_indices, HS_indptr, HS_shape = get_HS(H_mat, S_data, S_indices, S_indptr, S_shape)
            HQ, R_adj = csr_mat_csr_T_mult(HS_data, HS_indices, HS_indptr, HS_shape, P_W_M)
            R_inverse_in = np.ascontiguousarray(np.linalg.pinv(np.diag(1./np.diag(R_inverse)) + R_adj))
        else:
            R_inverse_in = R_inverse
        if fourD:
            obs_bias_correction = get_hybrid_bias_obs_output_3D(bias_features, model_states, W_H, output_size,
                                                                num_regions,
                                                                model_interp_fun, model_grid_ratio, model_in_feature,
                                                                stride)
            analysis, illConditioned = etkf_cycle_from_background(
                observations[i * hybrid_int_steps:(i + 1) * hybrid_int_steps],
                obs_indices[i * hybrid_int_steps:(i + 1) * hybrid_int_steps],
                R_inverse, background, output_size, H,
                obs_bias_correction,
                inflation_vec=inflation_vec)
        else:
            obs_bias_correction = get_hybrid_bias_obs_output_3D(bias_features[-1].reshape(1, bias_features.shape[1],
                                                                                          bias_features.shape[2],
                                                                                          -1),
                                                                model_states[-1].reshape(1, model_states.shape[1], -1),
                                                                W_H, output_size,
                                                                num_regions,
                                                                model_interp_fun, model_grid_ratio, model_in_feature,
                                                                stride)
            analysis, illConditioned = etkf_cycle_from_background(observations[i].reshape(1, observations.shape[1]),
                                                                  np.array([obs_indices[i]]),
                                                                  R_inverse_in,
                                                                  background[-1].reshape(1, background.shape[1],
                                                                                         background.shape[2]),
                                                                  output_size, H, obs_bias_correction,
                                                                  inflation_vec=inflation_vec)
        if r_in_ensemble:
            reservoir_prev_background = np.ascontiguousarray(analysis[-1,:,D:D+reservoir_vec_size]).\
                reshape(-1, num_regions, A_shape[0])
        else:
            reservoir_prev_background = reservoir_background[-1]
        prev_model_states = model_states[-1]
        prev_bias_features = bias_features[-1]
        if illConditioned:
            print(i)
            if i == 0:
                return analyses[:i * hybrid_int_steps + 1], np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1)),\
                    W_M, P_W_M
            else:
                return analyses[
                       :i * hybrid_int_steps + 1], prev_bias_features, reservoir_prev_background, prev_model_states,\
                    W_M, P_W_M
        if W_M_kf:
            if method == '1a':
                W_M_flat, P_W_M, illConditioned_W_M = kf_cycle_from_analysis_method1a(analysis[0, :, :D],
                                                                                      model_states[-1],
                                                                                      S_data, S_indices, S_indptr,
                                                                                      S_shape, W_M.flatten(),
                                                                                      P_W_M, R = Q,
                                                                                      fourD = fourD)
            elif method == '1b':
                W_M_flat, P_W_M, illConditioned_W_M = kf_cycle_from_analysis_method1b(analysis[0, :, :D],
                                                                                      model_states[-1],
                                                                                      S_data, S_indices, S_indptr,
                                                                                      S_shape, W_M.flatten(),
                                                                                      P_W_M, fourD=fourD)
            elif method == '1c':
                W_M_flat, P_W_M, illConditioned_W_M = \
                    kf_cycle_from_analysis_method1c(observations[i].flatten(),
                                                    np.ascontiguousarray(background[-1]),
                                                    np.ascontiguousarray(model_states[-1]),
                                                    HS_data, HS_indices, HS_indptr, HS_shape, H_mat_T, W_M.flatten(),
                                                    P_W_M, R = np.diag(1./np.diag(R_inverse)), fourD = fourD)
                analysis[:, :, :D] = analysis[:, :, :D] + mult_vec_csr(S_data, S_indices, S_indptr, S_shape,
                                                                       W_M_flat - W_M.flatten())
            elif method == '2':
                delta_W_M_flat, P_W_M, illConditioned_W_M = \
                    kf_cycle_from_analysis_method2(observations[i].flatten(), background[-1, :, :D], model_states[-1],
                                                   HS_data, HS_indices, HS_indptr, HS_shape,  P_W_M, H_mat_T,
                                                   R=np.diag(1. / np.diag(R_inverse)), fourD=fourD)
                analysis[:, :, :D] = analysis[:, :, :D] + mult_vec_csr(S_data, S_indices, S_indptr, S_shape,
                                                                       delta_W_M_flat)
                if r_in_ensemble:
                    analysis[:, :, D + reservoir_vec_size:D + reservoir_vec_size + N] = \
                        analysis[:, :, D + reservoir_vec_size:D + reservoir_vec_size + N] + delta_W_M_flat
                else:
                    analysis[:, :, D:D + N] = analysis[:, :, D:D + N] + delta_W_M_flat
            else:
                print('W_M_kf method is not recognized.')
                raise ValueError

            if illConditioned_W_M:
                print('W_M is ill-conditioned...')
                print(i)
                if i == 0:
                    return analyses[:i * hybrid_int_steps + 1], np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros(
                        (1, 1)), W_M, P_W_M
                else:
                    return analyses[
                           :i * hybrid_int_steps + 1], prev_bias_features, reservoir_prev_background, prev_model_states,\
                        W_M, P_W_M
            if method in ['1a', '1b', '1c']:
                W_M = W_M_flat.reshape(1, bias_features.shape[-1], stride)
        if return_ensemble:
            if fourD:
                analyses[i * hybrid_int_steps + 1:(i + 1) * hybrid_int_steps + 1] = analysis
            else:
                analyses[(i + 1) * hybrid_int_steps] = analysis[-1]
        else:
            if fourD:
                for j in range(hybrid_int_steps):
                    analyses[i * hybrid_int_steps + 1 + j, 0] = mean_numba_axis0(analysis[j])
            else:
                analyses[(i + 1) * hybrid_int_steps, 0] = mean_numba_axis0(analysis[-1])
        progress.update(1)
    return analyses, prev_bias_features, reservoir_prev_background, prev_model_states, W_M, P_W_M

@njit()
def etkf_cycle_from_background(observation, obs_idx, R_inverse, background, D, H,
                               obs_bias_correction = np.zeros((1, 1, 1), dtype = np.float64),
                               inflation_vec = np.array([1.0], dtype = np.float64),
                               max_cond = 1e6):
    if np.any(np.isinf(background)) or np.any(np.isnan(background)):
        print('Background has infs or nans, ending cycle...')
        return background, True
    y_ensemble = np.ascontiguousarray(np.transpose(H(background[:, :, :D] + obs_bias_correction, obs_idx), (1, 0, 2)))\
        .reshape(background.shape[1], -1)
    y_avg = mean_numba_axis0(y_ensemble)
    Y = y_ensemble - y_avg
    background_reshape = np.ascontiguousarray(np.transpose(background, (1, 0, 2))).reshape(background.shape[1], -1)
    x_avg = mean_numba_axis0(background_reshape)
    X = background_reshape - x_avg

    C = R_inverse @ Y.T
    #temp = self.inflation_mat + Y @ C
    temp = Y @ C + (background.shape[1] - 1)*np.eye(background.shape[1])#/inflation_vec[0]
    temp = (temp + temp.T)/2
    if np.any(np.isinf(temp)) or np.any(np.isnan(temp)):
        print('Temp has infs or nans, ending cycle...')
        return background, True
    ill_conditioned = check_condition(temp, max_condition = max_cond)
    if ill_conditioned:
        print('Ensemble is ill-conditioned, ending cycle...')
        return background, ill_conditioned
    P_tilde = np.ascontiguousarray(fast_positive_definite_inverse(temp))
    #P_tilde = np.linalg.inv(temp)
    W = numba_sqrtm((background.shape[1] - 1) * P_tilde)
    w = (observation.flatten() - y_avg) @ C @ P_tilde
    w_a = W + w
    analyses = np.transpose(np.ascontiguousarray(w_a @ X + x_avg).reshape(background.shape[1], background.shape[0],
                                                                          background.shape[2]), (1, 0, 2))
    return analyses, False
@njit()
def background_lorenz(analyses, dxdt, int_steps, h, params, hybrid_int_steps):
    background_preds = np.zeros((hybrid_int_steps, analyses.shape[0], analyses.shape[1]))
    background_preds[0] = lorenz_model_1_run_array(analyses, dxdt, int_steps, h, params)
    for i in range(1, hybrid_int_steps):
        background_preds[i] = lorenz_model_1_run_array(background_preds[i-1], dxdt, int_steps, h, params)
    return background_preds

@njit()
def background_lorenz_2(analyses, dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr, s_mat_shape, params,
                        hybrid_int_steps):
    background_preds = np.zeros((hybrid_int_steps, analyses.shape[0], analyses.shape[1]))
    background_preds[0] = lorenzmodel2_run_array(analyses, dxdt, int_steps, h, s_mat_data, s_mat_indices, s_mat_indptr,
                                                s_mat_shape, params)
    for i in range(1, hybrid_int_steps):
        background_preds[i] = lorenzmodel2_run_array(background_preds[i-1], dxdt, int_steps, h, s_mat_data,
                                                     s_mat_indices, s_mat_indptr, s_mat_shape, params)
    return background_preds

@njit(parallel = False)
def synchronize_background_bias_hybrid_local(analyses,  dxdt, int_steps, h, params,
                                             reservoir_background, leakage, A_data, A_indices, A_indptr, A_shape,
                                             B_data, B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes,
                                             input_pass, output_bias, num_regions, stride, overlap_size, local_overlap,
                                             model_in_feature, model_interp_fun, model_grid_ratio, hybrid_int_steps):
    if model_in_feature:
        bias_features = np.zeros((analyses.shape[1], num_regions, feature_size + stride))
    else:
        bias_features = np.zeros((analyses.shape[1], num_regions, feature_size))
    model_states = np.zeros((analyses.shape[1], analyses.shape[2] // model_grid_ratio))
    for j in range(analyses.shape[0]):
        background_rolled = roll_data_2d(analyses[j], num_regions, stride, overlap_size, local_overlap)
        for i in range(analyses.shape[1]):
            bias_features[i], reservoir_background[i], model_states[i] = \
                get_feature_bias_hybrid_local(background_rolled[i], dxdt, int_steps, h, params,
                                              reservoir_background[i], leakage, A_data, A_indices, A_indptr,
                                              A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                              squarenodes, input_pass, output_bias, num_regions, stride,
                                              overlap_size, local_overlap, model_in_feature, model_interp_fun,
                                              model_grid_ratio)
            #background_in[i] = get_hybrid_bias_output(bias_features[i], model_states[i], W[0], num_regions*stride,
            #                                         num_regions, model_interp_fun, model_grid_ratio, stride)
    #print('Background after synchronization')
    #print(background_in[1,:5])
    return bias_features, reservoir_background, model_states

@njit(parallel = False)
def background_bias_hybrid_local(analyses,  dxdt, int_steps, h, params,
                                 reservoir_background, leakage, A_data, A_indices, A_indptr, A_shape,
                                 B_data, B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes,
                                 input_pass, output_bias, num_regions, stride, overlap_size, local_overlap,
                                 model_in_feature, assimilate_model, output_size, model_interp_fun,
                                 model_grid_ratio, hybrid_int_steps):
    background_preds = np.zeros((hybrid_int_steps, analyses.shape[0], analyses.shape[1]))
    background_preds[0] = analyses
    if model_in_feature:
        bias_features = np.zeros((hybrid_int_steps, analyses.shape[0], num_regions, feature_size + stride))
    else:
        bias_features = np.zeros((hybrid_int_steps, analyses.shape[0], num_regions, feature_size))
    reservoir_background_full = np.zeros((hybrid_int_steps+1, reservoir_background.shape[0],
                                          reservoir_background.shape[1], reservoir_background.shape[2]))
    reservoir_background_full[0] = reservoir_background
    model_states = np.zeros((hybrid_int_steps, analyses.shape[0], analyses.shape[1] // model_grid_ratio))
    if assimilate_model:
        background_rolled = roll_data_2d(analyses, num_regions, stride, overlap_size, local_overlap)
        for k in range(hybrid_int_steps):
            for i in range(analyses.shape[0]):
                bias_features[k, i], reservoir_background_full[k+1, i], model_states[k, i] = \
                    get_feature_bias_hybrid_local(background_rolled[i], dxdt, int_steps, h, params,
                                             reservoir_background_full[k, i], leakage, A_data, A_indices, A_indptr,
                                             A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                             squarenodes, input_pass, output_bias, num_regions, stride,
                                             overlap_size, local_overlap, model_in_feature, model_interp_fun,
                                                  model_grid_ratio)
                background_preds[k, i] = get_hybrid_bias_output(bias_features[k, i], model_states[k, i], W[i], output_size,
                                                             num_regions, model_interp_fun, model_grid_ratio, stride)
            if k < hybrid_int_steps - 1:
                background_rolled = roll_data_2d(background_preds[k], num_regions, stride, overlap_size, local_overlap)
    else:
        background_rolled = roll_data_2d(analyses, num_regions, stride, overlap_size, local_overlap)
        for k in range(hybrid_int_steps):
            for i in range(analyses.shape[0]):
                bias_features[k, i], reservoir_background_full[k+1, i], model_states[k, i] = \
                    get_feature_bias_hybrid_local(background_rolled[i], dxdt, int_steps, h, params,
                                             reservoir_background_full[k,i], leakage, A_data, A_indices, A_indptr,
                                             A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                             squarenodes, input_pass, output_bias, num_regions, stride,
                                             overlap_size, local_overlap, model_in_feature, model_interp_fun,
                                                  model_grid_ratio)
                background_preds[k, i] = get_hybrid_bias_output(bias_features[k, i], model_states[k, i], W[0], output_size,
                                                                num_regions, model_interp_fun, model_grid_ratio, stride)
            if k < hybrid_int_steps - 1:
                background_rolled = roll_data_2d(background_preds[k], num_regions, stride, overlap_size, local_overlap)
    return background_preds, bias_features, reservoir_background_full[1:], model_states

#@njit()
def background_bias_hybrid_local_standardized(analyses,  dxdt, int_steps, h, params, mean, std,
                                              reservoir_prev_background, leakage, A_data, A_indices, A_indptr, A_shape,
                                              B_data, B_indices, B_indptr, B_shape, C, W, feature_size, squarenodes,
                                              input_pass, output_bias, num_regions, stride, overlap_size, local_overlap,
                                              model_in_feature):
    analyses_rolled = roll_data_2d(analyses, num_regions, stride, overlap_size, local_overlap)
    if model_in_feature:
        bias_features = np.zeros((analyses.shape[0], num_regions, feature_size + stride))
    else:
        bias_features = np.zeros((analyses.shape[0], num_regions, feature_size))
    model_states = np.zeros(analyses.shape)
    reservoir_background = np.zeros(reservoir_prev_background.shape)
    background_preds = np.zeros(analyses.shape)
    for i in range(analyses.shape[0]):
        bias_features[i], reservoir_background[i], model_states[i] = \
            get_feature_hybrid_local_standardized(analyses_rolled[i], dxdt, int_steps, h, params, mean, std,
                                                  reservoir_prev_background[i], leakage, A_data, A_indices, A_indptr,
                                                  A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                                  squarenodes, input_pass, output_bias, num_regions, stride,
                                                  overlap_size, local_overlap, model_in_feature)
        background_preds[i] = get_hybrid_bias_output(bias_features[i], model_states[i], W, num_regions, stride)
    return background_preds, bias_features, reservoir_background, model_states



@njit()
def background_hybrid_local_standardized(analyses,  dxdt, int_steps, h, params, mean, std, reservoir_prev_background,
                                               leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                               B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                               output_bias, num_regions, stride, overlap_size, local_overlap):
    analyses_rolled = roll_data_2d(analyses, num_regions, stride, overlap_size, local_overlap)
    bias_features = np.zeros((analyses.shape[0], num_regions, feature_size + stride))
    reservoir_background = np.zeros(reservoir_prev_background.shape)
    background_preds = np.zeros(analyses.shape)
    for i in range(analyses.shape[0]):
        bias_features[i], reservoir_background[i] = \
            get_feature_hybrid_local_standardized(analyses_rolled[i], dxdt, int_steps, h, params, mean, std,
                                                  reservoir_prev_background[i], leakage, A_data, A_indices, A_indptr,
                                                  A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                                  squarenodes, input_pass, output_bias, num_regions, stride,
                                                  overlap_size, local_overlap)
        background_preds[i] = (bias_features[i] @ W).flatten()
    return background_preds, bias_features, reservoir_background
@njit()
def background_hybrid_local(analyses, dxdt, int_steps, h, params, reservoir_prev_background,
                                               leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                               B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                               output_bias, num_regions, stride, overlap_size, local_overlap):
    analyses_rolled = roll_data_2d(analyses, num_regions, stride, overlap_size, local_overlap)
    bias_features = np.zeros((analyses.shape[0], num_regions, feature_size + stride))
    reservoir_background = np.zeros(reservoir_prev_background.shape)
    background_preds = np.zeros(analyses.shape)
    for i in range(analyses.shape[0]):
        bias_features[i], reservoir_background[i] = \
            get_feature_hybrid_local(analyses_rolled[i], dxdt, int_steps, h, params,
                                     reservoir_prev_background[i], leakage, A_data, A_indices, A_indptr,
                                     A_shape, B_data, B_indices, B_indptr, B_shape, C, feature_size,
                                     squarenodes, input_pass, output_bias, num_regions, stride,
                                     overlap_size, local_overlap)
        background_preds[i] = (bias_features[i] @ W).flatten()
    return background_preds, bias_features, reservoir_background

@njit()
def run_etkf_cycle_hybrid_local(observations, R_inverse, H, analysis, inflation_vec, dxdt,
                                                  int_steps, h, params, reservoir_prev_background,
                                                  leakage, A_data, A_indices, A_indptr, A_shape, B_data, B_indices,
                                                  B_indptr, B_shape, C, W, feature_size, squarenodes, input_pass,
                                                  output_bias, num_regions, stride, overlap_size, local_overlap):
    analyses = np.zeros((observations.shape[0]+1, analysis.shape[0], analysis.shape[1]))
    analyses[0] = analysis
    for i in range(observations.shape[0]):
        mean_analysis = mean_numba_axis0(analyses[i])
        background_init = (analyses[i] - mean_analysis) * inflation_vec + mean_analysis
        background, bias_features, reservoir_background = \
            background_hybrid_local(background_init,  dxdt, int_steps, h, params,
                                                 reservoir_prev_background, leakage, A_data, A_indices,
                                                 A_indptr, A_shape, B_data, B_indices, B_indptr, B_shape, C, W,
                                                 feature_size, squarenodes, input_pass, output_bias,
                                                 num_regions, stride, overlap_size, local_overlap)
        analyses[i+1], illConditioned = etkf_cycle_from_background(observations[i], R_inverse, background, H)
        if illConditioned:
            break
        reservoir_prev_background = reservoir_background
        prev_bias_features = bias_features
    return analyses[:i+1], prev_bias_features, reservoir_prev_background

@njit()
def get_S_kf(bias_features, D, stride):
    S_shape = np.array([D, bias_features.shape[-1] * D], dtype = np.int64)
    S_data_coo = np.zeros((bias_features.shape[0], bias_features.shape[-1] * D), dtype = np.float64)
    S_rows_coo = np.zeros(bias_features.shape[-1] * D, dtype = np.float32)
    S_cols_coo = np.zeros(bias_features.shape[-1] * D, dtype = np.float32)
    for i in range(D // stride):
        for j in range(stride):
            idx = stride*i + j
            S_rows_coo[idx*bias_features.shape[-1]:(idx+1)*bias_features.shape[-1]] = \
                np.ones(bias_features.shape[-1], dtype = np.float32) * idx
            S_cols_coo[idx * bias_features.shape[-1]:(idx + 1) * bias_features.shape[-1]] = \
                np.arange(0, bias_features.shape[-1] * D, D, dtype = np.float32) + idx
            for k in range(bias_features.shape[0]):
                S_data_coo[k, idx * bias_features.shape[-1]:(idx + 1) * bias_features.shape[-1]] = \
                    bias_features[k, i]

    with objmode(data='float64[:,:]', indices='int32[:]', indptr='int32[:]'):
        S_csr_i = coo_matrix((S_data_coo[0], (S_rows_coo, S_cols_coo)), shape = (S_shape[0], S_shape[1])).tocsr()
        data = np.zeros((bias_features.shape[0], S_csr_i.data.size), dtype = np.float64)
        indices = S_csr_i.indices
        indptr = S_csr_i.indptr
        data[0] = S_csr_i.data
        for i in range(1, bias_features.shape[0]):
            S_csr_i = coo_matrix((S_data_coo[i], (S_rows_coo, S_cols_coo)), shape = (S_shape[0], S_shape[1])).tocsr()
            data[i] = S_csr_i.data

    return data, indices, indptr, S_shape

@njit()
def get_HS(H_mat, S_data_mean, S_indices, S_indptr, S_shape):
    HS_dense = mat_csr_mult(H_mat, S_data_mean, S_indices, S_indptr, S_shape)
    with objmode(data='float64[:]', indices='int32[:]', indptr='int32[:]', shape='int32[:]'):
        HS = csr_matrix(HS_dense)
        data, indices, indptr, shape = HS.data, HS.indices, HS.indptr, \
                                       np.array([HS.shape[0], HS.shape[1]],
                                                dtype=np.int32)
    return data, indices, indptr, shape

@njit()
def kf_cycle_from_analysis(analysis, model_forecasts, bias_features, W_M, P_W_M, D, stride, max_cond = 1e6,
                           Q = np.zeros((1,1), dtype = np.float64),
                           fourD = False):
    assert(not fourD)
    P_W_M = P_W_M + Q
    deltax = analysis[:, :D] - model_forecasts
    mean_deltax = mean_numba_axis0(deltax)
    S_data, S_indices, S_indptr, S_shape = get_S_kf(bias_features, D, stride)
    S_data_mean = mean_numba_axis0(S_data)
    innovation = mean_deltax - mult_vec_csr(S_data_mean, S_indices, S_indptr, S_shape, W_M)
    #S_data_diff = S_data - S_data_mean
    #mean_model_forecast = mean_numba_axis0(model_forecasts)
    #model_forecast_diff = model_forecasts - mean_model_forecast
    #mean_analysis = mean_numba_axis0(analysis[:, :D])
    analysis_diff = deltax - mean_deltax
    #hybrid_diff = np.zeros((analysis.shape[0], D))
    #for i in range(analysis.shape[0]):
    #    hybrid_diff[i] = mult_vec_csr(S_data_diff[i], S_indices, S_indptr, S_shape, W_M)
    diff = analysis_diff #+ hybrid_diff
    R = diff.T @ diff / (analysis.shape[0] - 1)
    HP, innovation_cov = csr_mat_csr_T_mult(S_data_mean, S_indices, S_indptr, S_shape, P_W_M)
    innovation_cov = innovation_cov + R
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.transpose(np.linalg.solve(innovation_cov, HP))
    W_M_next = W_M + kalman_gain @ innovation
    P_W_M_next = (np.eye(W_M.size) - mat_csr_mult(kalman_gain, S_data_mean, S_indices, S_indptr, S_shape)) @ P_W_M
    return W_M_next, P_W_M_next, ill_conditioned

@njit()
def kf_cycle_from_analysis_method1a(analysis, model_forecasts, S_data, S_indices, S_indptr, S_shape, W_M, P_W_M,
                                    max_cond = 1e6, R = np.zeros((1,1), dtype = np.float64), fourD = False):
    assert(not fourD)
    deltax = analysis - model_forecasts
    mean_deltax = mean_numba_axis0(deltax)
    innovation = mean_deltax - mult_vec_csr(S_data, S_indices, S_indptr, S_shape, W_M)
    HP, innovation_cov = csr_mat_csr_T_mult(S_data, S_indices, S_indptr, S_shape, P_W_M)
    innovation_cov = innovation_cov + R
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.transpose(np.linalg.solve(innovation_cov, HP))
    W_M_next = W_M + kalman_gain @ innovation
    P_W_M_next = (np.eye(W_M.size) - mat_csr_mult(kalman_gain, S_data, S_indices, S_indptr, S_shape)) @ P_W_M
    return W_M_next, P_W_M_next, ill_conditioned

@njit()
def kf_cycle_from_analysis_method1b(analysis, model_forecasts, S_data, S_indices, S_indptr, S_shape, W_M, P_W_M,
                                    max_cond = 1e6, R = np.zeros((1,1), dtype = np.float64), fourD = False):
    assert(not fourD)
    P_W_M = P_W_M + R
    deltax = analysis - model_forecasts
    mean_deltax = mean_numba_axis0(deltax)
    innovation = mean_deltax - mult_vec_csr(S_data, S_indices, S_indptr, S_shape, W_M)
    diff = deltax - mean_deltax
    R = diff.T @ diff / (analysis.shape[0] - 1)
    HP, innovation_cov = csr_mat_csr_T_mult(S_data, S_indices, S_indptr, S_shape, P_W_M)
    innovation_cov = innovation_cov + R
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.transpose(np.linalg.solve(innovation_cov, HP))
    W_M_next = W_M + kalman_gain @ innovation
    P_W_M_next = (np.eye(W_M.size) - mat_csr_mult(kalman_gain, S_data, S_indices, S_indptr, S_shape)) @ P_W_M
    return W_M_next, P_W_M_next, ill_conditioned

@njit()
def kf_cycle_from_analysis_method1c(observations, background, model_forecasts, HS_data, HS_indices, HS_indptr, HS_shape,
                                    H, W_M, P_W_M, max_cond = 1e6, R = np.zeros((1,1), dtype = np.float64),
                                    fourD = False):
    assert(not fourD)
    H_model_forecasts = model_forecasts @ H
    mean_deltax = observations - mean_numba_axis0(H_model_forecasts)
    H_background_diff = background @ H - H_model_forecasts
    diff = H_background_diff - mean_numba_axis0(H_background_diff)
    H_background_cov = diff.T @ diff / (diff.shape[0] - 1)
    innovation = mean_deltax - mult_vec_csr(HS_data, HS_indices, HS_indptr, HS_shape, W_M)
    HP, innovation_cov = csr_mat_csr_T_mult(HS_data, HS_indices, HS_indptr, HS_shape, P_W_M)
    innovation_cov = innovation_cov + R + H_background_cov
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.transpose(np.linalg.solve(innovation_cov, HP))
    W_M_next = W_M + kalman_gain @ innovation
    P_W_M_next = (np.eye(W_M.size) - mat_csr_mult(kalman_gain, HS_data, HS_indices, HS_indptr, HS_shape)) @ P_W_M
    return W_M_next, P_W_M_next, ill_conditioned
@njit()
def kf_cycle_from_analysis_method2(observations, background, model_forecasts, HS_data, HS_indices, HS_indptr, HS_shape,
                                   P_W_M, H, max_cond = 1e6, R = np.zeros((1,1), dtype = np.float64),
                                   fourD = False):
    assert(not fourD)
    H_model_forecasts = np.ascontiguousarray(model_forecasts) @ H
    H_background = np.ascontiguousarray(background) @ H
    mean_deltax = np.ascontiguousarray(observations - mean_numba_axis0(H_background))
    H_background_diff = H_background - H_model_forecasts
    diff = np.ascontiguousarray(H_background_diff - mean_numba_axis0(H_background_diff))
    H_background_cov = np.ascontiguousarray(diff.T) @ diff / (diff.shape[0] - 1)
    innovation = mean_deltax #- mult_vec_csr(HS_data, HS_indices, HS_indptr, HS_shape, delta_M)
    HP, innovation_cov = csr_mat_csr_T_mult(HS_data, HS_indices, HS_indptr, HS_shape, P_W_M)
    innovation_cov = innovation_cov + R + H_background_cov
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.ascontiguousarray(np.transpose(np.linalg.solve(innovation_cov, HP)))
    delta_M_next = kalman_gain @ innovation # + W_M
    P_W_M_next = (np.eye(P_W_M.shape[0]) - mat_csr_mult(kalman_gain, HS_data, HS_indices, HS_indptr, HS_shape)) @ P_W_M
    return delta_M_next, P_W_M_next, ill_conditioned

@njit()
def kf_cycle(analysis, model_forecasts, bias_features, W_M, P_W_M, R, D, stride, max_cond = 1e6,
            Q = np.zeros((1,1), dtype = np.float64)):
    W_M = W_M.flatten()
    P_W_M = P_W_M + Q
    deltax = analysis.flatten() - model_forecasts.flatten()
    #mean_deltax = mean_numba_axis0(deltax)
    S_data, S_indices, S_indptr, S_shape = get_S_kf(bias_features.reshape(1, 1, -1), D, stride)
    #S_data_mean = mean_numba_axis0(S_data)
    innovation = deltax - mult_vec_csr(S_data[0], S_indices, S_indptr, S_shape, W_M)
    #S_data_diff = S_data - S_data_mean
    #mean_model_forecast = mean_numba_axis0(model_forecasts)
    #model_forecast_diff = model_forecasts - mean_model_forecast
    #mean_analysis = mean_numba_axis0(analysis[:, :D])
    #analysis_diff = deltax - mean_deltax
    #hybrid_diff = np.zeros((analysis.shape[0], D))
    #for i in range(analysis.shape[0]):
    #    hybrid_diff[i] = mult_vec_csr(S_data_diff[i], S_indices, S_indptr, S_shape, W_M)
    #diff = analysis_diff #+ hybrid_diff
    #R = diff.T @ diff / (analysis.shape[0] - 1)
    HP, innovation_cov = csr_mat_csr_T_mult(S_data[0], S_indices, S_indptr, S_shape, P_W_M)
    innovation_cov = innovation_cov + R
    ill_conditioned = False
    #ill_conditioned = check_condition(innovation_cov, max_condition=max_cond)
    #if ill_conditioned:
    #    print('Ensemble is ill-conditioned, ending cycle...')
    #    return W_M, P_W_M, ill_conditioned
    kalman_gain = np.transpose(np.linalg.solve(innovation_cov, HP))
    W_M_next = W_M + kalman_gain @ innovation
    P_W_M_next = (np.eye(W_M.size) - mat_csr_mult(kalman_gain, S_data[0], S_indices, S_indptr, S_shape)) @ P_W_M
    return W_M_next.reshape(bias_features.shape[-1], -1), P_W_M_next, ill_conditioned





