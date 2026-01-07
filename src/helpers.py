import os
import numpy as np
from numba import njit, objmode
from scipy.sparse import csr_matrix, csc_matrix
from scipy.linalg import sqrtm as sqrtm_scipy
from scipy.linalg import lapack
import math

def set_numba(root_folder, disable_jit):
    config_file = open(os.path.join(root_folder, '.numba_config.yaml'), 'w')
    config_file.write('---\n')
    config_file.write('disable_jit: %d' % int(disable_jit))
    config_file.close()
@njit()
def getsmat(N,K):
    """
    getsmat - obtain the summation matrix used in calculating the coupling term in Lorenz Models 2 and 3

    Inputs:
      N - total number of grid points

      K - coupling distance
    Outputs: s_mat - summation matrix
    """
    mask = np.zeros(N)
    J = K // 2

    mask[:J+1] = 1.0
    mask[-J:] = 1.0
    if K % 2 == 0:
        mask[-J] = 1/2
        mask[J]  = 1/2

    s_mat = np.zeros((N,N))
    for i in range(N):
        s_mat[i] = np.roll(mask, i)

    with objmode(data = 'float64[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        s_mat_sparse = csr_matrix(s_mat)
        data, indices, indptr, shape = s_mat_sparse.data, s_mat_sparse.indices, s_mat_sparse.indptr, \
                                       np.array([s_mat_sparse.shape[0], s_mat_sparse.shape[1]],
                                                dtype = np.int32)
    return data, indices, indptr, shape

@njit(fastmath=False)
def RBFK(X,Y,sigma): # let X be an N by 1 vector, Y be an M by 1 vector
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            K[i,j]=np.exp(-1/(2*sigma**2)*np.sum((X[i]-Y[j])**2))
    return K
@njit()
def getz2x(N, I, alpha, beta):
    z2xmat = np.zeros((N,N))

    row1 = np.zeros(N)
    row1[:I+1] = alpha - beta*np.abs(np.arange(I+1))
    row1[-I:]  = alpha - beta*np.abs(np.arange(I, 0, -1))
    row1[I] = row1[I]/2
    row1[-I] = row1[-I]/2
    for i in range(N):
        z2xmat[i] = np.roll(row1, i)

    with objmode(data = 'float64[:]', indices = 'int32[:]', indptr = 'int32[:]', shape = 'int32[:]'):
        z2x_sparse = csr_matrix(z2xmat)
        data, indices, indptr, shape = z2x_sparse.data, z2x_sparse.indices, z2x_sparse.indptr, \
                                       np.array([z2x_sparse.shape[0], z2x_sparse.shape[1]],
                                                dtype = np.int32)
    return data, indices, indptr, shape


@njit()
def XYquadratic(X,Y,K,s_mat):
    """
    XYquadratic - calculated the quadractic coupling term in Lorenz Models 2
    and 3 given some coupling distance.

    Inputs:
    X - first coupling input
    Y - second coupling input
    K - coupling distance in grid points
    s_mat - summation matrix used in calculating coupling

    Output: XY - quadratic coupling
    """
    s_mat = np.ascontiguousarray(s_mat)
    W = (s_mat @ X)/K
    V = (s_mat @ Y)/K
    return -np.roll(W, -2*K)*np.roll(V, -K) + s_mat @ np.ascontiguousarray(np.roll(W, -K) * np.roll(Y, K))/K

@njit()
def XYquadratic_sparse(X,Y,K,s_data, s_indices, s_indptr, s_shape):
    """
    XYquadratic - calculated the quadractic coupling term in Lorenz Models 2
    and 3 given some coupling distance.

    Inputs:
    X - first coupling input
    Y - second coupling input
    K - coupling distance in grid points
    s_mat - summation matrix used in calculating coupling

    Output: XY - quadratic coupling
    """
    W = (mult_vec_csr(s_data, s_indices, s_indptr, s_shape, X))/K
    V = (mult_vec_csr(s_data, s_indices, s_indptr, s_shape, Y))/K
    return -np.roll(W, -2*K)*np.roll(V, -K) + \
           (mult_vec_csr(s_data, s_indices, s_indptr, s_shape,
                                   np.ascontiguousarray(np.roll(W, -K) * np.roll(Y, K))))/K

@njit()
def mult_mat(mat, data, indices, indptr, shape):
    with objmode(out = 'float64[:,:]'):
        out = mat @ csr_matrix((data, indices, indptr), shape=[shape[0], shape[1]])
    return out

@njit()
def mat_csr_mult(dense_matrix, csr_values, col_idx, row_ptr, shape):
    #csr_row = 0  # Current row in CSR matrix
    res = np.zeros((dense_matrix.shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            for k in range(dense_matrix.shape[0]):
                dense_value = dense_matrix[k, i]
                res[k, col_idx[j]] += csr_values[j] * dense_value
        #csr_row += 1
    return res

@njit()
def csr_mat_csr_T_mult(csr_values, col_idx, row_ptr, shape, dense_matrix):
    with objmode(out_1='float64[:,:]', out_2='float64[:,:]'):
        csr_mat = csr_matrix((csr_values, col_idx, row_ptr), shape=[shape[0], shape[1]])
        out_1 = csr_mat @ dense_matrix
        out_2 = out_1 @ csr_mat.T
    return out_1, out_2
@njit()
def mult_vec_sparsemat_csr(sparsemat, vec):
    """Evaluates the matrix-vector product with a CSR matrix."""
    # Get the rows and columns

    # m, n = shape

    data    = sparsemat[0]
    indices = sparsemat[1]
    indptr  = sparsemat[2,:int(sparsemat[3,2])]
    shape   = np.array([int(sparsemat[3,0]), int(sparsemat[3,1])], dtype = np.int32)

    return mult_vec_csr(data, indices, indptr, shape, vec)
@njit()
def mult_vec_csr(data, indices, indptr, shape, vec):
    """Evaluates the matrix-vector product with a CSR matrix."""
    # Get the rows and columns

    # m, n = shape

    y = np.zeros(shape[0])

    for row_index in range(shape[0]):
        col_start = indptr[row_index]
        col_end = indptr[row_index + 1]
        for col_index in range(col_start, col_end):
            y[row_index] += data[col_index] * vec[indices[col_index]]

    return y

@njit()
def mult_vec(data, indices, indptr, shape, mat):
    assert(shape[1] == mat.size)
    out = np.zeros(shape[0])
    for i in range(mat.size):
        for k in range(indptr[i], indptr[i+1]):
            out[indices[k]] += data[k] * mat[i]
    return out

@njit()
def mean_numba_axis0(mat):
    # Computes the mean over axis 0 in numba compiled functions
    res = np.zeros(mat.shape[1])
    for i in range(mat.shape[1]):
        res[i] = np.mean(mat[:,i])
    return res

@njit()
def mean_numba_axis0_3D(mat):
    res = np.zeros((mat.shape[1], mat.shape[2]))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[2]):
            res[i, j] = np.mean(mat[:, i, j])
    return res
@njit()
def mean_numba_axis1(mat):
    # Computes the mean over axis 1 in numba compiled functions
    res = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        res[i] = np.mean(mat[i])
        #res[i] = np.nanmean(mat[i])
    return res

@njit()
def sum_numba_axis2_3D(mat):
    # Computes the mean over axis 0 in numba compiled functions
    res = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[0]):
            res[j, i] = np.sum(mat[j,i])
    return res

@njit()
def mean_numba_axis3_4D(mat):
    # Computes the mean over axis 3 in numba compiled functions
    res = np.zeros((mat.shape[0], mat.shape[1], mat.shape[2]))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            for k in range(mat.shape[2]):
                res[i, j, k] = np.mean(mat[i, j, k])
    return res

@njit()
def sum_numba_axis1(mat):
    # Computes the mean over axis 1 in numba compiled functions
    res = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        res[i] = np.sum(mat[i])
    return res

@njit()
def check_condition(matrix, max_condition=1e2):
    if np.linalg.cond(matrix) >= max_condition:
        return True
    else:
        return False

@njit()
def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    for r in range(1, n):
        for c in range(r):
            ut[r, c] = ut[c, r]
@njit()
def fast_positive_definite_inverse(m):
    with objmode(inv = 'float64[:,:]'):
        cholesky, info = lapack.dpotrf(m)
        #if info != 0:
        #    raise ValueError('dpotrf failed on input {}'.format(m))
        inv, info = lapack.dpotri(cholesky)
        #if info != 0:
        #    raise ValueError('dpotri failed on input {}'.format(cholesky))
        upper_triangular_to_symmetric(inv)
    return inv

@njit()
def numba_sqrtm(matrix):
    with objmode(out = 'float64[:,:]'):
        out = np.real(sqrtm_scipy(matrix))
    return out

@njit()
def valid_time(truth, prediction, cutoff = 0.5, normalization = 1.):
    norm_rmse = np.sqrt(mean_numba_axis1((truth - prediction)**2.0))/normalization
    #norm_rmse = np.sqrt(np.nanmean((truth - prediction) ** 2.0),axis=1) / normalization
    for i in range(truth.shape[0]):
        if norm_rmse[i] > cutoff or math.isnan(norm_rmse[i]):
            return max(i-1, 0), norm_rmse
    print('Max VT reached')
    return i, norm_rmse

@njit(fastmath=False)
def get_train_error_fast(W,features=None, output=None):
    train_preds = features @ W
    train_error = np.sqrt(mean_numba_axis1((train_preds - output) ** 2.0))
    return train_error

def average_valid_time(true_data, model, num_predictions = 1, sync_length = 50, prediction_length = 100,
                       cutoff = 0.5, normalization = 1., check_stability = False):
    valid_times = np.zeros(num_predictions)
    pred_sync_length = sync_length + prediction_length
    stability = np.zeros(num_predictions)
    for i in range(num_predictions):
        data = true_data[i*pred_sync_length:(i+1)*pred_sync_length]
        prediction = model.sync_and_predict(data, prediction_length - 1, sync_length) #Why prediction_length-1?
        valid_times[i], norm_rmse = valid_time(data[sync_length:], prediction, cutoff, normalization)
        #print(valid_time(data[sync_length:], prediction, cutoff, normalization))
        if check_stability:
            if np.all(norm_rmse < 1.5):
                stability[i] = 1.0
    if check_stability:
        return valid_times, stability
    else:
        return valid_times
    #else:
    #    return np.mean(valid_times)
@njit()
def norm_rmse(truth, prediction, normalization = 1.):
    norm_rmse = np.sqrt(mean_numba_axis1((truth - prediction)**2.0))/normalization
    return norm_rmse

def average_norm_mse_last_step(true_data, model, num_predictions = 1, sync_length = 50, prediction_length = 100,
                    normalization = 1.,Over_Time=False):
    norm_mse_arr = np.zeros(num_predictions)
    pred_sync_length = sync_length + prediction_length
    for i in range(num_predictions):
        data = true_data[i*pred_sync_length:(i+1)*pred_sync_length]
        prediction = model.sync_and_predict(data, prediction_length - 1, sync_length)
        norm_mse_arr[i] = np.mean(((data[-1]-prediction[-1])**2.0)/normalization)
    return np.mean(norm_mse_arr)

def average_norm_mse_open_loop(true_data,true_data_with_noise, model, sync_length = 50, prediction_length = 100,
                    normalization = 1.,return_features=False):
    prediction, features = model.sync_and_predict_open_loop(true_data_with_noise, prediction_length - 1, sync_length)
    norm_mse = ((true_data[sync_length:sync_length+prediction_length]- prediction)**2.0)/normalization
    if return_features:
        return np.mean(norm_mse,axis=1), features, true_data[sync_length:sync_length+prediction_length]
    return np.mean(norm_mse,axis=1)


def average_valid_time_and_norm_rmse(true_data, model, num_predictions = 1, sync_length = 50, prediction_length = 100,
                       cutoff = 0.5, normalization = 1., check_stability = False):
    valid_times = np.zeros(num_predictions)
    norm_rmse_arr = []
    pred_sync_length = sync_length + prediction_length
    stability = np.zeros(num_predictions)
    for i in range(num_predictions):
        data = true_data[i*pred_sync_length:(i+1)*pred_sync_length]
        prediction = model.sync_and_predict(data, prediction_length - 1, sync_length) #Why prediction_length-1?
        valid_times[i], norm_rmse = valid_time(data[sync_length:], prediction, cutoff, normalization)
        norm_rmse_arr.append(norm_rmse)
        #print(valid_time(data[sync_length:], prediction, cutoff, normalization))
        if check_stability:
            if np.all(norm_rmse < 1.5):
                stability[i] = 1.0
    if check_stability:
        return valid_times, stability,np.mean(norm_rmse_arr,axis=0)
    else:
        return valid_times,np.mean(norm_rmse_arr,axis=0)
def average_one_step_error(true_data, model, num_steps = 1, sync_length = 50,normalization = 1,test=False):
    pred_sync_length = sync_length + 1
    error=np.zeros(num_steps)
    for i in range(num_steps):
        data = true_data[i * pred_sync_length:(i + 1) * pred_sync_length]
        prediction = model.sync_and_predict(data, 1, sync_length)
        error[i] = np.mean((prediction[0, :]-data[-1])**2)/normalization
        if(test):
            print(prediction)
            print(data[-1])
    return np.mean(error)

@njit()
def local_input_periodic(B_data, B_indices, B_indptr, B_shape, x, local_overlap):
    """
    Function for applying the input matrix to each local region (assuming periodic boundary conditions).
    The size of the input matrix second dimension must be 2*locality + 1.
    :param B_data:
    :param B_indices:
    :param B_indptr:
    :param B_shape:
    :param x:
    :param locality:
    :return:
    """
    output = np.zeros((x.size, B_shape[0]))
    x_expand = np.concatenate((x[-local_overlap:], x, x[:local_overlap]))
    for i in range(x.size):
        output[i] = mult_vec(B_data, B_indices, B_indptr, B_shape, x_expand[i:B_shape[1]+i])
    return output

@njit()
def default_H(X):
    return X

def etkf_test_name(ic_seed, input_size, rho_err_str, num_regions, local_overlap, reservoir_size, ensemble_members,
                   num_observations, hybrid_steps, observation_type, obs_noise_std, init_ensemble_var_state,
                   assimilate_model_str, init_ensemble_var_W_M, assimilate_obs_str, init_ensemble_var_W_H,
                   average_W_str, fourD_str, H_err_str, W_M_kf_str, W_M_kf_method_str, W_M_kf_var_str, Q_var_str):
    if assimilate_model_str == '_model' and assimilate_obs_str == '_obs':
        test_name = \
            'seed%d_D%d%s_%dregions_l%d_N%d_e%d_%dobs_%dsteps_%s_%0.1estd_var%0.1e%s_var%0.1e%s_var%0.1e%s%s%s%s%s%s%s' % \
                    (ic_seed, input_size, rho_err_str, num_regions, local_overlap, reservoir_size, ensemble_members,
                     num_observations, hybrid_steps, observation_type, obs_noise_std, init_ensemble_var_state,
                     assimilate_model_str, init_ensemble_var_W_M, assimilate_obs_str, init_ensemble_var_W_H,
                     H_err_str, average_W_str, fourD_str, W_M_kf_str, W_M_kf_method_str, W_M_kf_var_str, Q_var_str)
    elif assimilate_model_str == '_model':
        test_name = 'seed%d_D%d%s_%dregions_l%d_N%d_e%d_%dobs_%dsteps_%s_%0.1estd_var%0.1e%s_var%0.1e%s%s%s%s%s%s%s%s' % \
                    (ic_seed, input_size, rho_err_str, num_regions, local_overlap, reservoir_size, ensemble_members,
                     num_observations, hybrid_steps, observation_type, obs_noise_std, init_ensemble_var_state,
                     assimilate_model_str, init_ensemble_var_W_M, assimilate_obs_str, H_err_str,
                     average_W_str, fourD_str, W_M_kf_str, W_M_kf_method_str, W_M_kf_var_str, Q_var_str)
    elif assimilate_obs_str == '_obs':
        test_name = 'seed%d_D%d%s_%dregions_l%d_N%d_e%d_%dobs_%dsteps_%s_%0.1estd_var%0.1e%s%s_var%0.1e%s%s%s%s%s%s%s' % \
                    (ic_seed, input_size, rho_err_str, num_regions, local_overlap, reservoir_size, ensemble_members,
                     num_observations, hybrid_steps, observation_type, obs_noise_std, init_ensemble_var_state,
                     assimilate_model_str, assimilate_obs_str, init_ensemble_var_W_H, H_err_str,
                     average_W_str, fourD_str, W_M_kf_str, W_M_kf_method_str, W_M_kf_var_str, Q_var_str)
    else:
        test_name = 'seed%d_D%d%s_%dregions_l%d_N%d_e%d_%dobs_%dsteps_%s_%0.1estd_var%0.1e%s%s%s%s%s%s%s%s%s' % \
                    (ic_seed, input_size, rho_err_str, num_regions, local_overlap, reservoir_size, ensemble_members,
                     num_observations, hybrid_steps, observation_type, obs_noise_std, init_ensemble_var_state,
                     assimilate_model_str, assimilate_obs_str, H_err_str, average_W_str, fourD_str, W_M_kf_str,
                     W_M_kf_method_str, W_M_kf_var_str, Q_var_str)

    return test_name

