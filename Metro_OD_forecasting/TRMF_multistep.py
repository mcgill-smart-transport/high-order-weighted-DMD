"""
Temporal regularized matrix factorization (TRMF) for metro OD forecasting,
Code adapted from https://github.com/xinychen/transdim.

Original paper for TRMF:
Hsiang-Fu Yu, Nikhil Rao, Inderjit S. Dhillon, 2016. Temporal regularized matrix factorization for
high-dimensional time series prediction. 30th Conference on Neural Information Processing Systems (NIPS 2016),
"""
from numpy.linalg import inv as inv
import numpy as np

def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8], [9, 10]])
print(kr_prod(A, B))


def TRMF(dense_mat, sparse_mat, init, time_lags, lambda_w, lambda_x, lambda_theta, eta, maxiter, multi_steps=1):
    start = time.time()
    W = init["W"]
    X = init["X"]
    theta = init["theta"]

    dim1, dim2 = sparse_mat.shape
    binary_mat = np.zeros((dim1, dim2))
    position = np.where((sparse_mat != 0))
    binary_mat[position] = 1
    pos = np.where((dense_mat != 0) & (sparse_mat == 0))
    d = len(time_lags)
    r = theta.shape[1]

    for iter in range(maxiter):
        if (iter + 1) % 10 == 0:
            print('Time step: {} time {}'.format(iter + 1, time.time() - start))
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = np.matmul(var2, binary_mat.T)
        var4 = np.matmul(var1, sparse_mat.T)
        for i in range(dim1):
            W[i, :] = np.matmul(inv((var3[:, i].reshape([r, r])) + lambda_w * np.eye(r)), var4[:, i])

        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = np.matmul(var2, binary_mat)
        var4 = np.matmul(var1, sparse_mat)
        for t in range(dim2):
            Mt = np.zeros((r, r))
            Nt = np.zeros(r)
            if t < max(time_lags):
                Pt = np.zeros((r, r))
                Qt = np.zeros(r)
            else:
                Pt = np.eye(r)
                Qt = np.einsum('ij, ij -> j', theta, X[t - time_lags, :])
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    theta0 = theta.copy()
                    theta0[k, :] = 0
                    Mt = Mt + np.diag(theta[k, :] ** 2)
                    Nt = Nt + np.multiply(theta[k, :], (X[t + time_lags[k], :]
                                                        - np.einsum('ij, ij -> j', theta0,
                                                                    X[t + time_lags[k] - time_lags, :])))
                X[t, :] = np.matmul(inv(var3[:, t].reshape([r, r])
                                        + lambda_x * Pt + lambda_x * Mt + lambda_x * eta * np.eye(r)),
                                    (var4[:, t] + lambda_x * Qt + lambda_x * Nt))
            elif t >= dim2 - np.min(time_lags):
                X[t, :] = np.matmul(inv(var3[:, t].reshape([r, r]) + lambda_x * Pt
                                        + lambda_x * eta * np.eye(r)), (var4[:, t] + Qt))
        for k in range(d):
            var1 = X[np.max(time_lags) - time_lags[k]: dim2 - time_lags[k], :]
            var2 = inv(np.diag(np.einsum('ij, ij -> j', var1, var1)) + (lambda_theta / lambda_x) * np.eye(r))
            var3 = np.zeros(r)
            for t in range(np.max(time_lags) - time_lags[k], dim2 - time_lags[k]):
                var3 = var3 + np.multiply(X[t, :],
                                          (X[t + time_lags[k], :]
                                           - np.einsum('ij, ij -> j', theta, X[t + time_lags[k] - time_lags, :])
                                           + np.multiply(theta[k, :], X[t, :])))
            theta[k, :] = np.matmul(var2, var3)

        mat_hat = np.matmul(W, X.T)
        rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2) / dense_mat[pos].shape[0])
        if (iter + 1) % 200 == 0:
            print('Iter: {}'.format(iter + 1))
            print('RMSE: {:.6}'.format(rmse))
            print()

    X_new = np.zeros((dim2 + multi_steps, rank))
    X_new[0: dim2, :] = X.copy()
    for step in range(multi_steps):
        X_new[dim2 + step, :] = np.einsum('ij, ij -> j', theta, X_new[dim2 + step - time_lags, :])

    return W, X_new, theta, np.matmul(W, X_new[dim2 : dim2 + multi_steps, :].T)


def OnlineTRMF(sparse_vec, init, lambda_x, time_lags):
    W = init["W"]
    X = init["X"]
    theta = init["theta"]
    dim = sparse_vec.shape[0]
    t, rank = X.shape
    position = np.where(sparse_vec != 0)
    binary_vec = np.zeros(dim)
    binary_vec[position] = 1

    xt_tilde = np.einsum('ij, ij -> j', theta, X[t - 1 - time_lags, :])
    var1 = W.T
    var2 = kr_prod(var1, var1)
    var_mu = np.matmul(var1, sparse_vec) + lambda_x * xt_tilde
    inv_var_Lambda = inv(np.matmul(var2, binary_vec).reshape([rank, rank]) + lambda_x * np.eye(rank))
    X[t - 1, :] = np.matmul(inv_var_Lambda, var_mu)
    return X


def st_prediction(dense_mat, sparse_mat, time_lags, lambda_w, lambda_x, lambda_theta, eta,
                  rank, pred_time_steps, maxiter, multi_steps=1):
    start = time.time()
    start_time = dense_mat.shape[1] - pred_time_steps
    dense_mat0 = dense_mat[:, 0: start_time]
    sparse_mat0 = sparse_mat[:, 0: start_time]
    dim1 = sparse_mat0.shape[0]
    dim2 = sparse_mat0.shape[1]
    max_time_lag = max(time_lags)
    results = {step + 1: np.zeros((dim1, pred_time_steps)) for step in range(multi_steps)}

    for t in range(pred_time_steps):
        if t == 0:
            init = {"W": 0.1 * np.random.rand(dim1, rank), "X": 0.1 * np.random.rand(dim2, rank),
                    "theta": 0.1 * np.random.rand(time_lags.shape[0], rank)}
            W, X, theta, mat_f = TRMF(dense_mat0, sparse_mat0, init, time_lags,
                                      lambda_w, lambda_x, lambda_theta, eta, maxiter, multi_steps)
            # Assign forecast to the corresponding step
            for step in range(multi_steps):
                results[step + 1][:, t] = mat_f[:, step]
            X0 = X[dim2-max_time_lag:dim2 + 1, :].copy()  # Keep recent max_time_lag + one-step forecast
        else:
            sparse_vec = sparse_mat[:, start_time + t - 1]
            if np.where(sparse_vec > 0)[0].shape[0] > rank:
                init = {"W": W, "X": X0, "theta": theta}
                X = OnlineTRMF(sparse_vec, init, lambda_x / dim2, time_lags)
                X0 = np.zeros((max_time_lag + multi_steps, rank))
                X0[0: max_time_lag, :] = X[1:, :]
                for step in range(multi_steps):
                    step_X = np.einsum('ij, ij -> j', theta, X0[max_time_lag + step - time_lags, :])
                    X0[max_time_lag + step, :] = step_X
                    results[step+1][:, t] = W @ step_X
                X0 = X0[:max_time_lag+1, :]   # Keep recent max_time_lag + one-step forecast
            else:
                X = X0.copy()
                X0 = np.zeros((max_time_lag + multi_steps, rank))
                X0[0: max_time_lag, :] = X[1:, :]
                for step in range(multi_steps):
                    step_X = np.einsum('ij, ij -> j', theta, X0[max_time_lag + step - time_lags, :])
                    X0[max_time_lag + step, :] = step_X
                    results[step + 1][:, t] = W @ step_X
                X0 = X0[:max_time_lag + 1, :]  # Keep recent max_time_lag + one-step forecast

        if (t + 1) % 40 == 0:
            print('Time step: {}, time {}'.format(t + 1, time.time() - start))
    return results


# %% Start my code, TRMF on mean subtracted data
from functions import *
import time

data = loadmat('..//data//OD_3m.mat')
data = data['OD']
data = remove_weekends(data, start=5)

train_idx = start_end_idx('2017-07-03', '2017-08-11', weekend=False, night=False)
validate_idx = start_end_idx('2017-07-31', '2017-08-11', weekend=False, night=False)
test_idx = start_end_idx('2017-08-14', '2017-08-25', weekend=False, night=False)

# Subtract the mean in the training set
data = data.astype(np.float64)
data_mean = data[:, 0:20 * 36].reshape([159 * 159, 36, -1], order='F')
data_mean = data_mean.mean(axis=2)
for i in range(65):
    data[:, i * 36:(i + 1) * 36] = data[:, i * 36:(i + 1) * 36] - data_mean

multi_steps = 3
pred_time_steps = 36*5*2 + (multi_steps-1)
train_data = data[:, train_idx[0]:test_idx[-1]+1]
# time_lags = np.array([1, 2, 3, 4, 7, 8, 9, 21, 33, 35, 36])
time_lags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
rank = 45
lambda_w = 6000
lambda_x = 6000
lambda_theta = 6000
eta = 0.03
d = time_lags.shape[0]
dense_mat = train_data
sparse_mat = train_data
maxiter = 300
results = st_prediction(dense_mat, dense_mat, time_lags, lambda_w, lambda_x, lambda_theta,
                        eta, rank, pred_time_steps, maxiter, multi_steps)


for step in range(3):
    print(RMSE(dense_mat[:, -360:], results[step+1][:, 2-step:2-step+360]))

#%% Save results to file
mat_hat1 = results[1][:, 2:2+360].copy()
mat_hat2 = results[2][:, 1:1+360].copy()
mat_hat3 = results[3][:, 0:0+360].copy()
for i in range(mat_hat1.shape[1]):
    mat_hat1[:, i] += data_mean[:, i % 36]
    mat_hat2[:, i] += data_mean[:, i % 36]
    mat_hat3[:, i] += data_mean[:, i % 36]

np.savez_compressed('..//data//OD_TRMF_step1.npz', data=mat_hat1)
np.savez_compressed('..//data//OD_TRMF_step2.npz', data=mat_hat2)
np.savez_compressed('..//data//OD_TRMF_step3.npz', data=mat_hat3)

#%%
# n = 14
# plt.plot(results[1][n,2:])
# plt.plot(results[2][n,1:-1])
# plt.plot(results[3][n,:-2])
# plt.plot(dense_mat[n, -360:])

