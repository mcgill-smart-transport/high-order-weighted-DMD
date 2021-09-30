import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.io import loadmat, savemat
from scipy.linalg import svd, svdvals, orth
from scipy.sparse.linalg import svds
from numpy import diag
from numpy.linalg import pinv, norm, eig, eigh
from sklearn.metrics import r2_score

figsize = (3, 3)
fontsize = 9
axislinewidth = 0.5
markersize = 5
text = None
limits = [-7, 7]
offset = [-44, 12]
projection = None
fontfamily = ["Helvetica", "Arial"]
contain_latex = False

matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": fontfamily, "size": fontsize})
matplotlib.rc('pdf', fonttype=42, use14corefonts=True, compression=6)
matplotlib.rc('ps', useafm=True, usedistiller='none', fonttype=42)
matplotlib.rc("axes", unicode_minus=False, linewidth=axislinewidth, labelsize='medium')
matplotlib.rc("axes.formatter", limits=limits)
matplotlib.rc('savefig', bbox='tight', format='eps', pad_inches=0.05)
matplotlib.rc('lines', marker=None, markersize=markersize)
matplotlib.rc('text', usetex=False)
matplotlib.rc('xtick', direction='in')
matplotlib.rc('xtick.major', width=axislinewidth)
matplotlib.rc('ytick', direction='in')
matplotlib.rc('lines', linewidth=1)
matplotlib.rc('ytick.major', width=axislinewidth)
matplotlib.rcParams['lines.solid_capstyle'] = 'round'
matplotlib.rcParams['lines.solid_joinstyle'] = 'round'
matplotlib.rc('mathtext', fontset='cm')  # stixsans, STIX,
matplotlib.rc('legend', fontsize='medium', frameon=False,
              handleheight=0.5, handlelength=1, handletextpad=0.4, numpoints=1)



def stagger_data(data, h):
    """
    >>> i = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    >>> stagger_data(i, [1, 3])
    (array([[ 3,  4,  5],
           [ 9, 10, 11],
           [ 1,  2,  3],
           [ 7,  8,  9]]), array([[ 4,  5,  6],
           [10, 11, 12]]))
    """
    h.sort()
    len_h = len(h)
    n, m = data.shape
    max_h = max(h)

    Y = data[:, max_h:]
    X = np.zeros((n * len_h, m - max_h), dtype=data.dtype)
    for i in range(len_h):
        X[i * n: i * n + n, :] = data[:, max_h - h[i]:m - h[i]]
    return X, Y


# %%
def get_flow1(od, s, dir='o', num_s=159):
    """Get the flow of station `s`"""
    n = od.shape[0]
    if dir == 'o':
        idx = np.arange(s, n, num_s)
    elif dir == 'd':
        idx = np.arange((s * num_s), (s * num_s + num_s))
    return np.sum(od[idx, :], axis=0)


def od2flow(od, s_list=None, dir='o', num_s=159):
    if s_list is None:
        s_list = range(num_s)

    n_s = len(s_list)
    flow = np.zeros((n_s, od.shape[1]), dtype=np.float32)
    for i, s in enumerate(s_list):
        flow[i, :] = get_flow1(od, s, dir, num_s)
    return flow


def RMSE(f0, f1, axis=None):
    return np.sqrt(np.mean((f0 - f1) ** 2, axis))


def SMAPE(real, predict):
    a = real.ravel().copy()
    b = predict.ravel().copy()
    mask = ((a > 0) | (b > 0))
    a = a[mask]
    b = b[mask]
    return 2 * np.mean(np.abs(a - b) / (a + b))


def WMAPE(real, predict):
    e = np.sum(np.abs(real - predict)) / np.sum(np.abs(real))
    return e


def MAE(real, predict):
    return np.mean(np.abs(real - predict))


def MSE(f0, f1, axis=None):
    return np.mean((f0 - f1) ** 2, axis)


def get_score(real, predict, real_flow, predict_flow):
    print('RMSE of OD: {}'.format(RMSE(real, predict)))
    print('WMAPE of OD: {}'.format(WMAPE(real, predict)))
    print('SMAPE of OD: {}'.format(SMAPE(real, predict)))
    print('MAE of OD: {}'.format(MAE(real, predict)))
    print('r2 of OD: {}'.format(r2_score(real.ravel(), predict.ravel())))
    print('\n')
    print('RMSE of flow: {}'.format(RMSE(real_flow, predict_flow)))
    print('WMAPE of flow: {}'.format(WMAPE(real_flow, predict_flow)))
    print('SMAPE of flow: {}'.format(SMAPE(real_flow, predict_flow)))
    print('MAE of flow: {}'.format(MAE(real_flow, predict_flow)))
    print('r2 of flow: {}\n'.format(r2_score(real_flow.ravel(), predict_flow.ravel())))


def remove_weekends(data, start=0, bs=36):
    """
    Remove the columns of weekends from data
    Parameters
    ----------
    data
    start: int, the weekday of the first column, default 0 (Monday)
    bs: the number of columns per day
    """
    _, m = data.shape
    n_day = int(m / bs)
    weekday = np.concatenate([np.arange(start, 7) % 7, np.arange(n_day) % 7])[:n_day]
    weekday = np.repeat(weekday, bs)
    return data[:, weekday < 5]


def start_end_idx(start, end, weekend=False, night=False):
    date = pd.period_range('2017-07-01', '2017-09-30 23:30', freq='30T')
    date = date.to_timestamp()
    if not night:
        date = date[date.hour >= 6]
    if not weekend:
        date = date[date.weekday < 5]
    idx = pd.DataFrame(data=np.arange(date.shape[0]), index=date)
    return idx.loc[start:end, :].values.ravel()


# %%
class HWDMD:
    def __init__(self, h, rx, ry, rho=1, bs=1):
        self.h = np.sort(h)
        self.rx = rx
        self.ry = ry
        self.rho = rho
        self.sigma = rho ** 0.5
        self.bs = bs  # batch size
        self.Ux = None
        self.Uy = None
        self.P = None
        self.Qx = None
        self.Qy = None
        self.Qx_inv = None
        self.n = None
        self.buffer_data = None  # The data used in batch update

    def Uxi(self, i):
        if i == len(self.h):  # The Ux for other features
            return self.Ux[(i * self.n):, :]
        else:  # The Ux for auto-regression
            return self.Ux[(i * self.n):(i * self.n + self.n), :]

    def fit(self, data, features, memory_save_mode=False):
        self.n, t = data.shape
        X, Y = stagger_data(data, self.h)
        X = np.concatenate([X, features], axis=0)
        self.buffer_data = data[:, -max(self.h):]
        m = Y.shape[1]

        num_bs = np.ceil(t / self.bs)  # Number of batches
        weight = self.sigma ** np.repeat(np.arange(num_bs), self.bs)[:m][::-1]

        X = X * weight
        Y = Y * weight

        if memory_save_mode:
            [Ux, _, _] = svds(np.float32(X), k=self.rx, which='LM', return_singular_vectors='u')
            [Uy, _, _] = svds(np.float32(Y), k=self.ry, which='LM', return_singular_vectors='u')
            self.Ux = Ux
            self.Uy = Uy
        else:
            [Ux, _, _] = svd(X, full_matrices=False)
            [Uy, _, _] = svd(Y, full_matrices=False)
            self.Ux = Ux[:, 0:self.rx]
            self.Uy = Uy[:, 0:self.ry]

        Xtilde = self.Ux.T @ X
        Ytilde = self.Uy.T @ Y
        self.P = Ytilde @ Xtilde.T
        self.Qx = Xtilde @ Xtilde.T
        self.Qy = Ytilde @ Ytilde.T
        self.Qx_inv = pinv(self.Qx)

    def apply(self, data, features, return_fit=True):
        """Apply the new data to the model. Optionally return one-step forecasting."""
        data = np.concatenate([self.buffer_data, data], axis=1)
        self.buffer_data = data[:, -max(self.h):]
        if return_fit:
            X, _ = stagger_data(data, self.h)
            X = np.concatenate([X, features], axis=0)
            return self._forecast(X)

    def _forecast(self, X):
        """Return the one_step forecast for **staggered** data."""
        nn = len(self.h)
        part2 = 0
        for i in range(nn):
            part2 += self.Uxi(i).T @ self.Uy @ (self.Uy.T @ X[i * self.n:(i * self.n + self.n), :])
        # For external features
        part2 += self.Uxi(nn).T @ X[nn * self.n:, :]
        return self.Uy @ self.P @ self.Qx_inv @ part2

    def update_model(self, X, Y):
        """Update the model coefficients using the new data (staggered). Does not change the buffer data"""
        if X.shape[1] != self.bs or Y.shape[1] != self.bs:
            raise ValueError('Number of columns does not equal to batchsize.')
        Ybar = self.Uy @ (self.Uy.T @ Y)
        Xbar = self.Ux @ (self.Ux.T @ X)

        Ux = orth(X - Xbar)
        Uy = orth(Y - Ybar)
        self.Ux = np.concatenate([self.Ux, Ux], axis=1)
        self.Uy = np.concatenate([self.Uy, Uy], axis=1)
        self.Qx = np.pad(self.Qx, ((0, Ux.shape[1]), (0, Ux.shape[1])), 'constant', constant_values=0)
        self.Qy = np.pad(self.Qy, ((0, Uy.shape[1]), (0, Uy.shape[1])), 'constant', constant_values=0)
        self.P = np.pad(self.P, ((0, Uy.shape[1]), (0, Ux.shape[1])), 'constant', constant_values=0)

        Ytilde = self.Uy.T @ Y
        Xtilde = self.Ux.T @ X
        self.P = self.rho * self.P + Ytilde @ Xtilde.T
        self.Qx = self.rho * self.Qx + Xtilde @ Xtilde.T
        self.Qy = self.rho * self.Qy + Ytilde @ Ytilde.T

        # Compress rx and ry
        evalue_x, evector_x = eigh(self.Qx)
        evalue_y, evector_y = eigh(self.Qy)
        evector_x = evector_x[:, -self.rx:]
        evector_y = evector_y[:, -self.ry:]
        self.Ux = self.Ux @ evector_x
        self.Uy = self.Uy @ evector_y
        self.Qx = np.diag(evalue_x[-self.rx:])
        self.Qy = np.diag(evalue_y[-self.ry:])
        self.P = evector_y.T @ self.P @ evector_x

        # Update
        self.Qx_inv = pinv(self.Qx)
