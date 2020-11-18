#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from collections import OrderedDict
import numpy as np
from numpy.polynomial import polynomial
from scipy import stats
import pandas as pd
from sklearn import linear_model
from . import six, utils


def nearest(x, parity='odd', round=np.round):
    if parity == 'even':
        return np.int_(round(x/2.0)) * 2
    elif parity == 'odd':
        return np.int_(round((x+1)/2.0)) * 2 - 1


def invert_affine(mat):
    M, v = mat[:,:3], mat[:,3]
    inv = np.linalg.inv(M)
    return np.c_[inv, -inv@v]


def concat_affine(mat2, mat1):
    '''
    mat @ v = mat2 @ mat1 @ v

    For afni's "fetching" convention, the first transform goes first.
    mat1 = io.read_affine('SurfVoltoT1.aff12.1D')   # Fetching SurfVol from T1 grid
    mat2 = io.read_affine('T1toEPI.aff12.1D')       # Fetching T1 from EPI grid
    mat = math.concat_affine(mat1, mat2) # Note the order! The combined transform needs to fetch SurfVol from EPI
    io.write_affine('SurfVoltoEPI.aff12.1D', mat)   # Fetching SurfVol from EPI
    '''
    return np.c_[mat2[:,:3]@mat1[:,:3], mat2[:,:3]@mat1[:,3]+mat2[:,3]]


def apply_affine(mat, xyz):
    '''
    xyz : 3xN array
    '''
    return mat[:,:3] @ xyz + mat[:,3:4]


def LPI2RAI_affine(mat):
    LPI2RAI = np.c_[np.diag([-1,-1,1]), np.zeros(3)] # Interestingly, the inverse of LPI2RAI is itself (i.e., LPI2RAI==RAI2LPI)
    return concat_affine(LPI2RAI, concat_affine(mat, LPI2RAI))

RAI2LPI_affine = LPI2RAI_affine
# fsl2afni_affine = LPI2RAI_affine
# afni2fsl_affine = RAI2LPI_affine


def polyfit3d(x, y, z, f, deg, method=None):
    # https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.polynomials.polynomial.html
    if method is None:
        method = 'ols'
    deg = np.array(deg) if utils.iterable(deg) else np.repeat(deg, 3)
    vander = polynomial.polyvander3d(x, y, z, deg)
    vander = vander.reshape(-1,vander.shape[-1])
    if method == 'ols':
        c = np.linalg.lstsq(vander, f.ravel(), rcond=None)[0]
    elif method == 'ridge':
        model = linear_model.Ridge(fit_intercept=False)
        model.fit(vander, f.ravel())
        c = model.coef_
    elif method == 'lasso':
        model = linear_model.Lasso(fit_intercept=False)
        model.fit(vander, f.ravel())
        c = model.coef_
    c = c.reshape(deg+1)
    return c 


class DomainMapper(object):
    def __init__(self, domain=None):
        self.domain = [0, 2*np.pi] if domain is None else domain

    def to2pi(self, x):
        # Mapping domain to [0, 2*pi]
        y = (x - self.domain[0]) / (self.domain[-1] - self.domain[0]) * 2*np.pi
        return y

    def from2pi(self, y):
        # Mapping domain back to original
        x = y / (2*np.pi) * (self.domain[-1] - self.domain[0]) + self.domain[0]
        return x


def circular_mean(x, domain=None, weight=None, axis=None):
    '''
    Circular mean for values from arbitary circular domain (not necessarily angles).
    '''
    if weight is None:
        weight = np.ones(x.shape)
    # Mapping domain into [0, 2*pi]
    mapper = DomainMapper(domain)
    y = mapper.to2pi(x)
    # Circular mean
    mean_y = np.sum(np.exp(1j*y) * weight, axis=axis) / np.sum(weight, axis=axis)
    mean_y = np.mod(np.angle(mean_y), 2*np.pi)
    # Mapping domain back
    mean_x = mapper.from2pi(mean_y)
    return mean_x


def circular_std(x, domain=None, weight=None, axis=None):
    '''
    Following scipy.stats.circstd()'s definition of circular standard deviation 
    that in the limit of small angles returns the 'linear' standard deviation.
    scipy.stats.circstd() doesn't support weight.
    pycircstat.std() doesn't support domain.
    astropy.stats.circvar() follows another definition.
    '''
    if weight is None:
        weight = np.ones(x.shape)
    # Mapping domain into [0, 2*pi]
    mapper = DomainMapper(domain)
    y = mapper.to2pi(x)
    # Circular std
    mean_y = np.sum(np.exp(1j*y) * weight, axis=axis) / np.sum(weight, axis=axis)
    std_y = np.sqrt(-2*np.log(np.abs(mean_y)))
    # Mapping domain back
    std_x = mapper.from2pi(std_y)
    return std_x


def corrcoef_along_axis(x, y, axis=None, norm=True):
    x = x - np.mean(x, axis=axis, keepdims=True)
    y = y - np.mean(y, axis=axis, keepdims=True)
    r = np.sum(x * y.conjugate(), axis=axis)
    if norm:
        r /= np.sqrt(np.sum(x*x.conjugate(), axis=axis) * np.sum(y*y.conjugate(), axis=axis))
    return r


def circular_corrcoef(x1, x2, domain=None, n_perm=1000, ci=0.95, n_boot=None):
    '''
    The complex corrcoef method used here is fundamentally different from 
    the (Fisher & Lee, 1983) method which is implemented by pycircstat.corrcc(). 
    For more details, read my note
    https://docs.google.com/document/d/1sl39YH3g3TFQu1zX-Ax477NULEyJE--MHMXg2gg0cyk/edit
    '''
    # Mapping domain into [0, 2*pi]
    mapper = DomainMapper(domain)
    y1 = mapper.to2pi(x1)
    y2 = mapper.to2pi(x2)
    # Compute circular corrcoef by treating angles as the phase of complex numbers
    f = lambda a1, a2, axis=None: corrcoef_along_axis(np.exp(1j*a1), np.exp(1j*a2), axis=axis)
    r = f(y1, y2)
    rM, rP = np.abs(r), np.angle(r)
    # Randomization test for significance
    if n_perm:
        n = y1.shape[0]
        Y1 = y1[np.random.choice(np.arange(n), size=[n_perm,n], replace=True)]
        Y2 = y2[np.random.choice(np.arange(n), size=[n_perm,n], replace=True)]
        R = f(Y1, Y2, axis=1)
        p = 1 - stats.percentileofscore(np.abs(R), rM)/100
    else:
        p = None
    # Bootstrap for confidence interval
    if ci is not None:
        if n_boot is None:
            n_boot = n_perm if n_perm else 1000
        idx = np.random.choice(np.arange(n), size=[n_boot,n], replace=True)
        Y1 = y1[idx]
        Y2 = y2[idx]
        R = f(Y1, Y2, axis=1)
        lb = (1-ci)/2
        ub = 1 - lb
        CI = np.percentile(np.abs(R), np.r_[lb, ub]*100)
    else:
        CI = None
    return rM, rP, p, CI


def normalize_logP(logP, axis=None):
    # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
    max_logP = np.max(logP, axis=axis, keepdims=True)
    logP = logP - max_logP # Make sure the larger elements (at least the largest one) no longer underflow
    P = np.exp(logP) # It is OK that the smaller elements still underflow
    return P / np.sum(P, axis=axis, keepdims=True)


def gaussian_logpdf(x, mean, cov, cov_inv=None, axis=-1):
    '''
    More efficient multivariate normal distribution log pdf than stats.multivariate_normal.logpdf()
    '''
    z = (x.swapaxes(axis,-1) - mean)[...,np.newaxis] # shape=(...,n,1)
    M = np.linalg.pinv(cov) if cov_inv is None else cov_inv # shape=(n,n). This can be slow, precompute if possible.
    # maxmul() used here is much more efficient (10x) for large matrices than dot() used in stats.multivariate_normal
    maha = (z.swapaxes(-1,-2) @ M @ z)[...,0,0] # (...,1,n) @ (n,n) @ (...,n,1) = (...,1,1)
    return -0.5 * (len(mean)*np.log(2*np.pi) + np.prod(np.linalg.slogdet(cov)) + maha)


def pinv(x):
    # This could be a numpy issue, since the same matrix works fine in Matlab.
    # https://github.com/numpy/numpy/issues/1588
    try:
        res = np.linalg.pinv(x)
    except np.linalg.LinAlgError as err:
        n_retry = 0
        while True:
            try:
                res = np.linalg.pinv(x + np.random.randn(*x.shape)*1e-15)
                break
            except np.linalg.LinAlgError:
                n_retry += 1
                if n_retry == 3:
                    from deepdish import io as dio
                    dio.save('debug.hdf', dict(x=x)) 
                    raise err
    return res


def median_argmax(x, axis=-1):
    y = x.swapaxes(axis, -1)
    res = [np.median(np.nonzero(yy == np.max(yy))[0]) for yy in y.reshape(-1, x.shape[axis])]
    return np.reshape(res, y.shape[:-1]).astype(int)


def tsarray2df(tsarray, t=None, ts_name='value', t_name='time', trial_name='trial', trial_df=None):
    '''
    Parameters
    ----------
    tsarray : ndarray, n_trials * n_times

    Returns
    -------
    df : DataFrame
    '''
    n_trials, n_times = tsarray.shape
    if t is None:
        t = np.arange(n_times)
    tsarray = tsarray.ravel()
    t = np.tile(t, [n_trials, 1]).ravel()
    trials = np.repeat(np.arange(n_trials), n_times)
    df = pd.DataFrame(OrderedDict([(trial_name, trials), (t_name, t), (ts_name, tsarray)]))
    if trial_df is not None:
        df = pd.concat([df, trial_df.iloc[trials].reset_index(drop=True)], axis=1)
    return df


def argsort_rows(rows, cols=None):
    if cols is None:
        cols = np.arange(rows.shape[1])[::-1]
    idx = np.arange(rows.shape[0])
    for col in cols:
        idx = idx[rows[idx,col].argsort(kind='stable')]
    fwd, rev = idx, np.zeros_like(idx)
    rev[fwd] = np.arange(rows.shape[0])
    return fwd, rev


if __name__ == '__main__':
    pass
