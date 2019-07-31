#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from numpy.polynomial import polynomial
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
fsl2afni_affine = LPI2RAI_affine
afni2fsl_affine = RAI2LPI_affine


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


if __name__ == '__main__':
    pass
