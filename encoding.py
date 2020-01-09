#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import time
import numpy as np
from scipy import optimize, stats
from . import six, utils, math


def basis_vanBergen2015(s, n_channels=8):
    '''
    Parameters
    ----------
    s : 1D array, n_trials
    
    Returns
    -------
    fs : 2D array, n_channels * n_trials

    References
    ----------
    {van Bergen2015}
    '''
    phi = np.arange(0, np.pi, np.pi/n_channels)
    fs = np.maximum(0, np.cos(2 * (s[np.newaxis,:] - phi[:,np.newaxis]))) # n_channels * n_trials
    return fs


def basis_Sprague2013(s, n_channels=6, spacing=2, size=None, power=7, dim=1):
    '''
    Parameters
    ----------
    s : 2D array, n_trials * dim | 1D array, n_trials
    
    Returns
    -------
    fs : 2D array, n_channels * n_trials

    References
    ----------
    {Sprague2013}
    '''
    if size is None:
        size = 5.8153/2.0940 * spacing # Ratio is chosen to avoid high corr between channels while accomplish smooth recon
    center = (np.arange(n_channels) - (n_channels-1)/2) * spacing
    if dim == 1:
        r = np.abs(s[np.newaxis,:] - center[:,np.newaxis]) # Distance from filter’s center
    elif dim == 2:
        X, Y = np.meshgrid(center, center)
        center = np.c_[X.ravel(), Y.ravel()] # 2D channel array is serialized in row-first order
        r = np.linalg.norm(s.T[np.newaxis,...] - center[...,np.newaxis], axis=1)
    fs = np.where(r<size, (0.5*np.cos(r/size*np.pi) + 0.5)**7, 0) # n_channels * n_trials
    return fs


class ChannelEncodingModel(object):
    def __init__(self, n_channels, basis_func, stimulus_domain, circular=False):
        '''
        After (Brouwer et al., 2009).
        '''
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.stimulus_domain = stimulus_domain
        self.circular = circular

    # get_params() and get_params() are required by sklearn
    def get_params(self, deep=True):
        return dict(n_channels=self.n_channels, basis_func=self.basis_func, 
            stimulus_domain=self.stimulus_domain, circular=self.circular)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : 2D array
            n_trials * n_voxels BOLD response pattern
            (e.g., beta for each trial, or delayed and detrended time points within block plateau)
        y : 1D array
            n_trials stimulus value (e.g., orientation, color)
        '''
        b = X.T # Voxel BOLD response, n_voxels * n_trials
        s = y # Stimulus, n_trials
        fs = self.basis_func(s) # Channel response, n_channels * n_trials
        # Step 1: Estimate W by OLS regression
        W = b @ fs.T @ np.linalg.pinv(fs @ fs.T) # Weight, n_voxels * n_channels
        # Store params
        self.W_ = W
        return self # Required by sklearn

    def predict(self, X, stimulus_domain=None, return_all=False):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        channel_resp = self.inverted_encoding(X) # n_trials * n_channels
        evidence = self.correlation_inversion(X, stimulus_domain=None) # n_trials * n_domain
        y_map = stimulus_domain[np.argmax(evidence, axis=1)] # n_trials
        # y_mean = np.sum(stimulus_domain * evidence, axis=1) / np.sum(evidence, axis=1) # n_trials
        # y_std = np.sqrt(np.sum((stimulus_domain[np.newaxis,:] - y_mean[:,np.newaxis])**2 * evidence, axis=1) \
        #     / np.sum(evidence, axis=1)) # n_trials
        # return (y_mean, y_std, y_map, evidence, channel_resp) if return_all else y_mean
        return (y_map, evidence, channel_resp) if return_all else y_map

    def pRF(self, stimulus_domain=None, method='ols', X=None, y=None):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        fs_domain = self.basis_func(stimulus_domain) # n_channels * n_domain
        if method == 'ols':
            pRF = self.W_ @ fs_domain # n_voxels * n_domain
        elif method == 'ridge': 
            # After {Sprague2013}, to ensure that most of the pRFs were sufficiently unimodal
            # However, the method doesn't seem to help much in my simulation...
            b = X.T # Voxel BOLD response, n_voxels * n_trials
            s = y # Stimulus, n_trials
            fs = self.basis_func(s) # Channel response, n_channels * n_trials
            n_trials = len(s) 
            # Find best lambda according to BIC
            Ws = []
            BICs = []
            for lambda_ in np.logspace(-3, 3, 19):
                M = fs.T @ np.linalg.pinv(fs @ fs.T + lambda_ * np.eye(self.n_channels)) # n_trials * n_channels
                W = b @ M # Weight, n_voxels * n_channels
                H = M @ fs # Projection (or "hat") matrix, n_trials * n_trials
                df = np.trace(H) # Degrees of freedom for ridge (less than the number of predictors as in OLS)
                RSS = np.sum((b @ H - b)**2) # Residual sum of squares
                BIC = n_trials * np.log(RSS) + df * np.log(n_trials) # Scaling problem???
                Ws.append(W)
                BICs.append(BIC)
            W_opt = Ws[np.argmin(BICs)]
            pRF = W_opt @ fs_domain
        return pRF # n_voxels * n_domain
        
    def inverted_encoding(self, X):
        b = X.T # Voxel BOLD response, n_voxels * n_trials
        fs = np.linalg.pinv(self.W_.T @ self.W_) @ self.W_.T @ b # Inverted channel response, n_channels * n_trials
        return fs.T # n_trials * n_channels

    def voxel_inversion(self, X, stimulus_domain=None):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        recon = X @ self.pRF(stimulus_domain=stimulus_domain) # n_trials * n_domain
        recon /= X.shape[1] # / n_voxels
        return recon

    def channel_inversion(self, X, stimulus_domain=None):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        fs_domain = self.basis_func(stimulus_domain) # n_channels * n_domain
        fs_data = self.inverted_encoding(X) # n_trials * n_channels
        recon = fs_data @ fs_domain # n_trials * n_domain
        return recon

    def correlation_inversion(self, X, stimulus_domain=None):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        # Iterate every possible stimulus in the domain
        fs_domain = self.basis_func(stimulus_domain) # n_channels * n_domain
        # Inverted encoding
        fs_data = self.inverted_encoding(X) # n_trials * n_channels
        # Correlation as evidence (similar to likelihood)
        n_trials, n_domain = fs_data.shape[0], fs_domain.shape[1]
        evidence = np.zeros([n_trials, n_domain]) # n_trials * n_domain
        for tid in range(n_trials):
            for did in range(n_domain):
                evidence[tid,did] = np.corrcoef(fs_data[tid,:], fs_domain[:,did])[0,1]
        return evidence


class BayesianChannelModel(object):
    def __init__(self, n_channels, basis_func, stimulus_domain, circular=False, stimulus_prior=None, global_search=False):
        '''
        After (van Bergen et al., 2015).
        '''
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.stimulus_domain = stimulus_domain
        self.circular = circular
        self.stimulus_prior = 1 if stimulus_prior is None else stimulus_prior
        self.global_search = global_search

    # get_params() is required by sklearn
    def get_params(self, deep=True):
        return dict(n_channels=self.n_channels, basis_func=self.basis_func, stimulus_domain=self.stimulus_domain,
            circular=self.circular, stimulus_prior=self.stimulus_prior, global_search=self.global_search)

    # set_params() is required by sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : 2D array
            n_trials * n_voxels BOLD response pattern
            (e.g., beta for each trial, or delayed and detrended time points within block plateau)
        y : 1D array
            n_trials stimulus value (e.g., orientation, color)
        '''
        b = X.T # Voxel BOLD response, n_voxels * n_trials
        s = y # Stimulus, n_trials
        fs = self.basis_func(s) # Channel response, n_channels * n_trials
        # Step 1: Estimate W by OLS regression
        # W = b @ fs.T @ np.linalg.inv(fs @ fs.T) # Weight, n_voxels * n_channels
        # It is critical to use pinv() here instead of inv():
        # 1) Result in much more accurate estimation for W, and hence tau, rho, sigma
        # 2) Avoid "LinAlgError: Singular matrix" at `inv(fs @ fs.T)`
        # 3) Avoid negative negloglikelihood (i.e., negative `slogdet(Omega)`) 
        #    and non positive semidefinite `Omega` (i.e., `all(eigvals(Omega)>0) == False`), 
        #    which could occur with randn W
        # Note that Gilles used `np.linalg.lstsq()` here, which should be numerically adept.
        W = b @ fs.T @ np.linalg.pinv(fs @ fs.T) # Weight, n_voxels * n_channels
        # Store params
        self.W_ = W
        # Step 2: Estimate tau, rho, sigma by ML optimization (gradient-based)
        z = b - W @ fs # n_voxels * n_trials
        # Initial params
        tau0 = np.std(b, axis=1)
        rho0 = np.mean(np.corrcoef(b)[np.triu_indices(len(tau0), k=1)])
        sigma0 = np.mean(np.std(fs, axis=1))
        params0 = np.r_[tau0, rho0, sigma0/5.0]
        bounds = np.c_[np.ones(len(params0))*1e-4, np.r_[tau0*5, 1, sigma0*5]]
        # Conjugate gradient algorithm, due to lack of support for bounds, requires multi-start to avoid/alleviate being trapped in local minima.
        print('>> Start maximum likelihood optimization...')
        if self.global_search:
            def accept_test(f_new, x_new, f_old, x_old):
                Omega = self._calc_Omega(self.W_, x_new[:-2], x_new[-2], x_new[-1])
                # is_pos_semi_def = np.all(np.linalg.eigvals(Omega) > 0)
                # return (f_new < f_old and f_new > 0 and is_pos_semi_def)
                is_singular = np.linalg.matrix_rank(Omega, hermitian=True) < Omega.shape[0]
                return (f_new < f_old and f_new > 0 and not is_singular)
            res = optimize.basinhopping(self._negloglikelihood, params0, accept_test=accept_test, 
                minimizer_kwargs=dict(args=(z, W), method='L-BFGS-B', jac=self._negloglikelihood_prime, bounds=bounds))
        else:
            class Counter(object):
                def __init__(self, model, args):
                    self.count = 0
                    self.last_time = time.time()
                    self.model = model
                    self.args = args
                def step(self, xk):
                    self.count += 1
                    cost = self.model._negloglikelihood(xk, *self.args)
                    curr_time = time.time()
                    duration = curr_time - self.last_time
                    self.last_time = curr_time
                    print(f"iter#{self.count:03d} ({utils.format_duration(duration)}): cost={cost:.4f}, tau[-3:]={xk[-5:-2]}, rho={xk[-2]:.4f}, sigma={xk[-1]:.4f}")
            res = optimize.minimize(self._negloglikelihood, params0, args=(z, W), method='L-BFGS-B', 
                jac=self._negloglikelihood_prime, bounds=bounds, callback=Counter(model=self, args=(z, W)).step)
        params = res.x
        print(f"cost={res.fun}, iter={res.nit}, func_eval={res.nfev}, success={res.success}, {res.message}")
        print(params0[-3:], '-->', params[-3:])
        # Store params
        self.tau_, self.rho_, self.sigma_ = params[:-2], params[-2], params[-1]
        self.Omega_ = self._calc_Omega(self.W_, self.tau_, self.rho_, self.sigma_)
        return self # Required by sklearn

    def predict(self, X, stimulus_domain=None, stimulus_prior=None, return_all=False):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        stimulus_prior = self.stimulus_prior if stimulus_prior is None else stimulus_prior
        posterior = self.bayesian_inversion(X, stimulus_domain=stimulus_domain, stimulus_prior=stimulus_prior) # n_trials * n_domain
        y_map = stimulus_domain[np.argmax(posterior, axis=1)] # n_trials
        if self.circular:
            y_mean = math.circular_mean(stimulus_domain, domain=stimulus_domain, weight=posterior, axis=1)
            y_std = math.circular_std(stimulus_domain, domain=stimulus_domain, weight=posterior, axis=1)
        else:
            y_mean = np.sum(stimulus_domain * posterior, axis=1) / np.sum(posterior, axis=1) # n_trials
            y_std = np.sqrt(np.sum((stimulus_domain[np.newaxis,:] - y_mean[:,np.newaxis])**2 * posterior, axis=1) \
                / np.sum(posterior, axis=1)) # n_trials
        return (y_mean, y_std, y_map, posterior) if return_all else y_mean

    def loglikelihood(self, X, y):
        b = X.T
        s = y
        fs = self.basis_func(s)
        z = b - self.W_ @ fs
        return self._calc_L(z, self.W_, self.tau_, self.rho_, self.sigma_)

    def bayesian_inversion(self, X, stimulus_domain=None, stimulus_prior=None, density=True):
        '''
        Parameters
        ----------
        X : 2D array, n_trials * n_voxels
        stimulus_domain : 1D array, n_domain
        stimulus_prior : 2D array, n_trials * n_domain (or 1D array, n_domain)
            None for a flat stimulus prior, same for all trials.

        Returns
        -------
        posterior : 2D array, n_trials * n_domain
        '''
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        stimulus_prior = self.stimulus_prior if stimulus_prior is None else stimulus_prior
        b = X.T[:,np.newaxis,:] # n_voxels * n_domain * n_trials
        fs = self.basis_func(stimulus_domain) # n_channels * n_domain
        predicted_mean_resp = (self.W_ @ fs)[...,np.newaxis] # n_voxels * n_domain * n_trials
        z = b - predicted_mean_resp
        # mv_norm = stats.multivariate_normal(np.zeros(self.Omega.shape[0]), self.Omega)
        mv_norm = stats.multivariate_normal(np.zeros(self.Omega_.shape[0]), self.Omega_, allow_singular=True)
        # likelihood = mv_norm.pdf(z.T) # n_trials * n_domain
        # posterior = likelihood * stimulus_prior # n_trials * n_domain
        # posterior /= np.sum(posterior, axis=1, keepdims=True)
        # The above code will underflow
        loglikelihood = mv_norm.logpdf(z.T) # n_trials * n_domain
        logposterior = loglikelihood + np.log(stimulus_prior) # n_trials * n_domain
        posterior = math.normalize_logP(logposterior, axis=1)
        if density:
            posterior /= stimulus_domain[-1] - stimulus_domain[0]
        return posterior

    def _negloglikelihood(self, params, z, W):
        tau, rho, sigma = params[:-2], params[-2], params[-1]
        return -self._calc_L(z, W, tau, rho, sigma)

    def _negloglikelihood_prime(self, params, z, W):
        tau, rho, sigma = params[:-2], params[-2], params[-1]
        Omega = self._calc_Omega(W, tau, rho, sigma)
        tau_prime = self._dL_dtau(z, Omega, W, tau, rho, sigma)
        rho_prime = self._dL_drho(z, Omega, W, tau, rho, sigma)
        sigma_prime = self._dL_dsigma(z, Omega, W, tau, rho, sigma)
        print(f"{tau_prime[-3:]}, {rho_prime}, {sigma_prime}")
        return -np.r_[tau_prime, rho_prime, sigma_prime]

    def _negloglikelihood_prime_numerical(self, params, z, W, h=1e-6):
        Hs = np.eye(len(params)) * h
        return [(self._negloglikelihood(params+H, z, W) - self._negloglikelihood(params-H, z, W)) / (2*h) for H in Hs]

    def _test_gradient(self, n_channels=6, n_voxels=10, n_trials=7):
        z = np.random.randn(n_voxels, n_trials)
        W = np.random.rand(n_voxels, n_channels)
        tau = np.random.rand(n_voxels)
        rho = 0.5
        sigma = 0.1
        params = np.r_[tau, rho, sigma]
        np.testing.assert_allclose(self._negloglikelihood_prime(params, z, W), 
            self._negloglikelihood_prime_numerical(params, z, W), rtol=1e-6)

    def _test_loglikelihood(self, n_channels=6, n_voxels=10, n_trials=7):
        z = np.random.randn(n_voxels, n_trials)
        W = np.random.rand(n_voxels, n_channels)
        tau = np.random.rand(n_voxels)
        rho = 0.5
        sigma = 0.1
        mv_norm = stats.multivariate_normal(np.zeros(n_voxels), self._calc_Omega(W, tau, rho, sigma))
        np.testing.assert_allclose(self._calc_L(z, W, tau, rho, sigma),
            np.sum(mv_norm.logpdf(z.T)), rtol=1e-6)

    def _calc_Omega(self, W, tau, rho, sigma):
        return (rho + (1-rho)*np.eye(len(tau))) * np.outer(tau, tau) + sigma**2 * W@W.T

    def _calc_L(self, z, W, tau, rho, sigma):
        '''
        L = log(p(b|s; W, Omega))
        z = b - W @ fs
        '''
        Omega = self._calc_Omega(W, tau, rho, sigma)
        M = np.linalg.pinv(Omega) # Although (4x) slower than inv, pinv is preferred in the numerical world (=inv if invertible and well conditioned)
        n_voxels, n_trials = z.shape
        # For a single sample: -0.5 * (z.T @ M @ z + np.log(np.linalg.det(Omega)) + n_voxels*np.log(2*np.pi))
        # May also use (by Gilles): np.sum(stats.multivariate_normal(np.zeros(n_voxels), Omega).logpdf(z.T))
        # This is less numerically robust: -0.5 * (np.trace(z.T @ M @ z) + n_trials*np.log(np.linalg.det(Omega)) + n_trials*n_voxels*np.log(2*np.pi))
        return -0.5 * (np.trace(z.T @ M @ z) + n_trials*np.prod(np.linalg.slogdet(Omega)) + n_trials*n_voxels*np.log(2*np.pi))
        
    def _dL_dOmega(self, z, Omega, chain=False):
        M = np.linalg.pinv(Omega) # Although (4x) slower than inv, pinv is preferred in the numerical world (=inv if invertible and well conditioned)
        n_voxels, n_trials = z.shape
        # deriv = 0.5 * (M.T @ np.outer(z, z) @ M.T - M.T)
        # For a single sample: deriv = 0.5 * (M @ np.outer(z, z) @ M - M) # M.T == M
        deriv = 0.5 * (M @ z @ z.T @ M - n_trials*M)
        if chain:
            return deriv
        else:
            return (2 - np.eye(n_voxels)) * deriv

    def _dL_dtau(self, z, Omega, W, tau, rho, sigma):
        N = len(tau)
        deriv = np.zeros(N)
        for n in range(N):
            e = np.zeros(N)
            e[n] = 1
            dOmega_dtau_n = (rho+(1-rho)*np.eye(N)) * (np.outer(e, tau) + np.outer(tau, e))
            deriv[n] = np.trace(self._dL_dOmega(z, Omega, chain=True) @ dOmega_dtau_n)
        return deriv

    def _dL_drho(self, z, Omega, W, tau, rho, sigma):
        dOmega_drho = (1 - np.eye(len(tau))) * np.outer(tau, tau)
        return np.trace(self._dL_dOmega(z, Omega, chain=True) @ dOmega_drho)

    def _dL_dsigma(self, z, Omega, W, tau, rho, sigma):
        dOmega_dsigma = W@W.T * 2*sigma
        return np.trace(self._dL_dOmega(z, Omega, chain=True) @ dOmega_dsigma)


if __name__ == '__main__':
    pass
