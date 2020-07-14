#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import time
import numpy as np
from scipy import optimize, stats
from deepdish import io as dio
from . import utils, math


def basis_vanBergen2015(s, n_channels=8, power=5):
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
    fs = np.maximum(0, np.cos(2 * (s[np.newaxis,:] - phi[:,np.newaxis])))**power # n_channels * n_trials
    return fs


def basis_Sprague2013(s, n_channels=6, spacing=2, center=None, size=None, power=7, dim=1, intercept=False):
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
    if center is None:
        center = 0
    centers = (np.arange(n_channels) - (n_channels-1)/2) * spacing + center
    if dim == 1:
        r = np.abs(s[np.newaxis,:] - centers[:,np.newaxis]) # Distance from filter’s center
    elif dim == 2:
        X, Y = np.meshgrid(centers, centers)
        centers = np.c_[X.ravel(), Y.ravel()] # 2D channel array is serialized in row-first order
        r = np.linalg.norm(s.T[np.newaxis,...] - centers[...,np.newaxis], axis=1)
    fs = np.where(r<size, (0.5*np.cos(r/size*np.pi) + 0.5)**power, 0) # n_channels * n_trials
    if intercept:
        fs = np.vstack([fs, np.ones(len(s))]) # Learnable bias
    return fs


class BaseModel(utils.Savable2):
    # get_params() is required by sklearn
    def get_params(self, deep=True):
        raise NotImplementedError('** You must implement this method by yourself in your model.') # Abstrast method

    # set_params() is required by sklearn
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k.endswith('_')}

    def from_dict(self, d):
        self.__dict__.update(d)
        return self


class ChannelEncodingModel(BaseModel):
    def __init__(self, n_channels, basis_func, stimulus_domain, circular=False, verbose=2):
        '''
        After (Brouwer et al., 2009).
        '''
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.stimulus_domain = stimulus_domain
        self.circular = circular
        self.verbose = verbose

    # get_params() is required by sklearn
    def get_params(self, deep=True):
        return dict(n_channels=self.n_channels, basis_func=self.basis_func, 
            stimulus_domain=self.stimulus_domain, circular=self.circular)

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
        W = b @ fs.T @ math.pinv(fs @ fs.T) # Weight, n_voxels * n_channels
        # Store params
        self.W_ = W
        return self # Required by sklearn

    def predict(self, X, stimulus_domain=None, return_all=False):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        channel_resp = self.inverted_encoding(X) # n_trials * n_channels
        evidence = self.correlation_inversion(X, stimulus_domain=None) # n_trials * n_domain
        # y_map = stimulus_domain[np.argmax(evidence, axis=1)] # n_trials
        y_map = stimulus_domain[math.median_argmax(evidence, axis=1)] # n_trials
        # y_mean = np.sum(stimulus_domain * evidence, axis=1) / np.sum(evidence, axis=1) # n_trials
        # y_std = np.sqrt(np.sum((stimulus_domain[np.newaxis,:] - y_mean[:,np.newaxis])**2 * evidence, axis=1) \
        #     / np.sum(evidence, axis=1)) # n_trials
        # return (y_mean, y_std, y_map, evidence, channel_resp) if return_all else y_mean
        return (y_map, evidence, channel_resp) if return_all else y_map

    _pidx = 1 # Index of posterior (for MAP) if return_all

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
                M = fs.T @ math.pinv(fs @ fs.T + lambda_ * np.eye(self.n_channels)) # n_trials * n_channels
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
        fs = math.pinv(self.W_.T @ self.W_) @ self.W_.T @ b # Inverted channel response, n_channels * n_trials
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


class BayesianChannelModel(BaseModel):
    def __init__(self, n_channels='required', basis_func='required', stimulus_domain='required', circular=False, stimulus_prior=None, global_search=False, verbose=2):
        '''
        After (van Bergen et al., 2015).

        Examples
        --------
        from mripy import encoding
        from sklearn import model_selection, preprocessing, pipeline

        stimulus_domain = np.linspace(0, pi, 181)
        n_channels = 8
        basis_func = lambda s: encoding.basis_vanBergen2015(s, n_channels=n_channels)
        model = encoding.BayesianChannelModel(n_channels=n_channels, 
            basis_func=basis_func, stimulus_domain=stimulus_domain, circular=True)
        model = pipeline.make_pipeline(preprocessing.StandardScaler(), model)
        cv = model_selection.LeaveOneGroupOut() # One group of each run
        y_hat = model_selection.cross_val_predict(model, X, y, groups, cv=cv, n_jobs=1)
        '''
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.stimulus_domain = stimulus_domain
        self.circular = circular
        self.stimulus_prior = 1 if stimulus_prior is None else stimulus_prior # TODO: This should be refactored for CV
        self.global_search = global_search
        self.verbose = verbose

    # get_params() is required by sklearn
    def get_params(self, deep=True):
        return dict(n_channels=self.n_channels, basis_func=self.basis_func, stimulus_domain=self.stimulus_domain,
            circular=self.circular, stimulus_prior=self.stimulus_prior, global_search=self.global_search)

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
        W = b @ fs.T @ math.pinv(fs @ fs.T) # Weight, n_voxels * n_channels
        # Store params
        self.W_ = W
        # Step 2: Estimate tau, rho, sigma by ML optimization (gradient-based)
        z = b - W @ fs # n_voxels * n_trials
        # Initial params
        tau0 = np.std(b, axis=1)
        rho0 = np.mean(np.corrcoef(b)[np.triu_indices(len(tau0), k=1)])
        sigma0 = np.mean(np.std(fs, axis=1))
        params0 = np.r_[tau0, rho0, sigma0/5.0]
        # bounds = np.c_[np.ones(len(params0))*1e-4, np.r_[tau0*5, 1, sigma0*5]]
        bounds = np.c_[np.ones(len(params0))*1e-3, np.r_[tau0*5, 0.99, sigma0*5]]
        # Conjugate gradient algorithm, due to lack of support for bounds, requires multi-start to avoid/alleviate being trapped in local minima.
        if self.verbose > 0:
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
                    # cost = self.model._negloglikelihood(xk, *self.args) # The pinv->svd here is too expensive...
                    curr_time = time.time()
                    duration = curr_time - self.last_time
                    self.last_time = curr_time
                    if self.model.verbose > 1:
                        # print(f"iter#{self.count:03d} ({utils.format_duration(duration)}): cost={cost:.4f}, tau[-3:]={xk[-5:-2]}, rho={xk[-2]:.4f}, sigma={xk[-1]:.4f}")
                        print(f"iter#{self.count:03d} ({utils.format_duration(duration)}): tau[-3:]={xk[-5:-2]}, rho={xk[-2]:.4f}, sigma={xk[-1]:.4f}")
                    elif self.model.verbose == 1:
                        print(f"iter#{self.count:03d} ({utils.format_duration(duration)}): tau[-3:]={xk[-5:-2]}, rho={xk[-2]:.4f}, sigma={xk[-1]:.4f}", end='\r')
            # res = optimize.minimize(self._negloglikelihood, params0, args=(z, W), method='L-BFGS-B', 
            #     jac=self._negloglikelihood_prime, bounds=bounds, callback=Counter(model=self, args=(z, W)).step)
            res = optimize.minimize(self._negloglikelihood, params0, args=(z, W, True), method='L-BFGS-B', 
                jac=True, bounds=bounds, callback=Counter(model=self, args=(z, W)).step)
        params = res.x
        if self.verbose > 0:
            print(f"cost={res.fun}, iter={res.nit}, func_eval={res.nfev}, success={res.success}, {res.message}")
            print(params0[-3:], '-->', params[-3:])
        # Store params
        self.tau_, self.rho_, self.sigma_ = params[:-2], params[-2], params[-1]
        self._Omega = self._calc_Omega(self.W_, self.tau_, self.rho_, self.sigma_)
        self._Omega_inv = math.pinv(self.Omega_) # Update cache
        return self # Required by sklearn

    # Cache backed properties (the "if else" construct is to prevent unnecessary expression evaluation)
    Omega_ = property(lambda self: self.__dict__.setdefault('_Omega', None if hasattr(self, '_Omega') else self._calc_Omega(self.W_, self.tau_, self.rho_, self.sigma_)))
    Omega_inv_ = property(lambda self: self.__dict__.setdefault('_Omega_inv', None if hasattr(self, '_Omega_inv') else math.pinv(self.Omega_)))

    def predict(self, X, stimulus_domain=None, stimulus_prior=None, return_all=False):
        stimulus_domain = self.stimulus_domain if stimulus_domain is None else stimulus_domain
        stimulus_prior = self.stimulus_prior if stimulus_prior is None else stimulus_prior
        posterior = self.bayesian_inversion(X, stimulus_domain=stimulus_domain, stimulus_prior=stimulus_prior) # n_trials * n_domain
        y_map = stimulus_domain[np.argmax(posterior, axis=1)] # n_trials
        # y_map = stimulus_domain[math.median_argmax(posterior, axis=1)] # n_trials
        if self.circular:
            y_mean = math.circular_mean(stimulus_domain, domain=stimulus_domain, weight=posterior, axis=1)
            y_std = math.circular_std(stimulus_domain, domain=stimulus_domain, weight=posterior, axis=1)
        else:
            y_mean = np.sum(stimulus_domain * posterior, axis=1) / np.sum(posterior, axis=1) # n_trials
            y_std = np.sqrt(np.sum((stimulus_domain[np.newaxis,:] - y_mean[:,np.newaxis])**2 * posterior, axis=1) \
                / np.sum(posterior, axis=1)) # n_trials
        return (y_mean, y_std, y_map, posterior) if return_all else y_mean

    _pidx = 3 # Index of posterior (for MAP) if return_all

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
        # # mv_norm = stats.multivariate_normal(np.zeros(self.Omega_.shape[0]), self.Omega_)
        # mv_norm = stats.multivariate_normal(np.zeros(self.Omega_.shape[0]), self.Omega_, allow_singular=True)
        # # likelihood = mv_norm.pdf(z.T) # n_trials * n_domain
        # # posterior = likelihood * stimulus_prior # n_trials * n_domain
        # # posterior /= np.sum(posterior, axis=1, keepdims=True)
        # # The above code will underflow
        # loglikelihood = mv_norm.logpdf(z.T) # n_trials * n_domain
        # The following implementation is 10x faster than stats.multivariate_normal:D
        loglikelihood = math.gaussian_logpdf(z.T, np.zeros(self.Omega_.shape[0]), self.Omega_, cov_inv=self.Omega_inv_) # n_trials * n_domain
        logposterior = loglikelihood + np.log(stimulus_prior) # n_trials * n_domain
        posterior = math.normalize_logP(logposterior, axis=1)
        if density:
            posterior /= stimulus_domain[-1] - stimulus_domain[0]
        return posterior

    def _negloglikelihood(self, params, z, W, return_prime=False):
        tau, rho, sigma = params[:-2], params[-2], params[-1]
        if not return_prime:
            return -self._calc_L(z, W, tau, rho, sigma)
        else:
            Omega = self._calc_Omega(W, tau, rho, sigma)
            Omega_inv = math.pinv(Omega) # np.linalg.pinv() may encounter "LinAlgError: SVD did not converge" for some matrices
            L = -self._calc_L(z, W, tau, rho, sigma, Omega=Omega, Omega_inv=Omega_inv)
            L_prime = self._negloglikelihood_prime(params, z, W, Omega=Omega, Omega_inv=Omega_inv)
            return L, L_prime

    def _negloglikelihood_prime(self, params, z, W, Omega=None, Omega_inv=None):
        tau, rho, sigma = params[:-2], params[-2], params[-1]
        Omega = self._calc_Omega(W, tau, rho, sigma) if Omega is None else Omega
        dL_dOmega = self._dL_dOmega(z, Omega, chain=True, Omega_inv=Omega_inv)
        tau_prime = self._dL_dtau(z, Omega, W, tau, rho, sigma, dL_dOmega=dL_dOmega)
        rho_prime = self._dL_drho(z, Omega, W, tau, rho, sigma, dL_dOmega=dL_dOmega)
        sigma_prime = self._dL_dsigma(z, Omega, W, tau, rho, sigma, dL_dOmega=dL_dOmega)
        # print(f"{tau_prime[-3:]}, {rho_prime}, {sigma_prime}")
        return -np.r_[tau_prime, rho_prime, sigma_prime]

    def _negloglikelihood_prime_numerical(self, params, z, W, h=1e-6):
        Hs = np.eye(len(params)) * h
        return np.array([(self._negloglikelihood(params+H, z, W) - self._negloglikelihood(params-H, z, W)) / (2*h) for H in Hs])

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

    def _calc_L(self, z, W, tau, rho, sigma, Omega=None, Omega_inv=None):
        '''
        L = log(p(b|s; W, Omega))
        z = b - W @ fs
        '''
        Omega = self._calc_Omega(W, tau, rho, sigma) if Omega is None else Omega
        M = math.pinv(Omega) if Omega_inv is None else Omega_inv # Although (4x) slower than inv, pinv is preferred in the numerical world (=inv if invertible and well conditioned)
        n_voxels, n_trials = z.shape
        # For a single sample: -0.5 * (z.T @ M @ z + np.log(np.linalg.det(Omega)) + n_voxels*np.log(2*np.pi))
        # May also use (by Gilles): np.sum(stats.multivariate_normal(np.zeros(n_voxels), Omega).logpdf(z.T))
        # This is less numerically robust: -0.5 * (np.trace(z.T @ M @ z) + n_trials*np.log(np.linalg.det(Omega)) + n_trials*n_voxels*np.log(2*np.pi))
        # return -0.5 * (np.trace(z.T @ M @ z) + n_trials*np.prod(np.linalg.slogdet(Omega)) + n_trials*n_voxels*np.log(2*np.pi))
        return -0.5 * ((z * (M @ z)).sum() + n_trials*np.prod(np.linalg.slogdet(Omega)) + n_trials*n_voxels*np.log(2*np.pi))
        
    def _dL_dOmega(self, z, Omega, chain=False, Omega_inv=None):
        M = math.pinv(Omega) if Omega_inv is None else Omega_inv # Although (4x) slower than inv, pinv is preferred in the numerical world (=inv if invertible and well conditioned)
        n_voxels, n_trials = z.shape
        # deriv = 0.5 * (M.T @ np.outer(z, z) @ M.T - M.T)
        # For a single sample: deriv = 0.5 * (M @ np.outer(z, z) @ M - M) # M.T == M
        deriv = 0.5 * (M @ z @ z.T @ M - n_trials*M)
        if chain:
            return deriv
        else:
            return (2 - np.eye(n_voxels)) * deriv

    def _dL_dtau(self, z, Omega, W, tau, rho, sigma, dL_dOmega=None):
        N = len(tau)
        dL_dOmega = self._dL_dOmega(z, Omega, chain=True) if dL_dOmega is None else dL_dOmega
        # deriv = np.zeros(N)
        # for n in range(N):
        #     e = np.zeros(N)
        #     e[n] = 1
        #     dOmega_dtau_n = (rho+(1-rho)*np.eye(N)) * (np.outer(e, tau) + np.outer(tau, e))
        #     # deriv[n] = np.trace(dL_dOmega @ dOmega_dtau_n)
        #     # The above expression is wasteful, trace(A@B) == sum(A*B.T), and Einstein summation is even faster (but numerically less stable???)
        #     deriv[n] = np.einsum('ij,ji->', dL_dOmega, dOmega_dtau_n)
        # This function is the bottleneck after cProfile.run()
        # https://stackoverflow.com/questions/18854425/what-is-the-best-way-to-compute-the-trace-of-a-matrix-product-in-numpy
        A = dL_dOmega * (rho+(1-rho)*np.eye(N)).T
        # deriv = (A*tau).sum(axis=1) + (A*tau[:,np.newaxis]).sum(axis=0)
        deriv = 2 * (A*tau).sum(axis=1) # A should be symmetric matrix
        return deriv

    def _dL_drho(self, z, Omega, W, tau, rho, sigma, dL_dOmega=None):
        dL_dOmega = self._dL_dOmega(z, Omega, chain=True) if dL_dOmega is None else dL_dOmega
        dOmega_drho = (1 - np.eye(len(tau))) * np.outer(tau, tau)
        # return np.trace(dL_dOmega @ dOmega_drho)
        return (dL_dOmega * dOmega_drho.T).sum()

    def _dL_dsigma(self, z, Omega, W, tau, rho, sigma, dL_dOmega=None):
        dL_dOmega = self._dL_dOmega(z, Omega, chain=True) if dL_dOmega is None else dL_dOmega
        dOmega_dsigma = W@W.T * 2*sigma
        # return np.trace(dL_dOmega @ dOmega_dsigma)
        return (dL_dOmega * dOmega_dsigma.T).sum()


class EnsembleModel(BaseModel):
    def __init__(self, n_ensemble=10, base_model='required', pred_method=None, pred_options=None):
        # Cannot use 1) **kwargs; 2) class as argument. Use instance instead (__class__ + get_params).
        # Otherwise you may get the misleading "TypeError: get_params() missing 1 required positional argument: 'self'".
        # Also cannot modify any argument, otherwise sklearn's clone() method will complain during cross-validation.
        self.n_ensemble = n_ensemble
        self.base_model = base_model
        self.pred_method = pred_method
        self.pred_options = pred_options

    # get_params() is required by sklearn
    def get_params(self, deep=True):
        return dict(n_ensemble=self.n_ensemble, base_model=self.base_model,
            pred_method=self.pred_method, pred_options=self.pred_options)

    def fit(self, X, y):
        # Perform argument validation here (as recommended by sklearn) so that get_params() and __init__() have the same effect 
        # This is refactored so that load() can work without fit()
        self._set_default_params()
        self.models_ = [self.base_model.__class__(**self.base_model.get_params()).fit(X[:,k::self.n_ensemble], y) for k in range(self.n_ensemble)]
        return self # Required by sklearn

    def predict(self, X, method=None, options=None, return_all=False, pred_kws=None):
        method = self.pred_method if method is None else method
        options = self.pred_options if options is None else options
        pred_kws = dict(dict(return_all=(True if method in ['map'] else False)), **({} if pred_kws is None else pred_kws))
        preds = [model.predict(X[:,k::self.n_ensemble], **pred_kws) for k, model in enumerate(self.models_)]
        if method == 'mean':
            y_hat = np.mean([pred[0] if isinstance(pred, tuple) else pred for pred in preds], axis=0)
        elif method == 'map':
            stimulus_domain = pred_kws['stimulus_domain'] if 'stimulus_domain' in pred_kws else self.base_model.get_params()['stimulus_domain']
            posterior = np.mean([prep[options['pidx']] for prep in preds], axis=0)
            y_hat = stimulus_domain[np.argmax(posterior, axis=1)]
        return (y_hat, preds) if return_all else y_hat

    def _set_default_params(self):
        if self.pred_method is None:
            self.pred_method = 'map' if hasattr(self.base_model, '_pidx') else 'mean' 
        self.pred_options = dict(dict(pidx=(self.base_model._pidx if hasattr(self.base_model, '_pidx') else -1)), 
            **({} if self.pred_options is None else self.pred_options))

    def to_dict(self):
        d = super().to_dict()
        d['models_'] = [model.to_dict() for model in d['models_']]
        return d

    def from_dict(self, d):
        self._set_default_params()
        d['models_'] = [self.base_model.__class__(**self.base_model.get_params()).from_dict(model) for model in d['models_']]
        self.__dict__.update(d)
        return self


def shift_distribution(d, stimulus_domain, center_on=None, circular=True):
    '''
    Parameters
    ----------
    d : n_trials * n_domain
    '''
    if center_on is not None:
        tgt_idx = len(stimulus_domain)//2
        src_idx = np.argmin(np.abs(center_on[:,np.newaxis] - stimulus_domain[np.newaxis,:]), axis=-1)
        shift_idx = tgt_idx - src_idx
    shifted = np.zeros_like(d)
    for k, dd in enumerate(d):
        dd = np.roll(dd, shift_idx[k])
        if not circular:
            if shift_idx[k] >= 0:
                dd[:shift_idx[k]] = np.nan
            else:
                dd[shift_idx[k]:] = np.nan
        shifted[k,:] = dd
    return shifted


def discretize_prediction(y_pred, targets, circular_domain=None):
    '''
    Discretize continous prediction to the nearest target.
    Can handle irregular target grid and also circular domain.

    E.g.,
    y_pred = encoding.discretize_prediction(y_hat, arange(8)/8*pi, circular_domain=[0, pi])
    correct = encoding.circular_correct(y_true, y_hat, domain=[0, pi], n_targets=8)
    assert(allclose(mean(y_pred==y_true), mean(correct)))
    '''
    if circular_domain: # Circular domain
        D = circular_domain[-1] - circular_domain[0] # Domain size
        augmented = np.r_[targets, D+targets[0]]
    else: # Non circular domain
        augmented = targets
    idx = np.argmin(np.abs(y_pred[:,np.newaxis] - augmented[np.newaxis,:]), axis=-1) % len(targets)
    return targets[idx]


def circular_correct(y_true, y_pred, domain=None, n_targets=None, tolerance=None, return_dist=False):
    if domain is None:
        domain = [0, 2*np.pi]
    D = domain[-1] - domain[0] # Domain size
    if n_targets is None:
        if tolerance is None:
            raise ValueError('You must provide either "n_targets" or "tolerance".')
        else:
            d = tolerance
    else:
        d = D / n_targets / 2
    dist = np.abs(y_pred - y_true)
    dist = np.minimum(dist, D-dist)
    correct = (dist < d)
    return (correct, dist) if return_dist else correct


if __name__ == '__main__':
    pass
