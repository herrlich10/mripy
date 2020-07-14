#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from os import path
from collections import OrderedDict
import numpy as np
from scipy import stats
import pandas as pd
from sklearn import model_selection, metrics
from . import six, utils


def standardize_within_group(X, groups, with_mean=True, with_std=True):
    '''
    This is an extension of the "mean centering" method proposed in [1].
    Can be used as a replacement for the training-set-wise standardization.
    Both with_mean and with_std may provide some extra performance.

    References
    ----------
    [1] Lee, S., & Kable, J. W. (2018). Simple but robust improvement in 
        multivoxel pattern classification. PloS One, 13(11), e0207083.
    '''
    X = X.copy()
    for g in np.unique(groups):
        indexer = (groups==g)
        u = X[indexer].mean(axis=0, keepdims=True)
        s = X[indexer].std(axis=0, keepdims=True) if with_std else 1
        if with_mean:
            X[indexer] = (X[indexer] - u) / s
        else:
            X[indexer] = (X[indexer] - u) / s + u
    return X


def permute_within_group(y, groups):
    y = y.copy()
    for g in np.unique(groups):
        indexer = (groups==g)
        y[indexer] = y[indexer][np.random.permutation(np.sum(indexer))]
    return y


def cross_validate_ext(model, X, y, groups=None, cv=None, pred_kws=None, method=None):
    if cv is None:
        cv = model_selection.LeaveOneGroupOut() # One group of each run
    if method is None:
        method = 'predict'
    pred_kws = dict(dict(), **({} if pred_kws is None else pred_kws))
    res = []
    idx = []
    for train_index, test_index in cv.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        res.append(getattr(model, method)(X_test, **pred_kws))
        idx.extend(test_index)
    sorter = np.argsort(idx)
    if isinstance(res[0], tuple) and len(res[0]) > 1: # predict() has more the one output
        res = tuple([np.concatenate([r[k] for r in res], axis=0)[sorter] for k in range(len(res[0]))])
    else: # predict() has only one output
        res = np.concatenate(res, axis=0)[sorter]
    return res


def cross_validate_with_permutation(model, X, y, groups, rois=None, n_permutations=1000, scoring=None, cv=None):
    if rois is None:
        X, y, groups, rois = [X], [y], [groups], ['NA']
    if cv is None:
        cv = model_selection.LeaveOneGroupOut() # One group of each run
    if scoring is None:
        scoring = {'performance': metrics.make_scorer(metrics.accuracy_score)}
    def cross_validate(X, y, groups, roi, permute):
        if permute:
            y = permute_within_group(y, groups)
        scores = model_selection.cross_validate(model, X, y, groups, \
            scoring=scoring, cv=cv, return_train_score=True, n_jobs=1)
        res = OrderedDict(roi=roi, permute=permute, train=np.mean(scores['train_performance']), 
            test=np.mean(scores['test_performance']))
        return res
    res = []
    for XX, yy, gg, roi in zip(X, y, groups, rois):
        for permute in range(n_permutations+1):
            res.append(cross_validate(XX, yy, gg, roi, permute))
    res = pd.DataFrame(res)
    return res


def compute_critical_value(x, y, permute='permute', data=None, alpha=0.05, tail=2):
    '''
    Get critical values based on permutation distribution, and
    account for multiple comparisons using extreme statistics.

    Parameters
    ----------
    x : str, list of str
        Columns along which multiple comparisons occur (e.g., roi, time).
    y : str
        Column for performance measurement (e.g., test_accuracy, PC, RT).
    data : pd.DataFrame(x, y, permute)
        permute == 0 is originally observed data, >= 1 is permutation data.
    '''
    # Mean performance for each condition and each permutation
    by = [x, permute] if isinstance(x, six.string_types) else list(x) + [permute]
    df = data[data[permute]>0].groupby(by=by)[y].mean() # This is a Series with MultiIndex
    # Globally corrected critical value
    max_dist = df.groupby(permute).max().values # Max distribution
    min_dist = df.groupby(permute).min().values # Min distribution
    if tail == 2:
        gmax = np.percentile(max_dist, (1-alpha/2)*100)
        gmin = np.percentile(min_dist, alpha/2*100)
    else:
        gmax = np.percentile(max_dist, (1-alpha)*100)
        gmin = np.percentile(min_dist, alpha*100)
    # Per-comparison (uncorrected) critical value
    if tail == 2:
        pmax = df.groupby(x).quantile(1-alpha/2)
        pmin = df.groupby(x).quantile(alpha/2)
    else:
        pmax = df.groupby(x).quantile(1-alpha)
        pmin = df.groupby(x).quantile(alpha)
    bounds = pd.concat([pmin, pmax], axis=1)
    bounds.columns = ['lower', 'upper']
    bounds = pd.concat([pd.DataFrame([{'lower': gmin, 'upper': gmax}], index=['overall']), bounds], axis=0)
    # Determine significance
    obs = data[data[permute]==0]
    if obs.size > 0: # Contain originally observed data
        bounds['obs_mean'] = pd.concat([pd.Series([np.nan], ['overall']), obs.groupby(by=x)[y].mean()], axis=0) # Mean response
        bounds['obs_std'] = pd.concat([pd.Series([np.nan], ['overall']), obs.groupby(by=x)[y].std()], axis=0)
        bounds['obs_n'] = pd.concat([pd.Series([-1], ['overall']), obs.groupby(by=x)[y].count()], axis=0)
        n_comparisons = len(obs[x].unique())
        if tail == 2: # The two-tailed p value is twice the one-tailed p value (assuming you correctly predicted the direction of the difference)
            bounds['corrected'] = (bounds.obs_mean < bounds.lower['overall']) | (bounds.upper['overall'] < bounds.obs_mean) # Significance (corrected)
            bounds['uncorrected'] = (bounds.obs_mean < bounds.lower) | (bounds.upper < bounds.obs_mean) # Significance (uncorrected)
        elif tail == 1:
            bounds['corrected'] = (bounds.upper['overall'] < bounds.obs_mean)
            bounds['p_corr'] = [np.nan] + [1-stats.percentileofscore(max_dist, v)/100 for v in bounds.obs_mean[1:]]
            bounds['uncorrected'] = (bounds.upper < bounds.obs_mean)
            bounds['p_uncorr'] = [np.nan] + [1-stats.percentileofscore(df[k], v)/100 for k, v in bounds.obs_mean[1:].iteritems()]
            bounds['bonferroni'] = bounds['p_uncorr'] * n_comparisons
        elif tail == -1:
            bounds['corrected'] = (bounds.obs_mean < bounds.lower['overall'])
            bounds['uncorrected'] = (bounds.obs_mean < bounds.lower)
    return bounds


# def plot_permutation(x, y, subject='subject', permute='permute', data=None, plot=None, 
#     color=None, x_order=None, xtick_format=None, chance=0, alpha=0.05, tail=2, mcc='extreme', 
#     figsize=None, star_shift=None, star_alpha=None,
#     dist_kws=None, scatter_kws=None, line_kws=None, bar_kws=None, point_kws=None, chance_kws=None, ref_kws=None, 
#     show_mcc=True, show_n=True, show_p=False, show_ref=False, show_num=False):
#     '''
#     Parameters
#     ----------
#     x : str
#     y : str
#     data : pd.DataFrame(x, y, permute)
#         permute == 0 is originally observed data, >= 1 is permutation data.
#     '''
#     if figsize is not None:
#         fig = plt.figure(figsize=figsize)
#     else:
#         fig = plt.gcf()
#     pmt = data[data[permute]>0]
#     has_pmt = (pmt.size > 0)
#     obs = data[data[permute]==0]
#     if x_order is None:
#         x_order = data[x].unique()
#     else:
#         x_order = [x_label for x_label in x_order if x_label in data[x].values]
#     if has_pmt:
#         bounds = compute_critical_value(x=x, y=y, data=data, alpha=alpha, tail=tail)
#         df_pmt = pmt.groupby(by=[x, permute])[y].mean().reset_index()
#     df_obs = obs.groupby(by=[x, subject])[y].mean().reset_index()
#     obs_mean = df_obs.groupby(x)[y].mean()
#     obs_n = df_obs.groupby(x)[y].count()
#     x_loc = np.arange(len(df_obs[x].unique()))
#     if plot is None:
#         plot = 'violinplot' if has_pmt else 'barplot'
#     if plot == 'violinplot':
#         # Plot permutation distribution
#         dist_kws = dict(dict(color='gray', inner=None, linewidth=0), **(dist_kws if dist_kws is not None else {}))
#         sns.violinplot(x=x, y=y, data=df_pmt, order=x_order, **dist_kws)
#         # Plot originally observed data
#         scatter_kws = dict(dict(color=color, s=100, linewidths=1, edgecolors='k'), **(scatter_kws if scatter_kws is not None else {}))
#         plt.scatter(np.arange(len(x_order)), bounds.loc[x_order,'obs_mean'], **scatter_kws)
#     elif plot == 'lineplot':
#         line_kws = dict(dict(), **(line_kws if line_kws is not None else {}))
#         sns.lineplot(x=x, y=y, data=df_obs, ci=(1-alpha)*100, palette=color, **line_kws)
#         x_loc = df_obs[x].unique()
#     elif plot == 'barplot':
#         color = 'gray' if color is None else color
#         bar_kws = dict(dict(), **(bar_kws if bar_kws is not None else {}))
#         sns.barplot(x=x, y=y, data=df_obs, order=x_order, ci=(1-alpha)*100, color=color, **bar_kws)
#     elif plot == 'finalplot':
#         # Plot permutation distribution
#         dist_kws = dict(dict(color='gray', alpha=0.5, inner=None, linewidth=0), **(dist_kws if dist_kws is not None else {}))
#         sns.violinplot(x=x, y=y, data=df_pmt, order=x_order, **dist_kws)
#         # Plot bootstrap errorbars
#         color = 'k' if color is None else color
#         point_kws = dict(dict(linestyles='', scale=0.5, errwidth=2, capsize=0.1, facecolors='r'), **(point_kws if point_kws is not None else {}))
#         sns.pointplot(x=x, y=y, data=df_obs, order=x_order, ci=(1-alpha)*100, color=color, **point_kws)
#         # Plot originally observed data
#         scatter_kws = dict(dict(s=50, marker='o', linewidths=1, edgecolors=color, zorder=10), **(scatter_kws if scatter_kws is not None else {}))
#         plt.scatter(np.arange(len(x_order)), bounds.loc[x_order,'obs_mean'], **scatter_kws)
#     # Shift long ticklabels
#     if xtick_format is None:
#         xtick_format = ('normal' if plot == 'lineplot' else 'rotated')
#     if xtick_format == 'rotated':
#         plt.setp(plt.gca().get_xticklabels(), rotation=-30)
#         dx = 20/72; dy = 0/72 
#         offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
#         for label in plt.gca().xaxis.get_majorticklabels():
#             label.set_transform(label.get_transform() + offset)
#     elif xtick_format == 'short':
#         plt.setp(plt.gca(), xticklabels=[label.get_text().split('_')[0] for label in plt.gca().get_xticklabels()])
#     elif xtick_format == 'normal':
#         pass
#     elif xtick_format == 'final':
#         plt.setp(plt.gca(), xticklabels=[label.get_text().split('_')[0] for label in plt.gca().get_xticklabels()])
#         plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
#         dx = 15/72; dy = 5/72 
#         offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
#         for label in plt.gca().xaxis.get_majorticklabels():
#             label.set_transform(label.get_transform() + offset)
#     # Plot chance level
#     chance_kws = dict(dict(color='C3', ls='--', zorder=1), **(chance_kws if chance_kws is not None else {}))
#     plt.axhline(chance, **chance_kws)
#     # Plot reference line
#     if show_ref:
#         ref_kws = dict(dict(ref=0.55, color='gray', lw=0.5, ls='--'), **(ref_kws if ref_kws is not None else {}))
#         ref = ref_kws.pop('ref')
#         plt.axhline(ref, **ref_kws)
#     if star_alpha is None:
#         star_alpha = [0.3, 1]
#     if has_pmt:
#         # Plot multiple comparison correction band
#         if show_mcc:
#             plt.axhspan(bounds.loc['overall','lower'] if tail==2 else chance, bounds.loc['overall','upper'], color='r', alpha=0.1)
#         # Plot significant stars
#         if star_shift is None:
#             star_shift = bounds.ix[1:,'obs_std']/np.sqrt(bounds.ix[1:,'obs_n']) * 2.2 # Ignore first row
#         else:
#             star_shift = pd.Series(star_shift, index=bounds.index)
#         for k, x_label in enumerate(x_order):
#             if bounds.loc[x_label,'uncorrected']:
#                 plt.text(x_loc[k], bounds.loc[x_label,'obs_mean']+star_shift[x_label], '*', ha='center', alpha=star_alpha[0])
#             if bounds.loc[x_label,'corrected']:
#                 plt.text(x_loc[k], bounds.loc[x_label,'obs_mean']+star_shift[x_label], '*', ha='center', alpha=star_alpha[1])
#             if show_p:
#                 if mcc == 'none':
#                     plt.text(x_loc[k], 0.15, f"{bounds.loc[x_label,'p_uncorr']:.3f}", transform=myplot.transHDVA(), ha='center', alpha=star_alpha[0], fontsize='xx-small')
#                 if mcc == 'extreme':
#                     plt.text(x_loc[k], 0.15, f"{bounds.loc[x_label,'p_corr']:.3f}", transform=myplot.transHDVA(), ha='center', alpha=star_alpha[0], fontsize='xx-small')
#                 elif mcc == 'bonferroni':
#                     plt.text(x_loc[k], 0.15, f"{bounds.loc[x_label,'bonferroni']:.3f}", transform=myplot.transHDVA(), ha='center', alpha=star_alpha[1], fontsize='xx-small')
#     # Plot performance
#     if show_num:
#         for k, x_label in enumerate(x_order):
#             plt.text(x_loc[k], bounds.loc['overall','upper']*1.1-0.05 if has_pmt else 0.9, f"{obs_mean[x_label]:.3f}", 
#                 transform=plt.gca().transData if has_pmt else myplot.transHDVA(), ha='center', alpha=star_alpha[0], fontsize='xx-small')
#     # Plot obs_n
#     if show_n:
#         if len(set(obs_n[x_order].values)) == 1: # All equal
#             plt.text(0.95, 0.05, f"$n={obs_n[x_order[0]]}$", transform=plt.gca().transAxes, ha='right', fontsize='x-small')
#         else:
#             for k, x_label in enumerate(x_order):
#                 plt.text(x_loc[k], 0.05, f"$n={obs_n[x_label]}$" if k == 0 else f"${obs_n[x_label]}$", 
#                     transform=myplot.transHDVA(), ha='center', fontsize='x-small')


if __name__ == '__main__':
    pass
