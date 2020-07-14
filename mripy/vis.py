#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import glob
import numpy as np
from matplotlib import pyplot as plt, transforms
# from matplotlib import transforms
from . import six, io


def get_color_list(cmap):
    '''
    This function is still wrong!
    '''
    r = np.array(cmap._segmentdata['red'])[:,1]
    g = np.array(cmap._segmentdata['green'])[:,1]
    b = np.array(cmap._segmentdata['blue'])[:,1]
    return list(zip(r, g, b))


def draw_color_circle(cmap, res=512):
    '''
    Draw circular colorbar (color index from 0 to 1) clockwise from 0 to 12 o'clock
    '''
    X, Y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    D = np.sqrt(X**2 + Y**2)
    A = np.rot90((-np.arctan2(-Y, X)+np.pi)/(2*np.pi), 3)
    A[(D<0.4)|(D>1)] = np.nan
    if isinstance(cmap, six.string_types):
        cmap = plt.get_cmap(cmap)
    cmap.set_bad('gray', alpha=1)
    plt.imshow(A, cmap=cmap)


def plot_volreg(dfiles, convention=None):
    '''
    The six columns from left to right are:
    - For 3dvolreg: roll(z), pitch(x), yaw(y), dS(z), dL(x), dP(y)
        Note that the interpretation for roll and yaw in afni is opposite from SPM (and common sense...)
        This program will follow the SPM convention.
    - For 3dAllineate shift_rotate: x-shift  y-shift  z-shift$ z-angle  x-angle$ y-angle$
        Note that the dollar signs in the end indicate parameters that are fixed.
    '''
    files = glob.glob(dfiles) if isinstance(dfiles, six.string_types) else dfiles
    labels = [r'$\Delta$R-L(x) [mm]', r'$\Delta$A-P(y) [mm]', r'$\Delta$I-S(z) [mm]',
              r'Pitch(x) [$^\circ$]', r'Roll(y) [$^\circ$]', r'Yaw(z) [$^\circ$]']
    colors = ['C3', 'C2', 'C0', 'C3', 'C2', 'C0'] # xyz -> RGB
    vrs, comments = zip(*[io.read_txt(f, return_comments=True) for f in files])
    vr = np.vstack(vrs)
    n_runs = len(vrs)
    n_TRs = [vr.shape[0] for vr in vrs]
    borders = np.r_[0, np.cumsum(n_TRs)]
    try:
        base_idx = np.nonzero(np.linalg.norm(vr, axis=1)==0)[0][0]
    except IndexError:
        base_idx = None
    if convention is None:
        if comments[0] and comments[0][0].startswith('# 3dAllineate'):
            convention = '3dAllineate'
        else:
            convention = '3dvolreg'
    if convention == '3dAllineate': # 3dAllineate shift_rotate
        col_order = [0, 1, 2, 4, 5, 3]
        col_sign = np.r_[1, 1, 1, 1, 1, 1]
    elif convention == '3dvolreg': # 3dvolreg
        col_order = [4, 5, 3, 1, 2, 0]
        col_sign = np.r_[-1, -1, -1, 1, 1, 1]
    vr = vr[:,col_order] * col_sign

    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, right=0.85)
    for k, ax in enumerate(axs):
        plt.sca(ax)
        plt.plot(vr[:,k], color=colors[k], lw=2)
        plt.axhline(0, color='gray', ls='--', lw=0.5)
        plt.text(1.01, 0.5, labels[k], color=colors[k], transform=ax.transAxes, va='center', fontsize='small')
        if base_idx is not None:
            plt.axvline(base_idx, color='gray', ls='--', lw=0.5)
        for run_idx in range(n_runs):
            if run_idx % 2:
                plt.axvspan(borders[run_idx], borders[run_idx+1], color='C1', alpha=0.2)
    transHDVA = transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)
    for run_idx in range(n_runs):
        axs[0].text((borders[run_idx]+borders[run_idx+1])/2, 1.05, '{0}'.format(run_idx+1), color='gray', transform=transHDVA, ha='center', fontsize='small')
    plt.sca(axs[-1])
    plt.xlabel('#TR')
    return vr
