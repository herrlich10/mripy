#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import glob
import numpy as np
from matplotlib import pyplot as plt, transforms
# from matplotlib import transforms
from . import six


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

def plot_volreg(dfiles):
    '''
    The six columns from left to right are: roll, pitch, yaw, dI-S, dR-L, dA-P
    '''
    files = glob.glob(dfiles) if isinstance(dfiles, six.string_types) else dfiles
    labels = [r'Roll [$^\circ$]', r'Pitch [$^\circ$]', r'Yaw [$^\circ$]', r'$\Delta$I-S [mm]', r'$\Delta$R-L [mm]', r'$\Delta$A-P [mm]']
    colors = ['C2', 'C3', 'C0', 'C0', 'C3', 'C2']
    vrs = [np.loadtxt(f) for f in files]
    n_runs = len(vrs)
    n_TRs = [vr.shape[0] for vr in vrs]
    borders = np.r_[0, np.cumsum(n_TRs)]
    vr = np.vstack(vrs)
    base = np.nonzero(np.linalg.norm(vr, axis=1)==0)[0][0]

    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0)
    for ax, k in zip(axs, [4, 5, 3, 1, 0, 2]):
        plt.sca(ax)
        plt.plot(vr[:,k], color=colors[k], lw=2)
        plt.axhline(0, color='gray', ls='--', lw=0.5)
        plt.text(1.01, 0.5, labels[k], color=colors[k], transform=ax.transAxes, va='center', fontsize='small')
        plt.axvline(base, color='gray', ls='--', lw=0.5)
        for r_idx in range(n_runs):
            if r_idx % 2:
                plt.axvspan(borders[r_idx], borders[r_idx+1], color='C1', alpha=0.2)
    transHDVA = transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes)
    for r_idx in range(n_runs):
        axs[0].text((borders[r_idx]+borders[r_idx+1])/2, 1.05, '{0}'.format(r_idx+1), color='gray', transform=transHDVA, ha='center', fontsize='small')
    plt.sca(axs[-1])
    plt.xlabel('TR')
