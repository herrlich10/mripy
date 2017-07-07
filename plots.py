#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from matplotlib import pyplot as plt
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
