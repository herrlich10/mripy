    #!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from os import path
import numpy as np
from . import six, afni, utils


class Surface(object):
    def __init__(self, suma_dir, surf_vol=None):
        self.surf_dir = suma_dir
        self.subj = afni.get_suma_subj(self.surf_dir)
        if surf_vol is None:
            self.surf_vol = self.subj + '_SurfVol+orig.HEAD'
        else:
            self.surf_vol = surf_vol
        self.hemis = ['lh', 'rh']
        self.surfs = ['pial', 'smoothwm', 'inflated', 'sphere.reg']

    def _get_surf2exp_transform(self, exp_anat):
        pass

    def _get_spherical_coordinates(self):
        verts = read_asc('../SUMA/rh.sphere.reg.asc')
        theta = np.arccos(verts[:,2]/100) # Polar (inclination) angle, [0,pi]
        phi = np.arctan2(verts[:,1], verts[:,0]) # Azimuth angle, (-pi,pi]
        return theta, phi



if __name__ == '__main__':
    pass
