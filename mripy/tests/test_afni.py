#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
from mripy import afni


class test_afni(unittest.TestCase):
    def test_get_prefix(self):
        # 3d dset
        self.assertEqual(afni.get_prefix('path/to/dset+orig.HEAD'), 'dset')
        self.assertEqual(afni.get_prefix('dset+orig.'), 'dset')
        self.assertEqual(afni.get_prefix('dset+tlrc'), 'dset')
        self.assertEqual(afni.get_prefix('my+dset'), 'my+dset') # Only +orig and +tlrc are treated specially
        self.assertEqual(afni.get_prefix('~/my+dset+orig'), 'my+dset')
        self.assertEqual(afni.get_prefix('path/to/my+dset+orig.BRIK', with_path=True), 'path/to/my+dset')
        # Surface dset
        self.assertEqual(afni.get_prefix('path/to/prefix.niml.dset'), 'prefix')
        self.assertEqual(afni.get_prefix('prefix.1D.dset'), 'prefix')
        self.assertEqual(afni.get_prefix('prefix.niml'), 'prefix')
        self.assertEqual(afni.get_prefix('path/to/prefix.1D'), 'prefix')

    def test_get_suma_spec(self):
        spec = afni.get_suma_spec('path/to/subj_both.spec')
        self.assertEqual(spec['lh'], 'path/to/subj_lh.spec')
        self.assertEqual(sorted(spec.keys()), sorted(afni.SPEC_HEMIS))
        spec = afni.get_suma_spec('subj.hd.lh.spec')
        self.assertEqual(spec['rh'], 'subj.hd.rh.spec')

    def test_substitute_hemi(self):
        self.assertEqual(afni.substitute_hemi('roi.V1.lh.niml.dset'), 'roi.V1.{0}.niml.dset') # In the middle
        self.assertEqual(afni.substitute_hemi('rh.smoothwm.asc'), '{0}.smoothwm.asc') # In the beginning
        self.assertEqual(afni.substitute_hemi('std.60.subj_both.spec', '*'), 'std.60.subj_*.spec') # Flankered by "_"
        self.assertEqual(afni.substitute_hemi('roi.rhymic'), 'roi.rhymic') # Non-standalone instance
        self.assertEqual(afni.substitute_hemi('both_V1.lh.niml.roi'), '{0}_V1.{0}.niml.roi') # No consistancy check for multiple instances

if __name__ == '__main__':
    unittest.main()
