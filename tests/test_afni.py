#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
from mripy import afni


class test_afni(unittest.TestCase):
    def test_get_prefix(self):
        self.assertEqual(afni.get_prefix('path/to/dset+orig.HEAD'), 'dset')
        self.assertEqual(afni.get_prefix('dset+orig.'), 'dset')
        self.assertEqual(afni.get_prefix('dset+tlrc'), 'dset')
        self.assertEqual(afni.get_prefix('my+dset'), 'my+dset')
        self.assertEqual(afni.get_prefix('~/my+dset+orig'), 'my+dset')
        self.assertEqual(afni.get_prefix('path/to/my+dset+orig.BRIK', with_path=True), 'path/to/my+dset')


if __name__ == '__main__':
    unittest.main()
