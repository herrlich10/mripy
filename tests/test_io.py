#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
from numpy.testing import assert_allclose
try:
    from .context import data_dir # If mripy is importable: python -m mripy.tests.test_io
except ValueError: # Attempted relative import in non-package
    from context import data_dir # If not importable: cd mripy/tests; python -m test_io
from mripy import io

from os import path
import os, glob, subprocess
import numpy as np


class test_io(unittest.TestCase):
    def test_Mask(self):
        mask_file = path.join(data_dir, 'brain_mask', 'brain_mask+orig')
        data_file = path.join(data_dir, 'brain_mask', 'gre*.volreg+orig.HEAD')
        mask = io.Mask(mask_file)
        # Test dump
        x = mask.dump(data_file)
        self.assertEqual(x.shape, (564361,4))
        mask2 = io.MaskDumper(mask_file)
        y = mask2.dump(data_file)
        assert_allclose(x, y)
        # Test undump
        max_idx = np.argmax(x, axis=1) + 1
        mask.undump('test_undump', max_idx, method='nibabel')
        self.assertEqual(subprocess.check_output('3dinfo -orient test_undump+orig', shell=True), b'RSA\n')
        assert_allclose(mask.dump('test_undump+orig.HEAD'), max_idx)
        for f in glob.glob('test_undump+orig.*'):
            os.remove(f)
        # Test constrain
        smaller, sel0 = mask.near(5, 45, -17, 12, return_selector=True)
        sel = mask.infer_selector(smaller)
        smaller.undump('test_constrain', max_idx[sel])
        assert_allclose(smaller.dump('test_constrain+orig.HEAD'), max_idx[sel0])
        for f in glob.glob('test_constrain+orig.*'):
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
