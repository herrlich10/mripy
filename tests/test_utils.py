#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
from mripy import utils


class test_utils(unittest.TestCase):
    def test_fname_with_ext(self):
        self.assertEqual(utils.fname_with_ext('prefix+orig', '.HEAD'), 'prefix+orig.HEAD')
        self.assertEqual(utils.fname_with_ext('prefix', '+orig.HEAD'), 'prefix+orig.HEAD')
        self.assertEqual(utils.fname_with_ext('prefix+orig', '+orig.HEAD'), 'prefix+orig.HEAD')
        self.assertEqual(utils.fname_with_ext('prefix+orig.', '+orig.HEAD'), 'prefix+orig.HEAD')
        self.assertEqual(utils.fname_with_ext('prefix+orig.HEAD', '+orig.HEAD'), 'prefix+orig.HEAD')

if __name__ == '__main__':
    unittest.main()
