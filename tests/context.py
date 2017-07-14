#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from os import sys, path

# Make sure the package "mripy" is importable during test time
package_dir = path.abspath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, path.dirname(package_dir))

# Make sure the test data (which is too big to fit in the package) is available
data_dir = path.abspath(path.join(package_dir, '../mripy_data'))
