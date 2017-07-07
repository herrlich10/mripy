#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import subprocess
from . import six, afni


def afni_costs(base_img, input_img):
    cmd = '3dAllineate -base {0} -input {1} -allcostX'.format(base_img, input_img)
    lines = afni.check_output(cmd, pattern=r'^\s+\S+\s+=\s+\S+')
    costs = (line.split('=') for line in lines)
    costs = dict([name.strip(), float(value)] for name, value in costs)
    return costs


if __name__ == '__main__':
    pass
