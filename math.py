#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from . import six


def nearest(x, parity='odd', round=np.round):
    if parity == 'even':
        return np.int_(round(x/2.0)) * 2
    elif parity == 'odd':
        return np.int_(round((x+1)/2.0)) * 2 - 1


if __name__ == '__main__':
    pass
