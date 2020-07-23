#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path as path

__author__ = 'herrlich10 <herrlich10@gmail.com>'
with open(path.join(path.dirname(path.realpath(__file__)), '__version__')) as f:
    __version__ = f.readline().strip()


if __name__ == '__main__':
    pass
