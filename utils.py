#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, re
from . import six


# Command line syntactic sugar
def expand_index_list(index_list, format=None):
    '''
    Expand a list of indices like: 1 3-5 7:10:2 8..10(2)
    '''
    flatten = []
    for x in index_list:
        if '-' in x: # 3-7
            start, end = x.split('-')
            flatten.extend(range(int(start), int(end)+1))
        elif ':' in x: # 1:10 or 1:10:2
            flatten.extend(range(*map(int, x.split(':'))))
        elif '..' in x: # 1..9 or 1..9(2)
            range_ = list(map(int, re.split('\.\.|\(|\)', x)[:3]))
            range_[1] += 1
            flatten.extend(range(*range_))
        else: # 2
            flatten.append(int(x))
    return flatten if format is None else [format % x for x in flatten]


def format_duration(duration, format='standard'):
    if format == 'short':
        units = ['d', 'h', 'm', 's']
    elif format == 'long':
        units = [' days', ' hours', ' minutes', ' seconds']
    else:
        units = [' day', ' hr', ' min', ' sec']
    values = [int(duration//86400), int(duration%86400//3600), int(duration%3600//60), duration%60]
    for K in range(len(values)): # values[K] would be the first non-zero value
        if values[K] > 0:
            break
    formatted = ((('%d' if k<len(values)-1 else '%.3f') % values[k]) + units[k] for k in range(len(values)) if k >= K)
    return ' '.join(formatted)
