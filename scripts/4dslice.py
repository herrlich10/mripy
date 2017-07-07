#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import io, utils

    parser = argparse.ArgumentParser(description='Extract slices in any of the x/y/z/t dimensions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              $ 4dslice -x -20 20 -t 0 134 -o func2 -i func+orig
            '''))
    parser.add_argument('-i', '--input', metavar='input_dset', required=True)
    parser.add_argument('-o', '--output', metavar='output_dset', required=True)
    parser.add_argument('-x', nargs='+', help='slice range(s) in x dimension')
    parser.add_argument('-y', nargs='+', help='slice range(s) in y dimension')
    parser.add_argument('-z', nargs='+', help='slice range(s) in z dimension')
    parser.add_argument('-t', nargs='+', help='slice range(s) in t dimension')
    args = parser.parse_args()

    v, img = io.read_afni(args.input, return_img=True)
    for d in ['x', 'y', 'z', 't']:
        if args.__getattribute__(d) is None:
            args.__setattr__(d, slice(None))
        else:
            args.__setattr__(d, utils.expand_index_list(args.__getattribute__(d)))
    io.write_afni(args.output, v[args.x][:,args.y][:,:,args.z][...,args.t], img)
