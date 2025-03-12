#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import vis

    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) Plot head motion across multiple runs
                $ 1dplot.py -vr dfile.epi*.1D
            '''))
    parser.add_argument('files', nargs='+', help='1D file(s)')
    parser.add_argument('-vr', '--volreg', action='store_true', help='plot head motion parameters')
    parser.add_argument('-y', '--ylim', metavar='L U', nargs='+', help='set ylim')
    parser.add_argument('-s', '--save', metavar='FNAME', help='save figure as *.png')
    parser.add_argument('--silent', action='store_true', help='do not show figure')
    args = parser.parse_args()

    if args.volreg:
        vis.plot_volreg(args.files)
    else:
        x = np.loadtxt(args.files[0])
        plt.plot(x)
    if args.ylim is not None:
        if len(args.ylim) == 1:
            # In case the two parameters are quoted together
            # Yeah, this is a little bit too much...
            plt.ylim(np.float_(args.ylim[0].split()))
        else:
            plt.ylim(np.float_(args.ylim))
    if args.save:
        plt.savefig(args.save)
    if not args.silent:
        plt.show()
        plt.close()
