#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import plots

    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) Plot head motion across multiple runs
                $ 1dplot.py -vr dfile.epi*.1D
            '''))
    parser.add_argument('files', nargs='+', help='1D file(s)')
    parser.add_argument('-vr', '--volreg', action='store_true', help='plot head motion')
    args = parser.parse_args()

    if args.volreg:
        plots.plot_volreg(args.files)
    else:
        x = np.loadtxt(args.files[0])
        plt.plot(x)
    plt.show()
    plt.close()
