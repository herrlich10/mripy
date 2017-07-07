#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    import script_utils # Append mripy to Python path

    parser = argparse.ArgumentParser(description='Sort raw dicom files and copy them into separate folders.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) List all datasets in ~/raw_data
                $ sort_dicom.py -l -i ~/raw_data
              2) Copy 0002 to ~/sorted_data/anat, 0003-0006 to ~/sorted_data/func01-04,
                 0008 as func05, 0010 as func06, 0012 as func07
                $ extract_phys.py -i ~/raw_data -o ~/sorted_data -a 2 -f 3-6 8..12(2)\
            '''))
    # parser.add_argument('-i', '--input', dest='input_dir', default='.', metavar='path/to/input/dir', help='path/to/input/dir containing raw dicom files, default: pwd')
    parser.add_argument('fname', metavar='1dfile')
    args = parser.parse_args()

    x = np.loadtxt(args.fname)
    plt.plot(x)
    plt.show()
    plt.close()
