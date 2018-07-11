#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap, glob, re, os
from os import path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch rename files using regexp.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) mripy_batch_rename.py '(.+)\.py'
              2) mripy_batch_rename.py -r '(.+)\.py' 'mripy_{0}.py' -e
            '''))
    parser.add_argument('pattern', help='old filename pattern, using regexp')
    parser.add_argument('replace', nargs='?', default='', help='new filename pattern, using {0}, {1}, ...')
    parser.add_argument('-r', '--recursive', action='store_true', help='check all files recursively')
    parser.add_argument('-e', '--execute', action='store_true', help='execute mv commands')
    args = parser.parse_args()

    args.pattern = re.compile(args.pattern)

    if args.recursive:
        files = glob.glob('**', recursive=True) # Require Python 3.5+
    else:
        files = glob.glob('*') # Current dir only
    for f in files:
        match = re.search(args.pattern, f)
        if match:
            if args.replace:
                fnew = args.replace.format(*match.groups())
                if args.execute:
                    os.rename(f, fnew)
                else:
                    print('"{0}" -> "{1}"'.format(f, fnew))
            else:
                print(f)
