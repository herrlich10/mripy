#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, argparse, textwrap, shlex, os, shutil, collections
from os import path


def parse_copy(args_copy):
    def is_folder(x):
        return not x[0].isdigit()
    to_copy = collections.OrderedDict()
    if len(args_copy) > 0:
        if not is_folder(args_copy[0]):
            raise ArgumentError("The first parameter following -c is expected to be a folder name, and mustn't start with 0-9.")
        for x in args.copy:
            if is_folder(x):
                folder = x
                to_copy[folder] = []
            else:
                to_copy[folder].append(x)
        for folder, index_list in to_copy.items():
            to_copy[folder] = utils.expand_index_list(index_list, '%04d')
    return to_copy


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import io, utils, afni, dicom

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
    parser.add_argument('-i', '--input', dest='input_dir', default='.', metavar='path/to/input/dir', help='path/to/input/dir containing raw dicom files, default: pwd')
    parser.add_argument('-o', '--output', dest='output_dir', default=None, help='path/to/output/dir, mkdir if not exists, default: same as input')
    parser.add_argument('-l', '--list', action='store_true', help='list dataset info')
    parser.add_argument('-a', '--anat', nargs='+', default=[], help='anat datasets to copy (e.g., -a 2)')
    parser.add_argument('-f', '--func', nargs='+', default=[], help='func datasets to copy (e.g., -f 3-6 9..15(2))')
    parser.add_argument('-c', '--copy', nargs='+', default=[], help='other datasets to copy (e.g., -c reverse 3 13 forward 4 14)')
    parser.add_argument('-g', '--log', nargs='?', help='write log to (specified) file') # Will be None whether you set -g, unless you specify -g a/new/path
    parser.add_argument('-s', '--select', default=-1, help='select which study to copy, default: last study (-1)')
    parser.add_argument('--pattern', default=io.SERIES_PATTERN, help='regular expression pattern capturing dataset index')
    parser.add_argument('-m', '--method', default='filename', help='method used to sort dicom files: [filename]|header')
    args = parser.parse_args()
    # args, unknowns = parser.parse_known_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    to_copy = collections.OrderedDict()
    to_copy['anat'] = utils.expand_index_list(args.anat, '%04d')
    to_copy['func'] = utils.expand_index_list(args.func, '%04d')
    to_copy.update(parse_copy(args.copy))
    n_to_copy = sum([len(v) for v in to_copy.values()])
    if args.log is None and n_to_copy > 0:
        args.log = path.join(args.output_dir, 'sort_dicom.log')
    if args.select > 0:
        args.select -= 1 # From one-based index to zero-based index
    # Sort files into studies of series
    if args.method == 'filename':
        studies = io.sort_dicom_series(args.input_dir)
        dicom_parser = io.parse_dicom_header
    elif args.method == 'header':
        print('>> Please wait while extracting headers...')
        studies = dicom.sort_dicom_series(args.input_dir)
        dicom_parser = dicom.parse_dicom_header
    if studies is None:
        print('>> No dicom file found in "{0}". Use -i to specify input dir.'.format(path.realpath(args.input_dir)))
        exit()
    # List all series in each study
    descriptions = []
    for k, study in enumerate(studies):
        descriptions.append(collections.OrderedDict())
        if args.list and len(studies) > 1:
            print('===== study #{0} ====='.format(k+1))
        for sn, files in study.items():
            if afni.has_afni:
                info = io.parse_series_info(files, parser=dicom_parser)
                desc = '{0} ({1}): {2}, {3}{4}'.format(sn,
                    info['n_volumes'] if info['n_volumes']>1 else len(files),
                    info['ProtocolName'],
                    'x'.join(['%.2g' % x for x in info['resolution']]),
                    (', TR=%.2f' % info['TR']) if info['TR'] is not None else '')
            else:
                desc = '{0} ({1})'.format(sn, len(files))
            descriptions[-1][sn] = desc
            if args.list:
                print(desc)
    # Overload print
    if args.log is not None:
        print = script_utils.get_log_printer(args.log)[0]
        print('# {0}'.format(os.getcwd()))
        print('# {0}'.format(' '.join([path.basename(sys.argv[0])] + sys.argv[1:])))
    # Copy files
    for kind, seqs in to_copy.items():
        if len(seqs) > 0:
            print('# {0}'.format(kind))
        for k, sn in enumerate(seqs):
            d = path.join(args.output_dir, '{0}{1:02d}'.format(kind, k+1) if len(seqs) > 1 else kind)
            if not path.exists(d):
                os.makedirs(d)
            print('- {0}'.format(descriptions[args.select][sn]))
            for f in studies[args.select][sn]:
                shutil.copy(f, d)
