#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import argparse, textwrap, glob, re, sys, os, shutil
from os import path
import numpy as np


def plot_match(physio_infos, series_infos):
    from matplotlib import pyplot as plt
    from matplotlib import transforms

    transHDVA = transforms.blended_transform_factory(plt.gca().transData, plt.gca().transAxes)
    n_plots = 0
    appended = False
    for k, sinfo in enumerate(series_infos):
        plt.axvspan(sinfo['start'], sinfo['stop'], n_plots*0.2, n_plots*0.2+0.2, color='k', alpha=0.3)
        plt.text(sinfo['start'], n_plots*0.2+0.05, '%d'%k, transform=transHDVA, clip_on=True)
        print('DICOM#{0:02d}: {1} ({2} volumes)'.format(k, path.basename(sinfo['files'][0]), sinfo['n_volumes']))
        appended = True
    if appended:
        plt.text(0.01, n_plots*0.2+0.1, 'DICOM', transform=plt.gca().transAxes)
        print('='*30)
        n_plots += 1
    for ch, color in zip(['resp', 'puls'], ['b', 'r']):
        appended = False
        for k, pinfo in enumerate(physio_infos):
            if pinfo is not None and ch in pinfo:
                plt.axvspan(pinfo[ch]['start'], pinfo[ch]['stop'], n_plots*0.2, n_plots*0.2+0.2, color=color, alpha=0.3)
                plt.text(pinfo[ch]['start'], n_plots*0.2+0.05, '%d'%k, transform=transHDVA, clip_on=True)
                print('{0}#{1:02d}: {2} ({3} samples)'.format(ch.upper(), k, path.basename(pinfo[ch]['file']), len(pinfo[ch]['data'])))
                appended = True
        if appended:
            plt.text(0.01, n_plots*0.2+0.1, ch.upper(), transform=plt.gca().transAxes)
            print('='*30)
            n_plots += 1
    plt.show()


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import io, utils, six
    print = six.print_

    parser = argparse.ArgumentParser(description='Extract physiological data sync with mri acquisition.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
              1) Physiological files are in physio/, and raw dicom files are in
                 func01/, func02/, func03/, etc. Simply list timing information.
                $ extract_physio.py -p physio -d func* -l
              2) Both physiological and dicom files are in 20170222_S18. Valid
                 functional runs are from 0006 to 0009. Output extracted 1D files
                 to 20170222_S18/extracted.
                $ extract_physio.py -p 20170222_S18 -d 20170222_S18 -f 6-9 -o\
            '''))
    parser.add_argument('-p', '--physio', default='.', metavar='path/to/physio/dir', help='folder that contains *.resp/*.puls/etc.')
    parser.add_argument('-d', '--dicom', default=['.'], nargs='+', metavar='path/to/dicom/dir', help='one or more folders containing *.IMA')
    parser.add_argument('-o', '--output_dir', nargs='?', const='default', metavar='path/to/output/dir', help='folder to output the 1D files')
    parser.add_argument('-l', '--list', action='store_true', help='list timing info')
    parser.add_argument('-C', '--channels', default=['resp', 'puls'], nargs='+', help='channels to list (resp/puls/etc.)')
    parser.add_argument('-f', '--func', default=[], nargs='+', help='series number of selected func runs')
    parser.add_argument('-u', '--dummy', default=0, help='number of dummy scans (will trim physio accordingly)')
    parser.add_argument('-c', '--copy', nargs='?', const='default', metavar='path/to/copy/files/to', help='copy raw physio files to specified folder')
    parser.add_argument('-nbf', '--name_by_folder', action='store_true', help='name output by dicom folder names')
    parser.add_argument('-g', '--graph', action='store_true', help='plot match between physio and dicom')
    parser.add_argument('-m', '--match', default='cover', help='matching method: (cover)|overlap')
    parser.add_argument('-M', '--CMRR', action='store_true', help='CMRR MB acquisition time correction')
    args = parser.parse_args()
    if args.output_dir == 'default':
        args.output_dir = path.join(args.physio, 'physio')
    if args.output_dir is not None and not path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.copy == 'default':
        folder_name = 'raw_physio'
        if args.output_dir is not None:
            args.copy = path.join(path.split(args.output_dir)[0], folder_name)
        else:
            args.copy = path.join(args.physio, folder_name)
    if args.copy is not None and not path.exists(args.copy):
        os.makedirs(args.copy)
    args.func = utils.expand_index_list(args.func)
    # Parse dicom files
    print('Parsing dicom info...', end='', flush=True)
    dicom_files = []
    for dicom_dir in args.dicom:
        dicom_files.extend(io.filter_dicom_files(dicom_dir, series_numbers=args.func, instance_numbers=[1]))
    series_infos = [io.parse_series_info(f, shift_time='CMRR' if args.CMRR else None) for f in dicom_files]
    print(' ({0} files)'.format(len(series_infos)))
    if len(series_infos) == 0:
        print('No dicom file found in {0}. Use -d to specify one or more dicom input dir(s).'
            .format(', '.join(['"{0}"'.format(path.realpath(p)) for p in args.dicom])))
        exit()
    # Parse physiological files
    print('Parsing physiological info...', end='', flush=True)
    physio_files = sorted(glob.glob(path.join(args.physio, '*.resp')))
    date = series_infos[0]['date'] # Potential bug: Assume there is no cross-day experiment
    physio_infos = [io.parse_physio_files(f, date=date, channels=args.channels) for f in physio_files]
    print(' ({0} files)'.format(len(physio_infos)))
    if len(physio_infos) == 0:
        print('No physiological file found in "{0}". Use -p to specify physiological input dir.'
            .format(path.realpath(args.physio)))
        exit()
    # Match physiological with
    if args.graph:
        plot_match(physio_infos, series_infos)
        # from mypy import mypy
        # mypy.saveObjectAsPickle(dict(physio_infos=physio_infos, series_infos=series_infos), 'debug.pickle')
    physio_infos, series_infos = io.match_physio_with_series(physio_infos, series_infos,
        channel=args.channels[0], method=args.match)
    print('{0} pairs were found.'.format(len(physio_infos)))
    # List timing
    if args.list:
        print('==================================================')
        for k, sinfo in enumerate(series_infos):
            for ch in args.channels:
                io._print_physio_timing(physio_infos[k][ch], sinfo, ch, k)
    # Extract 1D files
    if args.output_dir is not None:
        print('Writing outputs...')
        for k, sinfo in enumerate(series_infos):
            res = io.extract_physio(physio_infos[k], sinfo, TR=sinfo['TR'], dummy=int(args.dummy), channels=args.channels, verbose=int(not args.list))
            dicom_folder_name = path.split(path.split(sinfo['files'][0])[0])[1]
            for c, ch in enumerate(args.channels):
                if args.name_by_folder:
                    fname = '{0}_{1}.1D'.format(ch, dicom_folder_name)
                else:
                    fname = '{0}{1:02d}.1D'.format(ch, k+1)
                np.savetxt(path.join(args.output_dir, fname), res[c], fmt=str('%d')) # Use str() because npyio.py is not 2/3 compatible
    # Copy raw physio files
    if args.copy is not None:
        print('Copying raw physio files...')
        for physio_info in physio_infos:
            for f in glob.glob(path.splitext(list(physio_info.values())[0]['file'])[0] + '.*'):
                shutil.copy(f, args.copy)
