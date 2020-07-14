#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals


if __name__ == '__main__':
    import argparse, textwrap
    import script_utils # Append mripy to Python path
    from mripy import dicom_report, utils

    parser = argparse.ArgumentParser(description='',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            examples:
                1) report_parameters.py ../raw_fmri/func01
                2) report_parameters.py -p mp2rage ../raw_fmri/'T1*'
            '''))
    parser.add_argument('dicom_folder', help='a folder that contains all dicom files for a single sequence')
    parser.add_argument('-p', '--preset', help=f"preset for what parameters to report: {', '.join(dicom_report.preset4sequence.values())}")
    args = parser.parse_args()

    print('>> Reading dicom headers...')
    report, preset, info = dicom_report.report_parameters(args.dicom_folder, preset=args.preset, return_preset=True, return_info=True)
    print(f'>> Using "{preset}" preset')
    print(report)
    print(f">> Coil: {info['TransmittingCoil']},   ReferenceAmplitude: {info['ReferenceAmplitude']} V,   TotalScanTime: {utils.format_duration(info['TotalScanTime'])}")