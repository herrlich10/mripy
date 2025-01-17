#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re
from os import path
import argparse, textwrap
from mripy import utils, io


def parse_fname(fname, pattern='hemi', trailing_exts=['.gz'], compound_exts=['.gz']):
    '''
    {dir}/{hemi}.{stem}{ext}
    {dir}/{stem}{view}{ext}
    '''
    res = dict()
    res['dir'], res['name'] = path.split(fname)
    res['dir/'] = res['dir'] + '/' if res['dir'] else ''
    for ext in compound_exts:
        if res['name'].lower().endswith(ext):
            res['prefix'], res['ext'] = path.splitext(res['name'][:-len(ext)])
            res['ext'] = res['ext'] + ext
            break
    else:
        res['prefix'], res['ext'] = path.splitext(res['name'])
    if pattern == 'hemi':
        pattern = r"(?P<hemi>lh|rh)\.(?P<stem>.+)$"
    elif pattern == 'view':
        pattern = r"(?P<stem>.+?)(?P<view>\+(:?orig|tlrc))?$"
    match = re.match(pattern, res['prefix'])
    if match:
        res.update(match.groupdict())
    else:
        raise ValueError(f"'{res['prefix']}' doesn't match '{pattern}'")
    return res


def curv2niml(fs_subj_dir, curv_file, niml_file=None):
    '''
    Convert a scalar overlay file (FreeSurfer curv file) to AFNI niml dset.
    E.g., path/to/lh.wang15_mplbl.mgz, or rh.benson14_varea.mgz
    '''
    parts = parse_fname(curv_file)
    gii_file = "{dir/}{hemi}.{stem}.gii".format(**parts)
    if niml_file is None:
        niml_file = "{dir/}{hemi}.{stem}.niml.dset".format(**parts)
    utils.run(f"mris_convert -c {curv_file} {fs_subj_dir}/surf/{parts['hemi']}.white {gii_file}")
    utils.run(f"ConvertDset -input {gii_file} -prefix {niml_file} -overwrite")
    os.remove(gii_file)


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    timer = script_utils.ScriptTimer()

    parser = argparse.ArgumentParser(description='Apply wang15 atlas to individual surface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f'''
        Examples
        --------
        1) mripy_run_wang15.py -s SubjID

        References
        ----------
        [1] 
        '''))
    parser.add_argument('-s', '--subj_dir', required=True, help='subject dir')
    parser.add_argument('--suma_dir', help='SUMA dir')
    args = parser.parse_args()
    args.subj_dir = path.realpath(args.subj_dir) # This is important in case link changes subject folder name 
    if args.suma_dir is None:
        args.suma_dir = f"{args.subj_dir}/SUMA"
    subjects_dir, subject = path.split(args.subj_dir)
    mri_dir = f"{args.subj_dir}/mri"
    surf_dir = f"{args.subj_dir}/surf" 

    # Run benson command
    os.environ['SUBJECTS_DIR'] = subjects_dir
    try:
        utils.run(f"python -m neuropythy atlas --verbose --volume-export --atlases wang15 {subject}")
    except RuntimeError: # Expecting "ValueError: lh hemi/property size mismatch"
        pass

    # Convert surface dataset
    # 'fplbl' raises "ERROR: number of vertices in lh.wang15_fplbl.mgz does not match surface (25,297651)"
    # I guess that mplbl is maximum probability map, which is a scalar map; 
    # while fplbl is full probability map, which contains 25 entries, each corresponding to one of the 25 areas
    for dset in ['mplbl']: 
        for hemi in ['lh', 'rh']:
            curv_file = f"{surf_dir}/{hemi}.wang15_{dset}.mgz"
            niml_file = f"{args.suma_dir}/{hemi}.wang15_{dset}.niml.dset"
            if path.exists(curv_file):
                curv2niml(args.subj_dir, curv_file, niml_file)

    # Convert volume dataset
    for dset in ['mplbl']:
        mgz_file = f"{mri_dir}/wang15_{dset}.mgz"
        nii_file = f"{args.suma_dir}/wang15_{dset}.nii"
        if path.exists(mgz_file):
            utils.run(f"mri_convert --in_type mgz --out_type nii {mgz_file} {nii_file}")





