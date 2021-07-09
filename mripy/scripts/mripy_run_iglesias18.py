#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re
from posixpath import basename
from os import path
import argparse, textwrap
import numpy as np


nuclei = {
    'unknown': 0,
    'AV': 3,
    'CeM': 4,
    'CL': 5,
    'CM': 6,
    'LD': 8,
    'LGN': 9,
    'LP': 10,
    'L-Sg': 11,
    'MDl': 12, 
    'MDm': 13,
    'MGN': 15,
    'MV(Re)': 16,
    'Pc': 17, 
    'Pf': 18, 
    'Pt': 19, 
    'PuA': 20, # Pulvinar Anterior
    'PuI': 21, # Pulvinar Inferior 
    'PuL': 22, # Pulvinar Lateral
    'PuM': 23, # Pulvinar Medial
    'R': 25,
    'VA': 26,
    'VAmc': 27,
    'VLa': 28,
    'VLp': 29,
    'VM': 30,
    'VPL': 33,
}


if __name__ == '__main__':
    import script_utils # Append mripy to Python path
    from mripy import utils, io
    timer = script_utils.ScriptTimer()

    parser = argparse.ArgumentParser(description='Create Iglesias2018 thalamus masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f'''
        Create thalamic nuclei masks (e.g., LGN, pulvinar) from FreeSurfer processed 
        T1 image based on a probabilistic atlas of the human thalamic nuclei 
        (Iglesias et al.,2018).

        Examples
        --------
        1) Convert HCP retinotopy atlas
        $ mripy_run_iglesias18.py -s fs_subjects/ZhangP --suma_dir my_analysis/S01/SUMA
        
        Thalamic nuclei code
        --------------------
        {nuclei}

        References
        ----------
        [1] https://freesurfer.net/fswiki/ThalamicNuclei
        '''))
    parser.add_argument('-s', '--subj_dir', required=True, help='')
    parser.add_argument('--suma_dir', help='')
    args = parser.parse_args()
    args.subj_dir = path.realpath(args.subj_dir) # This is important in case link changes subject folder name 
    if args.suma_dir is None:
        args.suma_dir = f"{args.subj_dir}/SUMA"
    mri_dir = f"{args.subj_dir}/mri"

    if not utils.exists(f"{mri_dir}/ThalamicNuclei.v12.T1.mgz"):
        # Check FreeSurfer version
        res = utils.run('freesurfer', shell=True, verbose=0)['output'][-2].strip()
        match = re.search(r'-v?(\d+)\.(\d+)\.(\d+)-', res)
        fs_ver = np.int_(match.groups())
        if fs_ver[0] < 7:
            raise RuntimeError(textwrap.dedent(f'''
                ** This function require FreeSurfer 7 or above.
                But you are currently running "{res}".
                Please check your $PATH or update your FreeSurfer.'''))
        # Check SUMA
        if not path.exists(args.suma_dir):
            raise RuntimeError(textwrap.dedent(f'''
                ** The output SUMA dir "{args.suma_dir}" does not exist.
                Please create your SUMA dir first.
                You can also use --suma_dir to specify a correct path to  your SUMA dir.'''))

        # Run segmentThalamicNuclei.sh
        os.environ['SUBJECTS_DIR'] = path.dirname(f"{args.subj_dir}")
        subject = path.basename(args.subj_dir)
        utils.run(f"segmentThalamicNuclei.sh {subject}")
    
    # Convert volume datasets
    print('>> Convert *.mgz files...')
    dset_list = ['ThalamicNuclei.v12.T1', 'ThalamicNuclei.v12.T1.FSvoxelSpace']
    for dset in dset_list:
        utils.run(f"mri_convert {mri_dir}/{dset}.mgz {args.suma_dir}/{dset}.nii.gz")

    # Create mask volumes using the subvoxel parcellation
    print('>> Create mask files...')
    v, img = io.read_vol(f"{args.suma_dir}/ThalamicNuclei.v12.T1.FSvoxelSpace.nii.gz", return_img=True)
    v[(v>8100)&(v<8200)] = 8100 - v[(v>8100)&(v<8200)] # Negative values for lh
    v[(v>8200)&(v<8300)] -= 8200 # Positive values for rh
    io.write_vol(f"{args.suma_dir}/iglesias18_nuclei.nii", v, base_img=img)
    for roi in ['LGN', 'PuI']:
        utils.run(f"3dcalc -a {args.suma_dir}/iglesias18_nuclei.nii \
            -expr '-equals(a,-{nuclei[roi]})+equals(a,{nuclei[roi]})' \
            -prefix {args.suma_dir}/iglesias18_{roi}.nii -overwrite")



