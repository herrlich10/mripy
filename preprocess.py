#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import os, glob, shutil, re, subprocess, warnings
from os import path
from collections import OrderedDict
import numpy as np
try:
    import pandas as pd
except ImportError:
    warnings.warn('Cannot import pandas, which is required for some functions.', ImportWarning)
from . import six, afni, io, utils


def parse_seq_info(raw_dir, return_dataframe=True):
    '''
    Assuming your rawdata folder hierarchy is like:
        raw_fmri
        ├── func01
        ├── func02
        ├── reverse01
        ├── reverse02
        ├── T1
    '''
    seq_dirs = [f for f in os.listdir(raw_dir) if path.isdir(path.join(raw_dir, f)) and not f.startswith('.')]
    seq_info = OrderedDict([(seq_dir, io.parse_series_info(path.join(raw_dir, seq_dir))) for seq_dir in seq_dirs])
    if return_dataframe:
        return pd.DataFrame(list(seq_info.values()), index=list(seq_info.keys()))
    else:
        return seq_info


def find_best_reverse(seq_info, forward='func', reverse='reverse'):
    '''
    Find best forward-reverse pair (as a mapping dict) according to temporal proximity, 
    assuming that head motion is usually smaller if the two images are closer in time.

    seq_info : DataFrame
    '''
    match = lambda x: seq_info.index.str.startswith(x) # Return boolean selector over named index
    def min2(x): # Return both argmin and min. x : pd.Series
        k = x.idxmin()
        v = x[k]
        return k, v
    best_reverse = OrderedDict()
    for f in seq_info.index[match(forward)]:
        bk, bv = min2(abs(seq_info.loc[f,'start'] - seq_info.loc[match(reverse),'stop'])) # Nearest reverse before forward
        ak, av = min2(abs(seq_info.loc[f,'stop'] - seq_info.loc[match(reverse),'start'])) # Nearest reverse after forward
        best_reverse[f] = (bk, 'before') if bv < av else (ak, 'after')
    return best_reverse # {forward_file: (reverse_file, reverse_loc)}


def check_outputs(outputs):
    finished = np.all([path.exists(f) for n, f in outputs.items()])
    outputs['finished'] = finished
    return finished


def blip_unwarp(forward_file, reverse_file, reverse_loc, out_file, PE_axis='AP', patch_size=15):
    '''
    abin/unWarpEPI.py do this through: unwarp estimate (before tshift) > tshift > unwarp apply > motion correction > concat apply
    '''
    temp_dir = utils.temp_folder()
    out_dir, prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'for2mid_file': f"{out_dir}{prefix}.for2mid.warp{ext}",
        'rev2mid_file': f"{out_dir}{prefix}.rev2mid.warp{ext}",
        'QC_forward_file': f"{out_dir}QC.{prefix}.forward{ext}",
        'QC_reverse_file': f"{out_dir}QC.{prefix}.reverse{ext}",
    }

    # Prepare forward template
    n_subs = 5
    fnt = afni.get_dims(forward_file)[3]
    fsub = {'before': f'[0..{min(fnt, n_subs)-1}]', 'after': f'[{max(0, fnt-n_subs)}..$]'}[reverse_loc]
    # There is no need to protect the paths against whitespace (yet) because AFNI doesn't support it.
    afni.call(f"3dvolreg -zpad 1 -base {n_subs//2} -prefix {temp_dir}/forwards.nii -overwrite {forward_file}'{fsub}'")
    afni.call(f"3dTstat -median -prefix {temp_dir}/forward.nii -overwrite {temp_dir}/forwards.nii")
    afni.call(f"3dAutomask -apply_prefix {temp_dir}/forward.masked.nii -overwrite {temp_dir}/forward.nii")

    # Prepare reverse template
    rnt = afni.get_dims(reverse_file)[3]
    rsub = {'before': f'[{max(0, rnt-n_subs)}..$]', 'after': f'[0..{min(rnt, n_subs)-1}]'}[reverse_loc]
    # Input datasets for 3dQwarp must be on the same 3D grid (unlike program 3dAllineate)!
    afni.call(f"3dresample -rmode NN -master {temp_dir}/forwards.nii \
        -prefix {temp_dir}/reverses.nii -overwrite -input {reverse_file}'{rsub}'")
    afni.call(f"3dvolreg -zpad 1 -base {n_subs//2} -prefix {temp_dir}/reverses.nii -overwrite {temp_dir}/reverses.nii")
    afni.call(f"3dTstat -median -prefix {temp_dir}/reverse.nii -overwrite {temp_dir}/reverses.nii")
    afni.call(f"3dAutomask -apply_prefix {temp_dir}/reverse.masked.nii -overwrite {temp_dir}/reverse.nii")
    
    # Estimate nonlinear midpoint transform
    fix_non_PE_axes = {'AP': '-noXdis -noZdis', 'LR': '-noYdis -noZdis', None: ''}[PE_axis]
    dx = np.median(afni.get_head_delta(forward_file))
    # Minimal value allowed by 3dQwarp is 9, which is 27 mm for 3x3x3 voxels as in unWarpEPI.py
    # We use patch_size=15 mm by default here for typical hires data (1.2, 1.0, or 0.8 mm iso)
    min_patch = max(9, int(patch_size/dx)) # voxel
    # Compute the midpoint warp from reverse to forward
    # Among the resultant files:
    # - *_For_WARP is for warping forward to midpoint
    # - *_Rev_WARP is for warping reverse to midpoint
    # Even though compiled with OpenMP, this step is very slow...
    afni.call(f"3dQwarp -plusminus -pmNAMES Rev For \
        -pblur 0.05 0.05 -blur -1 -1 \
        -noweight {fix_non_PE_axes} -minpatch {min_patch} \
        -base   {temp_dir}/forward.masked.nii \
        -source {temp_dir}/reverse.masked.nii \
        -prefix {temp_dir}/{prefix}{ext} -overwrite")
    
    # Copy transform files
    afni.call(f"3dcopy {temp_dir}/{prefix}_For_WARP{ext} {outputs['for2mid_file']}")
    afni.call(f"3dcopy {temp_dir}/{prefix}_Rev_WARP{ext} {outputs['rev2mid_file']}")

    # Apply warp to unmasked templates for quality check
    afni.call(f"3dNwarpApply -quintic -nwarp {temp_dir}/{prefix}_For_WARP{ext} \
        -source {temp_dir}/forward.nii -prefix {outputs['QC_forward_file']} -overwrite")
    afni.call(f"3dNwarpApply -quintic -nwarp {temp_dir}/{prefix}_Rev_WARP{ext} \
        -source {temp_dir}/reverse.nii -prefix {outputs['QC_reverse_file']} -overwrite")

    shutil.rmtree(temp_dir)
    check_outputs(outputs)
    return outputs


def correct_motion(base_file, in_file, out_file, algorithm='3dvolreg', mode='rigid'):
    '''
    algorithm : {'3dvolreg', '3dAllineate'}
    '''
    out_dir, prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'param_file': f"{out_dir}{prefix}.motion.param.1D",
        'xform_file': f"{out_dir}{prefix}.motion.aff12.1D",
        'out_file': f"{out_dir}{prefix}{ext}",
    }

    if algorithm.lower() == '3dvolreg':
        if mode in ['rigid', 6]:
            afni.call(f"3dvolreg -base {base_file} \
                -verbose -zpad 2 \
                -1Dfile {outputs['param_file']} -1Dmatrix_save {outputs['xform_file']} \
                -prefix {out_file} -overwrite {in_file}")
        else:
            raise ValueError('Valid mode includes {"rigid"} for "3dvolreg" algorithm.')
    elif algorithm.lower() == '3dallineate':
        if mode in ['rigid', 6]:
            # The parameters are based on unWarpEPI.py and align_epi_anat.py
            # '-onepass': If you expect only small-ish (< 2 voxels?) image movement, skipping the coarse resolution first pass makes sense.
            # '-fineblur': The blur used in the refining pass.
            # '-norefinal': Skip restart after convergence (in case of local minima) in the interest of time 
            afni.call(f"3dAllineate -final quintic -base {base_file} \
                -warp shift_rotate -lpa -automask+2 -source_automask+2 \
                -onepass -fineblur 2 -norefinal \
                -1Dparam_save {outputs['param_file']} -1Dmatrix_save {outputs['xform_file']} \
                -prefix {out_file} -overwrite -source {in_file}")
        else:
            raise ValueError('Valid mode includes {"rigid"} for "3dAllineate" algorithm.')

    check_outputs(outputs)
    return outputs


def apply_transforms(transforms, base_file, in_file, out_file):
    if isinstance(transforms, six.string_types):
        transforms = [transforms]
    out_dir, prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{out_dir}{prefix}{ext}",
    }

    has_nwarp = not np.all([f.endswith('.1D') for f in transforms])
    transform_list = ' '.join(transforms)
    if has_nwarp:
        # '-interp wsinc5' is the default
        afni.call(f"3dNwarpApply -master {base_file} \
            -nwarp {transform_list} -source {in_file} \
            -prefix {out_file} -overwrite")
    else:
        combined = utils.temp_prefix(suffix='.1D')
        with open(f'{combined}', 'w') as fo:
            for line in afni.check_output(f"cat_matvec -ONELINE {transform_list}"):
                fo.write(line)
        # 'wsinc5' is 8x slower than 'quintic', but is highly accurate 
        # and should reduce the smoothing artifacts (see 3dAllineate)
        afni.call(f"3dAllineate -final wsinc5 -base {base_file} \
            -1Dmatrix_apply {combined} -input {in_file} \
            -prefix {out_file} -overwrite")
        os.remove(combined)

    check_outputs(outputs)
    return outputs


def manual_transform(in_file, out_file, shift=None, rotate=None, scale=None, shear=None, interp=None):
    '''
    shift : [x, y, z] in mm
    rotate : [I, R, A] in degrees (i.e., -z, -x, -y axes), right-hand rule

    For example, 'shift=[1,0,0]' shifts in_file to the left by 1 mm (not necessarily 1 voxel).
    The acutal applied param will be negated. Because if the source is shifted to the left 
    compared with the base, the resulted param will be x=1 indicating that source is shifted
    to the left, and apply that param will actually cause the source to shift rightward by 1
    to match the base. So to shift in_file leftward, the actual param will be x=-1.
    
    The matrix is specified in DICOM-ordered (RAI) coordinates (x=-R+L,y=-A+P,z=-I+S).
    By default the shift is applied AFTER matrix transform, as in augmented 4x4 affine.
    By default the shear matrix is LOWER triangle.

    For '-1Dmatrix_save' and '-1Dmatrix_apply', the matrix specifies 
    coordinate transformation from base to source DICOM coordinates. 
    In other words, with the estimated matrix at hand, you are ready to build the 
    transformed image by filling the base grid with source data.
    
    Refer to "DEFINITION OF AFFINE TRANSFORMATION PARAMETERS"@3dAllineate for details.
    '''
    if shift is None:
        shift = [0, 0, 0]
    if rotate is None:
        rotate = [0, 0, 0]
    if scale is None:
        scale = [0, 0, 0]
    if shear is None:
        shear = [0, 0, 0]
    if interp is None:
        interp = 'wsinc5'
    param_file = utils.temp_prefix(suffix='.1D')
    nt = afni.get_dims(in_file)[3]
    param = np.tile(np.r_[shift, rotate, scale, shear] * -1, [nt, 1])
    if np.all(param == 0):
        afni.call(f"3dcopy {in_file} {out_file} -overwrite")
    else:
        np.savetxt(param_file, param, fmt='%.6e')
        afni.call(f"3dAllineate -final {interp} -base {in_file} \
            -1Dparam_apply {param_file} -input {in_file} \
            -prefix {out_file} -overwrite")
        os.remove(param_file)


def nudge_cmd2mat(nudge_cmd, in_file):
    '''
    Refer to "Example 4"@SUMA_AlignToExperiment for details.
    '''
    match = re.search(r'(-?\d+\.\d{2}I) (-?\d+\.\d{2}R) (-?\d+\.\d{2}A).*(-?\d+\.\d{2}S) (-?\d+\.\d{2}L) (-?\d+\.\d{2}P)', nudge_cmd)
    if match:
        I, R, A, S, L, P = match.groups()
        temp_file = utils.temp_prefix(suffix='.nii')
        afni.call(f"3drotate -NN -clipit -rotate {I} {R} {A} -ashift {S} {L} {P} -prefix {temp_file} {in_file}")
        res = afni.check_output(f"cat_matvec '{temp_file}::ROTATE_MATVEC_000000' -I -ONELINE")[-2]
        os.remove(temp_file)
        return np.float_(res.split()).reshape(3,4)
    else:
        raise ValueError(f"`nudge_cmd` should contain something like '-rotate 0.00I -20.00R 0.00A -ashift 13.95S -2.00L -11.01P'")


def motion_correction(in_files, out_files, reverse_files=None):
    pass
    

def skullstrip_mp2rage(in_files, out):
    '''
    in_files : list or str
    '''
    pass


if __name__ == '__main__':
    pass
