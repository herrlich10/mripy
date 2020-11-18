#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import os, glob, shutil, shlex, re, subprocess, multiprocessing, warnings, time
import json, copy
from os import path
from collections import OrderedDict
import numpy as np
from numpy.polynomial import polynomial
from scipy import stats
from scipy.ndimage import interpolation
from sklearn import mixture
try:
    import pandas as pd
except ImportError:
    warnings.warn('Cannot import pandas, which is required for some functions.', ImportWarning)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    warnings.warn('Cannot import seaborn, which is required for some functions.', ImportWarning)
import nibabel
from . import six, afni, io, utils, dicom, dicom_report, math


DEFAULT_JOBS = multiprocessing.cpu_count() * 3 // 4


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


def find_best_reverse(seq_info, forward='func', reverse='reverse', for_out='epi{run}.nii', rev_out='reverse{run}.nii', return_single_best=False):
    '''
    Find best forward-reverse pair (as a mapping dict) according to temporal proximity, 
    assuming that head motion is usually smaller if the two images are closer in time.

    seq_info : DataFrame
    '''
    def match(x): # Return boolean selector over named index
        return seq_info.index.str.startswith(x) 
    def min2(x): # Return both argmin and min.    x : pd.Series
        k = x.idxmin()
        v = x[k]
        return k, v
    def subs(d, s): # Substitute "func01" with something like "epi01.nii".    d for direction, s for string
        if d == 'forward':
            return for_out.format(run=re.match(f"{forward}(\d+)", s).group(1))
        elif d == 'reverse':
            return rev_out.format(run=re.match(f"{reverse}(\d+)", s).group(1))
    best_reverse = OrderedDict()
    best_distance = dict()
    for f in seq_info.index[match(forward)]:
        before_index, before_value = min2(abs(seq_info.loc[f,'start'] - seq_info.loc[match(reverse),'stop'])) # Nearest reverse before forward
        after_index, after_value = min2(abs(seq_info.loc[f,'stop'] - seq_info.loc[match(reverse),'start'])) # Nearest reverse after forward
        best_reverse[subs('forward', f)] = (subs('reverse', before_index), 'before') if before_value < after_value else (subs('reverse', after_index), 'after')
        best_distance[subs('forward', f)] = before_value if before_value < after_value else after_value
    if return_single_best:
        forward_file = sorted(best_distance.items(), key=lambda kv: kv[1])[0][0]
        return (forward_file, *best_reverse[forward_file]) # forward_file, reverse_file, reverse_loc
    else:
        print('>> ---------- forward-reverse pairs ----------')
        for forward, (reverse, loc) in best_reverse.items():
            print({'before': f"{reverse} -> {forward}", 'after': f"{' '*len(reverse)}    {forward} <- {reverse}"}[loc])
        print('>> -------------------------------------------')
        return best_reverse # {forward_file: (reverse_file, reverse_loc)}


def all_finished(outputs):
    if isinstance(outputs, dict):
        outputs = [outputs]
    def check_exist(f):
        try:
            return path.exists(f)
        except TypeError:
            return all(path.exists(ff) for ff in f)
    for output in outputs:
        output['finished'] = np.all([(check_exist(f) if n.endswith('_file') else (f is not None)) for n, f in output.items()])
    finished = np.all([output['finished'] for output in outputs])
    return finished


def calculate_min_patch(fname, warp_res=10):
    '''
    warp_res : float
        Effective "warp resolution" in mm.
    '''
    dx = np.median(afni.get_head_delta(fname)) # Get in-plane resolution (assumed to be equal in two directions), regardless of slice orientation
    min_patch = max(9, math.nearest(warp_res*2/dx, 'odd'))
    return min_patch


def blip_unwarp(forward_file, reverse_file, reverse_loc, out_file, PE_axis='AP', min_patch=None):
    '''
    abin/unWarpEPI.py do this through: unwarp estimate (before tshift) > tshift > unwarp apply > motion correction > concat apply
    '''
    temp_dir = utils.temp_folder()
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = {
        'out_file': f"{out_dir}{prefix}{ext}",
        'for2mid_file': f"{out_dir}{prefix}.for2mid.warp{ext}",
        'rev2mid_file': f"{out_dir}{prefix}.rev2mid.warp{ext}",
        'QC_forward_file': f"{out_dir}QC.{prefix}.forward{ext}",
        'QC_reverse_file': f"{out_dir}QC.{prefix}.reverse{ext}",
        'template_idx': None,
    }

    # Prepare forward template
    n_subs = 5
    fwd_nt = afni.get_dims(forward_file)[3]
    fwd_sub = {'before': f"[0..{min(fwd_nt, n_subs)-1}]", 'after': f"[{max(0, fwd_nt-n_subs)}..$]"}[reverse_loc]
    fwd_n = min(n_subs, fwd_nt)
    outputs['template_idx'] = {'before': fwd_n//2, 'after': (max(0, fwd_nt-n_subs) + fwd_nt)//2}[reverse_loc]
    if fwd_n > 1:
        # There is no need to protect the paths against whitespace (yet) because AFNI doesn't support it.
        utils.run(f"3dvolreg -zpad 1 -base {fwd_n//2} -prefix {temp_dir}/forwards.nii -overwrite {forward_file}'{fwd_sub}'")
        utils.run(f"3dTstat -median -prefix {temp_dir}/forward.nii -overwrite {temp_dir}/forwards.nii")
    else:
        utils.run(f"3dcopy {temp_dir}/forwards.nii {temp_dir}/forward.nii -overwrite ")
    utils.run(f"3dAutomask -apply_prefix {temp_dir}/forward.masked.nii -overwrite {temp_dir}/forward.nii")

    # Prepare reverse template
    rev_nt = afni.get_dims(reverse_file)[3]
    rev_sub = {'before': f"[{max(0, rev_nt-n_subs)}..$]", 'after': f"[0..{min(rev_nt, n_subs)-1}]"}[reverse_loc]
    rev_n = min(n_subs, rev_nt)
    # Input datasets for 3dQwarp must be on the same 3D grid (unlike program 3dAllineate)!
    utils.run(f"3dresample -rmode NN -master {temp_dir}/forwards.nii \
        -prefix {temp_dir}/reverses.nii -overwrite -input {reverse_file}'{rev_sub}'")
    if rev_n > 1:
        utils.run(f"3dvolreg -zpad 1 -base {rev_n//2} -prefix {temp_dir}/reverses.nii -overwrite {temp_dir}/reverses.nii")
        utils.run(f"3dTstat -median -prefix {temp_dir}/reverse.nii -overwrite {temp_dir}/reverses.nii")
    else:
        utils.run(f"3dcopy {temp_dir}/reverses.nii {temp_dir}/reverse.nii -overwrite ")
    utils.run(f"3dAutomask -apply_prefix {temp_dir}/reverse.masked.nii -overwrite {temp_dir}/reverse.nii")
    
    # Estimate nonlinear midpoint transform
    fix_non_PE_axes = {'AP': '-noXdis -noZdis', 'LR': '-noYdis -noZdis', None: ''}[PE_axis]
    # Minimal value allowed by 3dQwarp is 9, which is 27 mm for 3x3x3 voxels as in unWarpEPI.py
    # We use patch_size=15 mm by default here for typical hires data (1.2, 1.0, or 0.8 mm iso)
    if min_patch is None:
        min_patch = calculate_min_patch(forward_file)
    # Compute the midpoint warp from reverse to forward
    # Among the resultant files:
    # - *_For_WARP is for warping forward to midpoint
    # - *_Rev_WARP is for warping reverse to midpoint
    # Even though compiled with OpenMP, this step is very slow...
    utils.run(f"3dQwarp -plusminus -pmNAMES Rev For \
        -pblur 0.05 0.05 -blur -1 -1 \
        -noweight {fix_non_PE_axes} -minpatch {min_patch} \
        -base   {temp_dir}/forward.masked.nii \
        -source {temp_dir}/reverse.masked.nii \
        -prefix {temp_dir}/{prefix}{ext} -overwrite")
    
    # Copy transform files
    utils.run(f"3dcopy {temp_dir}/{prefix}_For_WARP{ext} {outputs['for2mid_file']}")
    utils.run(f"3dcopy {temp_dir}/{prefix}_Rev_WARP{ext} {outputs['rev2mid_file']}")

    # Apply warp to get output
    utils.run(f"3dNwarpApply -quintic -nwarp {temp_dir}/{prefix}_For_WARP{ext} \
        -source {forward_file} -prefix {out_file} -overwrite")

    # Apply warp to unmasked templates for quality check
    utils.run(f"3dNwarpApply -quintic -nwarp {temp_dir}/{prefix}_For_WARP{ext} \
        -source {temp_dir}/forward.nii -prefix {outputs['QC_forward_file']} -overwrite")
    utils.run(f"3dNwarpApply -quintic -nwarp {temp_dir}/{prefix}_Rev_WARP{ext} \
        -source {temp_dir}/reverse.nii -prefix {outputs['QC_reverse_file']} -overwrite")

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def correct_bias_field(in_file, out_file, n_jobs=None):
    if n_jobs is None:
        n_jobs = DEFAULT_JOBS
    if shutil.which('N4BiasFieldCorrection') is None:
        raise ValueError('>> ANTs is not (correctly) installed. So cannot use N4BiasFieldCorrection.')
    prefix, ext = afni.split_out_file(out_file)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(n_jobs*2) # N4BiasFieldCorrection tends to use half of max
    utils.run(f"N4BiasFieldCorrection -d 3 -i {in_file} -s 2 -o \
        '[{prefix}{ext},{prefix}_bias{ext}]'")


def correct_motion(base_file, in_file, out_file, algorithm='3dvolreg', mode='rigid'):
    '''
    algorithm : {'3dvolreg', '3dAllineate'}
    '''
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'param_file': f"{prefix}.param.1D",
        'xform_file': f"{prefix}.aff12.1D",
        'template_idx': None,
    }

    if algorithm.lower() == '3dvolreg':
        if mode in ['rigid', 6]:
            utils.run(f"3dvolreg -base {base_file} \
                -verbose -zpad 2 \
                -1Dfile {outputs['param_file']} -1Dmatrix_save {outputs['xform_file']} \
                -prefix {out_file} -overwrite {in_file}")
        else:
            raise ValueError('Valid mode includes {"rigid"} for "3dvolreg" algorithm.')
    elif algorithm.lower() == '3dallineate':
        if mode in ['rigid', 6]:
            # The parameters are based on unWarpEPI.py and align_epi_anat.py
            # '-onepass': If you expect only small-ish (< 2 voxels?) image movement, skipping the coarse resolution first pass mfter_indexes sense.
            # '-fineblur': The blur used in the refining pass.
            # '-norefinal': Skip restart after convergence (in case of local minima) in the interest of time 
            utils.run(f"3dAllineate -final quintic -base {base_file} \
                -warp shift_rotate -lpa -automask+2 -source_automask+2 \
                -onepass -fineblur 2 -norefinal \
                -1Dparam_save {outputs['param_file']} -1Dmatrix_save {outputs['xform_file']} \
                -prefix {out_file} -overwrite -source {in_file}")
        else:
            raise ValueError('Valid mode includes {"rigid"} for "3dAllineate" algorithm.')

    if isinstance(base_file, six.string_types):
        pass
    else:
        outputs['template_idx'] = base_file

    all_finished(outputs)
    return outputs


def combine_affine_transforms(transforms, out_file=None):
    '''
    Note that: 
        1. Each transform can be modified by -I, -S, -P, e.g., transforms=['T1_al.aff12.1D -I']
        2. An oneline transform could contain multiple lines, one for each time point.
    '''
    if isinstance(transforms, six.string_types):
        transforms = [transforms]
    if out_file is None:
        out_file = utils.temp_prefix(suffix='.aff12.1D')
    outputs = {'out_file': out_file}
    with open(out_file, 'w') as fo:
        for line in utils.run(f"cat_matvec {' '.join(transforms)} -ONELINE")['output']:
            fo.write(line)
    all_finished(outputs)
    return outputs


def is_affine_transform(fname):
    return fname.endswith('.1D') or fname.endswith('.1D -I')


def apply_transforms(transforms, base_file, in_file, out_file, interp=None, res=None, save_xform=None):
    '''
    Note that last transform applys first, as in AFNI.
    Transforms can be modified as "xform.aff12.1D -I" for inversion.
    '''
    if isinstance(transforms, six.string_types):
        transforms = [transforms]
    if interp is None:
        interp = 'wsinc5'
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    if save_xform is not None:
        outputs['xform_file'] = save_xform
    
    has_nwarp = not np.all([is_affine_transform(f) for f in transforms])
    res_cmd = f"-newgrid {res}" if res is not None else ''
    if has_nwarp:
        transform_list = ' '.join([f"INV({t[:-3]})" if t.endswith(' -I') else t for t in transforms])
        # 'transform_list' must be quoted
        # -nwarp requires at least one nonlinear transform, can be IDENT(base_file.nii)
        utils.run(f"3dNwarpApply -interp {interp} -master {base_file} {res_cmd} \
            -nwarp '{transform_list}' -source {in_file} \
            -prefix {out_file} -overwrite")
    else:
        combined = combine_affine_transforms(transforms, out_file=save_xform)['out_file'] # This will handle "-I"
        # 'wsinc5' is 8x slower than 'quintic', but is highly accurate 
        # and should reduce the smoothing artifacts (see 3dAllineate)
        utils.run(f"3dAllineate -final {interp} -base {base_file} {res_cmd} \
            -1Dmatrix_apply {combined} -input {in_file} \
            -prefix {out_file} -overwrite")
        if not save_xform:
            os.remove(combined)

    all_finished(outputs)
    return outputs


def resample(in_file, out_file, res=None, base_file=None, interp=None):
    temp_dir = utils.temp_folder()
    identity = f"{temp_dir}/identity.aff12.1D"
    io.write_affine(identity, np.c_[np.diag([1,1,1]), np.zeros(3)])
    if base_file is None:
        if res is None:
            raise ValueError('>> You must either specify `res` or `base_file`.')
        base_file = f"{temp_dir}/base.nii"
        dxyz = ' '.join([f"{d:g}" for d in res])
        utils.run(f"3dresample -rmode NN -dxyz {dxyz} -prefix {base_file} -overwrite -input {in_file}")
        apply_transforms([identity], base_file, in_file, out_file, interp=interp)
    else:
        if res is None:
            apply_transforms([identity], base_file, in_file, out_file, interp=interp)
        else:
            raise NotImplementedError

    shutil.rmtree(temp_dir)


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
        utils.run(f"3dcopy {in_file} {out_file} -overwrite")
    else:
        np.savetxt(param_file, param, fmt='%.6e')
        utils.run(f"3dAllineate -final {interp} -base {in_file} \
            -1Dparam_apply {param_file} -input {in_file} \
            -prefix {out_file} -overwrite")
        os.remove(param_file)


def nudge_cmd2mat(nudge_cmd, in_file, return_inverse=False):
    '''
    Refer to "Example 4"@SUMA_AlignToExperiment for details.
    
    The 1st return (MAT) is what you want to write as 'src2base.aff12.1D' and use 
    apply_transforms to get the same effect as your manual nudge. Remember, the 
    AFNI way is to store inverse matrix under forward name (to "pull" data from src).
    So the mathematical map from moveable to tempalte is in the 2nd return (INV).
    '''
    match = re.search(r'(-?\d+\.\d{2}I) (-?\d+\.\d{2}R) (-?\d+\.\d{2}A).+?(-?\d+\.\d{2}S) (-?\d+\.\d{2}L) (-?\d+\.\d{2}P)', nudge_cmd)
    if match:
        I, R, A, S, L, P = match.groups()
        temp_file = utils.temp_prefix(suffix='.nii')
        # Note that we have to explicitly specify output dir, otherwise it will be put in input dir...
        utils.run(f"3drotate -NN -clipit -rotate {I} {R} {A} -ashift {S} {L} {P} -prefix ./{temp_file} {in_file}")
        # Extract the matrix that will apply the same rotation as the command
        mat = afni.check_output(f"cat_matvec '{temp_file}::ROTATE_MATVEC_000000' -I -ONELINE")[-2]
        # As well as its inverse for convenience
        inv = afni.check_output(f"cat_matvec '{temp_file}::ROTATE_MATVEC_000000' -ONELINE")[-2]
        os.remove(temp_file)
        mat = np.float_(mat.split()).reshape(3,4)
        inv = np.float_(inv.split()).reshape(3,4)
        if return_inverse:
            return mat, inv
        else:
            return mat
    else:
        raise ValueError(f"`nudge_cmd` should contain something like '-rotate 0.00I -20.00R 0.00A -ashift 13.95S -2.00L -11.01P'")


def copy_S2E_mat(src, dst):
    # This is the S2E mat in afni sense (because when aligning sv to exp, source=sv, base=exp)
    utils.run(f"3drefit -atrcopy {src} ALLINEATE_MATVEC_S2B_000000 {dst}")
    # This is the E2S mat
    utils.run(f"3drefit -atrcopy {src} ALLINEATE_MATVEC_B2S_000000 {dst}")


def align_epi(in_files, out_files, best_reverse=None, blip_results=None, blip_kws=None, volreg_kws=None, 
    template=None, template_pool=None, template_candidate_runs=None, final_resample=True, final_res=None):
    '''
    Parameters
    ----------
    blip_results : list or dict
        E.g., {'warp_file': '*.volreg.warp.nii', 'blip_file': '*.volreg.blip.nii'}
    '''
    temp_dir = utils.temp_folder()
    pc = utils.PooledCaller()
    outputs = []
    if isinstance(out_files, six.string_types): # If out_files is specified as a suffix
        out_dir, suffix = afni.split_out_file(out_files, split_path=True, trailing_slash=True)[:-1]
        insert_suffix = lambda prefix, ext: f"{out_dir}{prefix}{suffix}{ext}"
        out_files = [insert_suffix(*afni.split_out_file(in_file, split_path=True)[1:]) for in_file in in_files] # Now out_files is a list of filenames
    for out_file in out_files:
        prefix, ext = afni.split_out_file(out_file)
        output = {
            'out_file': f"{prefix}{ext}",
            'param_file': f"{prefix}.param.1D",
            'xform_file': f"{prefix}.aff12.1D",
        }
        if best_reverse is not None and blip_results is None:
            output['warp_file'] = f"{prefix}.warp{ext}"
            output['blip_file'] = f"{prefix}.blip{ext}"
        outputs.append(output)
    temp_prefixs = [path.join(temp_dir, afni.split_out_file(in_file, split_path=True)[1]) for in_file in in_files]
    dt = template_pool//2 if template_pool is not None else 0
    
    # Correct phase encoding distortion
    # `files` is a list of (file_to_be_volreg, volreg_template_idx)
    if blip_results is not None:
        if isinstance(blip_results, dict):
            # Construct full blip_results from glob pattern
            if isinstance(blip_results['blip_file'], six.string_types):
                if utils.contain_wildcard(blip_results['blip_file']): # Wildcard
                    blip_results['blip_file'] = sorted(glob.glob(blip_results['blip_file']))
                # 'blip_file' should not be a list of constant
            if isinstance(blip_results['warp_file'], six.string_types):
                if utils.contain_wildcard(blip_results['warp_file']): # Wildcard
                    blip_results['warp_file'] = sorted(glob.glob(blip_results['warp_file']))
                else: # Constant
                    blip_results['warp_file'] = [blip_results['warp_file']] * len(blip_results['blip_file'])
            if 'template_idx' not in blip_results:
                blip_results['template_idx'] = [afni.get_dims(blip_file)[3]//2 for blip_file in blip_results['blip_file']]
            blip_results = [{key: val[k] for key, val in blip_results.items()} for k in range(len(blip_results['blip_file']))]
        files = [(blip_result['blip_file'], blip_result['template_idx']) for blip_result in blip_results]
    elif best_reverse is not None:
        blip_kws = {} if blip_kws is None else blip_kws
        for k, forward in enumerate(in_files):
            reverse, loc = best_reverse[forward]
            pc.run(blip_unwarp, forward, reverse, loc, f"{temp_prefixs[k]}.blip.nii", **blip_kws)
        blip_outputs = pc.wait(pool_size=4)
        files = [(output['out_file'], output['template_idx']) for output in blip_outputs]
    else: # volreg only
        files = [(in_file, afni.get_dims(in_file)[3]//2) for in_file in in_files]

    # # Correct within-run motion
    # volreg_kws = {} if volreg_kws is None else volreg_kws
    # for k, (fi, local_idx) in enumerate(files):
    #     pc.run(correct_motion, local_idx, fi, f"{temp_prefixs[k]}.within.nii", **volreg_kws)
    # volreg_outputs = pc.wait()

    # # Correct between-run motion
    # mean5_files = []
    # for k, output in enumerate(volreg_outputs):
    #     sel = f"{output['template_idx']-2}..{output['template_idx']+2}"
    #     mean5_files.append(f"{temp_prefixs[k]}.mean5.nii")
    #     pc.run(f"3dTstat -mean -prefix {mean5_files[-1]} -overwrite {output['out_file']}'[{sel}]'")
    # pc.wait()
    # utils.run(f"3dTcat -prefix {temp_dir}/allrun.mean5.nii -overwrite {' '.join(mean5_files)}")
    # correct_motion(0, f"{temp_dir}/allrun.mean5.nii", f"{temp_dir}/allrun.volreg.nii", **volreg_kws)
    # X = np.loadtxt(f"{temp_dir}/allrun.volreg.param.1D")
    # global_idx = np.argmin([np.sqrt(np.sum(np.linalg.norm(X-x, axis=1)**2)) for x in X])
    # os.remove(f"{temp_dir}/allrun.volreg.param.1D") # If not removed, this will give a warning which startswith('**')
    # correct_motion(global_idx, f"{temp_dir}/allrun.mean5.nii", f"{temp_dir}/allrun.volreg.nii", **volreg_kws)

    # # Concat linear transforms for within- and between-run motion correction (*.volreg.aff12.1D)
    # # The transforms are applied in backward order: bottom to top, right to left
    # for k, output in enumerate(outputs):
    #     utils.run(f"3dTcat -prefix {temp_prefixs[k]}.between.aff12.1D -overwrite {temp_dir}/allrun.volreg.aff12.1D'{{{k}}}'")
    #     utils.run(f"cat_matvec -ONELINE {temp_prefixs[k]}.between.aff12.1D {temp_prefixs[k]}.within.aff12.1D > {output['xform_file']}", shell=True)

    # # Generate overall motion correction parameters (*.volreg.param.1D)
    # # At present, the only way to do this is to run 3dvolreg again...
    # fi, local_idx = files[global_idx]
    # sel = f"{local_idx-2}..{local_idx+2}"
    # utils.run(f"3dTstat -median -prefix {temp_dir}/template.volreg.nii -overwrite {fi}'[{sel}]'")
    # pc(pc.run(correct_motion, f"{temp_dir}/template.volreg.nii", fi, f"{temp_prefixs[k]}.volreg.nii", **volreg_kws) for k, (fi, idx) in enumerate(files))

    volreg_kws = {} if volreg_kws is None else volreg_kws
    if not isinstance(template, six.string_types):
        if template is None:
            # Correct motion: first pass
            utils.run(f"3dTcat -prefix {temp_dir}/template.pass1.nii -overwrite {files[0][0]}'[{files[0][1]}]'")
            pc(pc.run(correct_motion, f"{temp_dir}/template.pass1.nii", files[k][0], f"{prefix}.pass1.nii", **volreg_kws) for k, prefix in enumerate(temp_prefixs))
            # Consider only template_candidate_runs
            n_all_runs = len(temp_prefixs)
            if template_candidate_runs is None:
                template_candidate_runs = list(range(n_all_runs))
            # Find best template
            Xs = [np.loadtxt(f"{prefix}.pass1.param.1D") for k, prefix in enumerate(temp_prefixs) if k in template_candidate_runs]
            XX = np.vstack(Xs)
            idx = np.argmin([np.sqrt(np.sum(np.linalg.norm(XX-x, axis=1)**2)) for x in XX])
            L = [X.shape[0] for X in Xs]
            D = idx - np.cumsum(L)
            run_idx = np.nonzero(D<0)[0][0]
            TR_idx = L[run_idx] + D[run_idx]
            # Get run_idx within all runs from run_idx within "candidate" runs
            all_runs = np.zeros(n_all_runs)
            sel_runs = all_runs[template_candidate_runs]
            sel_runs[run_idx] = 1
            all_runs[template_candidate_runs] = sel_runs
            run_idx = np.nonzero(all_runs)[0][0]
        else:
            run_idx, TR_idx = template
        base_file = files[run_idx][0]
        n_TRs = afni.get_dims(base_file)[3]
        template = f"{base_file}'[{max(0, TR_idx-dt)}..{min(n_TRs, TR_idx+dt)}]'"
    # Correct motion: second pass (based on best templated)
    utils.run(f"3dTstat -median -prefix {temp_dir}/template.pass2.nii -overwrite {template}")
    pc(pc.run(correct_motion, f"{temp_dir}/template.pass2.nii", files[k][0], f"{prefix}.pass2.nii", **volreg_kws) for k, prefix in enumerate(temp_prefixs))

    # Generate final outputs in one resample step
    if final_resample:
        if blip_results is not None:
            pc((pc.run(apply_transforms, [f"{prefix}.pass2.aff12.1D", blip_result['warp_file']], \
                f"{temp_dir}/template.pass2.nii", in_file, output['out_file'], res=final_res) \
                for k, (prefix, blip_result, in_file, output) in enumerate(zip(temp_prefixs, blip_results, in_files, outputs))), pool_size=4)
        elif best_reverse is not None:
            pc((pc.run(apply_transforms, [f"{prefix}.pass2.aff12.1D", f"{prefix}.blip.for2mid.warp.nii"], \
                f"{temp_dir}/template.pass2.nii", in_file, output['out_file'], res=final_res) \
                for k, (prefix, in_file, output) in enumerate(zip(temp_prefixs, in_files, outputs))), pool_size=4)
        else: # volreg only
            pc((pc.run(apply_transforms, [f"{prefix}.pass2.aff12.1D"], \
                f"{temp_dir}/template.pass2.nii", in_file, output['out_file'], res=final_res) \
                for k, (prefix, in_file, output) in enumerate(zip(temp_prefixs, in_files, outputs))), pool_size=4)
    else:
        pc(pc.run(f"3dcopy {prefix}.pass2.nii {output['out_file']} -overwrite") for prefix, output in zip(temp_prefixs, outputs))

    # Copy other results
    if best_reverse is not None and blip_results is None:
        pc(pc.run(f"3dcopy {prefix}.blip.for2mid.warp.nii {output['warp_file']} -overwrite") for prefix, output in zip(temp_prefixs, outputs))
        pc(pc.run(f"3dcopy {prefix}.blip.nii {output['blip_file']} -overwrite") for prefix, output in zip(temp_prefixs, outputs))
    pc(pc.run(shutil.copy, f"{prefix}.pass2.aff12.1D", output['xform_file']) for prefix, output in zip(temp_prefixs, outputs))
    pc(pc.run(shutil.copy, f"{prefix}.pass2.param.1D", output['param_file']) for prefix, output in zip(temp_prefixs, outputs))

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def retrieve_mp2rage_labels(dicom_dirs, dicom_ext='.IMA'):
    '''
    Retrieve mp2rage subvolume labels like UNI, ND, etc.
    
    Parameters
    ----------
    dicom_dirs : list or str
        A list of dicom file folders, e.g., ['T101', 'T102', ...], or 
        a glob pattern like 'raw_fmri/T1??'
    
    Returns
    -------
    label2dicom_dir : OrderedDict
        label -> (index, dicom_dir)
    '''
    if isinstance(dicom_dirs, six.string_types):
        dicom_dirs = [d for d in sorted(glob.glob(dicom_dirs)) if path.isdir(d)]
    label2dicom_dir = OrderedDict()
    for index, dicom_dir in enumerate(dicom_dirs):
        f = glob.glob(f"{dicom_dir}/*{dicom_ext}")[0]
        header = dicom.parse_dicom_header(f)
        label = header['SeriesDescription'][len(header['ProtocolName'])+1:]
        label2dicom_dir[label] = (index, dicom_dir)
    return label2dicom_dir


def assign_mp2rage_labels(T1s, dicom_dirs, dicom_ext='.IMA'):
    if isinstance(T1s, six.string_types):
        T1s = sorted(glob.glob(T1s))
    labels = list(retrieve_mp2rage_labels(dicom_dirs, dicom_ext=dicom_ext).keys())
    assert(len(T1s) == len(labels))
    for T1, label in zip(T1s, labels):
        afni.set_attribute(T1, 'mp2rage_label', label)


def create_mp2rage_SNR_mask(T1s, out_file):
    '''
    Need to call prep.assign_mp2rage_labels() first.
    '''
    temp_dir = utils.temp_folder()
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = {
        'out_file': f"{out_dir}{prefix}{ext}",
        'thres_file': f"{out_dir}{prefix}_thres{ext}",
        'ths': None,
        'y': None,
        'th': None,
    }
    # Retrieve mp2rage labels like INV2_ND, UNI_Images, etc.
    if isinstance(T1s, six.string_types):
        T1s = sorted(glob.glob(T1s))
    label2T1 = OrderedDict((afni.get_attribute(T1, 'mp2rage_label'), (k, T1)) for k, T1 in enumerate(T1s))
    # Smooth INV2_ND to estimate the intensity profile
    utils.run(f"3dmerge -1blur_fwhm 6 -doall -prefix {outputs['thres_file']} -overwrite {label2T1['INV2_ND'][1]}")
    INV2 = io.read_vol(outputs['thres_file']).ravel()
    UNI = io.read_vol(label2T1['UNI_Images'][1]).ravel()
    # Determine best threshold
    # Exploit the fact that: 1) Noisy UNI is like Gaussian 2) Gaussian has large entropy/std
    ths = []
    y = []
    intensities = np.percentile(INV2, np.arange(100))
    for k, th in enumerate(intensities[1:], start=1):
        if th - intensities[k-1] < 5:
            continue
        ths.append(th)
        # y.append(stats.entropy(np.unique(UNI[INV2>th], return_counts=True)[1]))
        y.append(np.std(UNI[INV2>th]))
    th = ths[np.argmax(y)]
    outputs['ths'] = ths
    outputs['y'] = y
    outputs['th'] = th
    # Create mask
    utils.run(f"3dcalc -a {outputs['thres_file']} -expr 'step(a-{th})' -prefix {outputs['out_file']} -overwrite")
    # Modified mask (restricting to region with high correlation)
    utils.run(f"3dLocalBistat -nbhd 'SPHERE(3)' -stat pearson -mask {outputs['out_file']} \
        -prefix {temp_dir}/corr.nii -overwrite {label2T1['INV2_ND'][1]} {label2T1['UNI_Images'][1]}")
    utils.run(f"3dcalc -a {temp_dir}/corr.nii -expr 'step(a-0.3)' -prefix {temp_dir}/good.nii -overwrite")
    utils.run(f"3dmerge -1blur_fwhm 10 -doall -prefix {temp_dir}/good_smoothed.nii -overwrite {temp_dir}/good.nii")
    utils.run(f"3dcalc -a {temp_dir}/good_smoothed.nii -expr 'step(a-0.3)' -prefix {temp_dir}/good_mask.nii -overwrite")
    utils.run(f"3dmask_tool -input {temp_dir}/good_mask.nii -prefix {temp_dir}/good_mask.nii -overwrite -dilate_input 5")
    utils.run(f"3dcalc -a {outputs['out_file']} -b {temp_dir}/good_mask.nii -expr 'step(a)*step(b)' \
        -prefix {outputs['out_file']} -overwrite")
    
    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def prep_mp2rage(dicom_dirs, out_file='T1.nii', unwarp=False, dicom_ext='.IMA'):
    '''Convert dicom files and remove the noise pattern outside the brain.

    dicom_dirs : list or str
        A list of dicom file folders, e.g., ['T101', 'T102', ...], or 
        a glob pattern like 'raw_fmri/T1??'
    '''
    pc = utils.PooledCaller()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'ns_file': f"{prefix}_ns{ext}",
    }
    if unwarp:
        outputs.update({
            'corr_file': f"{prefix}_corr{ext}",
            'corr_ns_file': f"{prefix}_corr_ns{ext}",
            'warp_file': f"{prefix}_corr.warp{ext}",
        })

    if not all_finished(outputs):
        temp_dir = utils.temp_folder()
        # Retrieve dicom information (for labels like UNI, ND, etc.)
        label2dicom_dir = retrieve_mp2rage_labels(dicom_dirs, dicom_ext=dicom_ext)

        # Convert dicom files
        for label in ['UNI_Images', 'INV2_ND', 'INV2']:
            pc.run(io.convert_dicom, label2dicom_dir[label][1], f"{temp_dir}/{label}.nii", dicom_ext=dicom_ext)
        pc.wait()

        # Generate skull strip mask
        utils.run(f"3dcalc -a {temp_dir}/UNI_Images.nii -b {temp_dir}/INV2_ND.nii \
            -expr 'a*b' -float -prefix {temp_dir}/INV2xUNI.nii -overwrite") # INV2*UNI is the recommended method by {Fujimoto2014}, but too aggressive with 3dSkullStrip
        for label in ['INV2_ND', 'INV2xUNI']:
            pc.run(f"3dSkullStrip -orig_vol -prefix {temp_dir}/{label}_ns.nii -overwrite -input {temp_dir}/{label}.nii")
        pc.wait()
        utils.run(f"3dcalc -a {temp_dir}/INV2_ND_ns.nii -b {temp_dir}/INV2xUNI_ns.nii \
            -expr 'max(step(a),step(b))' -prefix {temp_dir}/mask.nii -overwrite") # Tfter_indexe the union of the two masks as the final brain mask
        utils.run(f"3dmask_tool -dilate_input -1 1 -prefix {temp_dir}/mask.nii -overwrite -input {temp_dir}/mask.nii") # Remove "spikes" on the surface of the mask

        # Generate merged T1
        utils.run(f"3dcalc -a {temp_dir}/UNI_Images.nii -m {temp_dir}/mask.nii \
            -expr 'a*m' -prefix {outputs['ns_file']} -overwrite")
        utils.run(f"3dcalc -a {temp_dir}/UNI_Images.nii -b {temp_dir}/INV2_ND.nii -m {temp_dir}/mask.nii \
            -expr 'a*m+2*b*(1-m)' -prefix {outputs['out_file']} -overwrite") # Generate "merged" file

        if unwarp:
            # Estimate distortion correction transform
            correction = 'DIS2D'
            min_patch = calculate_min_patch(f"{temp_dir}/INV2.nii")
            utils.run(f"3dQwarp -blur 1 1 -minpatch {min_patch} \
                -base {temp_dir}/INV2.nii \
                -source {temp_dir}/INV2_ND.nii \
                -prefix {temp_dir}/{correction}.nii -overwrite")

            # Apply transform
            for fi, fo in zip(['out_file', 'ns_file'], ['corr_file', 'corr_ns_file']):
                pc.run(apply_transforms, f"{temp_dir}/{correction}_WARP.nii", f"{temp_dir}/INV2.nii", outputs[fi], outputs[fo])
            pc.wait()
            os.rename(f"{temp_dir}/{correction}_WARP.nii", outputs['warp_file'])
        shutil.rmtree(temp_dir)
    else:
        print('>> Reuse existing results.')

    all_finished(outputs)
    return outputs


def prep_mp2rages(data_dir, sessions=None, subdir_pattern='T1??', unwarp=True, **kwargs):
    if sessions is None:
        sessions = dicom_report.inspect_mp2rage(data_dir, subdir_pattern=subdir_pattern).session
    pc = utils.PooledCaller(pool_size=4)
    for session_dir in [f'{data_dir}/{session}' for session in sessions]:
        out_file = kwargs.pop('out_file') if 'out_file' in kwargs else 'T1.nii'
        pc.run(prep_mp2rage, f'{session_dir}/{subdir_pattern}', out_file=f'{session_dir}/{out_file}', unwarp=unwarp, **kwargs)
    outputs = pc.wait()
    return OrderedDict([(session, output) for session, output in zip(sessions, outputs)])


def average_anat(T1s, out_file, template_idx=0, T1s_ns=None, weight=None):
    prefix, ext = afni.split_out_file(out_file)
    os.makedirs(prefix, exist_ok=True)
    N = len(T1s)
    pc = utils.PooledCaller()
    outputs = {
        'out_file': [f"{prefix}/T1{k+1:02d}.nii" for k in range(N)],
        'ns_file': [f"{prefix}/T1{k+1:02d}_ns.nii" for k in range(N)],
        'cost': None,
    }
    if T1s_ns is None:
        T1s_ns = [f"{prefix}/T1{k+1:02d}_ns.nii" for k in range(N)]
        pc(pc.run(skullstrip, T1, T1_ns) for T1, T1_ns in zip(T1s, T1s_ns))
    elif T1s_ns == 'default':
        T1s_ns = [afni.insert_suffix(T1, '_ns') for T1 in T1s]
    for k in range(N):
        if k == template_idx:
            pc.run(copy_dset, T1s_ns[k], outputs['ns_file'][k])
        else:
            pc.run(align_anat, T1s_ns[template_idx], T1s_ns[k], outputs['ns_file'][k], strip=False)
    align_outputs = pc.wait(pool_size=4)
    outputs['cost'] = [(o['cost']['lpa'] if o is not None else np.nan) for o in align_outputs]
    for k in range(N):
        if k == template_idx:
            pc.run(copy_dset, T1s[template_idx], outputs['out_file'][k])
        else:
            pc.run(apply_transforms, align_outputs[k]['xform_file'], T1s[template_idx], T1s[k], outputs['out_file'][k])
    pc.wait()
    if weight is None:
        pc.run1(f"3dMean -prefix {prefix}{ext} -overwrite {' '.join(outputs['out_file'])}")
    else:
        raise NotImplementedError()

    all_finished(outputs)
    return outputs


def fs_recon(T1s, out_dir, T2=None, FLAIR=None, NIFTI=True, hires=True, fs_ver=None, V1=True, HCP_atlas=True, n_jobs=None):
    '''
    Parameters
    ----------
    T1s : list of str | 'brainmask_edit' | 'wm_edit'
    fs_ver : {'v6', 'v6.hcp', 'skip'}
    '''
    start_time = time.time()
    assert(utils.has_hcp_retino_docker())
    edits = ['brainmask_edit', 'wm_edit']
    if isinstance(T1s, six.string_types) and T1s not in edits:
        T1s = [T1s]
    out_dir = path.realpath(out_dir)
    subjects_dir, subj = path.split(out_dir) # Environment variable may need full path
    temp_dir = utils.temp_folder()
    outputs = {
        'subj_dir': out_dir,
        'suma_dir': f"{out_dir}/SUMA",
    }
    if n_jobs is None:
        n_jobs = DEFAULT_JOBS
    # Setup FreeSurfer SUBJECTS_DIR
    if not path.exists(subjects_dir):
        os.makedirs(subjects_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    if not path.exists(f'{subjects_dir}/V1_average'):
        os.symlink(f"{os.environ['FREESURFER_HOME']}/subjects/V1_average", f"{subjects_dir}/V1_average")
    # Run recon-all
    if fs_ver is None:
        fs_ver = 'v6' # {'v7', 'v6', 'v6.hcp', 'skip'}
    if T1s == 'brainmask_edit': # Manual edit brainmask.mgz
        if fs_ver == 'v6':
            hires_cmd = f"-hires" if hires else ''
            utils.run(f"recon-all -s {subj} \
                -autorecon-pial {hires_cmd} \
                -parallel -openmp {n_jobs}",
                error_pattern='', goal_pattern='recon-all .+ finished without error')
        elif fs_ver == 'skip':
            pass
    elif T1s == 'wm_edit': # Manual edit control points and wm.mgz (in addition to brainmask.mgz)
        if fs_ver == 'v6':
            hires_cmd = f"-hires" if hires else ''
            utils.run(f"recon-all -s {subj} \
                -autorecon2-cp -autorecon2-wm -autorecon-pial {hires_cmd} \
                -parallel -openmp {n_jobs}",
                error_pattern='', goal_pattern='recon-all .+ finished without error')
        elif fs_ver == 'skip':
            pass
    else: # Standard recon-all
        if hires:
            expert_file = f"{temp_dir}/expert_options.txt"
            with open(expert_file, 'w') as fo:
                fo.write('mris_inflate -n 30\n')
        if fs_ver in ['v7', 'v6']:
            hires_cmd = f"-hires -expert {expert_file}" if hires else ''
            utils.run(f"recon-all -s {subj} \
                {' '.join([f'-i {T1}' for T1 in T1s])} \
                {f'-T2 {T2} -T2pial' if T2 is not None else ''} \
                {f'-FLAIR {FLAIR} -FLAIRpial' if FLAIR is not None else ''} \
                -all {hires_cmd} \
                -parallel -openmp {n_jobs} \
                {'-label-v1' if V1 else ''}", 
                error_pattern='', goal_pattern='recon-all .+ finished without error')
        elif fs_ver == 'v6.hcp':
            hires_cmd = f"-conf2hires -expert {expert_file}" if hires else ''
            utils.run(f"recon-all.v6.hires -s {subj} \
                {' '.join([f'-i {T1}' for T1 in T1s])} \
                {f'-T2 {T2} -T2pial' if T2 is not None else ''} \
                {f'-FLAIR {FLAIR} -FLAIRpial' if FLAIR is not None else ''} \
                -all {hires_cmd} \
                -parallel -openmp {n_jobs} \
                {'-label-v1' if V1 else ''}", 
                error_pattern='', goal_pattern='recon-all .+ finished without error')
        elif fs_ver == 'skip':
            pass
    print('\n==============================\n')
    # Make SUMA dir and viewing script
    create_suma_dir(out_dir, NIFTI=False)
    os.rename(outputs['suma_dir'], outputs['suma_dir']+'_woNIFTI')
    create_suma_dir(out_dir, NIFTI=True)
    os.rename(outputs['suma_dir'], outputs['suma_dir']+'_NIFTI')
    os.symlink('SUMA'+('_NIFTI' if NIFTI else '_woNIFTI'), outputs['suma_dir']) # This will create a relative link
    if HCP_atlas:
        # Create HCP retinotopic atlas (benson14 template) using docker
        create_hcp_retinotopic_atlas(out_dir, NIFTI=NIFTI)

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    print(f'>> The surface reconstruction process for "{subj}" took {utils.format_duration(time.time() - start_time)}.')
    return outputs


def create_suma_dir(subj_dir, NIFTI=True):
    '''
    Notes about NIFTI
    -----------------
    If NIFTI=True, @SUMA_Make_Spec_FS will use mri_convert to convert 
        $surf_dir/orig.mgz (which is averaged and hires) to SurfVol.nii.
        Volumes will be in .nii and surfaces will be in .gii format.
        This is the preferable way to go.
    If NIFTI=False, @SUMA_Make_Spec_FS will use to3d to convert COR- files
        to SurfVol+orig.HEAD which is 1x1x1.
        Volumes will be in +orig.HEAD and surfaces will be in .asc format.
        This is only provided for backward compatibility.
    '''
    subj = path.split(subj_dir)[1]
    outputs = {
        'suma_dir': f"{subj_dir}/SUMA",
        'viewing_script': f"{subj_dir}/SUMA/run_suma",
    }
    # Make SUMA dir
    utils.run(f"@SUMA_Make_Spec_FS -sid {subj} -fspath {subj_dir} {'-NIFTI' if NIFTI else ''}",
        error_pattern='', goal_pattern='@SUMA_Make_Spec_FS .+ finished')
    # Create viewing script
    with open(outputs['viewing_script'], 'w') as fo:
        fo.write("afni -niml &\n")
        fo.write(f"suma -spec {subj}_both.spec -sv {subj}_SurfVol{'.nii' if NIFTI else '+orig'} &\n")
    all_finished(outputs)
    return outputs


def create_suma_script(spec_file, surf_vol, out_file, use_relpath=False):
    if use_relpath:
        out_dir = path.split(out_file)[0]
        spec_file = path.relpath(spec_file, out_dir)
        surf_vol = path.relpath(surf_vol, out_dir)
    with open(out_file, 'w') as fo:
        fo.write("afni -niml &\n")
        fo.write(f"suma -spec {spec_file} -sv {surf_vol} &\n")


def create_hcp_retinotopic_atlas(subj_dir, suma='SUMA', NIFTI=True):
    '''
    Create HCP retinotopic atlas (benson14 template) using docker,
    and convert the volume and surface datasets into SUMA format.
    '''
    if not utils.has_hcp_retino_docker():
        print('*+ Skip HCP retinotopic atlas generation for now...') # Report as a warning
        return
    subj_dir = path.realpath(subj_dir)
    outputs = {
        'varea_file': f"{subj_dir}/{suma}/benson14_varea.nii.gz",
        'angle_file': f"{subj_dir}/{suma}/benson14_angle.nii.gz",
        'eccen_file': f"{subj_dir}/{suma}/benson14_eccen.nii.gz",
        'sigma_file': f"{subj_dir}/{suma}/benson14_sigma.nii.gz",
    }
    # Generate benson14 atlas (if necessary)
    subjects_dir, subjid = path.split(subj_dir)
    if not utils.exists(f'{subj_dir}/mri/benson14_varea.mgz'):
        # To get the correct (old) version of docker image:
        # $ docker pull nben/neuropythy@sha256:2541ee29a8d6bc676d9c3622ef4a38a258dd90e06c02534996a1c8354f9ac888
        # $ docker tag b38ebfcf6477 nben/neuropythy:mripy
        # # Specifying -it will result in "the input device is not a TTY"
        # utils.run(f"docker run -it --rm -v {subjects_dir}:/subjects nben/neuropythy:mripy \
        #     benson14_retinotopy --verbose {subjid}")
        # # Without -i, it will not be able to be interrupted by Ctrl+C
        # utils.run(f"docker run -t --rm -v {subjects_dir}:/subjects nben/neuropythy:mripy \
        #     benson14_retinotopy --verbose {subjid}")
        # Without -t, it will not return outputs on-the-go
        utils.run(f"docker run -i --rm -v {subjects_dir}:/subjects nben/neuropythy:mripy \
            benson14_retinotopy --verbose {subjid}")
    # Convert benson14 atlas to SUMA format
    utils.run(f"mripy_curv2dset.ipy -s {subj_dir} --suma_dir {subj_dir}/{suma} -i benson14")
    # Redo conversion if high density SUMA is detected
    nodes1 = io.read_surf_mesh(f"{subj_dir}/{suma}/lh.inflated{'.gii' if NIFTI else '.asc'}")[0]
    nodes2 = io.read_surf_data(f"{subj_dir}/{suma}/lh.benson14_varea.niml.dset")[0]
    if len(nodes1) != len(nodes2): # Likely to be a high density SUMA, redo convertion
        utils.run(f"mripy_curv2dset.ipy -s {subj_dir} --suma_dir {subj_dir}/{suma} \
            -i benson14 -m lh.inflated{'.gii' if NIFTI else '.asc'}")

    all_finished(outputs)
    return outputs


def irregular_resample(transforms, xyz, in_file, order=3):
    '''
    Parameters
    ----------
    transforms : list of str
        In order to project raw EPI in volume (in RAI) to surface coordinates (in LPI aka RAS+),
        i.e., surf_xyz = [dicom2nifti, exp2surf, volreg2template, unwarp2template, ijk2xyz] @ ijk,
        what we really do is mapping backward from xyz (assumed already in RAI):
        xyz -> E2A storing inv(exp2surf) -> ... -> inv(MAT).
    xyz : Nx3 array, assumed in DICOM RAI as with AFNI volumes.
        Note that FreeSurfur surface vertices are in NIFTI LPI aka RAS+, 
        whereas AFNI uses DICOM RAI internally.
    '''
    vol, xyz2ijk = (io.read_vol(in_file), math.invert_affine(afni.get_affine(in_file))) \
        if isinstance(in_file, six.string_types) else in_file
    transforms = [(io.read_affine(f) if is_affine_transform(f) else io.read_warp(f)) \
        if isinstance(f, six.string_types) else f for f in transforms]
    transforms += [xyz2ijk, None] # 'None' makes the end of all transforms
    xyz = xyz.T # For easier algebraic manipulation
    mat = None
    for xform in transforms:
        if xform is not None and not isinstance(xform, tuple):
            # Accumulate linear transforms into a combined affine matrix
            mat = xform if mat is None else math.concat_affine(xform, mat)
        else:
            # Apply accumulated linear transforms until now all at once
            xyz = math.apply_affine(mat, xyz)
            mat = None
            if xform is not None:
                # Apply non-linear transform
                dX, dY, dZ, iMAT = xform
                dx = interpolation.map_coordinates(dX, math.apply_affine(iMAT, xyz))
                dy = interpolation.map_coordinates(dY, math.apply_affine(iMAT, xyz))
                dz = interpolation.map_coordinates(dZ, math.apply_affine(iMAT, xyz))
                xyz = xyz + np.array([dx, dy, dz]) # Note the sign here
    v = interpolation.map_coordinates(vol, xyz, order=order, mode='constant', cval=0.0)
    return v


def resample_to_surface(transforms, surfaces, in_file, out_files=None, mask_file=None, n_jobs=1, **kwargs):
    '''
    Parameters
    ----------
    mask_file : str, dict
        Surface mask. Either a file name or dict(lh='lh.mask.niml.dset', rh='rh.mask.niml.dset').
        This will generate a partial surface dataset.

    Examples
    --------
    resample_to_surface(transforms=[f'SurfVol_Alnd_Exp.E2A.1D', f'epi{run}.volreg.aff12.1D', f'epi{run}.volreg.warp.nii'], 
        surfaces=[f'{suma_dir}/lh.pial.asc', f'{suma_dir}/rh.smoothwm.asc'], in_file=f'epi{run}.tshift.nii')
    '''
    if out_files is None:
        def out_files(in_file, surf_file):
            out_dir, prefix = afni.split_out_file(in_file, split_path=True, trailing_slash=True)[:2]
            return f"{out_dir}{prefix.split('.')[0]}.{'.'.join(path.basename(surf_file).split('.')[:2])}.niml.dset"
    if callable(out_files):
        out_files = [out_files(in_file, surf_file) for surf_file in surfaces]
    # TODO: Eliminate code copy and paste!
    vol, xyz2ijk = (io.read_vol(in_file), math.invert_affine(afni.get_affine(in_file))) \
        if isinstance(in_file, six.string_types) else in_file
    transforms = [(io.read_affine(f) if is_affine_transform(f) else io.read_warp(f)) \
        if isinstance(f, six.string_types) else f for f in transforms]
    if vol.ndim == 3: # For non 3D+t dset
        vol = vol[...,np.newaxis]
    n_vols = vol.shape[-1]
    if mask_file is not None:
        mask_nodes, mask_values = io.read_surf_data(mask_file)
        mask_nodes = mask_nodes[mask_values!=0]
    else:
        mask_nodes = slice(None)
    pc = utils.PooledCaller(pool_size=n_jobs)
    for sid, surf_file in enumerate(surfaces):
        print(f">> Mapping {path.basename(in_file)} onto {path.basename(surf_file)}...")
        verts = io.read_surf_mesh(surf_file)[0] # Assume that this is NOT a partial surface mesh
        n_verts = verts.shape[0]
        nodes = np.arange(n_verts)[mask_nodes]
        assert(isinstance(mask_nodes, slice) or np.all(nodes==mask_nodes)) # TODO: Better compatibility check between mesh and mask
        xyz = verts[mask_nodes,:] * np.r_[-1,-1,1] # From FreeSurfer's RAS+ to AFNI/DICOM's RAI
        n_xyz = xyz.shape[0]
        if n_jobs == 1:
            v = np.zeros(shape=[n_xyz, n_vols])
            for vid in range(n_vols):
                xforms = [xform[vid] if isinstance(xform, np.ndarray) and xform.ndim==3 else xform for xform in transforms]
                v[:,vid] = irregular_resample(xforms, xyz, (vol[...,vid], xyz2ijk), **kwargs)
        else:
            # Potential bug: For large dataset, this is strangely slow (even with only 1 job, is still 7x slower than non-shared version)
            v = utils.SharedMemoryArray.zeros(shape=[n_xyz, n_vols]) # lock=False won't speed it up
            def work(v, vids, transforms, xyz, vol, xyz2ijk):
                for vid in vids:
                    xforms = [xform[vid] if isinstance(xform, np.ndarray) and xform.ndim==3 else xform for xform in transforms]
                    v[:,vid] = irregular_resample(xforms, xyz, (vol[...,vid], xyz2ijk), **kwargs)
            for vids in pc.idss(n_vols, int(np.ceil(n_vols/pc.pool_size))):
                pc.run(work, v, vids, transforms, xyz, vol, xyz2ijk)
            pc.wait()
        io.write_surf_data(out_files[sid], nodes, v)


def deoblique(in_file, out_file=None, template=None):
    def save_mat(output):
        inv_file = utils.temp_prefix(suffix='.1D') # Caution: *.aff12.1D must be of oneline form
        idx = output.index('# mat44 Obliquity Transformation ::\n')
        with open(inv_file, 'w') as fo:
            for line in output[idx+1:idx+4]:
                fo.write(line)
        return inv_file
    if template is not None: # Deoblique by resampling over rotated grid
        prefix, ext = afni.split_out_file(out_file)
        outputs = {'out_file': f"{prefix}{ext}", 'xform_file': f"{prefix}.aff12.1D"}
        # Rotate and resample T1 to plumb, according to its oblique angle stored in the header
        oblique2card = save_mat(utils.run(f"3dWarp -verb -oblique2card -prefix {out_file} -overwrite {in_file}")['output'])
        # Rotate and resample T1 from plumb to EPI oblique. AFNI bug: the resultant dataset will be incorrectly labeled as plumb
        card2oblique = save_mat(utils.run(f"3dWarp -verb -card2oblique {template} -prefix {out_file} -overwrite {out_file}")['output'])
        # Crop extra margin after rotation
        utils.run(f"3dAutobox -prefix {out_file} -overwrite {out_file}")
        # Drop rotational information in T1 header. This is redundant due to previous bug...
        utils.run(f"3drefit -deoblique {out_file}")
        # Concat deoblique mat
        combine_affine_transforms([card2oblique+' -I', oblique2card+' -I'], outputs['xform_file'])
        os.remove(oblique2card)
        os.remove(card2oblique)
    else: # Deoblique by refit
        outputs = {'out_file': in_file}
        # Drop rotational information in EPI header, and pretend it is plumb
        utils.run(f"3drefit -deoblique {in_file}")
    all_finished(outputs)
    return outputs


def create_extent_mask(transforms, base_file, in_files, out_file):
    if isinstance(transforms[0], six.string_types):
        transforms = list(zip(*[sorted(glob.glob(pattern)) for pattern in transforms]))
    pc = utils.PooledCaller()
    temp_dir = utils.temp_folder()
    temp_prefixs = [path.join(temp_dir, afni.split_out_file(in_file, split_path=True)[1]) for in_file in in_files]
    # Create 3D+t all-1 datasets to mark the extent of the original datasets
    pc(pc.run(f"3dcalc -a {in_file} -expr 1 -prefix {prefix}.all1.nii -overwrite") \
        for in_file, prefix in zip(in_files, temp_prefixs))
    # Warp the all-1 datasets
    pc(pc.run(apply_transforms, transform, base_file, f"{prefix}.all1.nii", f"{prefix}.extent.nii", interp='NN') \
        for transform, prefix in zip(transforms, temp_prefixs))
    # Compute intersection across runs
    utils.run(f"3dTstat -min -prefix {out_file} -overwrite '{' '.join(sorted(glob.glob(f'{temp_dir}/*.extent.nii')))}'")   
    shutil.rmtree(temp_dir)


def create_brain_mask(in_files, out_file):
    pc = utils.PooledCaller()
    temp_dir = utils.temp_folder()
    temp_prefixs = [path.join(temp_dir, afni.split_out_file(in_file, split_path=True)[1]) for in_file in in_files]
    # Create brain mask of each run
    pc(pc.run(f"3dAutomask -dilate 1 -prefix {prefix}.brain.nii -overwrite {in_file}") \
        for in_file, prefix in zip(in_files, temp_prefixs))
    # Compute union across runs
    utils.run(f"3dTstat -max -prefix {out_file} -overwrite '{' '.join(sorted(glob.glob(f'{temp_dir}/*.brain.nii')))}'")
    shutil.rmtree(temp_dir)


def create_vessel_mask_BOLD(beta, out_file, th=10):
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
    }
    utils.run(f"3dcalc -a {beta} -expr 'step(a-{th})' -prefix {outputs['out_file']} -overwrite")
    all_finished(outputs)
    return outputs


def unifize_epi(in_file, out_file, method='N4', ribbon=None):
    '''
    Remove spatial trend/inhomogeneity in (mean) EPI data
        to better identify vessels using find_epi_dark_voxel_threshold().
        
    The default 'N4' method requires you have ANTs installed.
    Only 'Kay2019' method need an additional cortical ribbon mask
        that is aligned with the EPI volume.

    Parameters
    ----------
    method : str, 'N4' | 'Kay2019' | '3dUnifize'
        
    '''
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    if method.lower() == 'kay2019':
        temp_file = utils.temp_prefix(suffix='.nii')
        # Dilate ribbon by 1 voxel to reduce the influence of boundary values in vol2surf(min)
        utils.run(f"3dmask_tool -input {ribbon} -prefix {temp_file} -dilate_input 1")
        # Fit the spatial trend in cortical ribbon
        ribbon = io.Mask(temp_file)
        epi = ribbon.dump(in_file)
        x, y, z = ribbon.xyz.T
        c = math.polyfit3d(x, y, z, epi, deg=4, method='ridge') # (ridge, deg=4) outperforms (ols, deg=4), similar to (ols, deg=2)
        fitted = polynomial.polyval3d(x, y, z, c)
        # Remove spatial trend by divisive normalization
        normed = epi/fitted * 1000 # Mean ~ 1000
        ribbon.undump(outputs['out_file'], normed)
        os.remove(temp_file)
    elif method.lower() == 'N4':
        correct_bias_field(in_file, outputs['out_file'])
    elif method.lower() == '3dunifize':
        utils.run(f"3dUnifize -EPI -prefix {outputs['out_file']} -overwrite {in_file}")
    all_finished(outputs)
    return outputs


def find_epi_dark_voxel_threshold(v, method='Liu2020'):
    '''
    Find EPI intensity threshold for blood vessels on surface dset,
        produced by Surface.vol2surf(func='min').

    Parameters
    ----------
    method : str, 'Liu2020' | 'Marquardt2018' | 'Kay2019'
        Surprisingly, the three methods usually produce very similar results.
        The mixture of Gaussian method ('Kay2019') is robust with 'N4' or 'Kay2019'
        unifized EPI data (using preprocess.unifize_epi), but may fail with 
        '3dUnifize' or raw EPI data.
    '''
    v = v[v>0]
    if method.lower() == 'kay2019':
        model = mixture.GaussianMixture(n_components=2)
        model.fit(v.reshape(-1,1))
        comps = np.argsort(model.means_.ravel()) # Components as sorted by mean value (ascending)
        x = np.arange(0, max(v), 0.1)
        p = model.predict_proba(x.reshape(-1,1))
        th = x[np.nonzero(p[:,comps[1]] > p[:,comps[0]])[0][0]]
    elif method.lower() == 'liu2020':
        th = np.mean(v) * 0.75
    elif method.lower() == 'marquardt2018':
        th = np.mean(v) * 0.7
    return th


def create_vessel_mask_EPI(mean_epi, ribbon, out_file, corr_file=None, th=None):
    '''
    Find vessel voxels based on bias-corrected EPI intensity (Kay's method).
    The result may also highlight misalignment between EPI and T1, as well as 
    errors in pial surface reconstruction.

    Parameters
    ----------
    mean_epi : str
    ribbon : str
        Mask for gray matter which is aligned with mean_epi.
    th : float or 'auto', default 0.75 (as in [1])

    References
    ----------
    [1] Kay, K., Jamison, K. W., Vizioli, L., Zhang, R., Margalit, E., & Ugurbil, K. (2019). A critical assessment of data quality and venous effects in sub-millimeter fMRI. NeuroImage, 189, 847–869.
    '''
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = {
        'out_file': f"{out_dir}{prefix}{ext}",
        'corr_file': f"{out_dir}{afni.split_out_file(mean_epi, split_path=True)[1]}.bias_corr{ext}" \
            if corr_file is None else corr_file,
        'auto_th': np.nan,
        'inspect_file': f"{out_dir}{prefix}.png",
    }
    # Fit 3D polynomial trend of EPI intensity using ridge regression
    ribbon = io.Mask(ribbon)
    epi = ribbon.dump(mean_epi)
    x, p, z = ribbon.xyz.T
    c = math.polyfit3d(x, p, z, epi, deg=4, method='ridge') # (ridge, deg=4) outperforms (ols, deg=4), similar to (ols, deg=2)
    fitted = polynomial.polyval3d(x,p, z, c)
    divided = epi/fitted
    # Fit Gaussian mixture model to determine the threshold to classify "dark voxels" as vessels
    model = mixture.GaussianMixture(n_components=2)
    model.fit(divided.reshape(-1,1))
    comps = np.argsort(model.means_.ravel()) # Components as sorted by mean value (ascending)
    x = np.arange(0, 1, 0.001)
    p = model.predict_proba(x.reshape(-1,1))
    outputs['auto_th'] = x[np.argmin(np.abs(p[:,comps[0]] - p[:,comps[1]]))]
    if th is None:
        th = 0.75 # {Kay2019}
    elif th == 'auto':
        th = outputs['auto_th']
    # Save vessel mask
    ribbon.undump(outputs['out_file'], divided<th)
    ribbon.undump(outputs['corr_file'], divided)
    # Save inspect figure
    sns.distplot(divided, color='purple', hist=False)
    predicted = model.predict(divided.reshape(-1,1))
    p1 = np.array([np.sum(predicted==k)/len(predicted) for k in range(model.n_components)]) # Probability of generating a specific class label
    x = np.arange(0, 2, 0.01)
    p2 =  np.array([stats.norm.pdf(x, loc=model.means_[k], scale=np.sqrt(model.covariances_[k,0,0])) for k in range(model.n_components)]).T # Probability of generating an intensity level given class label
    p = p1 * p2
    with sns.plotting_context('talk'):
        for k, comp in enumerate(comps):    
            plt.plot(x, p[:,comp], color=['C3', 'C0', 'C2'][k])
        plt.xlim(0, 2)
        plt.axvline(outputs['auto_th'], color='purple', ls='--')
        plt.axvline(th, color='purple')
        plt.xlabel('Bias-corrected EPI intensity')
        plt.ylabel('$p(x)$')
        plt.text(th-0.1, 2, f'th = {th:.3g}', ha='right')
        plt.title("Kay's method to find vessels")
        sns.despine()
        plt.savefig(outputs['inspect_file'], bbox_inches='tight')

    all_finished(outputs)
    return outputs


def create_vessel_mask_PDGRE(PD, GRE, out_file, PD_div_GRE=None, ath=300, rth=5, nath=100, nrth=-5, base_file=None, strip_PD=True, strip_GRE=True):
    '''
    Find vessel voxels based on PD/GRE contrast.

    Parameters
    ----------
    PD : str, aligned with EXP
    GRE : str, aligned with PD
    ath : float
        Global absolute threshold for PD/GRE
    rth : float
        Local (within neighborhood) relative deviation for PD/GRE
    '''
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = {
        'out_file': f"{out_dir}{prefix}{ext}",
        'PD_div_GRE': f"{out_dir}PD_div_GRE{ext}" if PD_div_GRE is None else PD_div_GRE,
        'PD_div_GRE_dev': f"{out_dir}PD_div_GRE_dev{ext}" \
            if PD_div_GRE is None else afni.insert_suffix(PD_div_GRE, '_dev'),
    }
    temp_dir = utils.temp_folder()
    pc = utils.PooledCaller()
    # Brain mask
    if strip_PD:
        pc.run(skullstrip, PD, f"{temp_dir}/PD_ns.nii")
    else:
        pc.run(f"3dcopy {PD} {temp_dir}/PD_ns.nii -overwrite")
    if strip_GRE:
        pc.run(skullstrip, GRE, f"{temp_dir}/GRE_ns.nii")
    else:
        pc.run(f"3dcopy {GRE} {temp_dir}/GRE_ns.nii -overwrite")
    pc.wait()
    utils.run(f"3dcalc -a {temp_dir}/PD_ns.nii -b {temp_dir}/GRE_ns.nii \
        -expr 'min(step(a),step(b))' -prefix {temp_dir}/brain_mask.nii -overwrite")
    # Resample according to base
    if base_file is not None:
        pc.run(f"3dresample -rmode Cu -master {base_file} -prefix {temp_dir}/PD_resam.nii -overwrite -input {PD}")
        pc.run(f"3dresample -rmode Cu -master {base_file} -prefix {temp_dir}/GRE_resam.nii -overwrite -input {GRE}")
        pc.run(f"3dresample -rmode Cu -master {base_file} -prefix {temp_dir}/brain_mask.nii -overwrite -input {temp_dir}/brain_mask.nii")
        pc.wait()
        PD = f"{temp_dir}/PD_resam.nii"
        GRE = f"{temp_dir}/GRE_resam.nii"
        utils.run(f"3dcalc -a {temp_dir}/brain_mask.nii -expr 'step(a-0.5)' -prefix {temp_dir}/brain_mask.nii -overwrite")
    # PD/GRE global absolute value
    utils.run(f"3dcalc -a {PD} -b {GRE} -expr 'a/b*100' -datum float -prefix {outputs['PD_div_GRE']} -overwrite")
    # PD/GRE local relative deviation
    pc.run(f"3dLocalstat -nbhd 'SPHERE(5)' -mask {temp_dir}/brain_mask.nii -stat median \
        -prefix {temp_dir}/PD_div_GRE_med.nii -overwrite {outputs['PD_div_GRE']}") # SPHERE(radius) is mm
    pc.run(f"3dLocalstat -nbhd 'SPHERE(5)' -mask {temp_dir}/brain_mask.nii -stat MAD \
        -prefix {temp_dir}/PD_div_GRE_mad.nii -overwrite {outputs['PD_div_GRE']}")
    pc.wait()
    utils.run(f"3dcalc -a {outputs['PD_div_GRE']} -b {temp_dir}/PD_div_GRE_med.nii -c {temp_dir}/PD_div_GRE_mad.nii \
        -expr '(a-b)/c' -prefix {outputs['PD_div_GRE_dev']} -overwrite")
    # Use both global absolute value and local (within neighborhood) relative deviation
    # (subtract local median and divide local MAD) to construct suprathreshold mask
    # Note that the mask can be either 1 or -1
    utils.run(f"3dcalc -a {outputs['PD_div_GRE']} -r {outputs['PD_div_GRE_dev']} -m {temp_dir}/brain_mask.nii \
        -expr 'max(step(a-{ath}),step(r-{rth}))*m-max(step({nath}-a),step({nrth}-r))*m' \
        -prefix {outputs['out_file']} -overwrite")

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def clusterize(in_file, out_file, cluster_size, neighbor=2):
    if out_file is None:
        out_file = in_file
    utils.run(f"3dclust -NN{neighbor} {cluster_size} \
        -prefix {out_file} -overwrite {in_file}")


def scale(in_file, out_file, mask_file=None, dtype=None):
    dtypes = {float: 'float', int: 'short'}
    if dtype in dtypes:
        dtype = dtypes[dtype]
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    mean_file = utils.temp_prefix(suffix=ext)
    utils.run(f"3dTstat -mean -prefix {mean_file} -overwrite {in_file}")
    if mask_file is not None:
        utils.run(f"3dcalc -a {in_file} -b {mean_file} -c {mask_file} \
            -expr 'min(200,a/b*100)*step(a)*step(b)*c' \
            {f'-datum {dtype}' if dtype is not None else ''} \
            -prefix {out_file} -overwrite")
    else:
        utils.run(f"3dcalc -a {in_file} -b {mean_file} \
            -expr 'min(200,a/b*100)*step(a)*step(b)' \
            {f'-datum {dtype}' if dtype is not None else ''} \
            -prefix {outputs['out_file']} -overwrite")
    os.remove(mean_file)
    all_finished(outputs)
    return outputs


def zscore(in_file, out_file):
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    mean_file = utils.temp_prefix(suffix=ext)
    utils.run(f"3dTstat -mean -stdev -prefix {mean_file} -overwrite {in_file}")
    utils.run(f"3dcalc -a {in_file} -b {mean_file}'[0]' -c {mean_file}'[1]' \
        -expr '(a-b)/c*step(b)' \
        -prefix {outputs['out_file']} -overwrite")
    os.remove(mean_file)
    all_finished(outputs)
    return outputs


def skullstrip(in_file, out_file=None):
    if out_file is None:
        prefix, ext = afni.split_out_file(in_file)
        out_file = f"{prefix}_ns{ext}"
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    utils.run(f"3dSkullStrip -orig_vol -prefix {out_file} -overwrite -input {in_file}")
    all_finished(outputs)
    return outputs


def _ants_sanitize_input(in_file, overwrite=True):
    '''
    It is important that the input is 3D (not 4D) dataset.
    But it is OK that the input is not in ORIG space (e.g., MNI), or RAS+/NIFTI orientation.
    Otherwise antsRegistration will spit WEIRD result, e.g., rotate cerebellum to forehead...
    '''
    if afni.get_dims(in_file)[3] > 1:
        print(f'+* WARNING: "{in_file}" contains multiple volumes. Only the first volume will be considered in the registration.')
        out_file = in_file if overwrite else utils.temp_prefix(suffix='.nii')
        is_temp = not overwrite
        utils.run(f"3dTcat -prefix {out_file} -overwrite {in_file}'[0]'", error_pattern='error')
    else:
        out_file, is_temp = in_file, False
    return out_file, is_temp


def align_ants(base_file, in_file, out_file, strip=None, base_mask=None, in_mask=None, 
    base_mask_SyN=None, in_mask_SyN=None, init_transform=None, preset=None, n_jobs=None):
    '''
    Nonlinearly align `in_file` to `base_file` using ANTs' SyN method via antsRegistration.

    Examples
    --------
    1. Align MNI template to T1_al using default preset. Skullstrip T1_al. 
        Apply inversed mask (1-mask) to MNI template at the nonlinear (SyN) stage.
        >>> prep.align_ants("T1_al.nii", "MNI152_2009_template.nii.gz", "MNI_al.nii", strip="base", in_mask_SyN="mask.nii -I")
    2. Align T1_ns_al.nii to MNI template using "test" preset.
        For a quick test of the parameters in a few minutes. The result will not be good, but should not be weird.
        >>> prep.align_ants("MNI152_2009_template.nii.gz", "T1_ns_al.nii", "T1_ns_MNI.nii", preset="test")

    Parameters
    ----------
    preset : str
        None | 'default' | 'test' | 'path/to/my/preset.json'
        For production, just leave it as None or use the 'default' presest (est. time: 3 hr).
        For quick test, use the 'test' presest (est. time: 3 min).

    Returns
    -------
    outputs : dict
        outputs['transform'] : ANTsTransform object
            You can apply the forward or inverse transform to other volumes. E.g., 
            >>> outputs['transform'].apply(in_file, out_file)
            >>> outputs['transform'].apply_inverse(in_file, out_file)
    '''
    temp_dir = utils.temp_folder()
    pc = utils.PooledCaller()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'fwd_affine': f"{prefix}_0GenericAffine.mat",
        'fwd_warp': f"{prefix}_1Warp.nii.gz",
        'inv_warp': f"{prefix}_1InverseWarp.nii.gz",
        'fwd_warped': f"{prefix}_fwd_warped.nii.gz",
        'inv_warped': f"{prefix}_inv_warped.nii.gz",
    }
    # Strip and sanitize base_file or in_file if required

    if strip is None:
        strip = [False, False]
    if isinstance(strip, str):
        strip = {'both': [True, True], 'base': [True, False], 'input': [False, True], 'none': [False, False]}[strip]
    strip = [((skullstrip if s else copy_dset) if isinstance(s, bool) else s) for s in strip]
    fixed = f"{temp_dir}/base.nii"
    moving = f"{temp_dir}/input.nii"
    D1 = pc.run(strip[0], base_file, out_file=fixed)
    D2 = pc.run(strip[1], in_file, out_file=moving)
    pc.run(_ants_sanitize_input, fixed, _depends=[D1])
    pc.run(_ants_sanitize_input, moving, _depends=[D2])
    pc.wait()
    # Prepare masks (e.g., strip, inversion, etc.)
    def populate_mask(mask_file, temp_dir=None, ref_file=None):
        if mask_file is None:
            out_file = 'None'
        elif mask_file == 'strip':
            out_file = utils.temp_prefix(prefix=f"{temp_dir}/tmp_", suffix='.nii')
            skullstrip(ref_file, out_file)
            utils.run(f"3dcalc -a {out_file} -expr 'step(a)' -prefix {out_file} -overwrite")
        elif mask_file.endswith(' -I'):
            out_file = utils.temp_prefix(prefix=f"{temp_dir}/tmp_", suffix='.nii')
            utils.run(f"3dcalc -a {mask_file[:-3]} -expr '1-a' -prefix {out_file} -overwrite")
        else:
            out_file = mask_file
        return out_file
    pc.run(populate_mask, base_mask, temp_dir, ref_file=base_file)
    pc.run(populate_mask, in_mask, temp_dir, ref_file=in_file)
    pc.run(populate_mask, base_mask_SyN, temp_dir, ref_file=base_file)
    pc.run(populate_mask, in_mask_SyN, temp_dir, ref_file=in_file)
    base_mask, in_mask, base_mask_SyN, in_mask_SyN = pc.wait()
    # Init moving transform
    if init_transform is None:
        # Align the geometric center of the images (=0), the image intensities (=1), or the origin of the images (=2).
        init_moving_cmd = f"--initial-moving-transform [ {fixed}, {moving}, 1 ]"
    elif init_transform in ['none', 'identity']:
        init_moving_cmd = ''
    else:
        init_moving_cmd = f"--initial-moving-transform {init_transform}"
    # Estimate transforms
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(DEFAULT_JOBS) if n_jobs is None else str(n_jobs)
    if preset is None:
        pc.run1(f"antsRegistration -d 3 --float 1 --verbose \
            --output [ {prefix}_, {outputs['fwd_warped']}, {outputs['inv_warped']} ] \
            --interpolation LanczosWindowedSinc \
            --collapse-output-transforms 1 \
            {init_moving_cmd} \
            --winsorize-image-intensities [0.005,0.995] \
            --use-histogram-matching 1 \
            --transform translation[ 0.1 ] \
                --metric mattes[ {fixed}, {moving}, 1, 32, regular, 0.3 ] \
                --convergence [ 1000x300x100, 1e-6, 10 ]  \
                --smoothing-sigmas 4x2x1vox  \
                --shrink-factors 8x4x2 \
                --use-estimate-learning-rate-once 1 \
                --masks [ {base_mask}, {in_mask} ] \
            -t rigid[ 0.1 ] \
                -m mattes[ {fixed}, {moving}, 1, 32, regular, 0.3 ] \
                -c [ 1000x300x100, 1e-6, 10 ]  \
                -s 4x2x1vox  \
                -f 4x2x1 -l 1 \
                -x [ {base_mask}, {in_mask} ] \
            -t affine[ 0.1 ] \
                -m mattes[ {fixed}, {moving}, 1, 32, regular, 0.3 ] \
                -c [ 1000x300x100, 1e-6, 10 ]  \
                -s 2x1x0vox  \
                -f 4x2x1 -l 1 \
                -x [ {base_mask}, {in_mask} ] \
            -t SyN[ 0.1, 3, 0 ] \
                -m mattes[ {fixed}, {moving}, 0.5 , 32 ] \
                -m cc[ {fixed}, {moving}, 0.5 , 4 ] \
                -c [ 100x100x50, 1e-6, 10 ]  \
                -s 1x0.5x0vox  \
                -f 4x2x1 -l 1 \
                -x [ {base_mask_SyN}, {in_mask_SyN} ]", _error_pattern='error')
    else:
        if isinstance(preset, str):
            if not preset.endswith('.json'): # Built-in preset (located in mripy/data/align_ants_presets)
                preset = f"{utils.package_dir}/data/align_ants_presets/{preset}.json"
            # Otherwise should be the fullpath to a custom .json file
            with open(preset) as json_file:
                preset = json.load(json_file)
        # `preset` is now a dict
        # Generate antsRegistration command line
        cmd = f"antsRegistration -d {preset['dimension']} --float 1 --verbose \
            --output [ {prefix}_, {outputs['fwd_warped']}, {outputs['inv_warped']} ] \
            --interpolation {preset['interpolation']} \
            --collapse-output-transforms 1 \
            --write-composite-transform {int(preset['write_composite_transform'])} \
            {init_moving_cmd} \
            --winsorize-image-intensities [ {preset['winsorize_lower_quantile']}, {preset['winsorize_upper_quantile']} ] "
        for k in range(len(preset['transforms'])):
            cmd += f"--transform {preset['transforms'][k]}[ {', '.join([f'{x:g}' for x in preset['transform_parameters'][k]])} ] \
                --metric {preset['metric'][k]}[ {fixed}, {moving}, {preset['metric_weight'][k]}, {preset['radius_or_number_of_bins'][k]}, {preset['sampling_strategy'][k]}, {preset['sampling_percentage'][k]} ] \
                --convergence [ {'x'.join([str(int(x)) for x in preset['number_of_iterations'][k]])}, {preset['convergence_threshold'][k]}, {preset['convergence_window_size'][k]} ]  \
                --smoothing-sigmas {'x'.join([str(int(x)) for x in preset['smoothing_sigmas'][k]])}{preset['sigma_units'][k]}  \
                --shrink-factors {'x'.join([str(int(x)) for x in preset['shrink_factors'][k]])} \
                --use-histogram-matching {int(preset['use_histogram_matching'][k])} \
                --use-estimate-learning-rate-once {int(preset['use_estimate_learning_rate_once'][k])} \
                --masks [ {base_mask_SyN if preset['transforms'][k].lower()=='syn' else base_mask}, {in_mask_SyN if preset['transforms'][k].lower()=='syn' else in_mask} ] "
        pc.run1(cmd, _error_pattern='error')
    # Apply transforms
    apply_ants([outputs['fwd_warp'], outputs['fwd_affine']], base_file, in_file, outputs['out_file'])
    shutil.rmtree(temp_dir)
    all_finished(outputs)
    outputs['base_file'] = base_file
    outputs['in_file'] = in_file
    outputs['transform'] = ANTsTransform.from_align_ants(outputs)
    outputs['transform_file'] = f"{prefix}_transform.json"
    outputs['transform'].to_json(outputs['transform_file'])
    outputs['pc'] = pc
    return outputs


def apply_ants(transforms, base_file, in_file, out_file, interp=None, dim=None, image_type=None, n_jobs=None):
    '''
    Parameters
    ----------
    transforms : list of file names
        Online matrix inversion is supported as "*_0GenericAffine.mat -I".
    base_file : str
        If None, apply transforms to point list using `antsApplyTransformsToPoints`,
        and `in_file` is expected to be a *.csv file.
        Otherwise, apply transforms to image using `antsApplyTransforms`.
    interp : str
        LanczosWindowedSinc, NearestNeighbor, Linear, BSpline[<order=3>], etc.
    
    Note that for volumes, last transform applies first (pulling from base grid), 
    as in AFNI, as well as in ANTs command line.
    However for point lists, FIRST transform applies first (pushing input points),
    and INVERSE transforms should be used compared with volume case, as in ANTs.
    '''
    if interp is None:
        interp = 'LanczosWindowedSinc'
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    xform_cmds = []
    for transform in transforms:
        if transform.endswith(' -I'):
            xform_cmds.append(f"-t [ {transform[:-3]}, 1 ]")
        else:
            xform_cmds.append(f"-t {transform}")
    if n_jobs is not None:
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(n_jobs)
    if base_file is None or path.splitext(in_file)[1] in ['.csv']: 
        # Apply transforms to point list (e.g., surface mesh)
        data = np.loadtxt(in_file, skiprows=1, delimiter=',')
        if dim is None:
            dim = 4 if any(data[:,3]) else 3
        utils.run(f"antsApplyTransformsToPoints --precision 0 \
            -i {in_file} -d {dim} \
            -o {outputs['out_file']} \
            {' '.join(xform_cmds)}")
    else: # Apply transforms to image (e.g., 3D volume)
        if dim is None:
            dim = 3
        if image_type is None: # 0/1/2/3 for scalar/vector/tensor/time-series
            image_type = 3 if afni.get_dims(in_file)[3] > 1 else 0
        # Make sure the image is timeseries, rather than afni bucket, during transform
        is_bucket = (io.get_dim_order(in_file) == 'bucket')
        if is_bucket:
            io.change_dim_order(in_file, dim_order='timeseries', method='afni') # Change in_file into timeseries
        # Sanitize base_file
        sanitized_base, is_temp = _ants_sanitize_input(base_file, overwrite=False)
        # Apply transforms
        utils.run(f"antsApplyTransforms --float 1 \
            -r {sanitized_base} -i {in_file} -d {dim} --input-image-type {image_type} \
            -o {outputs['out_file']} --interpolation {interp} \
            {' '.join(xform_cmds)}")
        if is_bucket:
            io.change_dim_order(in_file, dim_order='bucket', method='afni') # Change in_file back to bucket
            io.change_dim_order(outputs['out_file'], dim_order='bucket', method='afni') # The output should follow input
        if is_temp:
            os.remove(sanitized_base)
        # Make sure out_file is of the same space as base_file
        io.change_space(outputs['out_file'], space=io.get_space(base_file), method='nibabel') # Causion: Cannot use "afni" here...
        # Copy afni subbrick labels
        # This must go last because io.change_space(method='nibabel') will recreate the volume without afni metadata
        afni.set_brick_labels(outputs['out_file'], afni.get_brick_labels(in_file))
    all_finished(outputs)
    return outputs


def ants2afni_affine(ants_affine, afni_affine):
    raise NotImplementedError


def afni2ants_affine(afni_affine, ants_affine):
    raise NotImplementedError


def ants2afni_warp(ants_warp, afni_warp):
    raise NotImplementedError


def afni2ants_warp(afni_warp, ants_warp):
    raise NotImplementedError


class Transform(object):
    def __init__(self, transforms, base_file=None, source_file=None):
        '''
        Parameters
        ----------
        transforms : list of (fwd_xform, inv_xform) pairs
            If only one transform (rather than a tuple) is given, it is assumed to be
            the fwd_xform, and "fwd_xform -I" is treated as inv_xform.
            Transform chain should be specified using "pulling" convention, i.e., 
            last transform applies first (as in AFNI and ANTs for volumes).
            Each transform should also be a "pulling" transform from moving to fixed 
            as generated by AFNI and ANTs.
            Transforms should be specified by their file names, and inverse transforms 
            can be specified as "*_0GenericAffine.mat -I" (esp. for affine).
        '''
        # Add inverse transform automatically for singleton (assumed linear)
        self.transforms = [(t, f"{t} -I") if isinstance(t, str) else tuple(t) for t in transforms]
        self.base_file = base_file
        self.source_file = source_file

    def inverse(self):
        transforms = [transform[::-1] for transform in self.transforms[::-1]]
        return self.__class__(transforms, base_file=self.source_file, source_file=self.base_file)

    def rebase(self, base_file):
        return self.__class__(self.transforms, base_file=base_file, source_file=self.source_file)

    def replace_path(self, p):
        f = lambda fname: path.join(p, path.basename(fname))
        self.transforms = [(f(fwd), f(inv)) for fwd, inv in self.transforms]
        self.base_file = f(self.base_file)
        self.source_file = f(self.source_file)

    def to_json(self, fname):
        with open(fname, 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=4)

    @classmethod
    def from_json(cls, fname, replace_path=False):
        '''
        Parameters
        ----------
        replace_path: bool
            Replace path of the transform files according to json file path
        '''
        with open(fname) as json_file:
            data = json.load(json_file)
        inst = cls(data['transforms'], base_file=data['base_file'], source_file=data['source_file'])
        if replace_path:
            inst.replace_path(path.dirname(fname))
        return inst

    def __repr__(self):
        source = path.basename(self.source_file) if self.source_file is not None else 'unknown'
        base = path.basename(self.base_file) if self.base_file is not None else 'unknown'
        return f"<{self.__class__.__name__} | from {source} to {base} >"

    def apply(self, in_file, out_file, base_file=None, interp=None, **kwargs):
        '''
        For volumes, forward transform (from input/moving to base/fixed)
        '''
        transforms = [xform_pair[0] for xform_pair in self.transforms]
        base_file = self.base_file if base_file is None else base_file
        return apply_transforms(transforms, base_file, in_file, out_file, interp=interp, **kwargs)

    def apply_inverse(self, in_file, out_file, base_file=None, interp=None, **kwargs):
        '''
        For volumes, inverse transform (from base/fixed to input/moving)
        '''
        transforms = [xform_pair[1] for xform_pair in self.transforms[::-1]] # Inverse
        base_file = self.source_file if base_file is None else base_file
        return apply_transforms(transforms, base_file, in_file, out_file, interp=interp, **kwargs)


class ANTsTransform(Transform):
    @classmethod
    def from_align_ants(cls, outputs):
        transforms = [(path.realpath(outputs['fwd_warp']), path.realpath(outputs['inv_warp'])), 
                      (path.realpath(outputs['fwd_affine']), path.realpath(outputs['fwd_affine'])+' -I')]
        return cls(transforms, base_file=path.realpath(outputs['base_file']), source_file=path.realpath(outputs['in_file']))

    def apply(self, in_file, out_file, base_file=None, interp=None, **kwargs):
        '''
        For volumes, forward transform (from input/moving to base/fixed)
        '''
        transforms = [xform_pair[0] for xform_pair in self.transforms]
        base_file = self.base_file if base_file is None else base_file
        return apply_ants(transforms, base_file, in_file, out_file, interp=interp, **kwargs)

    def apply_inverse(self, in_file, out_file, base_file=None, interp=None, **kwargs):
        '''
        For volumes, inverse transform (from base/fixed to input/moving)
        '''
        transforms = [xform_pair[1] for xform_pair in self.transforms[::-1]] # Inverse
        base_file = self.source_file if base_file is None else base_file
        return apply_ants(transforms, base_file, in_file, out_file, interp=interp, **kwargs)

    def apply_to_points(self, in_file, out_file):
        '''
        For list of points, forward transform (from input/moving to base/fixed)
        
        Parameters
        ----------
        in_file, out_file : *.csv file with "x,y,z,t" header line.
        '''
        return self.apply_inverse(in_file, out_file, base_file=None)

    def apply_inverse_to_points(self, in_file, out_file):
        '''
        For list of points, inverse transform (from base/fixed to input/moving)

        Parameters
        ----------
        in_file, out_file : *.csv file with "x,y,z,t" header line.
        '''
        return self.apply(in_file, out_file, base_file=None)

    def _apply_transform_to_xyz(self, xyz, convention='DICOM', transform='forward'):
        '''
        Parameters
        ----------
        xyz : Nx3 array
        convention : 'DICOM' | 'NIFTI'
        transform : 'forward' | 'inverse'
        '''
        temp_file = utils.temp_prefix(suffix='.csv')
        if convention.upper() in ['NIFTI', 'LPI', 'RAS+']:
            xyz = xyz * [-1, -1, 1] # To DICOM or RAI or LPS+
        np.savetxt(temp_file, np.c_[xyz, np.zeros(xyz.shape[0])], delimiter=',', header='x,y,z,t', comments='')
        if transform == 'forward':
            self.apply_to_points(temp_file, temp_file)
        elif transform == 'inverse':
            self.apply_inverse_to_points(temp_file, temp_file)
        xyz = np.loadtxt(temp_file, skiprows=1, delimiter=',')[:,:3]
        if convention.upper() in ['NIFTI', 'LPI', 'RAS+']:
            xyz = xyz * [-1, -1, 1] # Back to NIFTI
        os.remove(temp_file)
        return xyz

    def apply_to_xyz(self, xyz, convention='DICOM'):
        '''
        Parameters
        ----------
        xyz : Nx3 array
        convention : 'DICOM' | 'NIFTI'
        '''
        return self._apply_transform_to_xyz(xyz, convention=convention, transform='forward')

    def apply_inverse_to_xyz(self, xyz, convention='DICOM'):
        '''
        Parameters
        ----------
        xyz : Nx3 array
        convention : 'DICOM' | 'NIFTI'
        '''
        return self._apply_transform_to_xyz(xyz, convention=convention, transform='inverse')


def align_anat(base_file, in_file, out_file, strip=None, N4=None, init_shift=None, init_rotate=None, init_xform=None,
    method=None, cost=None, n_params=None, interp=None, max_rotate=None, max_shift=None, 
    emask=None, save_weight=None):
    '''
    emask : fname
        Mask to exclude from analysis.
    '''
    def parse_cost(output):
        pattern = re.compile(r'\+\+ allcost output: final fine #0')
        k = 0
        while k < len(output):
            match = pattern.match(output[k])
            k += 1
            if match:
                cost = {}
                while True:
                    match = re.search('(\S+)\s+= (\S+)', output[k])
                    k += 1
                    if match:
                        cost[match.group(1)] = float(match.group(2))
                    else:
                        break
                return cost
    if method is None:
        method = '3dallineate'
    else:
        method = method.lower()
    assert(method in ['3dallineate', 'align_epi_anat'])
    if cost is None:
        cost = 'lpa'
    elif cost == 'within':
        cost = 'lpa'
    elif cost == 'cross':
        cost = 'lpc'
    elif cost == 'edge':
        cost = 'lpa -edge'
        if method != 'align_epi_anat':
            raise ValueError('cost="edge" only works with method="align_epi_anat"')
    if n_params is None:
        n_params = 'affine_general'
    elif n_params in ['affine', 12]:
        n_params = 'affine_general'
    elif n_params in ['rigid', 6]: 
        n_params = 'shift_rotate'
    if interp is None:
        interp = 'wsinc5'
    init_shift_cmd = ''
    if init_shift is None:
        if init_rotate is None and max_shift is None:
            init_shift_cmd = '-cmass'
    if max_rotate is None:
        max_rotate = 90
    temp_dir = utils.temp_folder()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'xform_file': f"{prefix}.aff12.1D",
        'cost': None,
    }
    if save_weight is not None:
        outputs['weight_file'] = save_weight if isinstance(save_weight, six.string_types) else f"{prefix}.autoweight{ext}"
    pc = utils.PooledCaller()
    # Strip skull
    if strip is None:
        strip = True
    if isinstance(strip, (str, bool)):
        strip = [strip]
    if not set(strip).isdisjoint({True, 'both', 'base', 'template'}):
        pc.run(skullstrip, base_file, f"{temp_dir}/base_ns.nii")
    else:
        pc.run(f"3dcopy {base_file} {temp_dir}/base_ns.nii")
    if not set(strip).isdisjoint({True, 'both', 'source', 'src', 'input', 'in'}):
        pc.run(skullstrip, in_file, f"{temp_dir}/in_ns.nii")
    else:
        pc.run(f"3dcopy {in_file} {temp_dir}/in_ns.nii")
    pc.wait()
    # Correct bias field using N4 method in ANTs
    # This may potentially enhance the performance of some cost functions.
    # Consider trying 6 dof rigid body transform instead of 12 dof affine.
    if N4 is None:
        N4 = False
    if N4 and shutil.which('N4BiasFieldCorrection') is None:
        raise ValueError('>> ANTs is not (correctly) installed. So cannot use N4BiasFieldCorrection. Set N4=None.')
    if isinstance(N4, (str, bool)):
        N4 = [N4]
    if not set(N4).isdisjoint({True, 'both', 'base', 'template'}):
        pc.run(f"N4BiasFieldCorrection -d 3 -i {temp_dir}/base_ns.nii -s 2 -o \
            '[{temp_dir}/base_ns.nii,{temp_dir}/base_bias.nii]'")
    if not set(N4).isdisjoint({True, 'source', 'input', 'in'}):
        pc.run(f"N4BiasFieldCorrection -d 3 -i {temp_dir}/in_ns.nii -s 2 -o \
            '[{temp_dir}/in_ns.nii,{temp_dir}/in_bias.nii]'")
    pc.wait()
    # Apply initial (manual) alignment and extract the parameters
    transforms = []
    if init_rotate is not None:
        if init_rotate:
            init_mat = nudge_cmd2mat(init_rotate, f"{temp_dir}/in_ns.nii")
            init_xform = f"{temp_dir}/init.aff12.1D"
            io.write_affine(init_xform, init_mat)
    if init_xform is not None:
        apply_transforms(init_xform, f"{temp_dir}/base_ns.nii",
            f"{temp_dir}/in_ns.nii", f"{temp_dir}/in_ns.nii")
        transforms.insert(0, init_xform)
    # Estimate best alignment parameters
    if method == '3dallineate':
        res = utils.run(f'''3dAllineate -final {interp} -cost {cost} -allcost -warp {n_params} \
            {init_shift_cmd} \
            -maxrot {max_rotate} {'' if max_shift is None else f'-maxshf {max_shift}'} \
            -base {temp_dir}/base_ns.nii -input {temp_dir}/in_ns.nii \
            -autoweight -source_automask+2 -twobest 11 -fineblur 1 \
            {f'-emask {emask}' if emask is not None else ''} \
            {f'-wtprefix {outputs["weight_file"]}' if save_weight is not None else ''} \
            -1Dmatrix_save {temp_dir}/in2base.aff12.1D \
            -prefix {temp_dir}/out_ns.nii -overwrite''')
        transforms.insert(0, f"{temp_dir}/in2base.aff12.1D")
        outputs['cost'] = parse_cost(res['output'])
    elif method == 'align_epi_anat':
        pass
    # Apply all transforms at once
    apply_transforms(transforms, base_file, in_file, outputs['out_file'], interp=interp, save_xform=outputs['xform_file'])

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def align_S2E(base_file, suma_dir, out_file=None, **kwargs):
    surf_vol = afni.get_surf_vol(suma_dir)
    if out_file is None:
        out_dir, prefix, ext = afni.split_out_file(base_file, split_path=True, trailing_slash=True)
        out_file = f"{out_dir}SurfVol_Alnd_Exp{ext}"
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = align_anat(base_file, surf_vol, out_file, **kwargs)
    outputs['script_file'] = f"{out_dir}run_suma"
    spec_file = path.relpath(afni.get_suma_spec(suma_dir)['both'], out_dir)
    create_suma_script(spec_file, path.split(out_file)[1], outputs['script_file'])
    all_finished(outputs)
    return outputs


def align_anat2epi(anat_file, epi_file, out_file, base_file=None, init_oblique=None, init_epi_rotate=None, init_anat_rotate=None):
    '''
    Different init methods are mutual exclusive (i.e., at most one init method could be used at a time).
    '''
    def parse_cost(output):
        pattern = re.compile(r' \+ - Final    cost = ([+-.\d]+)')
        for line in output:
            match = pattern.match(line)
            if match:
                return float(match.group(1))
    temp_dir = utils.temp_folder()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'xform_file': f"{prefix}.aff12.1D",
        'cost': None,
    }

    transforms = []
    utils.run(f"3dcopy {anat_file} {temp_dir}/anat_init.nii -overwrite")
    utils.run(f"3dcopy {epi_file} {temp_dir}/epi_init.nii -overwrite")
    if init_oblique is not None:
        deoblique_outputs = deoblique(anat_file, out_file=f"{temp_dir}/anat_init.nii", template=f"{init_oblique}")
        transforms.insert(0, deoblique_outputs['xform_file'])
    if init_anat_rotate is not None:
        anat2init_mat = nudge_cmd2mat(init_anat_rotate, anat_file)
        anat2init_xform = f"{temp_dir}/anat2init.aff12.1D"
        io.write_affine(anat2init_xform, anat2init_mat)
        apply_transforms(anat2init_xform, anat_file, anat_file, f"{temp_dir}/anat_init.nii")
        transforms.insert(0, anat2init_xform)
    if init_epi_rotate is not None:
        epi2init_mat, init2epi_mat = nudge_cmd2mat(init_epi_rotate, epi_file, return_inverse=True)
        epi2init_xform = f"{temp_dir}/epi2init.aff12.1D"
        io.write_affine(epi2init_xform, epi2init_mat)
        apply_transforms(epi2init_xform, epi_file, epi_file, f"{temp_dir}/epi_init.nii")
        init2epi_xform = f"{temp_dir}/init2epi.aff12.1D"
        io.write_affine(init2epi_xform, init2epi_mat)
        transforms.insert(0, init2epi_xform)
    old_dir = os.getcwd()
    try:
        os.chdir(temp_dir)
        res = utils.run(f"align_epi_anat.py -anat2epi -anat anat_init.nii -epi epi_init.nii \
            -epi_base 0 -epi_strip 3dAutomask -tshift off -volreg off -deoblique off \
            -suffix _al -overwrite", error_pattern='error')
        outputs['cost'] = parse_cost(res['output'])
    finally:
        os.chdir(old_dir)
    if init_epi_rotate is not None:
        transforms.append(f"{temp_dir}/anat_init_al_mat.aff12.1D")
    else:
        transforms.insert(0, f"{temp_dir}/anat_init_al_mat.aff12.1D")
    if base_file is None:
        base_file = f"{temp_dir}/anat_init.nii"
    apply_transforms(transforms, base_file, anat_file, \
        out_file=outputs['out_file'], save_xform=outputs['xform_file'])

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def combine_censors(censor_files, out_file, method='intersect'):
    '''Combine multiple censor files: 1=keep, 0=discard

    Parameters
    ----------
    method : str, {'intersect', 'concatinate'}
    '''
    if method == 'intersect':
        combined = np.all(np.vstack([np.loadtxt(f) for f in censor_files]), axis=0)
    elif method == 'concatinate':
        combined = np.hstack([np.loadtxt(f) for f in censor_files])
    np.savetxt(out_file, combined, fmt='%d')


def detrend(motion_file, in_file, out_file, censor=True, motion_th=0.3, censor_file=None, regressor_file=None):
    temp_dir = utils.temp_folder()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'poly_order': None,
    }
    if censor:
        outputs['censor_file'] = f"{prefix}.censor.1D"
        outputs['n_censored'] = None
    # Compute demeaned motion parameters for regression
    utils.run(f"1d_tool.py -infile {motion_file} \
        -demean -write {temp_dir}/motion_demean.1D -overwrite")
    if censor:
        # Construct TR censor excluded from analysis based on Euclidean norm of motion derivative greater than 0.3
        # Create *_censor.1D, *_enorm.1D, and *_CENSORTR.txt for use in 3dDeconvolve
        # -censor_prev_TR: because motion may span two TRs
        # -censor_motion: implies '-derivative', '-collapse_cols euclidean_norm', '-moderate_mask -LIMIT LIMIT', and '-prefix'
        utils.run(f"1d_tool.py -infile {motion_file} \
            -show_censor_count -censor_prev_TR \
            -censor_motion {motion_th} {temp_dir}/motion -overwrite")
        # Combine multiple censor files: 1=keep, 0=discard
        censors = [f"{temp_dir}/motion_censor.1D"]
        if censor_file is not None:
            censors.append(censor_file)
        combine_censors(censors, outputs['censor_file'])
        outputs['n_censored'] = np.sum(np.loadtxt(outputs['censor_file'])==0)
    # Regress out various "nuisance" time series
    D = afni.get_dims(in_file)[3] * afni.get_TR(in_file)
    outputs['poly_order'] = 1 + int(D/150) # According to "3dDeconvolve -polort A"
    regressor_cmd = f"-ort {regressor_file}" if regressor_file is not None else ''
    censor_cmd = f"-censor {outputs['censor_file']} -cenmode NTRP" if censor else ''
    utils.run(f"3dTproject -input {in_file} \
        -polort {outputs['poly_order']} -ort {temp_dir}/motion_demean.1D \
        {regressor_cmd} {censor_cmd} \
        -prefix {outputs['out_file']} -overwrite")

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def glm(in_files, out_file, design, model='BLOCK', contrasts=None, TR=None, pick_runs=None,
    motion_files=None, censor=True, motion_th=0.3, censor_files=None, 
    regressor_file=None, poly=None,
    fitts=True, errts=True, REML=True, perblock=False, FDR=None, check_TR=True):
    '''
    Parameters
    ----------
    design : OrderedDict(L=('../stimuli/L.txt', 24), R="../stimuli/R.txt 'BLOCK(24,1)'")
    model : str
    contrasts : OrderedDict([('L+R', '+0.5*L +0.5*R'), ...])

    Examples
    --------
    1. Canonical GLM
        design = OrderedDict()
        design['L'] = ('../stimuli/L_localizer.txt', 24)
        design['R'] = ('../stimuli/R_localizer.txt', 24)
        contrasts = OrderedDict()
        contrasts['L+R'] = '+0.5*L +0.5*R'
        contrasts['L-R'] = '+L -R'
        model = 'BLOCK' 

    2. FIR estimation
        design = OrderedDict()
        design['L'] = f"{stim_dir}/L_{condition}_deconv.txt 'CSPLINzero(0,24,11)'"
        design['R'] = f"{stim_dir}/R_{condition}_deconv.txt 'CSPLINzero(0,24,11)'"
        contrasts = None
        model = 'BLOCK' # This is not used for TENT/CSPLIN
    '''
    def parse_txt_fname(in_file):
        # Handle time selection like `dset.1D{0..10}`
        match = re.search(r"(.+)\'?\{(\d+)..(\d+|\$)\}\'?$", in_file)
        fname = match.group(1) if match else in_file
        begin = int(match.group(2)) if match else None
        end = int(match.group(3)) if match and match.group(3)!='$' else None
        return fname, begin, end
    def vcat_txt(in_files, out_file):
        # with open(out_file, 'wb') as fo:
        #     for in_file in in_files:
        #         with open(in_file, 'rb') as fi:
        #             shutil.copyfileobj(fi, fo)
        with open(out_file, 'w') as fo:
            for in_file in in_files:
                fname, begin, end = parse_txt_fname(in_file)
                with open(fname, 'r') as fi:
                    for k, line in enumerate(fi):
                        if begin is None or k >= begin:
                            fo.write(line)
                        if end is not None and k >= end:
                            break
    def load_txt(in_file, **kwargs):
        fname, begin, end = parse_txt_fname(in_file)
        return np.loadtxt(fname, **kwargs)[slice(begin, (end+1 if end is not None else end))]
    def pick_txt_rows(in_file, pick_runs, out_file):
        if np.array(pick_runs).dtype == bool:
            pick_runs = np.nonzero(pick_runs)[0]
        with open(out_file, 'w') as fo:
            with open(in_file, 'r') as fi:
                for k, line in enumerate(fi):
                    if k in pick_runs:
                        fo.write(line)
    temp_dir = utils.temp_folder()
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True, trailing_slash=True)
    outputs = {
        'motion_file': f"{out_dir}{prefix}.motion.1D",
        'X_image': f"{out_dir}X.{prefix}.jpg",
        'X_file': f"{out_dir}X.{prefix}.1D",
        'X_nocensor': f"{out_dir}X.{prefix}.nocensor.1D",
        'stats_file': f"{out_dir}stats.{prefix}{'_REML' if REML else ''}{ext}",
    }
    if FDR is None:
        FDR = not perblock
    if REML:
        outputs['stats_var'] = f"{out_dir}stats.{prefix}_REMLvar{ext}"
    if fitts:
        outputs['fitts_file'] = f"{out_dir}fitts.{prefix}{'_REML' if REML else ''}{ext}"
    if errts:
        outputs['errts_file'] = f"{out_dir}errts.{prefix}{'_REML' if REML else ''}{ext}"
    if censor:
        outputs['censor_file'] = f"{out_dir}{prefix}.censor.1D"
        outputs['n_censored'] = None
        censors = []
    default_ext = '.1D' if ext == '.1D' else '+orig.HEAD'
    # Check input/output extension compatibility
    input_ext = path.splitext(in_files[0])[1]
    if input_ext != ext:
        raise ValueError(f'Input extension "{input_ext}" is incompatible with output extension "{ext}"')
    # Pick runs
    if pick_runs is None:
        pick_runs = list(range(len(in_files)))
    in_files = np.array(in_files)[pick_runs]
    # Check TR
    TRs = np.array([afni.get_TR(f) for f in in_files])
    if TR is None:
        TR = TRs[0]
    try:
        assert(np.allclose(TRs, TR) or np.allclose(TRs, 0)) # TR=0 for 1D files
    except AssertionError as err:
        if check_TR:
            raise err
        else:
            print(f">> It seems that not all TRs are equal to {TR:.3f} sec: {TRs==TR}")
    # Check run lengths
    # 1D files should be OK if specified as dset.1D\'
    func_lens = np.array([afni.get_dims(f)[3] for f in in_files])
    if not np.all(func_lens == func_lens[0]):
        print('\x1b[7m*+ WARNING:\x1b[0m Not all runs have equal length:', func_lens)
    # Create demeaned motion regressor
    if motion_files is not None:
        motion_files = np.array(motion_files)[pick_runs]
        motion_lens = [load_txt(f).shape[0] for f in motion_files]
        assert(np.all(motion_lens == func_lens))
        run_lengths = ' '.join(f"{motion_len}" for motion_len in motion_lens)
        vcat_txt(motion_files, f"{temp_dir}/motion_param.1D")
        utils.run(f"1d_tool.py -infile {temp_dir}/motion_param.1D -set_run_lengths {run_lengths} \
            -demean -write {outputs['motion_file']} -overwrite")
        # Create motion censor
        if censor:
            utils.run(f"1d_tool.py -infile {temp_dir}/motion_param.1D -set_run_lengths {run_lengths} \
                -show_censor_count -censor_prev_TR \
                -censor_motion {motion_th} {temp_dir}/motion -overwrite")
            censors.append(f"{temp_dir}/motion_censor.1D")
    # Combine other censor (e.g., outlier censor)
    if censor:
        if censor_files is not None:
            censor_files = np.array(censor_files)[pick_runs]
            censor_lens = [load_txt(f).shape[0] for f in censor_files]
            assert(np.all(censor_lens == motion_lens))
            combine_censors(censor_files, f"{temp_dir}/other_censor.1D", method='concatinate')
            censors.append(f"{temp_dir}/other_censor.1D")
        combine_censors(censors, outputs['censor_file'], method='intersect')
        b = np.cumsum(np.r_[0, func_lens]) # Boundaries
        c = np.loadtxt(outputs['censor_file']) # Censors
        outputs['n_censored'] = [np.sum(c[b[k]:b[k+1]]==0) for k in range(len(b)-1)]
    # Prepare motion and censor commands
    total_regressors = 0
    if motion_files is not None:
        X = np.loadtxt(outputs['motion_file'])
        all_zero_cols = ~np.any(X, axis=0) 
        motion_labels = ['roll', 'pitch', 'yaw', 'dI-S', 'dR-L', 'dA-P']
        motion_cmds = []
        for k, motion_label in enumerate(motion_labels):
            if not all_zero_cols[k]: # Exclude all-zero columns from regressors
                total_regressors += 1
                motion_cmds.append(f"-stim_file {total_regressors} {outputs['motion_file']}'[{k}]' \
                    -stim_base {total_regressors} -stim_label {total_regressors} {motion_label}")
    else:
        motion_cmds = []
    censor_cmd = f"-censor {outputs['censor_file']}" if censor else ''
    # Other direct (non-HRF-convolved) regressors
    # TODO: Refactor this code
    # TODO: Check the compatibility of this code with varioius selection, etc.
    regressor_cmds = []
    if regressor_file is not None:
        X = np.loadtxt(regressor_file, ndmin=2)
        all_zero_cols = ~np.any(X, axis=0) 
        regressor_labels = [f'regressor{k+1:02d}' for k in range(X.shape[1])]
        for k, regressor_label in enumerate(regressor_labels):
            if not all_zero_cols[k]: # Exclude all-zero columns from regressors
                total_regressors += 1
                regressor_cmds.append(f"-stim_file {total_regressors} {regressor_file}'[{k}]' \
                    -stim_label {total_regressors} {regressor_label}")
    # Prepare model
    stim_cmds = []
    IRF_labels = []
    IRF_params = []
    IRF_cmds = []
    for condition, timing in design.items():
        total_regressors += 1
        if perblock:
            stim_cmd = f"-stim_times_IM {total_regressors} "
        else:
            stim_cmd = f"-stim_times {total_regressors} "
        stim_file_picked = f"{temp_dir}/stim_{condition}.txt"
        if isinstance(timing, six.string_types):
            stim_file, model_spec = shlex.split(timing)
            pick_txt_rows(stim_file, pick_runs, stim_file_picked)
            stim_cmd += ' '.join([stim_file_picked, model_spec])
        else:
            starts, duration = timing
            if isinstance(starts, six.string_types):
                pick_txt_rows(starts, pick_runs, stim_file_picked)
                stim_cmd += f"{stim_file_picked} "
            else:
                raise NotImplementedError
            if np.isscalar(duration):
                if model.lower() == 'block':
                    stim_cmd += f"'BLOCK({duration},1)'" # Must be quoted because of the parentheses (may not be true here...)
            else:
                raise NotImplementedError
        stim_cmd += f" -stim_label {total_regressors} {condition} "    
        stim_cmds.append(stim_cmd)
        # Prepare IRF output for nonparametric models like TENT, etc.
        match = re.search(r'(TENT|TENTzero|CSPLIN|CSPLINzero)\(([^,]+),([^,]+),(\d+)\)', stim_cmd)
        if match:
            IRF_label = f"{'irp' if REML else 'irf'}.{condition}"
            IRF_labels.append(IRF_label)
            model_ = match.group(1)
            n = int(match.group(4))
            IRF_params.append([float(match.group(2)), float(match.group(3)), n, (n-2) if 'zero' in model_ else n])
            IRF_cmds.append(f"-iresp {total_regressors} {temp_dir}/{IRF_label}")
            outputs[IRF_label] = f"{out_dir}{IRF_label}.{prefix}{'_REML' if REML else ''}{ext}" # Usually, this should be specified at the beginning of the function...

    # Prepare contrasts
    contrast_cmds = []
    if contrasts is not None:
        for k, (label, expr) in enumerate(contrasts.items()):
            contrast_cmds.append(f"-gltsym 'SYM: {expr}' -glt_label {k+1} {label} ")
    # Perform GLM
    # "-TR_times" specifies the time step for FIR/TENT output
    # "-local_times" means each run uses its own time base
    utils.run(f"3dDeconvolve -force_TR {TR} -TR_times {TR} -local_times \
        -input {' '.join(in_files)} {censor_cmd} \
        -polort {'A' if poly is None else str(poly)} \
        -num_stimts {total_regressors} {' '.join(motion_cmds)} {' '.join(regressor_cmds)} {' '.join(stim_cmds)} \
        {' '.join(contrast_cmds) if contrasts is not None else ''} \
        -xjpeg {outputs['X_image']} -x1D {outputs['X_file']} -x1D_uncensored {outputs['X_nocensor']} \
        -tout -bucket {temp_dir}/stats {' '.join(IRF_cmds)} \
        {f'-fitts {temp_dir}/fitts' if fitts else ''} {f'-errts {temp_dir}/errts' if errts else ''} \
        {'-noFDR' if not FDR else ''} \
        {'-x1D_stop' if REML else ''} -jobs {DEFAULT_JOBS} -overwrite", error_pattern=r'^\*{2}\w')
        
    if REML:
        utils.run(f"tcsh {temp_dir}/stats.REML_cmd", error_pattern=r'^\*{2}\w')
        copy_dset(f"{temp_dir}/stats_REMLvar{default_ext}", outputs['stats_var'])
        for k, IRF_label in enumerate(IRF_labels):
            # Select beta values (not t values)
            L = IRF_params[k][3]
            try: # Check if there are labels associated with each subbrick
                afni.get_brick_labels(f"{temp_dir}/stats_REML{default_ext}")
            except subprocess.CalledProcessError: # There are no label: select subbrick by index (if )
                utils.run(f"3dTcat -tr {TR} -prefix {outputs[IRF_label]} -overwrite \
                    {temp_dir}/stats_REML{default_ext}'[{','.join([f'{1+k*2*L+kk*2:d}' for kk in range(L)])}]'")
            else: # There are labels: select subbrick by label (if )
                utils.run(f"3dTcat -tr {TR} -prefix {outputs[IRF_label]} -overwrite \
                    {temp_dir}/stats_REML{default_ext}'[{','.join([f'{IRF_label[4:]}#{kk}_Coef' for kk in range(L)])}]'")
    else:
        for IRF_label in IRF_labels:
            copy_dset(f"{temp_dir}/{IRF_label}{default_ext}", outputs[IRF_label])
    copy_dset(f"{temp_dir}/stats{'_REML' if REML else ''}{default_ext}", outputs['stats_file'])
    if fitts:
        copy_dset(f"{temp_dir}/fitts{'_REML' if REML else ''}{default_ext}", outputs['fitts_file'])
    if errts:
        copy_dset(f"{temp_dir}/errts{'_REML' if REML else ''}{default_ext}", outputs['errts_file'])

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def collect_perblock_stats(mask_file, stats_file, n_conditions, n_repeats, n_runs=None, n_stats=2, stat_index=0, skip_first=1):   
    sel = slice(skip_first, skip_first+stat_index+n_conditions*n_repeats*n_stats, n_stats)
    labels = afni.get_brick_labels(stats_file)[sel]
    mask = io.Mask(mask_file)
    X = mask.dump(stats_file)[:,sel].T
    y = np.repeat(np.arange(n_conditions), n_repeats)
    if n_runs is not None:
        groups = np.tile(np.repeat(np.arange(n_runs), n_repeats/n_runs), n_conditions)
    else:
        groups = None
    # Sanity check
    assert(len(set((yy, label.split('#')[0]) for yy, label in zip(y, labels)))) # y is consistent with labels
    return X, y, groups, labels


def copy_dset(in_file, out_file):
    prefix, ext = afni.split_out_file(in_file)
    if ext == '.1D':
        shutil.copy(in_file, out_file)
    else:
        utils.run(f"3dcopy {prefix}{ext} {out_file} -overwrite")


def align_center(base_file, in_file, out_file):
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'xform_file': f"{prefix}.aff12.1D",
    }

    centerize = lambda n, d: -(n-1)*d/2
    data = np.float_(utils.run(f"3dinfo -n4 -d3 -o3 {base_file}", verbose=0)['output'][-1].split())
    nb, db, ob = data[:3].astype(int), data[4:7], data[7:10]
    cb = centerize(nb, db)
    data = np.float_(utils.run(f"3dinfo -n4 -d3 -o3 {in_file}", verbose=0)['output'][-1].split())
    ni, di, oi = data[:3].astype(int), data[4:7], data[7:10]
    ci = centerize(ni, di)
    oo = ci + (ob - cb)
    utils.run(f"3dcopy {in_file} {outputs['out_file']} -overwrite")
    utils.run(f"3drefit -xorigin_raw {oo[0]} -yorigin_raw {oo[1]} -zorigin_raw {oo[2]} {outputs['out_file']}")
    utils.run(f"3dresample -rmode NN -master {base_file} -prefix {outputs['out_file']} -overwrite -input {outputs['out_file']}")
    shift = oo - oi # Amount of translation that is needed to transform in_file to base_file
    io.write_affine(outputs['xform_file'], np.c_[np.eye(3), -shift]) # Afni needs the inverse transform, thus "-shift"

    all_finished(outputs)
    return outputs







if __name__ == '__main__':
    pass
