#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import os, glob, shutil, shlex, re, subprocess, multiprocessing, warnings
from os import path
from collections import OrderedDict
import numpy as np
try:
    import pandas as pd
except ImportError:
    warnings.warn('Cannot import pandas, which is required for some functions.', ImportWarning)
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
    for output in outputs:
        output['finished'] = np.all([(path.exists(f) if n.endswith('_file') else (f is not None)) for n, f in output.items()])
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
    fnt = afni.get_dims(forward_file)[3]
    fsub = {'before': f"[0..{min(fnt, n_subs)-1}]", 'after': f"[{max(0, fnt-n_subs)}..$]"}[reverse_loc]
    outputs['template_idx'] = {'before': min(fnt, n_subs)//2, 'after': (max(0, fnt-n_subs) + fnt)//2}[reverse_loc]
    # There is no need to protect the paths against whitespace (yet) because AFNI doesn't support it.
    utils.run(f"3dvolreg -zpad 1 -base {n_subs//2} -prefix {temp_dir}/forwards.nii -overwrite {forward_file}'{fsub}'")
    utils.run(f"3dTstat -median -prefix {temp_dir}/forward.nii -overwrite {temp_dir}/forwards.nii")
    utils.run(f"3dAutomask -apply_prefix {temp_dir}/forward.masked.nii -overwrite {temp_dir}/forward.nii")

    # Prepare reverse template
    rnt = afni.get_dims(reverse_file)[3]
    rsub = {'before': f"[{max(0, rnt-n_subs)}..$]", 'after': f"[0..{min(rnt, n_subs)-1}]"}[reverse_loc]
    # Input datasets for 3dQwarp must be on the same 3D grid (unlike program 3dAllineate)!
    utils.run(f"3dresample -rmode NN -master {temp_dir}/forwards.nii \
        -prefix {temp_dir}/reverses.nii -overwrite -input {reverse_file}'{rsub}'")
    utils.run(f"3dvolreg -zpad 1 -base {n_subs//2} -prefix {temp_dir}/reverses.nii -overwrite {temp_dir}/reverses.nii")
    utils.run(f"3dTstat -median -prefix {temp_dir}/reverse.nii -overwrite {temp_dir}/reverses.nii")
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


def apply_transforms(transforms, base_file, in_file, out_file, interp=None, res=None, save_xform=None):
    if isinstance(transforms, six.string_types):
        transforms = [transforms]
    if interp is None:
        interp = 'wsinc5'
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    if save_xform is not None:
        outputs['xform_file'] = save_xform

    
    has_nwarp = not np.all([f.endswith('.1D') for f in transforms])
    res_cmd = f"-newgrid {res}" if res is not None else ''
    if has_nwarp:
        transform_list = ' '.join(transforms)
        # 'transform_list' must be quoted
        utils.run(f"3dNwarpApply -interp {interp} -master {base_file} {res_cmd} \
            -nwarp '{transform_list}' -source {in_file} \
            -prefix {out_file} -overwrite")
    else:
        combined = combine_affine_transforms(transforms, out_file=save_xform)['out_file']
        # 'wsinc5' is 8x slower than 'quintic', but is highly accurate 
        # and should reduce the smoothing artifacts (see 3dAllineate)
        utils.run(f"3dAllineate -final {interp} -base {base_file} {res_cmd} \
            -1Dmatrix_apply {combined} -input {in_file} \
            -prefix {out_file} -overwrite")
        if not save_xform:
            os.remove(combined)

    all_finished(outputs)
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
    '''
    match = re.search(r'(-?\d+\.\d{2}I) (-?\d+\.\d{2}R) (-?\d+\.\d{2}A).*(-?\d+\.\d{2}S) (-?\d+\.\d{2}L) (-?\d+\.\d{2}P)', nudge_cmd)
    if match:
        I, R, A, S, L, P = match.groups()
        temp_file = utils.temp_prefix(suffix='.nii')
        utils.run(f"3drotate -NN -clipit -rotate {I} {R} {A} -ashift {S} {L} {P} -prefix {temp_file} {in_file}")
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


def align_epi(in_files, out_files, best_reverse=None, blip_results=None, blip_kws=None, volreg_kws=None, 
    template=None, template_pool=None, final_resample=True, final_res=None):
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
            # Find best template
            Xs = [np.loadtxt(f"{prefix}.pass1.param.1D") for prefix in temp_prefixs]
            XX = np.vstack(Xs)
            idx = np.argmin([np.sqrt(np.sum(np.linalg.norm(XX-x, axis=1)**2)) for x in XX])
            L = [X.shape[0] for X in Xs]
            D = idx - np.cumsum(L)
            run_idx = np.nonzero(D<0)[0][0]
            TR_idx = L[run_idx] + D[run_idx]
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
        if isinstance(dicom_dirs, six.string_types):
            dicom_dirs = glob.glob(dicom_dirs)    
        label2dicom_dir = OrderedDict()
        for dicom_dir in dicom_dirs:
            f = glob.glob(f"{dicom_dir}/*{dicom_ext}")[0]
            header = dicom.parse_dicom_header(f)
            label = header['SeriesDescription'][len(header['ProtocolName'])+1:]
            label2dicom_dir[label] = dicom_dir

        # Convert dicom files
        for label in ['UNI_Images', 'INV2_ND', 'INV2']:
            pc.run(io.convert_dicom, label2dicom_dir[label], f"{temp_dir}/{label}.nii", dicom_ext=dicom_ext)
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


def prep_mp2rages(data_dir, sessions=None, subdir_pattern='T1??', **kwargs):
    if sessions is None:
        sessions = dicom_report.inspect_mp2rage(data_dir, subdir_pattern=subdir_pattern).session
    pc = utils.PooledCaller(pool_size=4)
    for session_dir in [f'{data_dir}/{session}' for session in sessions]:
        out_file = kwargs.pop('out_file') if 'out_file' in kwargs else 'T1.nii'
        pc.run(prep_mp2rage, f'{session_dir}/{subdir_pattern}', out_file=f'{session_dir}/{out_file}', **kwargs)
    outputs = pc.wait()
    return OrderedDict([(session, output) for session, output in zip(sessions, outputs)])


def fs_recon(T1s, out_dir, T2=None, FLAIR=None, NIFTI=False, V1=True):
    if isinstance(T1s, six.string_types):
        T1s = [T1s]
    out_dir = path.realpath(out_dir)
    subjects_dir, subj = path.split(out_dir) # Environment variable may need full path
    temp_dir = utils.temp_folder()
    outputs = {
        'suma_dir': f"{out_dir}/SUMA",
    }
    # Setup FreeSurfer SUBJECTS_DIR
    if not path.exists(subjects_dir):
        os.makedirs(subjects_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    if not path.exists(f'{subjects_dir}/V1_average'):
        os.symlink(f"{os.environ['FREESURFER_HOME']}/subjects/V1_average", f"{subjects_dir}/V1_average")
    # Run recon-all
    expert_file = f"{temp_dir}/expert_options.txt"
    with open(expert_file, 'w') as fo:
        fo.write('mris_inflate -n 30\n')
    utils.run(f"recon-all -s {subj} \
        {' '.join([f'-i {T1}' for T1 in T1s])} \
        {f'-T2 {T2} -T2pial' if T2 is not None else ''} \
        {f'-FLAIR {FLAIR} -FLAIRpial' if FLAIR is not None else ''} \
        -all -hires -expert {expert_file} \
        -parallel -openmp {DEFAULT_JOBS} \
        {'-label-v1' if V1 else ''}", 
        error_pattern='', goal_pattern='recon-all .+ finished without error')
    # Make SUMA dir and viewing script
    create_suma_dir(out_dir, NIFTI=False)
    os.rename(outputs['suma_dir'], outputs['suma_dir']+'_woNIFTI')
    create_suma_dir(out_dir, NIFTI=True)
    os.rename(outputs['suma_dir'], outputs['suma_dir']+'_NIFTI')
    os.symlink('SUMA'+('_NIFTI' if NIFTI else '_woNIFTI'), outputs['suma_dir'])

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def create_suma_dir(subj_dir, NIFTI=False):
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


def scale(in_file, out_file, mask_file=None):
    mean_file = utils.temp_prefix(suffix='.nii')
    utils.run(f"3dTstat -mean -prefix {mean_file} -overwrite {in_file}")
    if mask_file is not None:
        utils.run(f"3dcalc -a {in_file} -b {mean_file} -c {mask_file} \
            -expr 'min(200,a/b*100)*step(a)*step(b)*c' \
            -prefix {out_file} -overwrite")
    else:
        utils.run(f"3dcalc -a {in_file} -b {mean_file} \
            -expr 'min(200,a/b*100)*step(a)*step(b)' \
            -prefix {out_file} -overwrite")
    os.remove(mean_file)


def skullstrip(in_file, out_file=None):
    if out_file is None:
        prefix, ext = afni.split_out_file(in_file)
        out_file = f"{prefix}_ns{ext}"
    prefix, ext = afni.split_out_file(out_file)
    outputs = {'out_file': f"{prefix}{ext}"}
    utils.run(f"3dSkullStrip -orig_vol -prefix {out_file} -overwrite -input {in_file}")
    all_finished(outputs)
    return outputs


def align_anat(base_file, in_file, out_file, strip_base=True, strip_in=True, method=None, cost=None, n_params=None, interp=None):
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
    temp_dir = utils.temp_folder()
    prefix, ext = afni.split_out_file(out_file)
    outputs = {
        'out_file': f"{prefix}{ext}",
        'xform_file': f"{prefix}.aff12.1D",
        # 'cost': None,
    }
    pc = utils.PooledCaller()
    if strip_base:
        pc.run(skullstrip, base_file, f"{temp_dir}/base_ns.nii")
    else:
        pc.run(f"3dcopy {base_file} {temp_dir}/base_ns.nii")
    if strip_in:
        pc.run(skullstrip, in_file, f"{temp_dir}/in_ns.nii")
    else:
        pc.run(f"3dcopy {in_file} {temp_dir}/in_ns.nii")
    pc.wait()
    if method == '3dallineate':
        utils.run(f"3dAllineate -final {interp} -cost {cost} -warp {n_params} -maxrot 90 \
            -base {temp_dir}/base_ns.nii -input {temp_dir}/in_ns.nii \
            -autoweight -source_automask+2 -twobest 11 -fineblur 1 \
            -1Dmatrix_save {outputs['xform_file']} \
            -prefix {temp_dir}/out_ns.nii -overwrite")
    elif method == 'align_epi_anat':
        pass
    if strip_in:
        apply_transforms(outputs['xform_file'], base_file, in_file, outputs['out_file'], interp=interp)
    else:
        utils.run(f"3dcopy {temp_dir}/out_ns.nii {outputs['out_file']}")

    shutil.rmtree(temp_dir)
    all_finished(outputs)
    return outputs


def align_anat2epi(anat_file, epi_file, out_file, init_oblique=None, init_epi_rotate=None, init_anat_rotate=None):
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
    apply_transforms(transforms, f"{temp_dir}/anat_init.nii", anat_file, \
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
    motion_files=None, censor=True, motion_th=0.3, censor_files=None, regressor_files=None, poly=None,
    fitts=True, errts=True, REML=True, perblock=False, FDR=None):
    '''
    Parameters
    ----------
    design : OrderedDict(L=('../stimuli/L.txt', 24), R="../stimuli/R.txt 'BLOCK(24,1)'")
    model : str
    contrasts : OrderedDict([('L+R', '+0.5*L +0.5*R'), ...])
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
        return np.loadtxt(fname, **kwargs)[slice(begin, (end+1 if end is not None else end)),:]
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
    # Pick runs
    if pick_runs is None:
        pick_runs = list(range(len(in_files)))
    in_files = np.array(in_files)[pick_runs]
    # Check TR
    TRs = np.array([afni.get_TR(f) for f in in_files])
    if TR is None:
        TR = TRs[0]
    assert(np.allclose(TRs, TR) or np.allclose(TRs, 0)) # TR=0 for 1D files
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
    X = np.loadtxt(outputs['motion_file'])
    all_zero_cols = ~np.any(X, axis=0) 
    motion_labels = ['roll', 'pitch', 'yaw', 'dI-S', 'dR-L', 'dA-P']
    motion_cmds = []
    for k, motion_label in enumerate(motion_labels):
        if not all_zero_cols[k]: # Exclude all-zero columns from regressors
            total_regressors += 1
            motion_cmds.append(f"-stim_file {total_regressors} {outputs['motion_file']}'[{k}]' \
                -stim_base {total_regressors} -stim_label {total_regressors} {motion_label}")
    censor_cmd = f"-censor {outputs['censor_file']}" if censor else ''
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
        -num_stimts {total_regressors} {' '.join(motion_cmds)} {' '.join(stim_cmds)} \
        {' '.join(contrast_cmds) if contrasts is not None else ''} \
        -xjpeg {outputs['X_image']} -x1D {outputs['X_file']} -x1D_uncensored {outputs['X_nocensor']} \
        -tout -bucket {temp_dir}/stats {' '.join(IRF_cmds)} \
        {f'-fitts {temp_dir}/fitts' if fitts else ''} {f'-errts {temp_dir}/errts' if errts else ''} \
        {'-noFDR' if not FDR else ''} \
        {'-x1D_stop' if REML else ''} -jobs {DEFAULT_JOBS} -overwrite", error_pattern=r'^\*{2}\w')
        
    if REML:
        utils.run(f"tcsh {temp_dir}/stats.REML_cmd")
        copy_dset(f"{temp_dir}/stats_REMLvar{default_ext}", outputs['stats_var'])
        for k, IRF_label in enumerate(IRF_labels):
            utils.run(f"3dTcat -tr {TR} -prefix {outputs[IRF_label]} -overwrite \
                {temp_dir}/stats_REML{default_ext}'[{','.join([f'{IRF_label[4:]}#{kk}_Coef' for kk in range(IRF_params[k][3])])}]'")
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


def copy_dset(src, dst):
    prefix, ext = afni.split_out_file(src)
    if ext == '.1D':
        shutil.copy(src, dst)
    else:
        utils.run(f"3dcopy {prefix}{ext} {dst} -overwrite")


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
