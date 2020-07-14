#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .. import utils, io, afni, preprocess as prep, surface


def align_suma2exp_mp2rage(suma_dir, T1_al, T1_al_affine=None, T1s=None, T1s_raw=None):
    res_dir, prefix, ext = afni.split_out_file(T1_al, split_path=True, trailing_slash=True)
    if T1_al_affine is None:
        T1_al_affine = f"{res_dir}{prefix}.aff12.1D"
    if T1s is None:
        T1s = f"{res_dir}T1??.nii"
    if T1s_raw is None:
        T1s_raw = f"{res_dir}../raw_fmri/T1??"
    print('>> ========== Create MP2RAGE high SNR region mask ==========')
    prep.assign_mp2rage_labels(T1s, T1s_raw)
    prep.create_mp2rage_SNR_mask(T1s, f"{res_dir}mp2rage_mask.nii")
    prep.apply_transforms(T1_al_affine, T1_al, \
        f"{res_dir}mp2rage_mask.nii", f"{res_dir}mp2rage_mask_al.nii", interp='NN')
    print('>> ========== ANTs align SurfVol to T1_al ==========')
    surf_vol = afni.get_surf_vol(suma_dir)
    outputs = prep.align_ants(T1_al, surf_vol, f"{res_dir}SurfVol_al.nii", 
        strip=[prep.copy_dset, prep.copy_dset], base_mask=f"{res_dir}mp2rage_mask_al.nii")
    print('>> ========== Transform SUMA surface meshes and volumes ==========')
    surface.transform_suma(outputs['transform'], suma_dir)
    suma_subj = afni.get_suma_subj(suma_dir)
    prep.create_suma_script(f"{suma_dir+'.al'}/{suma_subj}_both.spec", \
        f"{res_dir}SurfVol_al.nii", f"{res_dir}run_suma_al")


if __name__ == '__main__':
    pass
