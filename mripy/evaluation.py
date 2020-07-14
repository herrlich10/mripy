#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import subprocess, os
import numpy as np
from collections import OrderedDict
from . import six, afni, io, utils


def afni_costs(base_file, in_file, mask=None):
    if mask is not None:
        emask = utils.temp_prefix(suffix='.nii')
        utils.run(f"3dcalc -a {mask} -expr '1-step(a)' -prefix {emask} -overwrite")
    emask_cmd = f'-emask {emask}' if mask is not None else ''
    cmd = f"3dAllineate -base {base_file} -input {in_file} -allcostX {emask_cmd}"
    lines = afni.check_output(cmd, pattern=r'^\s+\S+\s+=\s+\S+')
    costs = OrderedDict()
    for line in lines:
        name, value = line.split('=')
        name = name.strip()
        if name not in costs:
            costs[name] = []
        costs[name].append(float(value))
    if np.all(np.array([len(v) for k, v in costs.items()]) == 1):
        costs = OrderedDict((k, v[0]) for k, v in costs.items())
    if mask is not None:
        os.remove(emask)
    return costs


def residual_motion(files, print_table=False):
    dsets = [io.read_afni(f) for f in files]
    scores = {'filename': [afni.get_prefix(f) for f in files]}

    # ===== Temporal variation =====
    # Caveates:
    # 1. Across dsets, those of different noise level have different inherent
    #    variability, and thus cannot be compared directly.
    # 2. Within dsets (and across different methods), different levels of
    #    spatial smoothing (caused by resampling) also affect this measure.
    scores['tSTD'] = [np.mean(np.std(dset, axis=-1)) for dset in dsets]

    # ===== Motion as simultaneous intensity change across many voxels =====
    # x = np.zeros((len(dsets), dsets[0].shape[-1]-1))
    # for k, dset in enumerate(dsets):
    #     d = np.diff(dset, axis=-1)
    #     np.abs()
    #     x[k,:] = np.mean(, axis=(0,1,2))


    # Print tabular results using pandas
    if print_table:
        import pandas as pd
        df = pd.DataFrame(scores)
        print(df)


if __name__ == '__main__':
    pass
