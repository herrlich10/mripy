#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import subprocess, os
import numpy as np
from collections import OrderedDict
from . import six, afni, io, utils


def afni_costs(base_file, in_file, mask=None):
    '''
    In general, the alignment algorithm want to **minimize** the cost.

    Some metrics in their canonical form are maximized when perfectly aligned 
    (e.g., mi and hel). These are usually negated when output by -allcost, 
    so that ALL cost (EXCEPT lss) values will decrease when alignment improves 
    (given that the base/source contrasts are valid as required by the metric, 
    e.g., lpc is ONLY meaningful if the base and source have opposite contrast).
    This behavior can be verified by manually applying a transform to a image.

    Based on global correlation coefficient
    ---------------------------------------
    ls : 1 - abs(Pearson corrcoef), near zero
    sp : 1 - abs(Spearman corrcoef), near zero
    lss (maximized): Pearson corrcoef, near one (the ONLY exception)

    Based on 2D joint histogram
    ---------------------------
    mi (negated) : - mutual information, large negative value
    nmi : 1/normalized MI, small positive value
    je : joint entropy, small positive value
    hel (negated) : - Hellinger distance, large negative value
    crM : near zero
    crA : near zero
    crU : near zero

    Based on local (a few mm) Pearson correlation coefficient (similar to boundary-based methods)
    ---------------------------------------------------------
    lpc : sum(w[i]*pc[i]*abs(pc[i])) / sum(w[i]), large negative value (FOR opposite contrast ONLY)
    lpa : 1 - abs(lpc), large negative value (FOR same contrast ONLY)
    lpc+ : lpc + hel*0.4 + crA*0.4 + nmi*0.2 + mi*0.2 + ov*0.4
    lpa+ : lpa + hel*1.0 + crA*0.4 + nmi*0.2 + mi*0.0 + ov*0.0

    References
    ----------
    [1] https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAllineate.html
        See "Cost functional descriptions (for use with -allcost output)" section
    [2] https://www.youtube.com/watch?v=PaZinetFKGY&list=PL_CD549H9kgqJ1GDXAs1BWkgEimAHZeNX
    '''
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
