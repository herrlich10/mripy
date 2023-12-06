#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance


def rot2trans(params, radius=50):
    '''
    Convert rotation (deg) to translation (mm) on a sphere with 50 mm radius (Power et al., 2014),
    which approximates the average distance from the cortex to the center of the head.
    
    This conversion makes summary statistics more interpretable and with appropriate units.
    
    Parameters
    ----------
    params : Nx6 array
        Rigid-body motion correction parameters [R-L(x), A-P(y), I-S(z), pitch(x), roll(y), yaw(z)].
        The translations are in mm, and the rotations are in degrees.
    radius : float
        Radius (mm) for converting rotation to translation.
        
    Returns
    -------
    trans : Nx6 array
        Motion parameters expressed as pure translation in a 6D space.
    '''
    trans = params[:,3:]/180*pi * radius
    return np.c_[params[:,:3], trans]


def compute_abs_disp(params, demean=True):
    '''
    Compute absolute displacement as Euclidean distance in a 6D space,
    relative to the mean location.
    
    Parameters
    ----------
    params : array, (N,6)
    demean : bool
        If True, the displacement is relative to the mean location.
        If False, the displacement is relative to the motion correction target.
    
    Returns
    -------
    displacements : array, (N,)
    '''
    if demean:
        params = params - np.mean(params, axis=0)
    return np.linalg.norm(params, ord=2, axis=1)


def compute_rel_disp(params):
    '''
    Compute relative displacement between neighboring volumes 
    as Euclidean distance in a 6D space.
    
    Parameters
    ----------
    params : array, (N,6)
    
    Returns
    -------
    displacements : array, (N,)
    '''
    d = np.r_[np.zeros(params[:1].shape), np.diff(params, axis=0)]
    return compute_abs_disp(d, demean=False)


def compute_head_motion_metrics(params_list):
    '''
    Compute various head motion metrics.
    
    Metrics
    -------
    RMS : RMS motion parameters (mm)
        Root mean squared values of the L2 norm (across dimensions) of 
        demeaned motion correction parameters.
    RMS_within : RMS within-run (mm)
        Same as the above but demean within each run.
    MaxRelDisp_within : Max relative displacement within-run (mm)
        Maximal relative displacement between neighboring volumes within runs.
    MaxRelDisp_between : Max relative displacement between-run (mm)
        Maximal relative displacement between the first volume of a run and 
        the last volume of the previous run.
    MaxDist : Max distance (mm)
        The largest distance between any two volumes in a session.
    MaxDist_within : Max distance within-run (mm)
        The largest distance between any two volumes within runs.     

    Rotation (deg) is converted to translation (mm) on a sphere with 
        50 mm radius (Power et al., 2014).
    Displacement is computed as the Euclidean distance (L2 norm) in the 6D space.

    Parameters
    ----------
    params_list : list of Nx6 arrays
        Rigid-body motion correction parameters from multiple runs.
        The six parameters are [R-L(x), A-P(y), I-S(z), pitch(x), roll(y), yaw(z)].
        The translations are in mm, and the rotations are in degrees.
    
    Returns
    -------
    metrics : dict
        Various head motion metrics.
    '''
    first_vol = [np.r_[1, np.zeros(len(params)-1)] for params in params_list]
    first_vol = np.concatenate(first_vol).astype(bool)
    trans_list = [rot2trans(params) for params in params_list]
    x = np.concatenate(trans_list, axis=0)
    metrics = {}
    # RMS
    abs_disp = compute_abs_disp(x)
    metrics['RMS'] = np.sqrt(np.mean(abs_disp**2))
    # RMS within-run
    # Or equivalently, norm(concatenate([params-mean(params, axis=0) for params in trans_list]), axis=1)
    abs_disp_within = np.concatenate([compute_abs_disp(params) for params in trans_list])
    metrics['RMS_within'] = np.sqrt(np.mean(abs_disp_within**2))
    # Maximal relative displacement between neighboring volumes
    rel_disp = compute_rel_disp(x)
    metrics['MaxRelDisp_within'] = max(rel_disp[~first_vol])
    metrics['MaxRelDisp_between'] = max(rel_disp[first_vol])
    # Maximal distance between any two volumes
    metrics['MaxDist'] = max(distance.pdist(x, metric='euclidean'))
    metrics['MaxDist_within'] = max([max(distance.pdist(params, metric='euclidean')) for params in trans_list])
    return metrics


if __name__ == '__main__':
    import doctest
    doctest.testmod()
