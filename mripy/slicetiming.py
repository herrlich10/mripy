#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def Siemens_produce_interleaved_slices(n_slices):
    '''
    Slices that are excited at each shot in the interleaved mode 
    for the Siemens produce sequences.
    
    Note that, Siemens varies the starting slice based on 
    the number of slices (even or odd).
    
    Parameters
    ----------
    n_slices : int
        The total number of slices.
        
    Returns
    -------
    slices : array
        The slices that are excited at each shot.
        
    >>> pprint_slices_at_each_shot(Siemens_produce_interleaved_slices(9))
    0,2,4,6,8,1,3,5,7
    
    >>> pprint_slices_at_each_shot(Siemens_produce_interleaved_slices(8))
    1,3,5,7,0,2,4,6
    '''
    if n_slices % 2 == 1:
        slices = np.r_[np.r_[0:n_slices:2], np.r_[1:n_slices:2]]
    else:
        slices = np.r_[np.r_[1:n_slices:2], np.r_[0:n_slices:2]]
    return slices


def CMRR_C2P_MB_interleaved_slices(n_slices, mb):
    '''
    Slices that are excited at each shot in the interleaved mode 
    for the CMRR C2P multiband sequence.
    
    Based on official document: "CMRR Multiband Sequence Slice Order"
    https://wiki.humanconnectome.org/download/attachments/40534057/CMRR_MB_Slice_Order.pdf
    
    Parameters
    ----------
    n_slices : int
        The total number of slices.
    mb : int
        Multiband accelatration factor.
        1 = single band acquisition
    
    Returns
    -------
    slices : array
        The slices that are excited at each shot.
        For single band acquisition, 
            array([1, 3, 5, 7, 0, 2, 4, 6])
        For multiband acquisition, 
            array([[0, 4],
                   [2, 6],
                   [1, 5],
                   [3, 7]])
    
    Single band acquisition: Siemens product behavior (for compatibility),
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(9, 1))
    0,2,4,6,8,1,3,5,7
    
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(8, 1))
    1,3,5,7,0,2,4,6
    
    Multiband acquisition: slice excitation always starts with slice0 
    in CMRR multiband C2P sequences.
    For 2, 3, and 4 shots, there is no way to avoid adjacent slice in adjacent excitation.
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(8, 2))
    0,4 - 2,6 - 1,5 - 3,7
    
    For number of shots >= 5, interleaved slice series have been implemented 
    to guarantee that no adjacent slices are exited in adjacent order.
    Odd number of shots,
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(10, 2))
    0,5 - 2,7 - 4,9 - 1,6 - 3,8
    
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(14, 2))
    0,7 - 2,9 - 4,11 - 6,13 - 1,8 - 3,10 - 5,12
    
    Even number of shots,
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(16, 2))
    0,8 - 3,11 - 6,14 - 1,9 - 4,12 - 7,15 - 2,10 - 5,13
    
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(20, 2))
    0,10 - 3,13 - 6,16 - 9,19 - 1,11 - 4,14 - 7,17 - 2,12 - 5,15 - 8,18
    
    Special case: n_shots=6,
    >>> pprint_slices_at_each_shot(CMRR_C2P_MB_interleaved_slices(12 2))
    0,6 - 2,8 - 4,10 - 1,7 - 5,11 - 3,9
    '''
    if mb == 1: # Single band acquisition (use Siemens product behavior, 2.2.1)
        slices = Siemens_produce_interleaved_slices(n_slices)
    else: # Multiband acquisition (2.2.2)
        assert(n_slices % 2 == 0)
        n_shots = n_slices // mb
        if n_shots == 6: # Special case in even number of shots,
            # CMRR uses a special order
            # to avoid adjacent slices in adjacent excitation (2.2.2.2)
            slices = np.reshape([0,2,4,1,5,3], [-1,1]) + np.arange(mb)*n_shots
        elif n_shots % 2 == 0 and n_shots >=5: # For even number of shots, 
            # CMRR uses an increment-n-slice interleave pattern 
            # to avoid adjacent slices in adjacent excitation (2.2.2.2)
            n_inc = n_shots // 2 - 1
            if n_inc % 2 == 0:
                n_inc -= 1
            # print(f"{n_shots}, {n_inc}")
            slices = []
            for k in range(n_inc):
                for s in range(k, n_shots, n_inc):
                    slices.append([s, s+n_shots])
            slices = np.array(slices)
        else:
            # For n_shots in [2, 3, 4],
            # there is no way to avoid adjacent slice in adjacent excitation.
            # For odd number of shots, 
            shots = np.r_[np.r_[0:n_shots:2], np.r_[1:n_shots:2]]
            slices = np.reshape(shots, [-1,1]) + np.arange(mb)*n_shots
    return slices


def infer_slices(slice_timing):
    '''
    Infer slices that are excited at each shot from slice timing.
    
    Parameters
    ----------
    slice_timing : array
        Acquisition timing of each slice (slice0, slice1, slice2, ...).
    
    Returns
    -------
    slices : array
        The slices that are excited at each shot.
        1D if single band and 2D if multiband (one row for each shot).
    dt : float
        Estimated interval between shots.

    >>> infer_slices([0, 1.2, 0.4, 1.6, 0.8])
    (array([0, 2, 4, 1, 3]), 0.4)
    
    >>> infer_slices([0, 1, 0, 1])[0]
    array([[0, 2],
           [1, 3]])
    '''
    shot_timing = np.unique(slice_timing) # Sorted
    slices = np.squeeze([np.nonzero(slice_timing==st)[0] for st in shot_timing])
    dt = np.median(np.diff(shot_timing))
    return slices, dt


def pprint_slices_at_each_shot(slices):
    '''
    Pretty print slices that are excited at each shot in the same format 
    as in the CMRR document: "CMRR Multiband Sequence Slice Order"
    '''
    if slices.size > len(slices): # Multiband acquisition
        slices_str = ' - '.join([','.join([f"{s}" for s in ss]) for ss in slices])
    else: # Single band acquisition
        slices_str = ','.join([f"{s}" for s in slices])
    print(slices_str)

    
def excitation_order(slices):
    '''
    Determine excitation order for each slice.
    
    Parameters
    ----------
    slices : array
        The slices that are excited at each shot.
    
    Returns
    -------
    order : array
        The excitation order for each slice (slice0, slice1, slice2, ...).
        Slice timing = order * dt
        
    >>> excitation_order([0, 2, 4, 1, 3])
    array([0, 3, 1, 4, 2])
    
    >>> excitation_order([[0, 2], [1, 3]])
    array([0, 1, 0, 1]))
    '''
    slices = np.asarray(slices)
    if slices.size > len(slices): # Multiband acquisition
        order = np.argsort([min(ss) for ss in slices])
        order = np.tile(order, slices.shape[1])
    else: # Single band acquisition
        order = np.argsort(slices)
    return order


def slice_timing(slices, dt=None, TR=None):
    '''
    Compute slice timing from slices that are excited at each shot
    and the interval between slices or the volume acquisition time.
    '''
    if dt is None and TR is None:
        raise ValueError('You must specify either `dt` or `TR`.')
    elif dt is None:
        dt = TR / len(slices)
    return excitation_order(slices) * dt


if __name__ == '__main__':
    import doctest
    doctest.testmod()
