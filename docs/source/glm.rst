GLM analysis for fMRI
=====================

We can use :meth:`mripy.preprocess.glm` to analyze our fMRI data using a General Linear Model.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from mripy import io, preprocess as prep

    data_dir = 'path/to/my/experiment/data'
    subject = 'sub-01'
    res_dir = f"{data_dir}/{subject}/results"
    stim_dir = f"{data_dir}/{subject}/stimuli"
    runs = [f"{k:02d}" for k in [1, 2, 3, 4]] # ['01', '02', '03', '04']
    TR = 2 # s


Block design
************

Fixed duration blocks 
---------------------

This is the most basic design, especially for functional localizers.

In the following example, we have Face and House blocks, and 
each block lasts 16 s. The model can be specified as ``BLOCK(16)``.

.. code-block:: python

    # Perform GLM using "BLOCK" model
    design = OrderedDict()
    design['Face'] = f"{stim_dir}/Face_block_onset.txt 'BLOCK(16)'"
    design['House'] = f"{stim_dir}/House_block_onset.txt 'BLOCK(16)'"
    contrasts = OrderedDict()
    contrasts['F-H'] = '+Face -House'
    prep.glm(in_files=[f"{res_dir}/epi{run}.scale.nii" for run in runs], 
        out_file=f"{res_dir}/localizer.nii", design=design, contrasts=contrasts, TR=TR, 
        motion_files=[f"{res_dir}/epi{run}.volreg.param.1D" for run in runs])

    # The above method will generate the following files:
    # - stats.localizer_REML.nii   # This is the usual stats file with F, beta, t values for each regressor


Variable duration blocks
------------------------

This is useful for modelling spontaneous change in states, e.g., binocular rivalry.

In this case, we can use the "dmUBLOCK" model in AFNI. "dm" stands for "duration modulated".

.. code-block:: python

    # Prepare event timing files with both block onset and block duration
    # t['L'] and d['L'] contain the onset and duration of "Left eye dominant" event for all runs
    t = {'L': [
                [1.2, 7.5, 15.7, 20.3, ...], # run01 "Left eye dominant" event onset
                [3.9, 9.4, 19.2, 25.5, ...], # run02 "Left eye dominant" event onset
                ...,
            ],
        'R': [...] # "Right eye dominant" event onset
    }
    d = {'L': [
                [4.3, 5.1, 2.6, 7.8, ...], # run01 "Left eye dominant" event duration
                [2.9, 6.2, 2.2, 5.3, ...], # run02 "Left eye dominant" event duration
                ...,
            ],
        'R': [...] # "Right eye dominant" event duration
    }
    for event in ['L', 'R']: # Left eye dominant vs Right eye dominant
        with open(f"{stim_dir}/{event}_rivalry.txt", 'w') as fout:
            # `tt` and `dd` contain the onset and duration for all events in a run
            # `ttt` and `ddd` are the onset and duration for each event (i.e., button press）
            for tt, dd in zip(t[event], d[event]): 
                fout.write(' '.join([f"{ttt:8.3f}:{ddd:<8.3f}" for ttt, ddd in zip(tt, dd)]) + '\n')

    # Perform GLM using "dmUBLOCK" model
    design = OrderedDict()
    design['L'] = f"{stim_dir}/L_rivalry.txt 'dmUBLOCK'"
    design['R'] = f"{stim_dir}/R_rivalry.txt 'dmUBLOCK'"
    contrasts = OrderedDict()
    contrasts['L+R'] = '+0.5*L +0.5*R'
    contrasts['L-R'] = '+L -R'
    prep.glm(in_files=[f"{res_dir}/epi{run}.scale.nii" for run in runs], 
        out_file=f"{res_dir}/rivalry.nii", design=design, contrasts=contrasts, TR=TR, 
        motion_files=[f"{res_dir}/epi{run}.volreg.param.1D" for run in runs])

    # The above method will generate the following files:
    # - stats.rivalry_REML.nii   # This is the usual stats file with F, beta, t values for each regressor


Event-related design
********************

Assume a particular shape for the HRF
-------------------------------------

We can assume the evoked fMRI response takes a particular shape of the HRF (Haemodynamic Response Function), 
with only one free parameter that we may adjust to fit our data, which is the amplitude of the peak response.
By convention, we call this free parameter :math:`\beta`.

The particular shape of HRF has many variants: GAM, SPM1, SPM2, SPM3, etc.

In the following example, we have two events: A and B.

.. code-block:: python

    # Perform GLM using "GAM" model
    design = OrderedDict()
    design['A'] = f"{stim_dir}/EventA_onset_time.txt 'GAM'"
    design['B'] = f"{stim_dir}/EventB_onset_time.txt 'GAM'"
    prep.glm(in_files=[f"{res_dir}/epi{run}.scale.nii" for run in runs], 
        out_file=f"{res_dir}/ER.nii", design=design, TR=TR, 
        motion_files=[f"{res_dir}/epi{run}.volreg.param.1D" for run in runs])

    # The above method will generate the following files:
    # - stats.ER_REML.nii   # This is the usual stats file with F, beta, t values for each regressor


No assumption about the shape of HRF 
------------------------------------

This is referred to as TENT or CSPLIN model in AFNI, and FIR model in SPM.

TENT (n parameter tent function) models the evoked fMRI response
by each event as a piecewise linear function.

CSPLIN (n parameter cubic spline function, n>=4) is a drop-in upgrade of TENT 
to a differentiable (i.e., smooth) set of functions. And this is our default choice.

Since we now have more parameters (i.e., beta values) to estimate for each brain location, 
we need more data to get a result with reasonably low variance. This can be achieved 
either by acquiring more time points (averaging over time), or by pooling all voxels in 
your ROI before running the GLM (averaging over space).

In the following example, we have two events: A and B. We first average all voxels 
in our ROI into a single time series, and then perform deconvolution to estimate 
the brain respones to event A and event B.

We assume the response starts at 0 s after the event onset, lasting 24 s, and 
we want to sample the response every 2 s (which does not need to be equal to the TR).
This results in 13 samples over the 24 s period. We can express the model as ``CSPLIN(0,24,13)``.
If we further assume the response starts from 0 at 0 s, and has already been settled to 0
at 24 s after the event onset, the resulting model can be written as ``CSPLINzero(0,24,13)``, 
although we now only have 11 free parameters to estimate.

.. code-block:: python

    # Average all voxels in the ROI into a single time series 
    mask = io.Mask(f"{res_dir}/V1.nii")
    for run in runs:
        data = mask.dump(f"{res_dir}/epi{run}.scale.nii") # Dump data within the ROI -> [n_voxels, n_times]
        data = data.mean(axis=0, keepdims=True) # Average across voxels
        # Note that the data must be 2D: [n_time_series, n_times], while n_time_series can be 1
        # This allow you to save data from multiple ROIs or multiple conditions in a single *.1D file
        np.savetxt(f"{res_dir}/epi{run}.1D", data, fmt='%.6f') # Save as *.1D file (plain text)

    # Perform GLM using "CSPLINzero" model
    design = OrderedDict()
    design['A'] = f"{stim_dir}/EventA_onset_time.txt 'CSPLINzero(0,24,13)'"
    design['B'] = f"{stim_dir}/EventB_onset_time.txt 'CSPLINzero(0,24,13)'"
    prep.glm(in_files=[f"{res_dir}/epi{run}.1D" for run in runs], 
        out_file=f"{res_dir}/V1_resp.1D", design=design, TR=TR, 
        motion_files=[f"{res_dir}/epi{run}.volreg.param.1D" for run in runs])

    # The above method will generate the following files:
    # - stats.V1_resp_REML.1D   # This is the usual stats file with F, beta, t values for each regressor
    # - irp.A.V1_resp_REML.1D   # The impulse response of event A \
    # - irp.B.V1_resp_REML.1D   # The impulse response of event B - These two are our estimated impulse responses
    irf = np.zeros([2,13]) # Impulse response function: [n_events, n_times]
    for k, event in enumerate(['A', 'B']):
        # Remember that the first (0th) and the last (12th) element are zero by construction
        irf[k,1:-1] = io.read_txt(f"{res_dir}/irp.{event}.V1_resp_REML.1D")
    
    # Plot the estimated evoked fMRI responses for event A and B
    t = np.linspace(0, 24, 13) # Time after event onset in seconds
    plt.plot(t, irf.T)


Need more flexibility in doing GLM?
***********************************

Calling 3dDeconvolve_ directly allows you to access more types of models, and control 
the behavior of the estimation process in more details.

If you need more help or details about the underlying algorithm, the ultimate source of 
reference is the AFNI documentation about its 3dDeconvolve_ command.

.. _3dDeconvolve: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html