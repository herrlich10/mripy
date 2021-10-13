GLM analysis for fMRI
=====================

We can use :meth:`mripy.preprocess.glm` to analyze our fMRI data using a General Linear Model.

Block design
************


Event-related design
********************

Assume a particular shape for the HRF
-------------------------------------




Do not assume any particular shape for the HRF
----------------------------------------------

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

We assume the response starts at 0 sec after the event onset, lasting 24 sec, and 
we want to sample the response every 2 sec (which does not need to be equal to the TR).
This results in 13 samples over the 24 sec period. We can express the model as ``CSPLIN(0,24,13)``.
If we further assume the response starts from 0 at 0 sec, and has already been back to 0
at 24 sec after the event onset, the resulting model can be written as ``CSPLINzero(0,24,11)``, 
since we are only left with 11 free parameters to estimate.


.. code-block:: python

    import numpy as np
    from mripy import io, preprocess as prep

    data_dir = 'path/to/my/experiment/data'
    subject = 'sub-01'
    res_dir = f"{data_dir}/{subject}/results"
    stim_dir = f"{data_dir}/{subject}/stimuli"
    runs = [f"{k:02d}" for k in [1, 2, 3, 4]] # ['01', '02', '03', '04']

    # Average all voxels in the ROI into a single time series 
    mask = io.Mask(f"{res_dir}/V1.nii")
    for run in runs:
        x = mask.dump(f"{res_dir}/epi{run}.scale.nii") # Dump data within the ROI -> [n_voxels, n_times]
        x = x.mean(axis=0) # Average across voxels
        np.savetxt(f"{res_dir}/epi{run}.1D", x, fmt='%.6f') # Save as *.1D file (plain text)

    # Perform GLM using CSPLINzero model
    design = OrderedDict()
    design['A'] = f"{stim_dir}/EventA_onset_time.txt 'CSPLINzero(0,24,11)'"
    design['B'] = f"{stim_dir}/EventB_onset_time.txt 'CSPLINzero(0,24,11)'"
    prep.glm(in_files=[f"{res_dir}/epi{run}.1D" for run in runs], 
        out_file=f"{res_dir}/V1_resp.1D", design=design, TR=TR, 
        motion_files=[f"{res_dir}/epi{run}.volreg.param.1D" for run in runs])

    # The above method will generate the following files:
    # - stats.V1_resp_REML.1D   # This is the usual stats file with F, beta, t values for each regressor
    # - irp.A.V1_resp_REML.1D   # The impulse response of event A \
    # - irp.B.V1_resp_REML.1D   # The impulse response of event B - These two are what we want
    irf = np.zeros([2,11]) # Impulse response function: [n_events, n_times]
    for k, event in enumerate(['A', 'B']):
        irf[k,:] = io.read_txt(f"{res_dir}/irp.{event}.V1_resp_REML.1D")


Need more flexibility in doing GLM?
-----------------------------------

Calling 3dDeconvolve_ directly allows you to access more types of models, and control 
the estimation process in more details.


Need more help or details about the underlying algorithm?
---------------------------------------------------------

The ultimate source of reference is the AFNI documentation about its 3dDeconvolve_ command.

.. _3dDeconvolve: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDeconvolve.html