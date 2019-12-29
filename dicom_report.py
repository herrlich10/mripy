#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect, re, glob
from os import path
from collections import OrderedDict
import pandas as pd
from . import dicom


preset4sequence = {
    'epfid2d1': 'ep2d',
    'ep3Dfid3d1': 'ep3d',
    'tfl3d1rr_ns': 'mprage',
    'tfl3d1_ns': 'mp2rage',
    'fl2d1': 'gre',
    'tfi2d1': 'ssfp2d',
}


def report_parameters(dicom_folder, preset=None, return_preset=False, return_info=False):
    # For multi-params sequence (e.g., MP2RAGE)
    if any([w in dicom_folder for w in ['*', '?']]):
        infos = [dicom.parse_series_info(folder) for folder in sorted(glob.glob(dicom_folder))]
    elif all([path.isdir(f) for f in sorted(glob.glob(path.join(dicom_folder, '*')))]):
        infos = [dicom.parse_series_info(folder) for folder in sorted(glob.glob(path.join(dicom_folder, '*')))]
    else: # For ordinary sequence
        infos = [dicom.parse_series_info(dicom_folder)]
    info = infos[0]
    oneline = lambda s: ''.join(inspect.cleandoc(s).split('\n'))
    # Try to automatically determine preset based on info['SequenceName']
    if preset is None:
        for pattern in preset4sequence:
            if re.search(pattern, info['SequenceName']):
                preset = preset4sequence[pattern]
                break
        else:
            raise ValueError(f"Please specify a preset for parameters reporting: {', '.join(preset4sequence.values())}")
    # Phrase
    ND = f"{info['MRAcquisitionType']}"
    TR = f"TR = {info['RepetitionTime']:.0f} ms"
    TE = f"TE = {info['EchoTime']:g} ms"
    FA = f"{info['FlipAngle']:.0f}° flip angle"
    voxel_size = f"voxel size {info['resolution'][0]:.3g} × {info['resolution'][1]:.3g} × {info['resolution'][2]:.3g} mm"
    FOV = f"field of view {info['FOV'][0]:g} × {info['FOV'][1]:g} mm"
    n_slices = f"{info['n_slices']} {info['orientation']} slices"
    bandwidth = f"receiver bandwidth {info['PixelBandwidth']:.0f} Hz/pix"
    PF = (f", {info['PhasePartialFourier']} phase partial Fourier" if info['PhasePartialFourier'] is not None else '') + \
         (f", {info['SlicePartialFourier']} slice partial Fourier" if info['SlicePartialFourier'] is not None else '')
    GRAPPA = (f", GRAPPA acceleration factor {info['GRAPPA']}" if info['GRAPPA'] else '')
    dist_corr = f"{dict(ND='No', DIS2D='2D', DIS3D='3D')[info['distortion_correction']]} distortion correction applied."
    # Sentence
    if preset.lower() in ['ep2d', 'ep3d']:
        effective_TR  = f"TR = {info['TR']*1000/1.004:.0f} ms" # 1.004 is corrected for the fast clock of IBP Siemens 7T
        MB = (f", MultiBand acceleration factor {info['MultiBand']}" if info['MultiBand'] else '')
        report = oneline(f'''T2*-weighted {ND} gradient-echo EPI sequence ({effective_TR}, {TE}, {FA}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}{MB}).''')
    elif preset.lower() in ['mprage']:
        if 'InversionTime' in info:
            TI = f"TI = {info['InversionTime']:.0f} ms"
            report = oneline(f'''T1-weighted {preset.upper()} sequence ({TR}, {TE}, {TI}, {FA}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}).''')
        else:
            report = oneline(f'''PD-weighted {preset.upper()} sequence ({TR}, {TE}, {FA}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}).''')
        report += '\n' + dist_corr
    elif preset.lower() in ['mp2rage']:
        labels = {info_['SeriesDescription'][len(info_['ProtocolName'])+1:]: k for k, info_ in enumerate(infos)}
        TI1 = f"TI1 = {infos[labels['INV1_ND']]['InversionTime']:.0f} ms"
        FA1 = f"{infos[labels['INV1_ND']]['FlipAngle']:.0f}° flip angle"
        TI2 = f"TI2 = {infos[labels['INV2_ND']]['InversionTime']:.0f} ms"
        FA2 = f"{infos[labels['INV2_ND']]['FlipAngle']:.0f}° flip angle"
        report = oneline(f'''T1-weighted {preset.upper()} sequence ({TR}, {TE}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}), yielding two inversion contrasts ({TI1}, {FA1}, {TI2}, {FA2}) which were combined into a single T1-weighted image.''')
    elif preset.lower() in ['gre']:
        report = oneline(f'''T2*-weighted {ND} gradient-echo sequence ({TR}, {TE}, {FA}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}).''')
    elif preset.lower() in ['ssfp2d']:
        report = oneline(f'''{ND} SSFP sequence ({TR}, {TE}, {FA}, {voxel_size}, {FOV}, {n_slices}, {bandwidth}{PF}{GRAPPA}).''')
    res = (report,) + ((preset,) if return_preset else ()) + ((info,) if return_info else ())
    return res if len(res) > 1 else res[0]


def inspect_mp2rage(data_dir, subdir_pattern='T1??'):
    sess_dirs = sorted([f for f in glob.glob(f"{data_dir}/*") if path.isdir(f)])
    df = []
    for sess_dir in sess_dirs:
        T1_folders = glob.glob(f"{sess_dir}/{subdir_pattern}")
        info = dicom.parse_series_info(T1_folders[0])
        df.append(OrderedDict(session=path.basename(sess_dir), resolution='x'.join(f'{d:g}' for d in info['resolution']), 
            ref_amp=info['ReferenceAmplitude'], coil=info['TransmittingCoil'], n_images=len(T1_folders)))
    return pd.DataFrame(df)


if __name__ == '__main__':
    pass