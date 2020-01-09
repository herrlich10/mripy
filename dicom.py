#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import glob, re, itertools, inspect
import gzip
from datetime import datetime
from collections import OrderedDict
from os import path
import numpy as np
import six


encoding = 'utf-8'

vr_parsers = {
    'CS': lambda x: str.strip(x.decode(encoding)), # Code String
    'SH': lambda x: str.strip(x.decode(encoding)), # Short String
    'LO': lambda x: str.strip(x.decode(encoding)), # Long String
    'ST': lambda x: str.strip(x.decode(encoding)), # Short Text
    'LT': lambda x: str.strip(x.decode(encoding)), # Long Text
    'UT': lambda x: str.strip(x.decode(encoding)), # Unlimited Text

    'PN': lambda x: str.strip(x.decode(encoding)), # Person Name
    'AS': lambda x: str.strip(x.decode(encoding)), # Age String ??? e.g., 039Y

    'DA': lambda x: datetime.strptime(x.decode(encoding), '%Y%m%d').date(), # Date
    'TM': lambda x: datetime.strptime(x.decode(encoding).strip(), '%H%M%S.%f').time(), # Time

    'IS': lambda x: np.int_(x.decode(encoding).split('\\')).squeeze()[()], # Integer String
    'DS': lambda x: np.float_(x.decode(encoding).split('\\')).squeeze()[()], # Decimal String

    'SS': lambda x: np.frombuffer(x, dtype=np.int16).squeeze()[()], # Signed Short
    'US': lambda x: np.frombuffer(x, dtype=np.uint16).squeeze()[()], # Unsigned Short
    'SL': lambda x: np.frombuffer(x, dtype=np.int32).squeeze()[()], # Signed Long
    'UL': lambda x: np.frombuffer(x, dtype=np.uint32).squeeze()[()], # Unsigned Long
    'FL': lambda x: np.frombuffer(x, dtype=np.float32).squeeze()[()], # Floating Point Single
    'FD': lambda x: np.frombuffer(x, dtype=np.float64).squeeze()[()], # Floating Point Double
}

# `gdcmdump --csa input.dcm`
# `dcmstack` in Python
# https://nipy.org/nibabel/dicom/siemens_csa.html
# https://neurostars.org/t/determining-bids-phaseencodingdirection-from-dicom/612 # For PhaseEncodingDirection
def parse_Siemens_CSA(b):
    return {}

def parse_Siemens_CSA2(b):
    def parse_value(pattern, b, default=None):
        match = re.search(pattern, b)        
        return match.group(1) if match else default
    CSA2 = {
        'ReferenceAmplitude': float(parse_value(rb'sTXSPEC\.asNucleusInfo\[0\]\.flReferenceAmplitude\s+=\s+(\S+)\s', b, default=np.nan)),
        'PhasePartialFourier': re.search(rb'sKSpace\.ucPhasePartialFourier\s+=\s+(\S+)\s', b).group(1).decode(encoding),
        'SlicePartialFourier': re.search(rb'sKSpace\.ucSlicePartialFourier\s+=\s+(\S+)\s', b).group(1).decode(encoding),
        'RefLinesPE': int(parse_value(rb'sPat\.lRefLinesPE\s+=\s+(\S+)\s', b, default=0)),
        'PATMode': re.search(rb'sPat.ucPATMode\s+=\s+(\S+)\s', b).group(1).decode(encoding),
        'RefScanMode': re.search(rb'sPat\.ucRefScanMode\s+=\s+(\S+)\s', b).group(1).decode(encoding),
        'TotalScanTimeSec': float(re.search(rb'lTotalScanTimeSec\s+=\s+(\S+)\s', b).group(1)),
        # 'SlicePosition': np.float_(re.findall(rb'sSliceArray\.asSlice\[\d+\]\.sPosition\.\w{4}\s+=\s(\S+)\s', b)).reshape(-1,3), # Unfortunately, no temporal order
    }
    return CSA2

Siemens = {
    'PartialFourier': {
        '0x10': None,
        '0x8': '7/8',
        '0x4': '6/8',
        '0x2': '5/8',
    },
    'PATMode': {
        '0x1': None, #??
        '0x2': 'GRAPPA',
    },
    'RefScanMode': {
        '0x1': None, #??
        '0x4': 'GRE',
    },
}


# http://dicom.nema.org/medical/dicom/current/output/chtml/part06/chapter_6.html
# https://dicom.innolitics.com/ciods
# Note that the hex codes must be in upper case. 
tag_parsers = {
    '0010,0010': ('PatientName', vr_parsers['PN']),
    '0010,0030': ('PatientBirthDate', vr_parsers['DA']),
    '0010,0040': ('PatientSex', vr_parsers['CS']),
    '0010,1010': ('PatientAge', vr_parsers['AS']),
    '0010,1030': ('PatientWeight', vr_parsers['DS']),
    '0002,0013': ('ImplementationVersionName', vr_parsers['SH']),
    '0008,0022': ('AcquisitionDate', vr_parsers['DA']),
    '0008,0032': ('AcquisitionTime', vr_parsers['TM']),
    '0008,103E': ('SeriesDescription', vr_parsers['LO']),
    '0018,0020': ('ScanningSequence', vr_parsers['CS']), # SE Spin Echo, IR Inversion Recovery, GR Gradient Recalled, EP Echo Planar, RM Research Mode
    '0018,0021': ('SequenceVariant', vr_parsers['CS']), # SK segmented k-space, MTC magnetization transfer contrast, SS steady state, TRSS time reversed steady state, SP spoiled, MP MAG prepared, OSP oversampling phase, NONE no sequence variant
    '0018,0023': ('MRAcquisitionType', vr_parsers['CS']), # 2D frequency x phase, 3D frequency x phase x phase
    '0018,0024': ('SequenceName', vr_parsers['SH']), # User defined name for the combination of Scanning Sequence (0018,0020) and Sequence Variant (0018,0021)
    '0018,0050': ('SliceThickness', vr_parsers['DS']),
    '0018,0080': ('RepetitionTime', vr_parsers['DS']),
    '0018,0081': ('EchoTime', vr_parsers['DS']),
    '0018,0082': ('InversionTime', vr_parsers['DS']),
    '0018,0084': ('ImagingFrequency', vr_parsers['DS']),
    # '0018,0086': ('EchoNumber', vr_parsers['IS']), # The echo number used in generating this image. In the case of segmented k-space, it is the effective Echo Number. (However, could be 1 for MB GE EPI)
    '0018,0087': ('MagneticFieldStrength', vr_parsers['DS']),
    '0018,0088': ('SpacingBetweenSlices', vr_parsers['DS']), # Thickness plus gap
    # '0018,0091': ('EchoTrainLength', vr_parsers['IS']), # Number of lines in k-space acquired per excitation per image. (However, could be 1 for MB GE EPI)
    '0018,0095': ('PixelBandwidth', vr_parsers['DS']),
    '0018,1020': ('SoftwareVersion', vr_parsers['LO']),
    '0018,1030': ('ProtocolName', vr_parsers['LO']),
    '0018,1251': ('TransmittingCoil', vr_parsers['SH']),
    '0018,1310': ('AcquisitionMatrix', lambda x : np.array([a for a in vr_parsers['US'](x) if a != 0])), # Dimensions of the acquired frequency/phase data before reconstruction: frequency rows\frequency columns\phase rows\phase columns
    '0018,1312': ('PhaseEncodingDirection', vr_parsers['CS']),
    '0018,1314': ('FlipAngle', vr_parsers['DS']),
    '0018,1316': ('SAR', vr_parsers['DS']),
    '0019,100A': ('n_slices', vr_parsers['US']),
    # https://wiki.humanconnectome.org/download/attachments/40534057/CMRR_MB_Slice_Order.pdf
    # https://wiki.humanconnectome.org/download/attachments/40534057/CMRR_MB_Slice_Order.pdf?version=2&modificationDate=1386950067494&api=v2
    # About Siemens slice timing: "For odd, the most inferior slice is acquired first. For even, the most inferior slice is acquired second."
    # However, "slice excitation always starts with slice0 in CMRR multiband C2P sequences" (i.e., different from Siemens default behavior).
    # "Slice cross-talk effects are minimized with interleaved slice series, hence this is the selected option in the default protocol."
    # "The most convenient and practical way to determine slice timing is by referencing the timing information for each slice under "MosaicRefAcqTimes" 
    # ([ms], ordered corresponding to the slice numbering) in vendor private field of the DICOM header. 
    # This slice-by-slice timing information is generic (transparent to the multiband factor) for any protocol."
    '0019,1029': ('MosaicRefAcqTimes', vr_parsers['FD']),
    '0020,0010': ('StudyID', lambda x : int(vr_parsers['SH'](x))),
    '0020,0011': ('SeriesNumber', vr_parsers['IS']),
    '0020,0012': ('AcquisitionNumber', vr_parsers['IS']),
    '0020,0013': ('InstanceNumber', vr_parsers['IS']),
    '0020,4000': ('ImageComments', vr_parsers['LT']),
    '0028,0030': ('PixelSpacing', vr_parsers['DS']),
    '0029,1010': ('CSA', parse_Siemens_CSA), # Siemens private element
    '0029,1020': ('CSA2', parse_Siemens_CSA2), # Siemens private element
}

Siemens_parsers = {
    '0051,100E': ('slice_orientation', vr_parsers['SH']),
    '0051,1011': ('acceleration_factor', vr_parsers['SH']),
    '0051,1016': ('reconstruction', vr_parsers['SH']),
}

custom_parsers = {
    'resolution': lambda header: np.r_[header['PixelSpacing'], header['SliceThickness']],
    'FOV': lambda header: header['resolution'][:2] * header['AcquisitionMatrix'], # Bug: Is "PixelSpacing" also ordered as frequency/phase like "AcquisitionMatrix" does??
    'orientation': lambda header: ('oblique-' if len(header['slice_orientation'])>3 else '') + {'Sag': 'sagittal', 'Cor': 'coronal', 'Tra': 'transversal'}[header['slice_orientation'][:3]],
    'GRAPPA': lambda header: int(header['acceleration_factor'].split()[0][1:]) if 'acceleration_factor' in header and header['acceleration_factor'].startswith('p') else 0, # TODO: The text can be "p2" or "p2 s4". What does "s4" mean?
    'PhasePartialFourier': lambda header: Siemens['PartialFourier'][header['CSA2']['PhasePartialFourier']],
    'SlicePartialFourier': lambda header: Siemens['PartialFourier'][header['CSA2']['SlicePartialFourier']],
    'MultiBand': lambda header: int(re.search('MB(\d+)', header['ImageComments']).group(1)) if 'MB' in header['ImageComments'] else None, # https://github.com/CMRR-C2P/MB/issues/223
    'distortion_correction': lambda header: re.search('(ND|DIS2D|DIS3D)', header['reconstruction']).group(1),
    'ReferenceAmplitude': lambda header: header['CSA2']['ReferenceAmplitude'],
    'PATMode': lambda header: Siemens['PATMode'][header['CSA2']['PATMode']],
    'RefLinesPE': lambda header: header['CSA2']['RefLinesPE'],
    'RefScanMode': lambda header: Siemens['RefScanMode'][header['CSA2']['RefScanMode']],
    'TotalScanTime': lambda header: header['CSA2']['TotalScanTimeSec'],
    'timestamp': lambda header: datetime.combine(header['AcquisitionDate'], header['AcquisitionTime']).timestamp(), 
}


def parse_SQ_data_element(fi):
    '''
    We only support Data Element with Explicit VR at present (Table 7.1-2).
    We don't support nested Item at present (2018-09-26).

    References
    ----------
    [1] http://dicom.nema.org/Dicom/2013/output/chtml/part05/chapter_7.html
    '''
    while True:
        # Parse an item
        item_tag = '{0:04X},{1:04X}'.format(*np.frombuffer(fi.read(4), dtype=np.uint16, count=2))
        if item_tag == 'FFFE,E000': # Item (Mark the start of an item)
            item_length = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            if item_length == 4294967295: # 0xFFFFFFFF: Undefined Length
                # Bruteforce scan the byte stream until we hit "FFFE,E00D"
                # Bug: However, this may fail if the payload also contains item with undefined length!
                while True:
                    # Until we hit "FFFE,E00D"
                    if fi.read(2) == b'\xfe\xff':
                        if fi.read(2) == b'\x0d\xe0':
                            if fi.read(4) == b'\x00\x00\x00\x00':
                                break
            else:
                fi.seek(item_length, 1)
        elif item_tag == 'FFFE,E00D': # Item Delimitation Item (Mark the end of an item with undefined length)
            item_length = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            assert(item_length == 0)
        elif item_tag == 'FFFE,E0DD': # Sequence Delimitation Item (Mark the end of an SQ with undefined length)
            item_length = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            assert(item_length == 0)
            break


def parse_dicom_header(fname, search_for_tags=None, **kwargs):
    '''
    Parameters
    ----------
    fname : str
    search_for_tags : set
        Search for specific dicom tags, and stop file scanning early if all tags of interest are seen.
        e.g., search_for_tags={'0020,0011', '0020,0013'} will search for SeriesNumber and InstanceNumber.
        This will save you some time, esp. when the remote file is accessed via slow data link.
    **kwargs : 
        This is only for backward compatibility.

    Notes
    -----
    "Implicit and Explicit VR Data Elements shall not coexist in a Data Set and Data Sets nested within it 
    (see Section 7.5). Whether a Data Set uses Explicit or Implicit VR, among other characteristics, 
    is determined by the negotiated Transfer Syntax (see Section 10 and Annex A)." [1]

    References
    ----------
    [1] http://dicom.nema.org/Dicom/2013/output/chtml/part05/chapter_7.html
    [2] https://stackoverflow.com/questions/119684/parse-dicom-files-in-native-python
    '''
    elements = []
    header = OrderedDict()
    n_tags_seen = 0
    tag_parsers.update(Siemens_parsers)
    if fname.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(fname, 'rb') as fi:
        # The preamble
        fi.seek(128) # The first 128 bytes are 0x00
        assert(fi.read(4).decode(encoding) == 'DICM') # The next 4 bytes are "DICM"
        # Data Elements
        while True:
            element = {}
            group = fi.read(2)
            if not group:
                break
            element['group'] = '{0:04X}'.format(int.from_bytes(group, byteorder='little', signed=False))
            element['element'] = '{0:04X}'.format(int.from_bytes(fi.read(2), byteorder='little', signed=False))
            tag = ','.join((element['group'], element['element']))
            # print(tag, end='')
            element['VR'] = fi.read(2).decode(encoding)
            # print(':', element['VR'])
            if element['VR'] in ['OB', 'OW', 'OF', 'SQ', 'UT', 'UN']:
                fi.seek(2, 1)
                element['length'] = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            else:
                element['length'] = int.from_bytes(fi.read(2), byteorder='little', signed=False)
            if element['length'] == 4294967295: # 0xFFFFFFFF: Undefined Length
                if element['VR'] == 'SQ':
                    parse_SQ_data_element(fi)
                else:
                    # print(element['VR'])
                    raise NotImplementedError('** Undefined Length')
            elif tag in tag_parsers:
                header[tag_parsers[tag][0]] = tag_parsers[tag][1](fi.read(element['length']))
            else:
                fi.seek(element['length'], 1)
            elements.append(element)
            if search_for_tags is not None and tag in search_for_tags:
                n_tags_seen += 1
                if n_tags_seen == len(search_for_tags):
                    break
    # elements = pd.DataFrame(elements, columns=['group', 'element', 'VR', 'length'])
    # Custom header fields
    for field, parser in custom_parsers.items():
        try:
            header[field] = parser(header)
        except KeyError:
            pass
    header['filename'] = path.realpath(fname)
    return header


def sort_dicom_series(folder):
    '''
    Parameters
    ----------
    folder : string
        Path to the folder containing all the dicom files.

    Returns
    -------
    studies : list of dicts
        [{'0001': [file0, file1, ...], '0002': [files], ...}, {study1}, ...]
    '''
    exts = ['.IMA', '.dcm', '.dcm.gz']
    files = sorted(itertools.chain.from_iterable(glob.glob(path.join(folder, '*'+ext)) for ext in exts))
    headers = [parse_dicom_header(f, search_for_tags={'0020,0010', '0020,0011', '0020,0013'}) for f in files]
    studies = []
    for study_id in np.unique([header['StudyID'] for header in headers]):
        study = OrderedDict()
        for series_id in np.unique([header['SeriesNumber'] for header in headers if header['StudyID'] == study_id]):
            series = sorted([(path.basename(header['filename']), header['InstanceNumber']) for header in headers \
                if header['StudyID']==study_id and header['SeriesNumber']==series_id], key=lambda x: x[1])
            study['{0:04d}'.format(series_id)] = [x[0] for x in series]
        studies.append(study)
    return studies


def parse_series_info(dicom_files, dicom_ext=None, parser=None, return_headers=False):
    '''
    Parameters
    ----------
    dicom_files : list or str
        A list of dicom files (e.g., as provided by sort_dicom_series), or
        a folder that contains a single series (e.g., "../raw_fmri/func01"), or 
        a single dicom file.
    '''
    if dicom_ext is None:
        dicom_ext = '.IMA'
    if parser is None:
        parser = parse_dicom_header
    if isinstance(dicom_files, six.string_types): # A single file or a folder
        if path.isdir(dicom_files):
            # Assume there is only one series in the folder
            dicom_files = sorted(glob.glob(path.join(dicom_files, '*'+dicom_ext)))
        else:
            dicom_files = [dicom_files]
    # Parse dicom headers
    headers = [parser(f) for f in dicom_files]
    info = OrderedDict(headers[0])
    assert(np.all(np.array([header['StudyID'] for header in headers])==info['StudyID']))
    assert(np.all(np.array([header['SeriesNumber'] for header in headers])==info['SeriesNumber']))
    # n_volumes, n_slices, TR
    info['n_volumes'] = headers[-1]['AcquisitionNumber'] - headers[0]['AcquisitionNumber'] + 1
    if 'n_slices' not in info:
        info['n_slices'] = int(len(headers) / info['n_volumes'])
    info['first'] = headers[0]['timestamp']
    info['last'] = headers[-1]['timestamp']
    info['TR'] = (info['last']-info['first'])/(info['n_volumes']-1) if info['n_volumes'] > 1 else None
    # Slice timing
    # if shift_time == 'CMRR':
    #     shift_time = 0
    #     if info['TR'] is not None and 'n_slices' in info and np.mod(info['n_slices'], 2)==0:
    #         slice_order = pares_slice_order(files)[0]
    #         if slice_order == 'interleaved':
    #             shift_time = -info['TR']/2
    # elif shift_time is None:
    #     shift_time = 0
    # info['first'] += shift_time
    # info['last'] += shift_time
    info['start'] = info['first']
    info['stop'] = (info['last'] + info['TR']) if info['TR'] is not None else info['last']
    info['time'] = np.array([header['timestamp'] for header in headers]) - info['timestamp']
    if return_headers:
        info['headers'] = headers
    return info



if __name__ == '__main__':
    print(parse_dicom_header('20180626_S18_EP2DBR_S07.MR.S18_APPROVED.0010.0001.2018.06.26.12.47.59.31250.120791562.IMA'))
    print(parse_dicom_header('20180918_S18_FACEID_VIS_S01.MR.S18_APPROVED.0007.0001.2018.09.18.15.52.59.828125.140479.IMA'))
    pass