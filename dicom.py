#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import glob, itertools
import gzip
from datetime import datetime
from collections import OrderedDict
from os import path
import numpy as np
# import pandas as pd


encoding = 'utf-8'

vr_parsers = {
    'CS': lambda x: str.strip(x.decode(encoding)), # Code String
    'SH': lambda x: str.strip(x.decode(encoding)), # Short String
    'LO': lambda x: str.strip(x.decode(encoding)), # Long String
    'ST': lambda x: str.strip(x.decode(encoding)), # Short Text
    'LT': lambda x: str.strip(x.decode(encoding)), # Long Text
    'UT': lambda x: str.strip(x.decode(encoding)), # Unlimited Text

    'DA': lambda x: datetime.strptime(x.decode(encoding), '%Y%m%d').date(), # Date
    'TM': lambda x: datetime.strptime(x.decode(encoding).strip(), '%H%M%S.%f').time(), # Time

    # 'IS': lambda x: int(x.decode(encoding)), # Integer String
    # 'DS': lambda x: float(x.decode(encoding)), # Decimal String
    'IS': lambda x: np.int_(x.decode(encoding).split('\\')).squeeze()[()], # Decimal String
    'DS': lambda x: np.float_(x.decode(encoding).split('\\')).squeeze()[()], # Decimal String

    # 'SS': lambda x: int.from_bytes(x, byteorder='little', signed=True), # Signed Short
    'SS': lambda x: np.frombuffer(x, dtype=np.int16).squeeze()[()], # Signed Short
    'US': lambda x: np.frombuffer(x, dtype=np.uint16).squeeze()[()], # Unsigned Short
    'SL': lambda x: np.frombuffer(x, dtype=np.int32).squeeze()[()], # Signed Long
    'UL': lambda x: np.frombuffer(x, dtype=np.uint32).squeeze()[()], # Unsigned Long
    'FL': lambda x: np.frombuffer(x, dtype=np.float32).squeeze()[()], # Floating Point Single
    'FD': lambda x: np.frombuffer(x, dtype=np.float64).squeeze()[()], # Floating Point Double
}

# http://dicom.nema.org/medical/dicom/current/output/chtml/part06/chapter_6.html
tag_parsers = {
    '0008,0022': ('AcquisitionDate', vr_parsers['DA']),
    '0008,0032': ('AcquisitionTime', vr_parsers['TM']),
    '0018,0024': ('SequenceName', vr_parsers['SH']),
    '0018,0050': ('SliceThickness', vr_parsers['DS']),
    '0018,0080': ('RepetitionTime', vr_parsers['DS']),
    '0018,0081': ('EchoTime', vr_parsers['DS']),
    '0018,0082': ('InversionTime', vr_parsers['DS']),
    '0018,0084': ('ImagingFrequency', vr_parsers['DS']),
    '0018,0087': ('MagneticFieldStrength', vr_parsers['DS']),
    '0018,0095': ('PixelBandwidth', vr_parsers['DS']),
    '0018,1030': ('ProtocolName', vr_parsers['LO']),
    '0018,1314': ('FlipAngle', vr_parsers['DS']),
    '0018,1316': ('SAR', vr_parsers['DS']),
    '0019,100A': ('n_slices', vr_parsers['US']),
    '0020,0010': ('StudyID', lambda x : int(vr_parsers['SH'](x))),
    '0020,0011': ('SeriesNumber', vr_parsers['IS']),
    '0020,0012': ('AcquisitionNumber', vr_parsers['IS']),
    '0020,0013': ('InstanceNumber', vr_parsers['IS']),
    '0028,0030': ('PixelSpacing', vr_parsers['DS']),
}

custom_parsers = {
    'resolution': lambda header: np.r_[header['PixelSpacing'], header['SliceThickness']],
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
        if item_tag == 'FFFE,E000': # Item
            item_length = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            if item_length == 4294967295: # 0xFFFFFFFF: Undefined Length
                # Bruteforce scan the byte stream until we hit "FFFE,E00D"
                while True:
                    if fi.read(2) == b'\xfe\xff':
                        if fi.read(2) == b'\x0d\xe0':
                            if fi.read(4) == b'\x00\x00\x00\x00':
                                break
            else:
                fi.seek(item_length, 1)
        elif item_tag == 'FFFE,E00D': # Item Delimitation Item
            item_length = int.from_bytes(fi.read(4), byteorder='little', signed=False)
            assert(item_length == 0)
        elif item_tag == 'FFFE,E0DD': # Sequence Delimitation Item
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
    header = {}
    n_tags_seen = 0
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
            element['group'] = '{0:04X}'.format(int.from_bytes(group, byteorder='little', signed=False)) # element['group'] = '{0:04X}'.format(np.frombuffer(group, dtype=np.uint16)[0])
            element['element'] = '{0:04X}'.format(int.from_bytes(fi.read(2), byteorder='little', signed=False)) # element['element'] = '{0:04X}'.format(np.frombuffer(fi.read(2), dtype=np.uint16)[0])
            tag = ','.join((element['group'], element['element']))
            # print(tag, end='')
            element['VR'] = fi.read(2).decode(encoding)
            # print(':', element['VR'])
            if element['VR'] in ["OB", "OW", "OF", "SQ", "UT", "UN"]:
                fi.seek(2, 1)
                element['length'] = int.from_bytes(fi.read(4), byteorder='little', signed=False) # element['length'] = np.frombuffer(fi.read(4), dtype=np.uint32)[0]
            else:
                element['length'] = int.from_bytes(fi.read(2), byteorder='little', signed=False) # element['length'] = np.frombuffer(fi.read(2), dtype=np.uint16)[0]
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
    header['filename'] = fname
    return header


def sort_dicom_series(folder):
    '''
    Parameters
    ----------
    folder : string
        Path to the folder containing all the *.IMA files.

    Returns
    -------
    studies : list of dicts
        [{'0001': [file0, file1, ...], '0002': [files], ...}, {study1}, ...]
    '''
    # files = sorted(glob.glob(path.join(folder, '*.IMA')))
    exts = ['*.IMA', '*.dcm', '*.dcm.gz']
    files = sorted(itertools.chain.from_iterable(glob.glob(path.join(folder, pattern)) for pattern in exts))
    headers = [parse_dicom_header(f, search_for_tags={'0020,0010', '0020,0011', '0020,0013'}) for f in files]
    studies = []
    for study_id in np.unique([header['StudyID'] for header in headers]):
        study = OrderedDict()
        for series_id in np.unique([header['SeriesNumber'] for header in headers if header['StudyID'] == study_id]):
            series = sorted([(path.split(header['filename'])[1], header['InstanceNumber']) for header in headers \
                if header['StudyID']==study_id and header['SeriesNumber']==series_id], key=lambda x: x[1])
            study['{0:04d}'.format(series_id)] = [x[0] for x in series]
        studies.append(study)
    return studies


if __name__ == '__main__':
    print(parse_dicom_header('20180626_S18_EP2DBR_S07.MR.S18_APPROVED.0010.0001.2018.06.26.12.47.59.31250.120791562.IMA'))
    print(parse_dicom_header('20180918_S18_FACEID_VIS_S01.MR.S18_APPROVED.0007.0001.2018.09.18.15.52.59.828125.140479.IMA'))
    pass