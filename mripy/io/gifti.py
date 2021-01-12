#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from lxml import etree
import base64, zlib


def read_gii_dset(fname):
    '''
    Read GIFTI surface data in XML format.
    '''
    et = etree.parse(fname)
    attrs = et.find('DataArray').attrib
    assert(attrs['Encoding']=='GZipBase64Binary')
    assert(attrs['Endian']=='LittleEndian')
    assert(attrs['DataType']=='NIFTI_TYPE_FLOAT32')
    assert(attrs['ArrayIndexingOrder']=='RowMajorOrder')
    assert(int(attrs['Dimensionality'])==1)
    data = et.find('DataArray/Data').text
    data = base64.b64decode(data)
    data = zlib.decompress(data)
    data = np.frombuffer(data, dtype='<f4')
    assert(int(attrs['Dim0'])==len(data))
    return data

    
if __name__ == '__main__':
    pass
