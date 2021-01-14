#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import shlex, uuid, io
from lxml import etree
import numpy as np

# Main specification about NIML: https://afni.nimh.nih.gov/pub/dist/src/niml/NIML_base.html

encoding = 'utf-8'

# ni_type -> dtype
NI_TYPE = {
    'byte': 'u1',
    'short': 'i2',
    'int': 'i4',
    'float': 'f4',
    'double': 'f8',
    'complex': 'c8', # [('x', 'f4'), ('y', 'f4')],
    'rgb': [('r', 'u1'), ('g', 'u1'), ('b', 'u1')],
    'RGBA': [('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')],
    'String': object, # dtype(str) cannot handle String with unknown max length well (e.g., np.fromregex)
    'Line': object,
}


def read_until(fi, end_token, batch_size=1024):
    '''
    Read until encountering a specific character.

    Useful for event-driven parsing, especially for large binary file like NIML dset.
    This method is designed to work with both str and bytes.
    '''
    ss = []
    while True:
        s = fi.read(batch_size)
        p = s.find(end_token)
        if p != -1 or not s: # Detect the end token or arrive at the EOF
            ss.append(s[:p+1])
            fi.seek(p+1-len(s), 1) # Rewind pointer to the first location after end_token
            return s[:0].join(ss)
        else: # Not finding the end token, keep going
            ss.append(s)


def parse_attr(attr):
    '''
    Split a single attribute into key and value.

    Specification
    -------------
    NIML attributes are in the general form "attname=string", separated by whitespace.
    XML allows whitespace to occur around the "=" that separates the attname from the string. 
    NIML does not allow this whitespace; the next character after attname must be "=", 
    and the next character after that must be a Name character or a quote character.

    We don't need to deal with the quotes around value here, leaving it to xml.etree.TreeBuilder.
    '''
    p = attr.find('=')
    if p != -1:
        key, value = attr[:p], attr[p+1:]
    else:
        key, value = attr, ''
    return key, value


def parse_ni_type(ni_type, flatten=None):
    '''
    Parse ni_type into numpy dtype.

    The method exploits the flexibilty of np.dtype() in a recursive manner.
    Assumptions which seem to be contradicting the NIML specification:
      1. The standard type is not abbreviated, e.g., int
      2. The multiple type is indicated by "*", e.g., 4*int
      3. The compound type is separated by ",", e.g., 4*float,int,String

    Parameters
    ----------
    ni_type : str
    flatten : bool
        Limited to create non-hierarchical structured dtype, e.g.,
        interpreting "4*float,int,String" as "float,float,float,float,int,String".
    '''
    if ',' in ni_type: # Compound type
        if not flatten:
            return np.dtype([('', parse_ni_type(nt)) for nt in ni_type.split(',')])
        else: # Cannot rely on recursion (TODO: only deal with single layer case for now)
            dt = []
            for nt in ni_type.split(','):
                if '*' in nt:
                    m, t = nt.split('*')
                    dt.extend([('', parse_ni_type(t))] * int(m))
                else:
                    dt.append(('', parse_ni_type(nt)))
            return np.dtype(dt)
    else: 
        if '*' in ni_type: # Multiple type
            m, nt = ni_type.split('*')
            return int(m) * parse_ni_type(nt)
        else: # Standard type
            return np.dtype(NI_TYPE[ni_type])


def parse_data_format(attrs):
    '''
    Parse data stream format based on ni_form, ni_type, ni_dimen, etc.
    for both binary, base64, and text data.
    '''
    fmt = dict(attrs) # Copy original attributes (can be overwritten by our definition)
    if 'ni_form' in attrs:
        fmt['form'] = attrs['ni_form'].split('.')[0] # text, binary, base64, ni_group
    else: # Specification: If the ni_form attribute is not present, then ni_form=text is assumed.
        fmt['form'] = 'text'
    if fmt['form'] in ['binary', 'base64']:
        if attrs['ni_form'].endswith('lsbfirst'):
            fmt['endian'] = '<'
        elif attrs['ni_form'].endswith('msbfirst'):
            fmt['endian'] = '>'
    if 'ni_type' in attrs:
        # For text data stream, compound type like "4*float,int,String" will be flatten, 
        # because np.fromregex can only deal with flat compound type.
        # For binary data stream, compound type is allowed to be hierarchical.
        # To remove this limitation, we'll have to implement our own np.fromregex,
        # but this seems to be unnecessary with current use cases in AFNI/SUMA.
        flatten = (fmt['form']=='text')
        fmt['dtype'] = parse_ni_type(attrs['ni_type'], flatten=flatten)
        if 'endian' in fmt:
            fmt['dtype'].newbyteorder(fmt['endian'])
        if 'ni_dimen' in attrs:
            fmt['shape'] = tuple(np.int_(attrs['ni_dimen'].split(',')))
        else: # Specification: If ni_dimen is not supplied, then ni_dimen=1 is assumed.
            fmt['shape'] = (1,)
        fmt['n'] = np.prod(fmt['shape']) # Number of data elements
        fmt['length'] = fmt['dtype'].itemsize * fmt['n'] # Length of the binary data stream (in bytes)
    return fmt


def parse_data(between, fmt):
    '''
    Parse data stream from between-tag content (can be empty).
    If ni_type exists, the data is converted into a numpy array.

    TODO: Handle escape sequence (<, >, ", &) in text data.
    '''
    if fmt is None or fmt['form'] not in ['binary', 'base64']:
        # Text data
        value = between.decode(encoding)
        stripped = value.strip()
        if not stripped: # No text data
            value = None
        elif fmt.get('ni_type') == 'String' and fmt.get('ni_dimen', '1') == '1': 
            # Special treatment for a single String: keep it as a Python str
            value = stripped[1:-1] # Strip quotes, can be empty str
        elif 'ni_type' in fmt: # Text data stream
            # Assumption: data elements are separated by "\n"
            try: # Simple array
                value = np.fromstring(value, dtype=fmt['dtype'], count=fmt['n'], sep='\n')
            except ValueError: # Structured array
                # Assumption: String is surrounded by "", and fields are separated by whitespace
                pattern = '\s+'.join([('"(.+)"' if dt==object else '(.+?)') 
                    for n, (dt, pos) in fmt['dtype'].fields.items()])
                value = np.fromregex(io.StringIO(value), pattern, dtype=fmt['dtype'])
        else: # Other text data (I don't think NIML allows this though)
            pass # Text data in its original form (without stripping)
    elif fmt['form'] == 'binary':
        # Binary data
        value = np.frombuffer(between, dtype=fmt['dtype'], count=fmt['n'])
        # For multiple type like 3*int, the shape of the element is combined into the final array.
        # But for compound type like int,int,int, the element is considered a singleton.
        value = value.reshape(fmt['shape']+value.shape[1:])
    elif fmt['form'] == 'base64':
        # base64 data (base64 encoded binary which allows binary data to be encoded 
        # in a pure text format, at the cost of a 33% expansion in size)
        raise NotImplementedError
    return value


def parse_niml(fname):
    '''
    Parse NIML file into Python xml.etree.Element using incremental event-driven parsing.

    Specification
    -------------
    https://afni.nimh.nih.gov/pub/dist/src/niml/NIML_base.html
    '''
    batch_size = 1024
    tb = etree.TreeBuilder()
    generate_key = lambda: uuid.uuid4().hex[:8]
    data = {}
    with open(fname, 'rb') as fi: # TODO: Check buffering for better read_until() performance
        fi.read_until = lambda end_token: read_until(fi, end_token, batch_size)
        fmt = None
        while True:
            # Read until < to get between-tag content
            # ---------------------------------------
            if fmt is None or fmt['form'] != 'binary':
                # If fmt is None, there should be no data before the tag.
                #   1. At the beginning of the file
                #   2. After a closing tag (assuming no trailiing data in NIML)
                # If fmt is not None, there can be data whose ni_form is
                #   text, base64, ni_group, or unspecified (no ni_form attribute).
                between = fi.read_until(b'<')
                if not between.endswith(b'<'):
                    # Arrive at the end of the file
                    break
                else:
                    between = between[:-1]
            else: # Binary data in NIML require special treatment (the bytes stream may contain b'<')
                between = fi.read(fmt['length']) 
                fi.read_until(b'<') # Also read the ending b'<'
                # Specification: After the proper ni_dimen number of data values have been read, 
                # any data bytes before the closing "</" will be discarded.
            # Parse data stream from between-tag content (can be empty)
            value = parse_data(between, fmt)
            # Push data (between-tag content) into xml.etree.TreeBuilder
            if value is not None:
                key = generate_key()
                data[key] = value
                tb.data(key)

            # Read until > to get within-tag content
            # --------------------------------------
            within = fi.read_until(b'>')
            within = within[:-1].decode(encoding) # Decode because shlex.split only works with str but not bytes
            # Parse tag and attributes from within-tag content
            empty = within.endswith('/') # Specification: Empty element's header ends with "/>"
            items = shlex.split(within) if not empty else shlex.split(within[:-1]) 
            closing = items[0].startswith('/')
            tag = items[0] if not closing else items[0][1:]
            attrs = dict([parse_attr(item) for item in items[1:]])
            # Push tag and attributes (within-tag content) into xml.etree.TreeBuilder
            if closing: # Closing tag contains no attrs, and should not have trailing data (my assumption about NIML)
                tb.end(tag)
                fmt = None
            else:
                tb.start(tag, attrs)
                if empty: # Self-closing tag doesn't need further processing
                    tb.end(tag)
                else: # Starting tag contains data format, and may be followed by actual data stream
                    fmt = parse_data_format(attrs)
        et = etree.ElementTree(tb.close()) # .close returns an Element, so we need to cast to an ElementTree
        return et, data

    
if __name__ == '__main__':
    pass
