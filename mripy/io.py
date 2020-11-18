#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, os, subprocess
import re, glob, shlex, shutil, tempfile, warnings
import collections, itertools, copy
import random, string
from os import path
from datetime import datetime
import numpy as np
from scipy import ndimage
from . import six, utils, afni, math, paraproc
# For accessing NIFTI files
try:
    import nibabel
except ImportError:
    print('You may need to install "nibabel" to read/write NIFTI (*.nii) files.')
try:
    from lxml import etree
except ImportError:
    print('You may need to install "lxml" to read/write niml datasets (*.niml.dset).')

# Timestamp
def _timestamp(dt):
    '''
    Work-around for python 2.7 not having dt.timestamp() yet.
    http://stackoverflow.com/questions/11743019/convert-python-datetime-to-epoch-with-strftime
    '''
    return (dt - datetime(1970,1,1)).total_seconds()


def hms2dt(hms, date=None, timestamp=False):
    '''
    Convert time string in hms format to datetime object.

    `hms` is like "102907.165000". This format is used in dicom header.
    '''
    if date is None:
        date = '19700101'
    dt = datetime.strptime(date+hms, '%Y%m%d%H%M%S.%f')
    return _timestamp(dt) if timestamp else dt


def mmn2dt(mmn, date=None, timestamp=False):
    '''
    Convert time string in mmn format to datetime object.

    `mmn` is "msec since midnight", like "37747165". This format is used in
    physiological measurement log file.
    '''
    if date is None:
        date = '19700101'
    t = datetime.utcfromtimestamp(float(mmn)/1000)
    d = datetime.strptime(date, '%Y%m%d')
    dt = datetime.combine(d.date(), t.time())
    return _timestamp(dt) if timestamp else dt


# Physiological data
def _parse_physio_raw(fname):
    # print('Parsing "{0}"...'.format(fname))
    n_pre = {'ecg': 5, 'ext': 4, 'puls': 4, 'resp': 4}
    with open(fname, 'r') as fin:
        info = {}
        ch = path.splitext(fname)[1][1:]
        info['file'] = path.realpath(fname)
        info['channel'] = ch
        lines = fin.read().splitlines() # Without \n unlike fin.readlines()
        if len(lines) == 0:
            print('*+ WARNING: "{0}" seems be empty...'.format(fname), file=sys.stderr)
            return None
        k = 0
        # Data line(s)
        while lines[k][-4:] != '5003': # There can be more than one data lines
            k += 1
            if k >= len(lines): # The file does not contain 5003
                print('*+ WARNING: "{0}" might be broken...'.format(fname), file=sys.stderr)
                return None
        else:
            k += 1
            data_line = ''.join(lines[:k])
            info['messages'] = re.findall('\s5002\s(.+?)\s6002', data_line)
            data_line = re.sub('\s5002\s(.+?)\s6002', '', data_line) # Remove messages inserted between 5002/6002
            info['rawdata'] = np.int_(data_line.split()[n_pre[ch]:-1])
        # Timing lines
        items = ['LogStartMDHTime', 'LogStopMDHTime', 'LogStartMPCUTime', 'LogStopMPCUTime']
        for item in items:
            while True:
                match = re.match('({0}):\s+(\d+)'.format(item), lines[k])
                k += 1
                if match:
                    info[match.group(1)] = int(match.group(2))
                    break
        return info


def parse_physio_file(fname, date=None):
    '''
    Implementation notes
    --------------------
    1. The first 4 (ext, puls, resp) or 5 (ecg) values are parameters (of
       unknown meanings).
    2. There can be multiple data lines, within which extra parameters is
       inclosed between 5002 and 6002, especially for ecg.
    3. The footer is inclosed between 5003 and 6003, following physiological
       data (and that's why the final data value always appears to be 5003).
    4. The MDH values are timestamps derived from the clock in the scanner (so
       do DICOM images), while the MPCU values are timestamps derived from the
       clock within the PMU recording system [1]. Use MDH time to synchronize
       physiological and imaging time series.
    5. The trigger values (5000) are "inserted" into the data, and have to be
       stripped out from the time series [1]. This fact is double checked by
       looking at the smooth trend of the puls waveform.
    6. The sampling rate is slightly (and consistently) slower than specified
       in the manual and in [1].

    Notes about timing
    ------------------
    The scanner clock is slightly faster than the wall clock so that 2 sec in
    real time is recorded as ~2.008 sec in the scanner, affacting both dicom
    header and physiological footer, even though the actual TR is precisely 2 s
    (as measured by timing the s triggers with psychtoolbox) and the actual
    sampling rate of physiological data is precisely 50 Hz (as estimated by
    dividing the total number of samples by the corrected recording duration).

    References
    ----------
    [1] https://cfn.upenn.edu/aguirre/wiki/public:pulse-oximetry_during_fmri_scanning
    '''
    # fs = {'ecg': 400, 'ext': 200, 'puls': 50, 'resp': 50}
    fs = {'ecg': 398.4, 'ext': 199.20, 'puls': 49.80, 'resp': 49.80}
    trig_value = 5000
    tag_values = np.r_[trig_value, 6000]
    info = _parse_physio_raw(fname)
    if info is None:
        return None
    ch = info['channel']
    info['fs'] = fs[ch]
    info['start'] = mmn2dt(info['LogStartMDHTime'], date, timestamp=True)
    info['stop'] = mmn2dt(info['LogStopMDHTime'], date, timestamp=True)
    x = info['rawdata']
    if ch != 'ecg':
        # y = x.copy()
        # y = x[x!=trig_value] # Strip trigger value (5000)
        y = x[~np.in1d(x, tag_values)] # Strip all tag values (5000, 6000, ...)
        trig = np.zeros_like(x)
        trig[np.nonzero(x==trig_value)[0]-1] = 1
        trig = trig[x!=trig_value]
    else:
        y = x[:len(x)//2*2].reshape(-1,2)
        trig = np.zeros_like(y)
    info['data'] = y
    info['trig'] = trig
    info['t'] = info['start'] + np.arange(len(y)) / fs[ch]
    try:
        assert(max(y) < 4096) # Valid data range is [0, 4095]
    except AssertionError as err:
        print('\n** Invalid data value detected: {0}'.format(np.unique(y[y>4095])))
        raise err
    try:
        assert(np.abs(info['t'][-1]-info['stop'])<2/info['fs']) # Allow 1 sampleish error
    except AssertionError as err:
        print('\n** {0}: Last sample = {1}, stop = {2}, error = {3}'.format(
            info['channel'], info['t'][-1], info['stop'], info['t'][-1]-info['stop']))
        raise err
    return info


def parse_physio_files(fname, date=None, channels=None):
    '''
    '''
    if channels is None:
        channels = ['ecg', 'ext', 'puls', 'resp']
    stem = path.splitext(fname)[0]
    info = collections.OrderedDict()
    for ch in channels:
        info[ch] = parse_physio_file('.'.join((stem, ch)), date=date)
        if info[ch] is None:
            print('*+ WARNING: "{0}" info is missing. Skip "{1}"...'.format(ch, stem), file=sys.stderr)
            return None
    return info


def match_physio_with_series(physio_infos, series_infos, channel=None, method='cover'):
    if channel is None:
        channel = 'resp'
    physio_t = np.array([[p[channel]['start'], p[channel]['stop']] if p is not None else [0, 0] for p in physio_infos])
    physio = []
    series = []
    for k, s in enumerate(series_infos):
        if method == 'cover':
            p_idx = (physio_t[:,0] < s['start']) & (s['stop'] < physio_t[:,1])
            if np.any(p_idx):
                # If there is more than one (which should not be the case), use only the first one
                physio.append(physio_infos[np.nonzero(p_idx)[0][0]])
                series.append(s)
        elif method == 'overlap':
            p_idx = (physio_t[:,0] < s['stop']) & (s['start'] < physio_t[:,1]) # Thanks to Prof. Zhang Jun
            if np.any(p_idx):
                # If there is more than one (which should not be the case), use the one with largest overlap
                overlap = np.maximum(physio_t[p_idx,0], s['start']) - np.minimum(physio_t[p_idx,1], s['stop'])
                idx = np.nonzero(p_idx)[0][np.argmax(overlap)]
                physio.append(physio_infos[idx])
                series.append(s)
    return physio, series


def _print_physio_timing(pinfo, sinfo, channel, index=None):
    prefix = channel if index is None else '#{0} ({1})'.format(index, channel)
    print('{0}: pre={1:.3f}, scan={2:.3f}, post={3:.3f}, total={4:.3f}'.format(
        prefix, sinfo['start']-pinfo['start'], sinfo['stop']-sinfo['start'],
        pinfo['stop']-sinfo['stop'], pinfo['stop']-pinfo['start']))


def extract_physio(physio_file, dicom_file, TR=None, dummy=0, channels=['resp', 'puls'], verbose=1):
    sinfo = parse_series_info(dicom_file) if isinstance(dicom_file, six.string_types) else dicom_file
    pinfo = parse_physio_files(physio_file, date=sinfo['date']) if isinstance(physio_file, six.string_types) else physio_file
    res = []
    for ch in channels:
        info = pinfo[ch]
        t = info['t']
        valid = (t >= sinfo['start']+dummy*TR) & (t < sinfo['stop']) # Assume timestamp indicates the start of the volume
        res.append(info['data'][valid])
        if verbose:
            _print_physio_timing(info, sinfo, ch)
    return res


# DICOM
def parse_dicom_header(fname, fields=None):
    '''
    Execute afni command `dicom_hdr` to readout most useful info from dicom header.

    Parameters
    ----------
    fname : str
    fields : {field: (matcher, extracter(match))}
        You can require additional fields in dicom header to be parsed.
        - field : e.g., 'ImageTime'
        - matcher : e.g. r'ID Image Time//(\S+)'
        - extracter : e.g., lambda match: io.hms2dt(match.group(1), date='20170706', timestamp=True)
    '''
    # print(fname)
    header = collections.OrderedDict()
    lines = subprocess.check_output(['dicom_hdr', fname]).decode('utf-8').split('\n')
    k = 0
    try:
        while True:
            match = re.search(r'ID Acquisition Date//(\S+)', lines[k])
            k += 1
            if match:
                header['AcquisitionDate'] = match.group(1)
                break
        while True:
            match = re.search(r'ID Acquisition Time//(\S+)', lines[k])
            k += 1
            if match:
                header['AcquisitionTime'] = match.group(1) # This marks the start of a volume
                header['timestamp'] = hms2dt(header['AcquisitionTime'], date=header['AcquisitionDate'], timestamp=True)
                break
        while True:
            match = re.search(r'ACQ Scanning Sequence//(.+)', lines[k])
            k += 1
            if match:
                header['sequence_type'] = match.group(1).strip()
                break
        while True:
            match = re.search(r'ACQ Sequence Variant//(.+)', lines[k])
            k += 1
            if match:
                header['sequence_type'] = ' '.join((match.group(1).strip(), header['sequence_type']))
                break
        while True:
            match = re.search(r'ACQ MR Acquisition Type //(.+)', lines[k])
            k += 1
            if match:
                header['sequence_type'] = ' '.join((match.group(1).strip(), header['sequence_type']))
                break
        while True:
            match = re.search(r'ACQ Slice Thickness//(\S+)', lines[k])
            k += 1
            if match:
                header['resolution'] = [float(match.group(1))]
                break
        while True:
            match = re.search(r'ACQ Repetition Time//(\S+)', lines[k])
            k += 1
            if match:
                header['RepetitionTime'] = float(match.group(1)) # ms
                break
        while True:
            match = re.search(r'ACQ Echo Time//(\S+)', lines[k])
            k += 1
            if match:
                header['TE'] = float(match.group(1)) # ms
                break
        while True:
            match = re.search(r'ACQ Imaging Frequency//(\S+)', lines[k])
            k += 1
            if match:
                header['Larmor'] = float(match.group(1)) # MHz
                break
        while True:
            match = re.search(r'ACQ Echo Number//(\S+)', lines[k])
            k += 1
            if match:
                header['EchoNumber'] = int(match.group(1)) # For multi-echo images
                break
        while True:
            match = re.search(r'ACQ Magnetic Field Strength//(\S+)', lines[k])
            k += 1
            if match:
                header['B0'] = float(match.group(1)) # Tesla
                break
        while True:
            match = re.search(r'ACQ Pixel Bandwidth//(\S+)', lines[k])
            k += 1
            if match:
                header['BW'] = float(match.group(1)) # Hz/pixel
                break
        while True:
            match = re.search(r'ACQ Protocol Name//(.+)', lines[k])
            k += 1
            if match:
                header['ProtocolName'] = match.group(1).strip()
                break
        while True:
            match = re.search(r'ACQ Flip Angle//(\S+)', lines[k])
            k += 1
            if match:
                header['FlipAngle'] = float(match.group(1))
                break
        while True:
            match = re.search(r'ACQ SAR//(\S+)', lines[k])
            k += 1
            if match:
                header['SAR'] = float(match.group(1))
                break
        while True: # This field is optional
            match = re.search(r'0019 100a.+//\s*(\d+)', lines[k])
            k += 1
            if match:
                header['n_slices'] = int(match.group(1))
                break
            if lines[k].startswith('0020'):
                break
        while True:
            match = re.search(r'REL Study ID//(\d+)', lines[k])
            k += 1
            if match:
                header['StudyID'] = int(match.group(1)) # Study index
                break
        while True:
            match = re.search(r'REL Series Number//(\d+)', lines[k])
            k += 1
            if match:
                header['SeriesNumber'] = int(match.group(1)) # Series index
                break
        while True:
            match = re.search(r'REL Acquisition Number//(\d+)', lines[k])
            k += 1
            if match:
                header['AcquisitionNumber'] = int(match.group(1)) # Volume index
                break
        while True:
            match = re.search(r'REL Instance Number//(\d+)', lines[k])
            k += 1
            if match:
                header['InstanceNumber'] = int(match.group(1)) # File index (whether it is one volume or one slice per file)
                break
        while True:
            match = re.search(r'IMG Pixel Spacing//(\S+)', lines[k])
            k += 1
            if match:
                header['resolution'] = list(map(float, match.group(1).split('\\'))) + header['resolution']
                break
        while True: # This field is optional
            match = re.search(r'0051 1011.+//(\S+)', lines[k])
            k += 1
            if match:
                header['iPAT'] = match.group(1)
                break
            if lines[k].startswith('Group'):
                break
    except IndexError as error:
        print('** Failed to process "{0}"'.format(fname))
        raise error
    if fields is not None:
        for line in lines:
            for field, (matcher, extracter) in fields.items():
                match = re.search(matcher, line)
                if match:
                    header[field] = extracter(match)
                    break
    header['gamma'] = 2*np.pi*header['Larmor']/header['B0']
    return header


SERIES_PATTERN = r'.+?\.(\d{4})\.' # Capture series number
MULTI_SERIES_PATTERN = r'.+?\.(\d{4})\.(\d{4}).+(\d{8,})' # Capture series number, slice number, uid
MULTI_SERIES_PATTERN2 = r'.+?\.(\d{4})\.(\d{4}).+(\d{5,}\.\d{8,})' # Capture series number, slice number, uid (5-6.8-9)
# MULTI_SERIES_PATTERN3 = r'.+?\.(\d{4})\.(\d{4}).+(\d{5,})\.\d{5,}' # Capture series number, slice number, uid (5-6).5-9
MULTI_SERIES_PATTERN3 = r'.+?\.(\d{4})\.(\d{4}).+(\d+)\.\d+' # Capture series number, slice number, uid

def _sort_multi_series(files):
    '''
    Sort multiple series sharing the same series number into different studies.
    '''
    series = []
    timestamps = []
    infos = []
    for f in files:
        match = re.search(MULTI_SERIES_PATTERN3, f)
        # infos.append((f, int(match.group(2)), int(match.group(3))))
        infos.append((f, int(match.group(2)), float(match.group(3))))
    prev_slice = sys.maxsize
    for f, curr_slice, timestamp in sorted(infos, key=lambda x: x[-1]):
        if curr_slice <= prev_slice and (prev_slice == sys.maxsize or curr_slice in slices):
            # We meet a new sequence (including the first one).
            # Note that slices within a study are unique but may not be strictly ordered.
            # The first clause is a shortcut, and the second one is the real condition.
            series.append([])
            slices = set()
            timestamps.append(timestamp)
        series[-1].append(f)
        slices.add(curr_slice)
        prev_slice = curr_slice
    return series, timestamps


def sort_dicom_series(folder, series_pattern=SERIES_PATTERN):
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
    # Sort files into series
    files = sorted(glob.glob(path.join(folder, '*.IMA')))
    series = collections.OrderedDict()
    for f in files:
        filename = path.basename(f)
        match = re.search(series_pattern, filename)
        sn = match.group(1)
        if sn not in series:
            series[sn] = []
        series[sn].append(filename) # Changed 2019-10-25: series[sn].append(f)
    # Separate potentially multiple series sharing the same series number into different studies
    studies = None
    for s_idx, (sn, files) in enumerate(series.items()):
        subsets, timestamps = _sort_multi_series(files)
        if s_idx == 0:
            n_folders = len(subsets)
            # Note that if the first series is single, all series must be single.
            if n_folders == 1:
                studies = [series]
                break
            else:
                studies = [collections.OrderedDict() for k in range(n_folders)]
            for k, subset in enumerate(subsets):
                studies[k][sn] = subset
            start_times = timestamps
        else:
            # Handle the case when a later study has more series than earlier studies
            for k, subset in enumerate(subsets):
                kk = n_folders - 1
                while start_times[kk] > timestamps[k]:
                    kk -= 1
                studies[kk][sn] = subset
    return studies


def filter_dicom_files(files, series_numbers=None, instance_numbers=None, series_pattern=MULTI_SERIES_PATTERN3):
    if isinstance(files, six.string_types) and path.isdir(files):
        files = glob.glob(path.join(files, '*.IMA'))
    if not isinstance(series_numbers, collections.Iterable):
        series_numbers = [series_numbers]
    if not isinstance(instance_numbers, collections.Iterable):
        instance_numbers = [instance_numbers]
    files = np.array(sorted(files))
    if len(files) == 0:
        return []
    infos = []
    for fname in files:
        filepath, filename = path.split(fname)
        match = re.match(series_pattern, filename)
        infos.append(list(map(int, match.groups()))) # series number, instance number, uid
    infos = np.array(infos)
    filtered = []
    if not series_numbers: # [] or None
        series_numbers = np.unique(infos[:,0])
    for series in series_numbers:
        if not instance_numbers:
            instance_numbers = np.unique(infos[infos[:,0]==series,1])
        for instance in instance_numbers:
            filtered.extend(files[(infos[:,0]==series)&(infos[:,1]==instance)])
    return filtered


def parse_slice_order(dicom_files):
    t = None
    if len(dicom_files) > 1:
        temp_dir = 'temp_pares_slice_order'
        os.makedirs(temp_dir)
        for k, f in enumerate(dicom_files[:2]):
            shutil.copyfile(f, path.join(temp_dir, '{0}.IMA'.format(k)))
        old_path = os.getcwd()
        try:
            os.chdir(temp_dir)
            afni.check_output('''Dimon -infile_pattern '*.IMA'
                -gert_create_dataset -gert_to3d_prefix temp -gert_quit_on_err''')
            res = afni.check_output(['3dAttribute', 'TAXIS_OFFSETS', 'temp+orig'])[-2]
            t = np.array(list(map(float, res.split())))
        finally:
            os.chdir(old_path)
            shutil.rmtree(temp_dir)
    if t is None:
        order = None
    elif np.all(np.diff(t) > 0):
        order = 'ascending'
    elif np.all(np.diff(t) < 0):
        order = 'descending'
    else:
        order = 'interleaved'
    return order, t


def parse_series_info(fname, timestamp=False, shift_time=None, series_pattern=SERIES_PATTERN, fields=None, parser=None):
    '''
    Potential bug: `dicom.parse_dicom_header` doesn't support `fields` as kwargs 
    '''
    if isinstance(fname, six.string_types): # A single file or a folder
        if path.isdir(fname):
            # Assume there is only one series in the folder, so that we only need to consider the first file.
            fname = sorted(glob.glob(path.join(fname, '*.IMA')))[0]
        # Select series by series number (this may fail if there is multi-series in the folder)
        filepath, filename = path.split(fname)
        match = re.match(series_pattern, filename)
        files = sorted(glob.glob(path.join(filepath, '{0}*.IMA'.format(match.group(0)))))
        findex = None
    else: # A list of files (e.g., as provided by sort_dicom_series)
        files = fname
        findex = 0
    if parser is None:
        parser = parse_dicom_header
    info = collections.OrderedDict()
    if timestamp:
        parse_list = range(len(files))
    else:
        parse_list = [0, -1]
    headers = [parser(files[k], fields=fields) for k in parse_list]
    if headers[0]['StudyID'] != headers[-1]['StudyID']:
        # There are more than one series (from different studies) sharing the same series number
        if parse_list == [0, -1]:
            headers = [headers[0]] + [parser(f) for f in files[1:-1]] + [headers[-1]]
        if findex is None:
            findex = files.index(fname)
        selected = [k for k, header in enumerate(headers) if header['StudyID']==headers[findex]['StudyID']]
        files = [files[k] for k in selected]
        headers = [headers[k] for k in selected]
    info.update(headers[0])
    info['date'] = info['AcquisitionDate']
    info['first'] = headers[0]['timestamp']
    info['last'] = headers[-1]['timestamp']
    info['n_volumes'] = headers[-1]['AcquisitionNumber'] - headers[0]['AcquisitionNumber'] + 1
    info['TR'] = (info['last']-info['first'])/(info['n_volumes']-1) if info['n_volumes'] > 1 else None
    if shift_time == 'CMRR':
        shift_time = 0
        if info['TR'] is not None and 'n_slices' in info and np.mod(info['n_slices'], 2)==0:
            slice_order = parse_slice_order(files)[0]
            if slice_order == 'interleaved':
                shift_time = -info['TR']/2
    elif shift_time is None:
        shift_time = 0
    info['first'] += shift_time
    info['last'] += shift_time
    info['start'] = info['first']
    info['stop'] = (info['last'] + info['TR']) if info['TR'] is not None else info['last']
    if timestamp:
        info['t'] = np.array([header['timestamp'] for header in headers]) + shift_time
    info['files'] = [path.realpath(f) for f in files]
    info['headers'] = headers
    return info


def convert_dicom(dicom_dir, out_file=None, dicom_ext=None, interactive=False):
    if dicom_ext is None:
        dicom_ext = '.IMA'
    if out_file is None:
        out_file = './*.nii'
    out_dir, prefix, ext = afni.split_out_file(out_file, split_path=True)
    out_dir = path.realpath(path.expanduser(out_dir)) # It is special here because we'll change dir later
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if prefix == '*': # Take the dicom folder name by default
        prefix = path.split(dicom_dir)[1]
    old_path = os.getcwd()
    try:
        os.chdir(dicom_dir)
        with open('uniq_image_list.txt', 'w') as fo:
            subprocess.check_call(['uniq_images'] + glob.glob('*'+dicom_ext), stdout=fo) # Prevent shell injection by not using shell=True with user defined string
        interactive_cmd = '' if interactive else '-gert_quit_on_err'
        utils.run("Dimon -infile_list uniq_image_list.txt \
            -gert_create_dataset -gert_outdir '{0}' -gert_to3d_prefix '{1}' -overwrite \
            -dicom_org -use_obl_origin -save_details Dimon.details {2}".format(out_dir, prefix+ext, interactive_cmd))
    finally:
        os.chdir(old_path)


def convert_dicoms(dicom_dirs, out_dir=None, prefix=None, out_type='.nii', dicom_ext='.IMA', **kwargs):
    '''
    Parameters
    ----------
    dicom_dirs : list or str
        1. A list of folders containing *.IMA files
        2. It can also be a glob pattern that describes a list of folders, e.g., "raw_fmri/func??"
        3. Finally, it can be a root folder (e.g., "raw_fmri") containing multiple sub-folders of *.IMA files, 
           raw_fmri/anat, raw_fmri/func01, raw_fmri/func02, etc.
    out_dir : str
        Output directory for converted datasets, default is current directory.
        The output would look like:
            out_dir/anat.nii, out_dir/func01.nii, out_dir/func02.nii, etc.
    '''
    original_dicom_dirs = dicom_dirs
    if isinstance(dicom_dirs, six.string_types):
        if utils.contain_wildcard(dicom_dirs):
            dicom_dirs = glob.glob(dicom_dirs)
        else:
            dicom_dirs = glob.glob(path.join(dicom_dirs, '*'))
    if len(dicom_dirs) == 0: # Sanity check after Yong's true story
        warnings.warn(f"\n>> Cannot find any dicom file to convert. Is the following path correct?\n{original_dicom_dirs}")
    if out_dir is None:
        out_dir = '.'
    idx = 0
    for f in dicom_dirs:
        if path.isdir(f) and len(glob.glob(path.join(f, '*'+dicom_ext))) > 0:
            idx += 1
            convert_dicom(f, path.join(out_dir, '*'+out_type if prefix is None else '{0}{1:02d}{2}'.format(prefix, idx, out_type)), dicom_ext=dicom_ext, **kwargs)


# ========== Generic read/write ==========
def read_vol(fname, return_img=False):
    img = nibabel.load(fname)
    vol = np.asanyarray(img.dataobj) # Equivalent to the deprecated "vol = img.get_data()" that gives minimum possible data size in memory
    # vol = img.get_fdata() # get_data() is deprecated in favor of get_fdata(), which has a more predictable return type
    return (vol, img) if return_img else vol


def write_vol(fname, vol, base_img=None):
    if fname.endswith('.nii'):
        write_nii(fname, vol, base_img)
    else:
        write_afni(fname, vol, base_img)


def read_surf_mesh(fname, return_img=False, **kwargs):
    if fname.endswith('.asc'):
        verts, faces = read_asc(fname, **kwargs)
        img = None
    elif fname.endswith('.gii'):
        verts, faces, img = read_gii(fname, return_img=True)
    return (verts, faces, img) if return_img else (verts, faces)


def write_surf_mesh(fname, verts, faces, **kwargs):
    if fname.endswith('.asc'):
        write_asc(fname, verts, faces, **kwargs)
    elif fname.endswith('.gii'):
        write_gii(fname, verts, faces, **kwargs)


def read_surf_data(fname):
    if fname.endswith('.niml.dset'):
        nodes, values = read_niml_bin_nodes(fname)
    return nodes, values


def write_surf_data(fname, nodes, values):
    if fname.endswith('.niml.dset'):
        write_niml_bin_nodes(fname, nodes, values)


def read_surf_info(fname):
    info = {}
    if fname.endswith('.asc'):
        with open(fname) as fi:
            for line in fi:
                if not line.startswith('#'):
                    info['n_verts'], info['n_faces'] = np.int_(line.split())
                    break
        info['hemi'] = afni.get_hemi(fname)
        info['ext'] = '.asc'
    elif fname.endswith('.gii'):
        img = nibabel.load(fname)
        info['n_verts'] = img.get_arrays_from_intent(nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].dims[0]
        info['n_faces'] = img.get_arrays_from_intent(nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].dims[0]
        info['hemi'] = afni.get_hemi(fname)
        info['ext'] = '.gii'
    return info


def read_txt(fname, dtype=float, comment='#', delimiter=None, skiprows=0, nrows=None, return_comments=False):
    '''Read numerical array from text file, much faster than np.loadtxt()'''
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    if return_comments:
        comments = [line for line in lines[skiprows:] if line.strip() and line.startswith(comment)]
    lines = [line for line in lines[skiprows:(nrows if not nrows else skiprows+nrows)] if line.strip() and not line.startswith(comment)]
    n_cols = len(lines[0].split(delimiter))
    x = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split(delimiter), lines)), dtype=dtype).reshape(-1,n_cols)
    if return_comments:
        return x, comments
    else:
        return x


def read_stim(fname):
    with open(fname) as fi:
        return [np.array([]) if line[0] == '*' else np.float_(line.split()) for line in fi if line.strip()]


# ========== NIFTI ==========
def read_nii(fname, return_img=False):
    if fname[-4:] != '.nii' and fname[-7:] != '.nii.gz':
        fname = fname + '.nii'
    img = nibabel.load(fname)
    vol = img.get_data()
    return (vol, img) if return_img else vol


def write_nii(fname, vol, base_img=None, space=None, dim=None):
    if fname[-4:] != '.nii' and fname[-7:] != '.nii.gz':
        fname = fname + '.nii'
    if base_img is None:
        affine = nibabel.affines.from_matvec(np.eye(3), np.zeros(3))
    elif isinstance(base_img, six.string_types):
        affine = nibabel.load(base_img).affine
    else:
        affine = base_img.affine
    if dim is not None:
        # img.header['dim'] = dim # This header field will be overwritten by vol.shape
        vol = vol.reshape(dim[1:1+dim[0]])
    img = nibabel.Nifti1Image(vol, affine)
    # https://afni.nimh.nih.gov/afni/community/board/read.php?1,149338,149340#msg-149340
    # 0 (unknown) sform not defined
    # 1 (scanner) RAS+ in scanner coordinates
    # 2 (aligned) RAS+ aligned to some other scan
    # 3 (talairach) RAS+ in Talairach atlas space
    # 4 (mni) RAS+ in MNI atlas space
    if space is None:
        space = 1 if base_img is None else base_img.header['sform_code']
    img.header['sform_code'] = space
    nibabel.save(img, fname)


SPACE_CODE = {
    'unknown': 0,
    'scanner': 1, 'orig': 1, 'ORIG': 1,
    'aligned': 2,
    'talairach': 3, 'tlrc': 3, 'TLRC': 3,
    'mni': 4, 'MNI': 4,
}


def get_space(in_file):
    space = int(nibabel.load(in_file).header['sform_code'])
    return {0: 'unknown', 1: 'orig', 2: 'aligned', 3: 'tlrc', 4: 'mni'}[space]


def change_space(in_file, out_file=None, space=None, method='nibabel'):
    '''
    >>> change_space('MNI152_2009_template.nii.gz', 'template.nii', space='ORIG')
    >>> change_space('test+tlrc.HEAD') # -> test.nii as ORIG
    '''
    if isinstance(space, str):
        space = SPACE_CODE[space]
    if method == 'nibabel':
        if out_file is None:
            prefix, ext = afni.split_out_file(in_file)
            out_file = f"{prefix}.nii"
        vol, img = read_vol(in_file, return_img=True)
        # write_nii(out_file, vol, base_img=img, space=space)
        # To work-around a strang bug in nibabel when overwriting an existing volume in linux,
        # an explicit copy of the memmap has to be made.
        write_nii(out_file, vol.copy(), base_img=img, space=space)
    elif method == 'afni':
        if space is None:
            space = 1
        # afni.set_nifti_field(in_file, 'sform_code', space, out_file=out_file)
        raise NotImplementedError()
        # Error message:
        # ** ERROR: EDIT_dset_items[244]: illegal new xyzdel
        # ** ERROR: EDIT_dset_items[244]: illegal new xyzorient
        # Before (with `nifti_tool -disp_hdr -field srow_x ...`):
        # srow_x               280      4    -0.8 -0.0 -0.0 62.294399
        # Aftre:
        # srow_x               280      4    0.0 0.0 0.0 0.0
        # With nibabel:
        # srow_x               280      4    -0.7 0.000183 -0.000183 110.674217


def get_dim_order(in_file):
    dim = nibabel.load(in_file).header['dim']
    if dim[0] > 4:
        return 'bucket'
    else:
        return 'timeseries'


def change_dim_order(in_file, out_file=None, dim_order=None, method='afni'):
    '''
    dim_order : 1D array with 8 numbers
        e.g., np.array([  5, 300, 300, 124,   1,   2,   1,   1], dtype=np.int16) # for stats 
        or, np.array([  4, 150, 150,  62, 158,   1,   1,   1], dtype=np.int16) # for epi
    method : str, 'afni' | 'nibabel'
    '''
    if dim_order is None:
        dim_order = 'timeseries'
    def get_new_dim(dim, dim_order):
        if isinstance(dim_order, str):
            if dim_order in ['timeseries']:
                new_dim = np.r_[4, dim[1:4], max(dim[4:]), 1, 1, 1]
            elif dim_order in ['stats', 'bucket']:
                new_dim = np.r_[5, dim[1:4], 1, max(dim[4:]), 1, 1]
            else:
                raise ValueError('** Only support "timeseries" and "bucket" dim_order.')
        else:
            new_dim = dim_order
        return new_dim.astype(np.int16)
    if method == 'nibabel':
        if out_file is None:
            prefix, ext = afni.split_out_file(in_file)
            out_file = f"{prefix}.nii"
        vol, img = read_vol(in_file, return_img=True)
        dim = img.header['dim']
        write_nii(out_file, vol, base_img=img, dim=get_new_dim(dim, dim_order))
    elif method == 'afni':
        dim = afni.get_nifti_field(in_file, 'dim', 'int')
        afni.set_nifti_field(in_file, 'dim', get_new_dim(dim, dim_order), out_file=out_file)


# ========== AFNI HEAD/BRIK ==========
def read_afni(fname, remove_nii=True, return_img=False):
    try:
        if fname[-5:] in ['.HEAD', '.BRIK']:
            pass
        elif fname[-1] == '.':
            fname = fname + 'HEAD'
        else:
            fname = fname + '.HEAD'
        img = nibabel.load(fname) # Start from nibabel 2.3.0 (with brikhead.py)
        # vol = img.get_data().squeeze()
        vol = img.get_data()
        return (vol, img) if return_img else vol
    except nibabel.filebasedimages.ImageFileError:
        print('*+ WARNING: Fail to open "{0}" with nibabel, fallback to 3dAFNItoNIFTI'.format(fname)) 
        match = re.match('(.+)\+', fname)
        nii_fname = match.group(1) + '.nii'
        subprocess.check_call(['3dAFNItoNIFTI', '-prefix', nii_fname, fname])
        res = read_nii(nii_fname, return_img)
        if remove_nii:
            os.remove(nii_fname)
        return res


def write_afni(prefix, vol, base_img=None):
    nii_fname = prefix + '.nii'
    write_nii(nii_fname, vol, base_img)
    subprocess.check_call(['3dcopy', nii_fname, prefix+'+orig', '-overwrite'])
    os.remove(nii_fname)




# ========== AFNI ASC ==========
def read_asc(fname, dtype=None):
    '''Read FreeSurfer/SUMA surface (vertices and faces) in *.asc format.'''
    if dtype is None:
        dtype = float
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    n_verts, n_faces = np.int_(lines[1].split())
    # verts = np.vstack(map(lambda line: np.float_(line.split()), lines[2:2+n_verts])) # As slow as np.loadtxt()
    # verts = np.float_(''.join(lines[2:2+n_verts]).split()).reshape(-1,4) # Much faster
    verts = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split()[:3], lines[2:2+n_verts])), dtype=dtype).reshape(-1,3)
    faces = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split()[:3], lines[2+n_verts:2+n_verts+n_faces])), dtype=int).reshape(-1,3)
    return verts, faces


def read_patch_asc(fname, dtype=None, index_type='multimap'):
    '''
    Read FreeSurfer/SUMA patch (noncontiguous vertices and faces) in *.asc format.
    
    index_type : str
        - "raw" or "array"
        - "map" or "dict"
        - "multimap" or "func"
    '''
    if dtype is None:
        dtype = float
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    n_verts, n_faces = np.int_(lines[1].split())
    # verts = np.vstack(map(lambda line: np.float_(line.split()), lines[2:2+n_verts])) # As slow as np.loadtxt()
    # verts = np.float_(''.join(lines[2:2+n_verts]).split()).reshape(-1,4) # Much faster
    verts = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split()[:3], lines[2+1:2+n_verts*2:2])), dtype=dtype).reshape(-1,3)
    faces = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split()[:3], lines[2+1+n_verts*2:2+n_verts*2+n_faces*2:2])), dtype=int).reshape(-1,3)
    vidx = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split('=')[-1:], lines[2:2+n_verts*2:2])), dtype=int)
    fidx = np.fromiter(itertools.chain.from_iterable(
        map(lambda line: line.split('=')[-1:], lines[2+n_verts*2:2+n_verts*2+n_faces*2:2])), dtype=int)
    vmap = {vidx[k]: k for k in range(n_verts)}
    fmap = {fidx[k]: k for k in range(n_faces)}
    if index_type in ['raw', 'array']:
        pass
    if index_type in ['map', 'dict']:
        vidx = vmap
        fidx = fmap
    elif index_type in ['multimap', 'func']:
        vidx = lambda K: vmap[K] if np.isscalar(K) else [vmap[k] for k in K]
        fidx = lambda K: fmap[K] if np.isscalar(K) else [fmap[k] for k in K]
    return verts, faces, vidx, fidx


def write_asc(fname, verts, faces):
    with open(fname, 'wb') as fout: # Binary mode is more compatible with older Python...
        fout.write('#!ascii version of surface mesh saved by mripy\n'.encode('ascii'))
        np.savetxt(fout, [[len(verts), len(faces)]], fmt='%d')
        np.savetxt(fout, np.c_[verts, np.zeros(len(verts))], fmt=['%.6f', '%.6f', '%.6f', '%d'])
        np.savetxt(fout, np.c_[faces, np.zeros(len(faces))], fmt='%d')    


# ========== GIFTI ==========
def read_gii(fname, return_img=False):
    img = nibabel.load(fname)
    # verts, faces = img.darrays[0].data, img.darrays[1].data
    verts = img.get_arrays_from_intent(nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data
    faces = img.get_arrays_from_intent(nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
    return (verts, faces, img) if return_img else (verts, faces)


def write_gii(fname, verts, faces):
    # NOTE: SUMA only work with float32 NIFTI_INTENT_POINTSET and int32 NIFTI_INTENT_TRIANGLE
    verts = nibabel.gifti.GiftiDataArray(data=verts.astype('float32'), intent=nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])
    faces = nibabel.gifti.GiftiDataArray(data=faces.astype('int32'), intent=nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])
    img = nibabel.gifti.GiftiImage(darrays=[verts, faces])
    nibabel.gifti.write(img, fname)


def read_label(fname):
    '''Read FreeSurfer label'''
    x = read_txt(fname)
    nodes = np.int_(x[:,0])
    coords = x[:,1:4]
    labels = x[:,4]
    return nodes, coords, labels


# ========== NIML ascii ==========
NIML_DSET_CORE_TAGS = ['INDEX_LIST', 'SPARSE_DATA']

def read_niml_dset(fname, tags=None, as_asc=True, return_type='list'):
    if tags is None:
        tags = NIML_DSET_CORE_TAGS
    if as_asc:
        temp_file = 'tmp.' + fname
        if not path.exists(temp_file):
            subprocess.check_call(['ConvertDset', '-o_niml_asc', '-input', fname, '-prefix', temp_file])
        root = etree.parse(temp_file).getroot()
        os.remove(temp_file)
        def get_data(tag):
            element = root.find(tag)
            return np.fromiter(element.text.split(), dtype=element.get('ni_type'))
        data = {tag: get_data(tag) for tag in tags}
    if return_type == 'list':
        return [data[tag] for tag in tags]
    elif return_type == 'dict':
        return data
    elif return_type == 'tree':
        return root


# def read_niml_bin_nodes(fname):
#     '''
#     Read "Node Bucket" (node indices and values) from niml (binary) dataset.
#     This implementation is experimental for one-column dset only.
#     '''
#     with open(fname, 'rb') as fin:
#         s = fin.read()
#         data = []
#         for tag in NIML_DSET_CORE_TAGS:
#             pattern = '<{0}(.*?)>(.*?)</{0}>'.format(tag)
#             match = re.search(bytes(pattern, encoding='utf-8'), s, re.DOTALL)
#             if match is not None:
#                 # attrs = match.group(1).decode('utf-8').split()
#                 # attrs = {k: v[1:-1] for k, v in (attr.split('=') for attr in attrs)}
#                 attrs = shlex.split(match.group(1).decode('utf-8')) # Don't split quoted string
#                 attrs = dict(attr.split('=') for attr in attrs)
#                 x = np.frombuffer(match.group(2), dtype=attrs['ni_type']+'32')
#                 data.append(x.reshape(np.int_(attrs['ni_dimen'])))
#             else:
#                 data.append(None)
#         if data[0] is None: # Non-sparse dataset
#             data[0] = np.arange(data[1].shape[0])
#         return data[0], data[1]


# ========== NIML binary ==========
def read_niml_bin_nodes(fname):
    '''
    Read "Node Bucket" (node indices and values) from niml (binary) dataset.
    '''
    with open(fname, 'rb') as fin:
        s = fin.read()
        data = []
        for tag in NIML_DSET_CORE_TAGS:
            pattern = '<{0}(.*?)>(.*?)</{0}>'.format(tag)
            match = re.search(bytes(pattern, encoding='utf-8'), s, re.DOTALL)
            if match is not None:
                attrs = shlex.split(match.group(1).decode('utf-8')) # Don't split quoted string
                attrs = dict(attr.split('=') for attr in attrs)
                if '*' in attrs['ni_type']: # Multi-colume dataset
                    n, t = attrs['ni_type'].split('*')
                    attrs['n_columes'] = int(n)
                    attrs['dtype'] = t+'32'
                else:
                    attrs['n_columes'] = int(1)
                    attrs['dtype'] = attrs['ni_type']+'32'
                x = np.frombuffer(match.group(2), dtype=attrs['dtype'])
                data.append(x.reshape(np.int_([attrs['ni_dimen'], attrs['n_columes']])))
            else:
                data.append(None)
        if data[0] is None: # Non-sparse dataset
            data[0] = np.arange(data[1].shape[0])
        return data[0].squeeze(), data[1].squeeze()


# def write_niml_bin_nodes(fname, idx, val):
#     '''
#     Write "Node Bucket" (node indices and values) as niml (binary) dataset.
#     This implementation is experimental for one-column dset only.

#     References
#     ----------
#     [1] https://afni.nimh.nih.gov/afni/community/board/read.php?1,60396,60399#msg-60399
#     [2] After some trial-and-error, the following components are required:
#         self_idcode, COLMS_RANGE, COLMS_TYPE (tell suma how to interpret val), 
#         no whitespace between opening tag and binary data.
#     '''
#     with open(fname, 'wb') as fout:
#         # AFNI_dataset
#         fout.write('<AFNI_dataset dset_type="Node_Bucket" self_idcode="{0}" \
#             ni_form="ni_group">\n'.format(generate_afni_idcode()).encode('utf-8'))
#         # COLMS_RANGE
#         fout.write('<AFNI_atr ni_type="String" ni_dimen="1" atr_name="COLMS_RANGE">\
#             "{0} {1} {2} {3}"</AFNI_atr>\n'.format(np.min(val), np.max(val), 
#             idx[np.argmin(val)], idx[np.argmax(val)]).encode('utf-8'))
#         # COLMS_TYPE
#         col_types = {'int': 'Node_Index_Label', 'float': 'Generic_Float'}
#         fout.write('<AFNI_atr ni_type="String" ni_dimen="1" atr_name="COLMS_TYPE">\
#             "{0}"</AFNI_atr>\n'.format(col_types[get_ni_type(val)]).encode('utf-8'))
#         # INDEX_LIST
#         # Important: There should not be any \n after the opening tag for the binary data!
#         fout.write('<INDEX_LIST ni_form="binary.lsbfirst" ni_type="int" ni_dimen="{0}" \
#             data_type="Node_Bucket_node_indices">'.format(len(idx)).encode('utf-8'))
#         fout.write(idx.astype('int32').tobytes())
#         fout.write(b'</INDEX_LIST>\n')
#         # SPARSE_DATA
#         fout.write('<SPARSE_DATA ni_form="binary.lsbfirst" ni_type="{0}" ni_dimen="{1}" \
#             data_type="Node_Bucket_data">'.format(get_ni_type(val), len(val)).encode('utf-8'))
#         fout.write(val.astype(get_ni_type(val)+'32').tobytes())
#         fout.write(b'</SPARSE_DATA>\n')
#         fout.write(b'</AFNI_dataset>\n')


def write_niml_bin_nodes(fname, idx, val):
    '''
    Write "Node Bucket" (node indices and values) as niml (binary) dataset.

    References
    ----------
    [1] https://afni.nimh.nih.gov/afni/community/board/read.php?1,60396,60399#msg-60399
    [2] After some trial-and-error, the following components are required:
        self_idcode, COLMS_RANGE, COLMS_TYPE (tell suma how to interpret val), 
        no whitespace between opening tag and binary data.
    '''
    idx = np.atleast_1d(idx.squeeze())
    val = np.atleast_2d(val)
    if val.shape[0] == 1 and val.shape[1] == len(idx):
        val = val.T
    n_columes = val.shape[1]
    with open(fname, 'wb') as fout:
        # AFNI_dataset
        fout.write('<AFNI_dataset dset_type="Node_Bucket" self_idcode="{0}" \
            ni_form="ni_group">\n'.format(generate_afni_idcode()).encode('utf-8'))
        # COLMS_RANGE
        colms_range = ';'.join(['{0} {1} {2} {3}'.format(np.min(val[:,k]), np.max(val[:,k]), 
            idx[np.argmin(val[:,k])], idx[np.argmax(val[:,k])]) for k in range(n_columes)])
        fout.write('<AFNI_atr ni_type="String" ni_dimen="1" atr_name="COLMS_RANGE">\
            "{0}"</AFNI_atr>\n'.format(colms_range).encode('utf-8'))
        # COLMS_TYPE
        col_types = {'int': 'Node_Index_Label', 'float': 'Generic_Float'}
        colms_type = ';'.join(['{0}'.format(col_types[get_ni_type(val[:,k])]) for k in range(n_columes)])
        fout.write('<AFNI_atr ni_type="String" ni_dimen="1" atr_name="COLMS_TYPE">\
            "{0}"</AFNI_atr>\n'.format(colms_type).encode('utf-8'))
        # INDEX_LIST
        # Important: There should not be any \n after the opening tag for the binary data!
        fout.write('<INDEX_LIST ni_form="binary.lsbfirst" ni_type="int" ni_dimen="{0}" \
            data_type="Node_Bucket_node_indices">'.format(len(idx)).encode('utf-8'))
        fout.write(idx.astype('int32').tobytes())
        fout.write(b'</INDEX_LIST>\n')
        # SPARSE_DATA
        fout.write('<SPARSE_DATA ni_form="binary.lsbfirst" ni_type="{0}" ni_dimen="{1}" \
            data_type="Node_Bucket_data">'.format(get_ni_type(val), len(val)).encode('utf-8'))
        fout.write(val.astype(get_ni_type(val[:,0])+'32').tobytes())
        fout.write(b'</SPARSE_DATA>\n')
        fout.write(b'</AFNI_dataset>\n')


def generate_afni_idcode():
    return 'AFN_' + ''.join(random.choice(string.ascii_letters + string.digits) for n in range(22))


def get_ni_type(x):
    multiple = '{0}*'.format(x.shape[1]) if x.squeeze().ndim > 1 else ''
    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_):
        return multiple+'int'
    elif np.issubdtype(x.dtype, np.floating):
        return multiple+'float'


def write_1D_nodes(fname, idx, val):
    if idx is None:
        idx = np.arange(len(val))
    formats = dict(int='%d', float='%.6f')
    np.savetxt(fname, np.c_[idx, val], fmt=['%d', formats[get_ni_type(val)]])
    

# ========== Affine matrix (matvec format, or aff12) ==========
def read_affine(fname, sep=None):
    '''
    Returns
    -------
    mat : 3x4 or Nx3x4
    '''
    mat = read_txt(fname, delimiter=sep).reshape(-1,3,4).squeeze()
    return mat


def write_affine(fname, mat, oneline=True, sep=None):
    '''
    TODO: Not support multivolume affine yet
    '''
    if sep is None:
        sep = ' '
    with open(fname, 'w') as fo:
        if oneline:
            fo.write(sep.join(['%.6f' % x for x in mat.flat]) + '\n')
        else:
            for row in mat:
                fo.write(sep.join(['%.6f' % x for x in row]) + '\n')


def read_warp(fname):
    '''
    References
    ----------
    [1] https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dQwarp.html
        "An AFNI nonlinear warp dataset stores the displacements (in DICOM mm) from
        the base dataset grid to the source dataset grid.
        AFNI stores a 3D warp as a 3-volume dataset (NiFTI or AFNI format), with the
        voxel values being the displacements in mm (32-bit floats) needed to
        'reach out' and bring (interpolate) another dataset into alignment -- that is,
        'pulling it back' to the grid defined in the warp dataset header."
    '''
    vol = read_vol(fname)
    dX, dY, dZ = np.rollaxis(vol.squeeze(), -1, 0)
    xyz2ijk = math.invert_affine(afni.get_affine(fname)) # iMAT
    return dX, dY, dZ, xyz2ijk


def read_register_dat(fname):
    mat = io.read_txt(fname, skiprows=4, nrows=3)
    return mat


class MaskDumper(object):
    def __init__(self, mask_file):
        self.mask_file = mask_file
        self.temp_file = 'tmp.dump.txt'
        subprocess.check_call(['3dmaskdump', '-mask', self.mask_file, '-index', '-xyz',
            '-o', self.temp_file, self.mask_file])
        x = np.loadtxt(self.temp_file)
        self.index = x[:,0].astype(int)
        self.ijk = x[:,1:4].astype(int)
        self.xyz = x[:,4:7]
        self.mask = x[:,7].astype(int)
        os.remove(self.temp_file)

    def dump(self, fname):
        files = glob.glob(fname) if isinstance(fname, six.string_types) else fname
        subprocess.check_call(['3dmaskdump', '-mask', self.mask_file, '-noijk',
            '-o', self.temp_file, ' '.join(files)])
        x = np.loadtxt(self.temp_file)
        os.remove(self.temp_file)
        return x

    def undump(self, prefix, x):
        np.savetxt(self.temp_file, np.c_[self.ijk, x])
        subprocess.check_call(['3dUndump', '-master', self.mask_file, '-ijk',
            '-prefix', prefix, '-overwrite', self.temp_file])
        os.remove(self.temp_file)


class Mask(object):
    def __init__(self, master=None, kind='mask'):
        self.master = master
        self.value = None
        if self.master is not None:
            self._infer_geometry(self.master)
            if master.endswith('.nii') or master.endswith('.nii.gz'):
                self.value = read_nii(self.master).ravel('F') # [x,y,z], x changes the fastest. Also, NIFTI read/write data in 'F'.
            else:
                self.value = read_afni(self.master).ravel('F')
            if kind == 'mask':
                idx = self.value > 0 # afni uses Fortran index here
                self.value = self.value[idx]
                self.index = np.arange(np.prod(self.IJK))[idx]
            elif kind == 'full':
                self.index = np.arange(np.prod(self.IJK))

    def _infer_geometry(self, fname):
        self.IJK = afni.get_DIMENSION(fname)[:3]
        self.MAT = afni.get_affine(fname)

    def to_dict(self):
        return dict(master=self.master, value=self.value, index=self.index, IJK=self.IJK, MAT=self.MAT)

    @classmethod
    def from_dict(cls, d):
        self = cls(None)
        for k, v in d.items():
            setattr(self, k, v)
        return self

    @classmethod
    def from_expr(cls, expr=None, **kwargs):
        master = list(kwargs.values())[0]
        mask = cls(master=None)
        mask.master = master
        mask._infer_geometry(master)
        data = {v: read_vol(f).squeeze() for v, f in kwargs.items()}
        idx = eval(expr, data).ravel('F') > 0
        mask.index = np.arange(np.prod(mask.IJK))[idx]
        return mask

    @classmethod
    def from_files(cls, files, combine='union'):
        mask = cls(files[0])
        mask.value[:] = 1
        for k in range(1, len(files)):
            mask2 = cls(files[k])
            mask2.value[:] = 2**k
            if combine == 'union':
                mask = mask + mask2
            elif combine == 'intersect':
                mask = mask * mask2
        return mask

    @classmethod
    def concat(cls, masks):
        # Check compatible and disjoint
        for m in masks[1:]:
            assert(masks[0].compatible(m))
            assert(len(np.intersect1d(masks[0].index, m.index))==0)
        # Concat index (in that order)
        mask = copy.deepcopy(masks[0])
        mask.index = np.concatenate([m.index for m in masks])
        mask.value = np.concatenate([m.value for m in masks])
        return mask

    def compatible(self, other):
        return np.all(self.IJK==other.IJK) and np.allclose(self.MAT, other.MAT)

    def __repr__(self):
        return 'Mask ({0} voxels)'.format(len(self.index))

    def __add__(self, other):
        '''Mask union. Both masks are assumed to share the same grid.'''
        assert(self.compatible(other))
        mask = copy.deepcopy(self)
        mask.index = np.union1d(self.index, other.index)
        value_dict = {idx: val for idx, val in zip(self.index, self.value)}
        for idx, val in zip(other.index, other.value):
            if idx in value_dict:
                value_dict[idx] += val
            else:
                value_dict[idx] = val
        mask.value = np.array([value_dict[idx] for idx in mask.index])
        return mask

    def __mul__(self, other):
        '''Mask intersection. Both masks are assumed to share the same grid.'''
        assert(self.compatible(other))
        mask = copy.deepcopy(self)
        mask.index = np.intersect1d(self.index, other.index)
        return mask

    def __sub__(self, other):
        '''
        Voxels that are in the 1st mask but not in the 2nd mask.
        Both masks are assumed to share the same grid.
        '''
        assert(self.compatible(other))
        mask = copy.deepcopy(self)
        mask.index = mask.index[~np.in1d(self.index, other.index, assume_unique=True)]
        return mask

    def __contains__(self, other):
        assert(self.compatible(other))
        return np.all(np.in1d(other.index, self.index, assume_unique=True))

    def pick(self, selector, inplace=False):
        mask = self if inplace else copy.deepcopy(self)
        mask.index = mask.index[selector]
        return mask

    def constrain(self, func, return_selector=False, inplace=False):
        '''
        Parameters
        ----------
        func : callable
            selector = func(x, y, z) is used to select a subset of self.index
        '''
        ijk1 = np.c_[np.unravel_index(self.index, self.IJK, order='F') + (np.ones_like(self.index),)]
        xyz = np.dot(self.MAT, ijk1.T).T # Yes, it is xyz here!
        selector = func(xyz[:,0], xyz[:,1], xyz[:,2])
        mask = self if inplace else copy.deepcopy(self)
        mask.index = mask.index[selector]
        return mask if not return_selector else (mask, selector)

    def infer_selector(self, smaller):
        assert(smaller in self)
        selector = np.in1d(self.index, smaller.index, assume_unique=True)
        return selector

    def near(self, x, y, z, r, **kwargs):
        '''mm'''
        if np.isscalar(r):
            r = np.ones(3) * r
        func = (lambda X, Y, Z: ((X-x)/r[0])**2 + ((Y-y)/r[1])**2 + ((Z-z)/r[2])**2 < 1)
        return self.constrain(func, **kwargs)

    def ball(self, c, r, **kwargs):
        # return self.near(*c, r, **kwargs) # For python 2.7 compatibility
        return self.near(c[0], c[1], c[2], r, **kwargs)

    def cylinder(self, c, r, **kwargs):
        '''The elongated axis is represented as nan'''
        if np.isscalar(r):
            r = np.ones(3) * r
        func = (lambda X, Y, Z: np.nansum(np.c_[((X-c[0])/r[0])**2, ((Y-c[1])/r[1])**2, ((Z-c[2])/r[2])**2], axis=1) < 1)
        return self.constrain(func, **kwargs)

    def slab(self, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None, **kwargs):
        limits = np.dot(self.MAT, np.c_[np.r_[0,0,0,1], np.r_[self.IJK-1,1]])
        x1 = np.min(limits[0,:]) if x1 is None else x1
        x2 = np.max(limits[0,:]) if x2 is None else x2
        y1 = np.min(limits[1,:]) if y1 is None else y1
        y2 = np.max(limits[1,:]) if y2 is None else y2
        z1 = np.min(limits[2,:]) if z1 is None else z1
        z2 = np.max(limits[2,:]) if z2 is None else z2
        func = (lambda X, Y, Z: (x1<X)&(X<x2) & (y1<Y)&(Y<y2) & (z1<Z)&(Z<z2))
        return self.constrain(func, **kwargs)

    def dump(self, fname, dtype=None):
        files = glob.glob(fname) if isinstance(fname, six.string_types) else fname
        # return np.vstack(read_afni(f).T.flat[self.index] for f in files).T.squeeze() # Cannot handle 4D...
        data = []
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                vol = read_nii(f)
            else:
                vol = read_afni(f)
            if dtype is not None:
                vol = vol.astype(dtype)
            S = vol.shape
            T = list(range(vol.ndim))
            T[:3] = T[:3][::-1]
            # TODO: Need to check compatibility here
            data.append(vol.transpose(*T).reshape(np.prod(S[:3]),int(np.prod(S[3:])))[self.index,:])
        return np.hstack(data).squeeze()

    def undump(self, prefix, x, method='nibabel', space=None):
        if method == 'nibabel': # Much faster
            temp_file = 'tmp.%s.nii' % next(tempfile._get_candidate_names())
            vol = np.zeros(self.IJK) # Don't support int64?？
            assert(self.index.size==x.size)
            vol.T.flat[self.index] = x
            mat = np.dot(np.diag([-1,-1, 1]), self.MAT) # AFNI uses DICOM's RAI, but NIFTI uses LPI aka RAS+
            aff = nibabel.affines.from_matvec(mat[:,:3], mat[:,3])
            img = nibabel.Nifti1Image(vol, aff)
            # https://afni.nimh.nih.gov/afni/community/board/read.php?1,149338,149340#msg-149340
            # 0 (unknown) sform not defined
            # 1 (scanner) RAS+ in scanner coordinates
            # 2 (aligned) RAS+ aligned to some other scan
            # 3 (talairach) RAS+ in Talairach atlas space
            # 4 (mni) RAS+ in MNI atlas space
            if space is None:
                space = 1
            img.header['sform_code'] = space
            if prefix.endswith('.nii'):
                nibabel.save(img, prefix)
            else:
                nibabel.save(img, temp_file)
                subprocess.check_call(['3dcopy', temp_file, prefix+'+orig', '-overwrite']) # However, still TLRC inside...
                os.remove(temp_file)
        elif method == '3dUndump': # More robust
            temp_file = 'tmp.%s.txt' % next(tempfile._get_candidate_names())
            ijk = np.c_[np.unravel_index(self.index, self.IJK, order='F')]
            np.savetxt(temp_file, np.c_[ijk, x])
            subprocess.check_call(['3dUndump', '-master', self.master, '-ijk',
                '-prefix', prefix, '-overwrite', temp_file])
            os.remove(temp_file)

    @property
    def ijk(self):
        return np.c_[np.unravel_index(self.index, self.IJK, order='F')]

    @property
    def xyz(self):
        return np.dot(self.MAT[:,:3], self.ijk.T).T + self.MAT[:,3]

    @property
    def xyz_nifti(self):
        return self.xyz * np.r_[-1,-1,1] # AFNI uses DICOM's RAI, but NIFTI uses LPI aka RAS+


class BallMask(Mask):
    def __init__(self, master, c, r):
        Mask.__init__(self, master, kind='full')
        self.ball(c, r, inplace=True)


class CylinderMask(Mask):
    def __init__(self, master, c, r):
        Mask.__init__(self, master, kind='full')
        self.cylinder(c, r, inplace=True)


class SlabMask(Mask):
    def __init__(self, master, x1=None, x2=None, y1=None, y2=None, z1=None, z2=None):
        Mask.__init__(self, master, kind='full')
        self.slab(x1, x2, y1, y2, z1, z2, inplace=True)


def filter_cluster(in_file, out_file, top=None, neighbor=2):
    '''
    neighbor : int
        1 : face touch
        2 : edge touch (default, as in afni)
        3 : corner touch
    '''
    d = np.mgrid[-1:2,-1:2,-1:2]
    structure = (np.linalg.norm(d, axis=0) <= (neighbor+1)/2).astype(int)
    im, img = read_vol(in_file, return_img=True)
    label, n = ndimage.label(im, structure)
    vol = [np.sum(label==k) for k in range(1, n+1)]
    if top is not None:
        keeped = np.argsort(vol)[::-1][:top] + 1
        for k in range(1, n+1):
            if k not in keeped:
                im[label==k] = 0
    write_vol(out_file, im, base_img=img)


if __name__ == '__main__':
    pass
