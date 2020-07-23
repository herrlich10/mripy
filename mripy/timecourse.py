#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import copy, warnings
from os import path
from collections import OrderedDict
import itertools
import numpy as np
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from . import six, afni, io, utils, dicom, math


def convolve_HRF(starts, lens, TR=2, scan_time=None, HRF=None):
    if np.isscalar(lens):
        lens = lens * np.ones(len(starts))
    numout_cmd = '' if scan_time is None else f"-numout {np.ceil(scan_time/TR)}"
    HRF_cmd = '-GAM' if HRF is None else HRF
    res = utils.run(f"waver -TR {TR} {numout_cmd} {HRF_cmd} -tstim {' '.join([f'{t}%{l}' for t, l in zip(starts, lens)])}", verbose=0)
    return np.float_(res['output'])


def create_ideal(stimuli, lens, **kwargs):
    '''
    Parameters
    ----------
    stimuli : list of fname
    '''
    starts = [io.read_stim(fname) for fname in stimuli]
    n_stims = len(starts)
    n_runs = len(starts[0])
    assert(np.all(np.array([len(runs) for runs in starts]) == n_runs))
    if lens == 'alterating':
        lens = [[[] for run in range(n_runs)] for stim in range(n_stims)]
        for run in range(n_runs):
            curr_state, curr_time = -1, np.nan
            n_events = np.array([len(starts[stim][run]) for stim in range(n_stims)])
            ids = np.zeros(n_stims, dtype=int)
            while np.any(ids < n_events):
                wavefront = [(starts[stim][run][ids[stim]] if ids[stim] < n_events[stim] else np.inf) for stim in range(n_stims)]
                next_state, next_time = np.argmin(wavefront), np.min(wavefront)
                if next_state != curr_state:
                    if curr_state != -1:
                        lens[curr_state][run].append(next_time - curr_time)
                    curr_state, curr_time = next_state, next_time
                ids[curr_state] += 1
    elif np.isscalar(lens):
        lens = [[[lens]*len(starts[stim][run]) for run in range(n_runs)] for stim in range(n_stims)]
    ideal = [[convolve_HRF(starts[stim][run], lens[stim][run], **kwargs) for run in range(n_runs)] for stim in range(n_stims)]
    return ideal


def create_times(tmin, tmax, dt):
    if tmin < 0 and tmax > 0:
        times = np.r_[np.arange(-dt, tmin-dt/2, -dt)[::-1], np.arange(0, tmax+dt/2, dt)]
    else:
        times = np.arange(tmin, tmax+dt/2, dt)
    return times


def create_ERP(t, x, events, tmin=-8, tmax=16, dt=0.1, baseline=[-2,0], interp='linear'):
    '''
    t : time for each data point (can be non-contiguous)
    x : [event, feature, time]
    events : event onset time (can be on non-integer time point)
    '''
    times = create_times(tmin, tmax, dt)
    f = interpolate.interp1d(t, x, axis=-1, kind=interp, fill_value=np.nan, bounds_error=False)
    base_corr = create_base_corr_func(times, baseline=baseline)
    ERP = np.zeros(np.r_[len(events), x.shape[1:-1], len(times)].astype(int), dtype=x.dtype)
    for k, t in enumerate(events):
        ERP[k] = base_corr(f(np.arange(t+tmin, t+tmax+dt/2, dt)))
    return ERP, times


class Attributes(object):
    def __init__(self, shape):
        super().__setattr__('attributes', {})
        self.shape = shape

    shape = property(lambda self: self._shape, lambda self, x: setattr(self, '_shape', np.array(x)))

    def add(self, name, value, axis):
        assert(len(value) == self.shape[axis])
        self.attributes[name] = {'axis': axis, 'value': np.array(value)}

    def drop(self, name):
        self.attributes.pop(name)

    def drop_all_with_axis(self, axis):
        axes = axis if np.iterable(axis) else [axis]
        for axis in axes:
            for name in self.names_with_axis(axis):
                self.drop(name)

    def __getattr__(self, name):
        return self.attributes[name]['value']

    def __setattr__(self, name, value):
        if name in self.attributes:
            assert(len(value) == self.shape[self.attributes[name]['axis']])
            self.attributes[name]['value'] = np.array(value)
        else:
            # If the attribute is not added before, it will become a instance attribute
            super().__setattr__(name, value)

    def names_with_axis(self, axis):
        return [name for name, attr in self.attributes.items() if attr['axis']==axis]

    def __repr__(self):
        names = '\n'.join([f" axis={axis} ({self.shape[axis]}) | {', '.join(self.names_with_axis(axis))}" for axis in range(len(self.shape))])
        return f"<Attributes  | shape = {self.shape}\n" + names

    def __copy__(self):
        inst = type(self)(self.shape)
        inst.attributes = copy.deepcopy(self.attributes)
        return inst

    def pick(self, index, axis):
        inst = copy.copy(self)
        if np.iterable(axis):
            indices, axes = index, axis
        else:
            indices, axes = [index], [axis]
        for index, axis in zip(indices, axes):
            # Update shape first via a virtual attribute (in case there is not attribute at all)
            inst.shape[axis] = len(np.arange(inst.shape[axis])[index])
            # Update attributes belonging to the axis
            for name in inst.names_with_axis(axis):
                attr = inst.attributes[name]
                attr['value'] = attr['value'][index]
        return inst

    @classmethod
    def concatinate(cls, attributes_list, axis):
        # Concat shape along axis, and check shape compatibility along other axis
        inst = attributes_list[0]
        self = cls(inst.shape)
        self.shape[axis] = np.sum([attributes.shape[axis] for attributes in attributes_list])
        other_axes = np.r_[0:axis, axis+1:len(self.shape)]
        for attributes in attributes_list[1:]:
            assert(np.all(attributes.shape[other_axes] == self.shape[other_axes]))
        # Concat attributes along axis, and check attributes compatibility (identity) along other axis
        for name, attr in inst.attributes.items():
            if attr['axis'] == axis:
                value = np.concatenate([attributes.attributes[name]['value'] for attributes in attributes_list])
            else:
                value = attr['value']
                for attributes in attributes_list[1:]:
                    assert(np.all(attributes.attributes[name]['value'] == value))
            self.add(name, value, axis)
        return self
                
    def to_dict(self):
        return dict(attributes=self.attributes, shape=self.shape)

    @classmethod
    def from_dict(cls, d):
        self = cls(None)
        for k, v in d.items():
            setattr(self, k, v)
        return self


class Raw(utils.Savable, object):
    def __init__(self, fname, mask=None, TR=None):
        if fname is None:
            return # Skip __init__(), create an empty Raw object, and manually initialize it later.
        if mask is not None:
            self.mask =  mask if isinstance(mask, io.Mask) else io.Mask(mask)
            self.data = self.mask.dump(fname)
        else:
            self.mask = None
            self.data = io.read_vol(fname)
        self.info = {}
        self.info['sfreq'] = 1 / (afni.get_TR(fname) if TR is None else TR)
        self.info['feature_name'] = 'voxel'
        self.info['value_name'] = 'value'
        self.times = np.arange(self.n_times) * self.TR

    shape = property(lambda self: self.data.shape)
    n_features = property(lambda self: self.data.shape[0])
    n_times = property(lambda self: self.data.shape[1])
    TR = property(lambda self: 1 / self.info['sfreq'])

    @classmethod
    def from_array(cls, data, TR):
        '''
        data : 2D array, [n_features, n_times]
        TR : in sec
        '''
        self = cls(None)
        self.mask = None
        self.data = np.array(data, copy=False)
        self.info = dict(sfreq=1/TR, feature_name='voxel', value_name='value')
        self.times = np.arange(self.n_times) * self.TR
        return self
        
    def __repr__(self):
        # return f"<Raw  | {self.n_features} {self.info['feature_name']}s, {self.times[0]:.3f} - {self.times[-1]:.3f} sec, TR = {self.TR} sec, {self.n_times} TRs>"
        return f"<Raw  | {self.n_features} {self.info['feature_name']}s, {self.times[0]:.3f} - {self.times[-1]+self.TR:.3f} sec, TR = {self.TR} sec, {self.n_times} TRs>"

    def copy(self):
        return _copy(self)

    def plot(self, events=None, event_id=None, color=None, palette=None, figsize=None, event_kws=None, **kwargs):
        # Plot mean time course
        data = np.mean(self.data, axis=0)
        if events is not None: # If going to plot events, plot data in black by default
            color = 'k' if color is None else color
        if figsize is not None:
            plt.gcf().set_figwidth(figsize[0])
            plt.gcf().set_figheight(figsize[1])
        plt.plot(self.times, data, color=color, **kwargs)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal change (%)')
        # Plot events
        if events is not None:
            if event_id is None:
                event_id = _default_event_id(events)
            if palette is None: # A palette is eventually a list of colors
                palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
            id2ev = {id: [eid, ev, None] for eid, (ev, id) in enumerate(event_id.items())}
            event_kws = dict(dict(), **(event_kws if event_kws is not None else {}))
            for event in events:
                t, id = event[0], event[-1]
                id2ev[id][2] = plt.axvline(t, color=palette[id2ev[id][0]], **event_kws)
            plt.legend(*zip(*[(h, ev) for eid, ev, h in id2ev.values()]))

    def to_dict(self):
        return dict(info=self.info, data=self.data, mask=self.mask.to_dict(), times=self.times)

    @classmethod
    def from_dict(cls, d):
        self = cls(None)
        for k, v in d.items():
            setattr(self, k, v)
        self.mask = io.Mask.from_dict(self.mask)
        return self


def _copy(self):
    '''Copy all object attributes other than `data`, which is simply referred to.'''
    # TODO: .info and events etc. should be deep copied
    data = self.data
    del self.data
    inst = copy.copy(self)
    inst.data = self.data = data
    return inst


class RawCache(utils.Savable, object):
    def __init__(self, fnames, mask, TR=None, cache_file=None, force_redo=False):
        if fnames is None:
            return # Skip __init__(), create an empty RawCache object, and manually initialize it later.
        if cache_file is None or not utils.exists(cache_file, force_redo=force_redo):
            self.mask = mask if isinstance(mask, io.Mask) else io.Mask(mask)
            self.raws = [Raw(fname, mask=self.mask, TR=TR) for fname in fnames]
            if cache_file is not None:
                self.save(cache_file)
        else:
            inst = self.load(cache_file)
            self.mask = inst.mask
            self.raws = inst.raws

    n_runs = property(lambda self: len(self.raws))

    def get_raws(self, mask, ids=None):
        return_scalar = False
        if ids is None:
            ids = range(self.n_runs)
        elif not utils.iterable(ids):
            return_scalar = True
            ids = [ids]
        if isinstance(mask, six.string_types):
            mask = io.Mask(mask)
            selector = self.mask.infer_selector(mask)
        elif isinstance(mask, io.Mask):
            selector = self.mask.infer_selector(mask)
        else: # boolean index
            selector = mask
            mask = self.mask.pick(selector)
        raws = []
        for idx in ids:
            raw = self.raws[idx].copy()
            raw.data = raw.data[selector]
            raw.mask = mask
            raws.append(raw)
        return raws[0] if return_scalar else raws

    def get_epochs(self, mask, events, event_id, cache_file=None, **kwargs):
        assert(len(events) == self.n_runs)
        if cache_file is None or not utils.exists(cache_file):
            epochs = [Epochs(raw, events[idx], event_id=event_id, **kwargs) for idx, raw in enumerate(self.get_raws(mask))]
            epochs = concatinate_epochs(epochs)
            if cache_file is not None:
                epochs.save(cache_file)
        else:
            epochs = Epochs.load(cache_file)
        return epochs

    def to_dict(self):
        return dict(raws=[raw.to_dict() for raw in self.raws], mask=self.mask.to_dict())

    @classmethod
    def from_dict(cls, d):
        self = cls(None, None)
        for k, v in d.items():
            setattr(self, k, v)
        self.raws = [Raw.from_dict(raw) for raw in self.raws]
        self.mask = io.Mask.from_dict(self.mask)
        return self


def read_events(event_files):
    '''
    Read events from AFNI style (each row is a run, and each element is an occurance) stimulus timing files.

    Parameters
    ----------
    event_files : dict
        e.g., OrderedDict(('Physical/Left', '../stimuli/phy_left.txt'), ('Physical/Right', '../stimuli/phy_right.txt'), ...)

    Returns
    -------
    events_list : list of n_events-by-3 arrays
        The three columns are [start_time(sec), reserved, event_id(int)]
    event_id : dict
    '''
    if not isinstance(event_files, dict):
        event_files = OrderedDict((path.splitext(path.basename(f))[0], f) for f in event_files)
    t = []
    e = []
    event_id = OrderedDict()
    for k, (event, event_file) in enumerate(event_files.items()):
        if isinstance(event_file, str):
            eid = k + 1
        else:
            event_file, eid = event_file
        with open(event_file, 'r') as fi:
            for rid, line in enumerate(fi):
                line = line.strip()
                if not line:
                    continue
                if k == 0:
                    t.append([])
                    e.append([])
                if not line.startswith('*'):
                    t_run = np.float_(line.split())
                    t[rid].extend(t_run)
                    e[rid].extend(np.ones_like(t_run)*(eid))
        event_id[event] = eid
    events_list = []
    for rid in range(len(t)):
        events_run = np.c_[t[rid], np.zeros_like(t[rid]), e[rid]]
        sorter = np.argsort(events_run[:,0])
        events_list.append(events_run[sorter,:])
    return events_list, event_id


def events_from_dataframe(df, run, time, conditions, duration=None, event_id=None):
    if event_id is None:
        levels = itertools.product(*[df[condition].unique() for condition in conditions])
        event_id = OrderedDict([('/'.join(level), k+1) for k, level in enumerate(levels)])
    events = []
    get_event_id = lambda trial: event_id['/'.join([getattr(trial, condition) for condition in conditions])]
    get_duration = lambda trial: 0 if duration is None else getattr(trial, duration)
    for run in df[run].unique():
        events.append(np.array([[trial.time, get_duration(trial), get_event_id(trial)] for trial in df[df.run==run].itertuples()]))
    return events, event_id


def events_to_dataframe(events_list, event_id, conditions):
    id2event = {eid: event.split('/') for event, eid in event_id.items()}
    trials = []
    for rid, events in enumerate(events_list):
        for tid in range(len(events)):
            trials.append(OrderedDict([('run', rid+1), ('trial', tid+1), ('time', events[tid,0])] \
                + list(zip(conditions, id2event[events[tid,2]]))))
    return pd.DataFrame(trials)


def _default_event_id(events):
    return {f"{id:g}": id for id in np.unique(events[:,2])}


def create_base_corr_func(times, baseline=None, method=None):
    '''
    Parameters
    ----------
    time : array-like
        Sampling time for your data.
    baseline : 'none', 'all', or (tmin, tmax)
        Baseline time interval.
    '''
    if baseline is None:
        baseline = 'none'
    if method is None:
        method = np.nanmean
    if isinstance(baseline, str):
        if baseline.lower() == 'none':
            return lambda x: x
        elif baseline.lower() == 'all':
            return lambda x: x - np.nanmean(x, axis=-1, keepdims=True)
    else: # `baseline` is like (tmin, tmax)
        tmin = times[0] if baseline[0] is None else baseline[0]
        tmax = times[-1] if baseline[1] is None else baseline[1]
        times = np.array(times)
        time_sel = (tmin<=times) & (times<=tmax)
        if method in [np.nanmean, np.nanmedian]:
            def base_corr(x):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    return x - method(x[...,time_sel], axis=-1, keepdims=True)
            return base_corr
        else:
            return lambda x: x - method(x[...,time_sel], axis=-1, keepdims=True)


class Epochs(utils.Savable, object):
    def __init__(self, raw, events, event_id=None, tmin=-5, tmax=15, baseline=(-2,0), dt=0.1, interp='linear', hamm=None, conditions=None):
        if raw is None:
            return # Skip __init__(), create an empty Epochs object, and manually initialize it later.
        self.events = events
        self.event_id = _default_event_id(events) if event_id is None else event_id
        self.info = raw.info.copy() # Caution: container datatype is assigned by reference by default
        self.info['sfreq'] = 1 / dt
        self.info['tmin'] = tmin
        self.info['tmax'] = tmax
        self.info['baseline'] = baseline
        self.info['interp'] = interp
        self.info['hamm'] = hamm
        if isinstance(conditions, six.string_types):
            conditions = conditions.split('/')
        self.info['conditions'] = conditions
        self.info['condition'] = None
        self.times = create_times(tmin, tmax, dt)
        x = raw.data
        if hamm is not None:
            h = signal.hamming(hamm)
            h = h/np.sum(h)
            x = signal.filtfilt(h, [1], x, axis=-1)
        f = interpolate.interp1d(raw.times, x, axis=-1, kind=interp, fill_value=np.nan, bounds_error=False)
        base_corr = create_base_corr_func(self.times, baseline=baseline)
        self.data = np.zeros([events.shape[0], raw.n_features, len(self.times)], dtype=raw.data.dtype)
        for k, t in enumerate(events[:,0]):
            self.data[k] = base_corr(f(np.arange(t+tmin, t+tmax+dt/2, dt)))
        self.attr = Attributes(shape=self.shape)

    shape = property(lambda self: self.data.shape)
    n_events = property(lambda self: self.data.shape[0])
    n_features = property(lambda self: self.data.shape[1])
    n_times = property(lambda self: self.data.shape[2])

    @classmethod
    def from_array(cls, data, TR=None, tmin=None, baseline=(-2,0), events=None, event_id=None, conditions=None):
        self = cls(None, None)
        self.data = np.array(data, copy=False)
        assert(self.data.ndim == 3) # n_events x n_features x n_times
        self.info = {}
        self.info['sfreq'] = 1 if TR is None else 1/TR
        self.info['feature_name'] = 'voxel'
        self.info['value_name'] = 'value'
        self.info['tmin'] = 0 if tmin is None else tmin
        self.times = self.info['tmin'] + np.arange(self.data.shape[-1])/self.info['sfreq']
        self.info['tmax'] = self.times[-1]
        self.info['baseline'] = baseline
        self.info['interp'] = None
        self.info['hamm'] = None
        self.info['conditions'] = conditions
        self.info['condition'] = None
        if events is not None:
            assert(self.data.shape[0] == events.shape[0])
            self.events = events
        else:
            self.events = np.zeros([self.data.shape[0], 3])
        if event_id is None:
            self.event_id = {'Event': 0} if events is None else _default_event_id(self.events)
        else:
            assert(np.all(np.in1d(self.events[:,2], list(event_id.values()))))
            self.event_id = event_id
        base_corr = create_base_corr_func(self.times, baseline=baseline)
        self.data = base_corr(self.data)
        self.attr = Attributes(shape=self.shape)
        return self

    def __repr__(self):
        event_str = '\n '.join(f"'{ev}': {np.sum(self.events[:,2]==id)}" for ev, id in self.event_id.items())
        return f"<Epochs  | {self.n_events:4d} events, {self.n_features} {self.info['feature_name']}s, {self.times[0]:.3f} - {self.times[-1]:.3f} sec, baseline {self.info['baseline']}, hamm = {self.info['hamm']},\n {event_str}>"
  
    # https://github.com/mne-tools/mne-python/blob/master/mne/utils/mixin.py
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.pick(*item)
        else:
            return self.pick(event=item)

    def add_event_attr(self, name, value):
        self.attr.add(name, value, axis=0)

    def add_feature_attr(self, name, value):
        self.attr.add(name, value, axis=1)

    def pick(self, event=None, feature=None, time=None):
        inst = self.copy()
        # Select event
        if event is None:
            sel_event = slice(None)
        elif isinstance(event, six.string_types) or (utils.iterable(event) and isinstance(event[0], six.string_types)): # e.g., 'Physical/Left'
            sel_event = self._partial_match_event(event)
            self.info['condition'] = event if isinstance(event, six.string_types) else ' | '.join(event) 
        else:
            sel_event = event
        inst.events = inst.events[sel_event]
        inst.event_id = {ev: id for ev, id in inst.event_id.items() if id in inst.events[:,2]}
        # Select feature
        if feature is None:
            sel_feature = slice(None)
        else:
            sel_feature = feature
        # Select time
        if time is None:
            sel_time = slice(None)
        elif utils.iterable(time) and len(time) == 2:
            sel_time = (time[0] <= self.times) & (self.times <= time[1])
        else:
            sel_time = time
        inst.times = inst.times[sel_time]
        inst.info['tmin'] = inst.times[0]
        inst.info['tmax'] = inst.times[-1]
        # inst.info['sfreq'] = ?
        # Make 3D selection
        inst.data = inst.data[sel_event][:,sel_feature][...,sel_time]
        inst.attr = inst.attr.pick([sel_event, sel_feature, sel_time], axis=[0, 1, 2])
        return inst

    def copy(self):
        inst = _copy(self)
        inst.attr = copy.copy(self.attr)
        return inst

    def drop_events(self, ids):
        inst = self.copy()
        inst.data = np.delete(inst.data, ids, axis=0)
        inst.events = np.delete(inst.events, ids, axis=0)
        inst.event_id = {ev: id for ev, id in inst.event_id.items() if id in inst.events[:,2]} # TODO: Need refactor
        inst.attr.pick(np.delete(np.arange(self.n_events), ids), axis=0)
        return inst

    def _partial_match_event(self, keys):
        if isinstance(keys, six.string_types):
            keys = [keys]
        matched = []
        for key in keys:
            key_set = set(key.split('/'))
            matched_id = [id for ev, id in self.event_id.items() if key_set.issubset(ev.split('/'))]
            matched.append(np.atleast_2d(np.in1d(self.events[:,2], matched_id)))
        return np.any(np.vstack(matched), axis=0)

    def apply_baseline(self, baseline):
        base_corr = create_base_corr_func(self.times, baseline=baseline)
        inst = self.copy()
        inst.data = base_corr(inst.data)
        inst.info['baseline'] = baseline
        return inst

    def aggregate(self, event=True, feature=False, time=False, method=np.nanmean, keepdims=np._globals._NoValue, return_index=False):
        axes = ((0,) if event else ()) + ((1,) if feature else ()) + ((2,) if time else ())
        values = method(self.data, axis=axes, keepdims=keepdims)
        events = self.events[:,2] if not event else None
        features = np.arange(self.n_features) if not feature else -1
        times = self.times if not time else np.mean(self.times)
        return (values, events, features, times) if return_index else values

    def average(self, feature=True, time=False, method=np.nanmean, error='bootstrap', ci=95, n_boot=1000, condition=None):
        '''
        Average data over event (and optionally feature and/or time) dimensions, and return an Evoked object.
        '''
        x, _, _, times = self.aggregate(event=False, feature=feature, time=time, method=method, return_index=True)
        data = method(x, axis=0)
        nave = x.shape[0]
        if error == 'bootstrap':
            error_type = (error, ci)
            boot_dist = method(x[np.random.randint(nave, size=[nave, n_boot]),...], axis=0)
            error = np.percentile(boot_dist, [50-ci/2, 50+ci/2], axis=0) - data
        elif error == 'instance':
            error_type = (error, None)
            error = x - data
        if condition is None:
            condition = self.info['condition']
        evoked = Evoked(self.info, data, nave, times, error=error, error_type=error_type, condition=condition)
        return evoked

    # def running_average(self, win_size, overlap=0.5, time=False, method=np.nanmean):
    #     data = []
    #     for start in range(0, self.n_events, int(win_size*overlap)):
    #         x, _, _, times = self.pick(event=slice(start, start+win_size)).aggregate(feature=True, time=time, method=method, return_index=True)
    #         data.append(x)
    #     evoked = Evoked2D(self.info, np.array(data), win_size, features, times)
    
    def transform(self, feature_name, feature_values, transformer):
        '''
        Transform the data into a new Epochs object.
        E.g., epochs.transform('depth', depth, tc.wcutter(depth, linspace(0, 1, 20), win_size=0.2, win_func='gaussian', exclude_outer=True))
        '''
        assert(len(feature_values)==self.n_features)
        inst = self.copy()
        inst.info['feature_name'] = feature_name
        inst.info['feature_values'] = feature_values
        inst.data = transform(inst.data, *transformer, axis=1)
        # TODO: update inst.attr
        return inst


    def summary(self, event=False, feature=True, time=False, method=np.nanmean, attributes=None):
        '''
        Summary data as a pandas DataFrame.
        '''
        assert(self.info['conditions'] is not None)
        dfs = []
        for ev in self.event_id:
            # import pdb; pdb.set_trace()
            x, _, features, times = self.pick(ev).aggregate(event=event, feature=feature, time=time, method=method, keepdims=True, return_index=True)
            events = ev.split('/')
            df = OrderedDict()
            for k, condition in enumerate(self.info['conditions']):
                df[condition] = events[k]
            df[self.info['feature_name']] = np.tile(np.repeat(features, x.shape[2]), x.shape[0])
            df['time'] = np.tile(times, np.prod(x.shape[:2]))
            df[self.info['value_name']] = x.ravel()
            dfs.append(pd.DataFrame(df))
        df = pd.concat(dfs, ignore_index=True)
        if attributes is not None:
            for name, value in attributes.items():
                df[name] = value
        return df

    def plot(self, hue=None, style=None, row=None, col=None, hue_order=None, style_order=None, row_order=None, col_order=None,
        palette=None, dashes=None, figsize=None, bbox_to_anchor=None, subplots_kws=None, average_kws=None, **kwargs):
        assert(self.info['conditions'] is not None)
        conditions = OrderedDict([(condition, np.unique(levels)) for condition, levels in zip(self.info['conditions'], np.array([ev.split('/') for ev in self.event_id]).T)])
        con_sel = [[hue, style, row, col].index(condition) for condition in conditions]
        n_rows = 1 if row is None else len(conditions[row])
        n_cols = 1 if col is None else len(conditions[col])
        subplots_kws = dict(dict(sharex=True, sharey=True, figsize=figsize, constrained_layout=True), **({} if subplots_kws is None else subplots_kws))
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, **subplots_kws)
        row_order = [None] if row is None else (conditions[row] if row_order is None else row_order)
        col_order = [None] if col is None else (conditions[col] if col_order is None else col_order)
        hue_order = [None] if hue is None else (conditions[hue] if hue_order is None else hue_order)
        style_order = [None] if style is None else (conditions[style] if style_order is None else style_order)
        if palette is None:
            palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
        elif isinstance(palette, str):
            palette = sns.color_palette(palette)
        if dashes is None:
            dashes = ['-', '--', ':', '-.']
        average_kws = dict(dict(), **({} if average_kws is None else average_kws))
        for rid, row_val in enumerate(row_order):
            for cid, col_val in enumerate(col_order):
                plt.sca(axs[rid,cid])
                show_info = True
                for hid, hue_val in enumerate(hue_order):
                    for sid, style_val in enumerate(style_order):
                        event = '/'.join(np.array([hue_val, style_val, row_val, col_val])[con_sel])
                        label = '/'.join([s for s in [hue_val, style_val] if s is not None])
                        event_sel = self._partial_match_event(event)
                        self[event_sel].average(**average_kws).plot(color=palette[hid], ls=dashes[sid], 
                            label=label, show_n='label' if label else 'info', info=show_info, **kwargs)
                        show_info = False
                        plt.axhline(0, color='gray', ls='--')
                plt.title('/'.join([s for s in [row_val, col_val] if s is not None]))
                if rid < n_rows-1:
                    plt.xlabel('')
                if cid > 0:
                    plt.ylabel('')
                if label:
                    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=None if bbox_to_anchor is None else 'center left')
        sns.despine()

    def to_dict(self):
        return dict(info=self.info, data=self.data, events=self.events, event_id=self.event_id, 
            times=self.times, attr=self.attr.to_dict())

    @classmethod
    def from_dict(cls, d):
        self = cls(None, None)
        for k, v in d.items():
            setattr(self, k, v)
        self.attr = Attributes.from_dict(self.attr)
        return self


def concatinate_epochs(epochs_list, axis=0):
    inst = epochs_list[0].copy()
    inst.data = np.concatenate([epochs.data for epochs in epochs_list], axis=axis)
    inst.attr = Attributes.concatinate([epochs.attr for epochs in epochs_list], axis=axis)
    if axis == 0: # Concat events
        inst.events = np.concatenate([epochs.events for epochs in epochs_list], axis=0)
        inst.event_id = {k: v for epochs in epochs_list for k, v in epochs.event_id.items()}
    elif axis == 1: # Concat voxels
        assert(np.all([np.all(epochs.events == epochs_list[0].events) for epochs in epochs_list[1:]]))
        assert(np.all([(epochs.event_id == epochs_list[0].event_id) for epochs in epochs_list[1:]]))
    return inst


def group_epochs(epochs_list):
    inst = epochs_list[0].copy()
    inst.data = np.concatenate([epochs[event].aggregate(event=True, feature=True, keepdims=True) 
        for epochs in epochs_list for event in inst.event_id], axis=0)
    inst.events = np.array(list(inst.event_id.values())*len(epochs_list))
    inst.events = np.c_[np.zeros((len(inst.events),2)), inst.events]
    inst.attr = Attributes(shape=inst.shape)
    return inst


# def transform(data, idx_gen, weight_gen=None, agg_func=None, axis=0):
#     if agg_func is None:
#         agg_func = np.nanmean if np.any(np.isnan(data)) else np.mean
#     if weight_gen is None:
#         ##### BUG!!!
#         return np.concatenate([agg_func(data[idx,...], axis=axis, keepdims=True) for idx in idx_gen], axis=axis)
#     else:
#         return np.concatenate([agg_func(data[idx,...]*weight, axis=axis, keepdims=True) 
#             for idx, weight in zip(idx_gen, weight_gen)], axis=axis)


def cut(data, val, bins, **kwargs):
    return transform(data, *cutter(val, bins), **kwargs)

def cutter(val, bins):
    idx_gen = ((bins[k]<val)&(val<=bins[k+1]) for k in range(len(bins)-1))
    return idx_gen, None


def qcut(data, val, q, **kwargs):
    return transform(data, *qcutter(val, q), **kwargs)

def qcutter(val, q):
    bins = np.percentile(val, q)
    idx_gen = ((bins[k]<val)&(val<=bins[k+1]) for k in range(len(bins)-1))
    return idx_gen, None


def wcut(data, val, v, win_size, win_func=None, exclude_outer=False, **kwargs):
    return transform(data, *wcutter(val, v, win_size, win_func=win_func, exclude_outer=exclude_outer), **kwargs)

def wcutter(val, v, win_size, win_func=None, exclude_outer=False):
    r = win_size/2
    if win_func == 'gaussian':
        sigma = win_size/4
        win_func = lambda c, x: np.exp(-(x-c)**2/sigma**2)
    if exclude_outer:
        bins = [[max(v[0], vv-r), min(v[-1], vv+r)] for vv in v]
    else:
        bins = [[vv-r, vv+r] for vv in v]
    idx_gen = ((lower<val)&(val<=upper) for lower, upper in bins)
    if win_func is not None:
        weight_gen = (win_func(vv, val[(lower<val)&(val<=upper)]) for vv, (lower, upper) in zip(v, bins))
    else:
        weigth_gen = None
    return idx_gen, weight_gen


class Evoked(object):
    def __init__(self, info, data, nave, times, error=None, error_type=None, condition=None):
        '''
        This class is intended for representing only a single subject and a single condition.
        Group data with multiple conditions together could be flexibly handled by `pd.DataFrame`.
        For the later purpose, instead of Epochs.average() -> Evoked, 
        consider using Epochs.summary() -> pd.DataFrame.
        '''
        self.info = info
        self.times = times
        self.data = data
        self.info['nave'] = nave
        self.error = error
        self.info['error_type'] = error_type
        self.info['condition'] = condition

    shape = property(lambda self: self.data.shape)

    def plot(self, color=None, error=True, info=True, error_kws=None, show_n='info', **kwargs):
        n_str = rf"$n={self.info['nave']}$"
        label = kwargs.pop('label') if 'label' in kwargs else self.info['condition']
        if show_n == 'label':
            label += rf"  ({n_str})"
        line = plt.plot(self.times, self.data, color=color, label=label, **kwargs)[0]
        if color is None:
            color = line.get_color()
        if error:
            if self.info['error_type'][0] == 'instance':
                error_kws = dict(dict(alpha=0.3, lw=0.3), **(error_kws if error_kws is not None else {}))
                plt.plot(self.times, (self.data+self.error).T, color=color, **error_kws)
            else:
                error_kws = dict(dict(alpha=0.3), **(error_kws if error_kws is not None else {}))
                plt.fill_between(self.times, self.data+self.error[0], self.data+self.error[1], color=color, **error_kws)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal change (%)')
        if info:
            ci = self.info['error_type'][1]
            err_str = r'$\pm SEM$' if ci == 68 else rf'${ci}\%\ CI$'
            s = rf"{n_str}$\quad${err_str}" if show_n == 'info' else err_str
            plt.text(0.95, 0.05, s, ha='right', transform=plt.gca().transAxes)


# class Evoked2D(object):
#     def __init__(self, info, data, nave, features, times):
#         self.info = info
#         self.features = features
#         self.times = times
#         self.data = data
#         self.info['nave'] = nave

#     @classmethod
#     def from_evoked_list(cls, evoked_list, features):
#         evoked = evoked_list[0]
#         data = np.array([evoked.data for evoked in evoked_list])
#         return cls(evoked.info, data, evoked.info['nave'], features, evoked.times)


if __name__ == '__main__':
    pass
    # stim_dir = '/Volumes/raid78/ccqian/LaminarRivalry2/LGN/S01/stimuli'
    # stim_files = [f"{stim_dir}/L_replay.txt", f"{stim_dir}/R_replay.txt"]
    # tc.create_ideal(stim_files, 'alterating')
