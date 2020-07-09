#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, os, re, glob, shlex, string
import subprocess, multiprocessing, ctypes, time, uuid
import json
import tables, warnings
from datetime import datetime
from itertools import chain
from collections import OrderedDict
from contextlib import contextmanager
from os import path
import numpy as np
from deepdish import io as dio
from . import six, afni


from .paraproc import format_duration
from .paraproc import cmd_for_exec, cmd_for_disp, run
from .paraproc import PooledCaller, SharedMemoryArray


package_dir = path.abspath(path.dirname(__file__))


def has_ants():
    pass


def has_N4():
    pass


def has_hcp_retino_docker():
    try:
        run(f"docker image list", goal_pattern='nben/neuropythy')
        return True
    except RuntimeError:
        print('>> Please check whether docker desktop is running.')
        return False


# Command line syntactic sugar
def expand_index_list(index_list, format=None):
    '''Expand a list of indices like: 1 3-5 7:10:2 8..10(2)
    '''
    flatten = []
    for x in index_list:
        if '-' in x: # 3-7 => 3,4,5,6,7 and 3-7-2 => 3,5,7
            range_ = list(map(int, x.split('-')))
            range_[1] += 1
            flatten.extend(range(*range_))
        elif ':' in x: # 3:7 => 3,4,5,6 and 3:7:2 => 3,5
            flatten.extend(range(*map(int, x.split(':'))))
        elif '..' in x: # 3..7 => 3,4,5,6,7 and '3..7(2)' => 3,5,7 (note that quotes are required in the shell because of the parentheses)
            range_ = list(map(int, re.split(r'\.\.|\(|\)', x)[:3])) # [:3] to get rid of the 4th, empty element
            range_[1] += 1
            flatten.extend(range(*range_))
        else: # 3
            flatten.append(int(x))
    return flatten if format is None else [format % x for x in flatten]




def select_and_replace_affix(glob_pattern, old_affix, new_affix):
    old_files = glob.glob(glob_pattern)
    pattern = re.compile(re.escape(old_affix))
    new_files = [re.sub(pattern, new_affix, f) for f in old_files]
    return old_files, new_files


def iterable(x): # Like the builtin `callable`
    try:
        iter(x)
        return not isinstance(x, six.string_types)
    except TypeError:
        return False


def contain_wildcard(fname):
    return any([w in fname for w in ['*', '?']])


class FilenameManager(object):
    @classmethod
    def from_glob(cls, fmt, **kwargs):
        self = cls(fmt, **kwargs)
        files = self.glob()
        self.parse(files)
        return self

    @classmethod
    def fmt2kws(cls, fmt, **kwargs):
        kws = {t[1]: kwargs.get(t[1], '*') for t in string.Formatter().parse(fmt) if t[1] is not None}
        return kws

    def __init__(self, fmt, **kwargs):
        self.fmt = fmt
        self.kws = self.fmt2kws(self.fmt, **kwargs)

    def glob(self, **kwargs):
        files = sorted(glob.glob(self.fmt.format(**dict(self.kws, **kwargs))))
        return files

    def format(self, fmt=None, keepdims=False, **kwargs):
        if fmt is None:
            fmt = self.fmt
        kws = self.fmt2kws(fmt, **dict(self.kws, **kwargs))
        kws = {k: (v if iterable(v) else [v]) for k, v in kws.items()}
        if not kws: # fmt contains no variable
            return fmt
        key2len = {k: len(v) for k, v in kws.items()}
        kws_lens = np.fromiter(key2len.values(), dtype=int)
        N = np.max(kws_lens)
        if not np.all(kws_lens[kws_lens>1] == N):
            raise ValueError('>> len() for different kwargs does not match.')
        files = []
        for n in range(N):
            kws_n = {k: (v[n] if key2len[k]>1 else v[0]) for k, v in kws.items()}
            files.append(fmt.format(**kws_n))
        if len(files) == 1 and not keepdims:
            return files[0]
        else:
            return files

    def _parse_one(self, fname):
        '''
        References
        ----------
        1. https://stackoverflow.com/questions/11844986/convert-or-unformat-a-string-to-variables-like-format-but-in-reverse-in-p
        '''
        fmt_escape = re.escape(self.fmt)
        fmt_regex = re.sub(r'\\{(.+?)\\}', r'(?P<\1>.+)', fmt_escape)
        fmt_regex_ = re.sub(r'\\{(.+?)\\}', r'(?P<\1>.+?)', fmt_escape)
        match = re.search(fmt_regex, fname)
        if match:
            kws = match.groupdict()
            match = re.search(fmt_regex_, fname)
            kws_ = match.groupdict()
            if kws != kws_:
                raise ValueError('>> Filename pattern is ambiguous.')
        else:
            raise ValueError('>> Filename pattern does not match provided files.')
        return kws

    def parse(self, files, multi_value='list'):
        '''
        files : list of filenames
        multi_value : {'wildcard', 'list'}
        '''
        all_kws = [self._parse_one(f) for f in files]
        kws = {}
        for key in all_kws[0]:
            values = [all_kws[k][key] for k in range(len(all_kws))]
            if np.all(np.array(values) == values[0]):
                kws[key] = all_kws[0][key]
            else:
                if multi_value == 'wildcard':
                    kws[key] = '*'
                elif multi_value == 'list':
                    kws[key] = values
        self.kws.update(kws)


def fname_with_ext(fname, ext):
    for k in reversed(range(len(ext)+1)):
        if fname.endswith(ext[:k]):
            return fname[:len(fname)-k] + ext


def exists(fname, force_redo=False):
    '''Lazy evaluation the body only if fname not already exists.'''
    if not path.exists(fname) or force_redo:
        return False
    else:
        print('>> Reuse existing "{0}"'.format(fname))
        return True


@contextmanager
def cd(d):
    old_dir = os.getcwd()
    os.chdir(d)
    yield
    os.chdir(old_dir)


class CacheManager(object):
    def __init__(self, persistent_file=None, ignore_init_run=True):
        self.persistent_file = 'cache_manager.json' if persistent_file is None else persistent_file
        self.ignore_init_run = ignore_init_run
        if self.persistent_file and path.exists(self.persistent_file):
            self.load_contexts()
        else:
            self.file_contexts = {}

    def load_contexts(self):
        if self.persistent_file:
            with open(self.persistent_file, 'r') as f:
                self.file_contexts = json.load(f)

    def save_contexts(self):
        if self.persistent_file:
            with open(self.persistent_file, 'w') as f:
                json.dump(self.file_contexts, f)

    def watch_files_updated(self, fname, watch_files):
        if watch_files is None:
            return False
        if isinstance(watch_files, six.string_types):
            watch_files = [watch_files]
        mtimes = self.file_contexts[fname]['watch_file_mtimes']
        flag = False
        for watch_file in watch_files:
            mtime = path.getmtime(watch_file)
            if watch_file not in mtimes:
                mtimes[watch_file] = mtime
                if not self.ignore_init_run:
                    print('>> File "{0}" is being watched'.format(watch_file))
                    flag = True
            elif mtime != mtimes[watch_file]:
                print('>> File "{0}" has been modified at {1}'.format(watch_file, 
                    datetime.fromtimestamp(mtime).strftime('%H:%M:%S %Y-%m-%d')))
                mtimes[watch_file] = mtime
                flag = True
        return flag

    def kwargs_updated(self, fname, kwargs):
        if not kwargs:
            return False
        kws = self.file_contexts[fname]['kwargs']
        flag = False
        for k, v in kwargs.items():
            if k not in kws:
                kws[k] = v
                if not self.ignore_init_run:
                    print('>> Variable "{0}" is being watched'.format(k))
                    flag = True
            elif v != kws[k]:
                print('>> Variable "{0}" has been modified from {1} -> {2}'.format(k, kws[k], v))
                kws[k] = v
                flag = True
        return flag

    def exists(self, fname, watch_files=None, force_redo=False, **kwargs):
        if fname not in self.file_contexts:
            self.file_contexts[fname] = {'watch_file_mtimes': {}, 'kwargs': {}}
        mtimes_updated = self.watch_files_updated(fname, watch_files)
        kws_updated = self.kwargs_updated(fname, kwargs)
        self.save_contexts()
        if not path.exists(fname) or mtimes_updated or kws_updated or force_redo:
            # Execute the code block
            return False
        else:
            print('>> Reuse existing "{0}"'.format(fname))
            return True


def temp_prefix(prefix='tmp_', n=4, suffix='.'):
    return prefix + uuid.uuid4().hex[:n] + suffix


def temp_folder(parent=None):
    folder = temp_prefix(suffix='')
    if parent is not None:
        folder = path.join(parent, folder)
    if not path.exists(folder):
        os.makedirs(folder)
    return folder


class ParallelCaller(object):
    def __init__(self):
        self.ps = []

    def check_call(self, cmd, **kwargs):
        '''
        Asynchronous check_call (launch and return immediately).
        Multiple commands can be stitched sequentially with ";" in linux/mac only if shell=True.
        '''
        cmd = cmd_for_exec(cmd, kwargs)
        if len(self.ps) == 0:
            self._start_time = time.time()
        print('>> job {0}: {1}'.format(len(self.ps), cmd_for_disp(cmd)))
        p = subprocess.Popen(cmd, **kwargs)
        self.ps.append(p)

    def wait(self):
        '''Wail for all parallel processes to finish (= wait for the slowest one).'''
        codes = [p.wait() for p in self.ps]
        duration = time.time() - self._start_time
        print('>> All {0} jobs done in {1}.'.format(len(self.ps), format_duration(duration)))
        self.ps = [] # Reset subprocess list
        return codes



def parallel_1D(cmd, in_file, prefix, n_jobs=1, combine_output=True, **kwargs):
    '''
    Parameters
    ----------
        cmd : str
            The command for parallel execution, must contain two placeholders
            {prefix} and {in_file}, which will be substituded with ''.format().
            As expected, other {} must be escaped as {{}}.
            For example,
                parallel_1D("3dTcat -prefix {prefix} -overwrite \
                    {in_file}'{{0:9}}'", 'xyz_list.1D', 'test', n_jobs=4)
    '''
    # Count the number of lines
    with open(in_file) as fi:
        n_lines = sum(1 for line in fi if line.strip() and not line.startswith('#'))
    # Split input 1D file
    files = [in_file+'-p{0:03d}'.format(j) for j in range(n_jobs)]
    job_size = (n_lines-1)//n_jobs + 1 # ceil
    with open(in_file) as fi:
        lines = (line for line in fi if line.strip() and not line.startswith('#'))
        for j in range(n_jobs):
            with open(files[j], 'w') as fj:
                for k in range(job_size):
                    try:
                        fj.write(next(lines))
                    except StopIteration:
                        pass
    # Parallel call
    prefixs = [prefix+'-p{0:03d}'.format(j) for j in range(n_jobs)]
    pc = ParallelCaller() # Execute n_jobs instances of cmd in parallel
    for j in range(n_jobs):
        pc.check_call(cmd.format(in_file=files[j], prefix=prefixs[j]), **kwargs)
    pc.wait() # Wait for all instances
    for j in range(n_jobs):
        os.remove(files[j])
    # Combine output 1D files
    if combine_output:
        outputs = glob.glob(prefix+'-*')
        match = re.match('{0}(.+)'.format(prefixs[0]), outputs[0])
        suffix = match.group(1)
        with open(prefix+suffix, 'w') as fo:
            for j in range(n_jobs):
                with open(outputs[j]) as fi:
                    for line in fi:
                        if line.strip() and not line.startswith('#'):
                            fo.write(line)
                os.remove(outputs[j])


def factorize(n):
    result = []
    for k in chain([2], range(3, n+1, 2)): # Test 2 and all odd integers between 3 and n
        while not n % k: # If n/k is an integer
            n = n // k
            result.append(k) # k is a prime factor
            if n == 1:
                return result
    return [1]


def parallel_3D(cmd, in_file, prefix, n_jobs=1, schema=None, fname_mapper=None, combine_output=True, **kwargs):
    '''
    Parameters
    ----------
    fname_mapper : dict or callable
    '''
    def n_factorize(factors, n):
        factors = sorted(factors, reverse=True) + [1]*(n-1)
        result = [1]*n
        for factor in factors:
            result[np.argmin(result)] *= factor
        return sorted(result, reverse=True)
    if fname_mapper is None:
        fname_mapper = lambda s: s
    elif isinstance(fname_mapper, dict):
        fname_map = fname_mapper
        def fname_mapper(s):
            for pattern, replace in fname_map.items():
                new_str, n = re.subn(pattern ,replace, s)
                if n > 0:
                    return new_str
            else:
                return s
    output_dir = path.dirname(prefix)
    # Determine split schema
    dims = afni.get_head_dims(in_file)[:3]
    if schema is None:
        factors = factorize(n_jobs)
        schema = np.r_[n_factorize(factors, 2), 1]
    dim_labels = ['RL', 'AP', 'IS']
    dim_order = np.argsort(dims)[::-1]
    split_schema = []
    for k in range(len(schema)):
        split_schema.append({'dim': dim_labels[dim_order[k]], 'splits': []})
        D = dims[dim_order[k]]
        delta = D // schema[k]
        for s in range(schema[k]):
            if s == schema[k] - 1:
                split_schema[k]['splits'].append([s*delta, 0])
            else:
                split_schema[k]['splits'].append([s*delta, D-(s+1)*delta])
    print('>> Split the volume into {0}x{1}x{2}={3} chunks...'.format(schema[0], schema[1], schema[2], np.prod(schema)))
    # Split input 3D file
    tmp_ = temp_prefix()
    s0, e0 = split_schema[0]['dim']
    s1, e1 = split_schema[1]['dim']
    s2, e2 = split_schema[2]['dim']
    split_params = {}
    pc = ParallelCaller()
    for m, (ns0, ne0) in enumerate(split_schema[0]['splits']):
        for n, (ns1, ne1) in enumerate(split_schema[1]['splits']):
            for l, (ns2, ne2) in enumerate(split_schema[2]['splits']):
                label = path.join(output_dir, '{0}{1:02d}_{2:02d}_{3:02d}'.format(tmp_, m, n, l))
                split_params[label] = (ns0, ne0, ns1, ne1, ns2, ne2, m, n, l)
                pc.check_call('''
                    3dZeropad -{s0} -{ns0} -{e0} -{ne0} -{s1} -{ns1} -{e1} -{ne1} -{s2} -{ns2} -{e2} -{ne2} \
                        -prefix {label} -overwrite {in_file}
                    '''.format(s0=s0, ns0=ns0, e0=e0, ne0=ne0, s1=s1, ns1=ns1, e1=e1, ne1=ne1, 
                        s2=s2, ns2=ns2, e2=e2, ne2=ne2, label=label, in_file=in_file))
    pc.wait()
    # Parallel call
    if not isinstance(cmd, six.string_types):
        cmd = ' '.join(cmd)
    for label in split_params:
        pc.check_call(cmd.format(in_file='{0}+orig'.format(label), prefix='{0}_out'.format(label)), **kwargs)
    pc.wait()
    # Combine output 3D files
    if combine_output:
        for label, (ns0, ne0, ns1, ne1, ns2, ne2, m, n, l) in split_params.items():
            fi = fname_mapper('{0}_out+orig'.format(label))
            pc.check_call('''
                3dZeropad -{s0} {ns0} -{e0} {ne0} -{s1} {ns1} -{e1} {ne1} -{s2} {ns2} -{e2} {ne2} \
                    -prefix {label}_pad -overwrite {fi}
                '''.format(s0=s0, ns0=ns0, e0=e0, ne0=ne0, s1=s1, ns1=ns1, e1=e1, ne1=ne1, 
                    s2=s2, ns2=ns2, e2=e2, ne2=ne2, label=label, fi=fi))
        pc.wait()
        glob_pattern = path.join(output_dir, '{0}*_pad+orig.HEAD'.format(tmp_))
        subprocess.check_call('''
            3dTstat -sum -prefix {prefix} -overwrite "{glob_pattern}"
            '''.format(prefix=prefix, glob_pattern=glob_pattern), shell=True)
    # Remove temp files
    for f in glob.glob(path.join(output_dir, tmp_+'*')):
        os.remove(f)


class Savable(object):
    def save(self, fname):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=tables.NaturalNameWarning)
            dio.save(fname, self.to_dict())

    @classmethod
    def load(cls, fname):
        return cls.from_dict(dio.load(fname))


class Savable2(object):
    def save(self, fname):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=tables.NaturalNameWarning)
            dio.save(fname, self.to_dict())

    def load(self, fname):
        self.from_dict(dio.load(fname))