#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys, os, re, glob, shlex, subprocess, time
from . import six


# Command line syntactic sugar
def expand_index_list(index_list, format=None):
    '''
    Expand a list of indices like: 1 3-5 7:10:2 8..10(2)
    '''
    flatten = []
    for x in index_list:
        if '-' in x: # 3-7
            start, end = x.split('-')
            flatten.extend(range(int(start), int(end)+1))
        elif ':' in x: # 1:10 or 1:10:2
            flatten.extend(range(*map(int, x.split(':'))))
        elif '..' in x: # 1..9 or 1..9(2)
            range_ = list(map(int, re.split('\.\.|\(|\)', x)[:3]))
            range_[1] += 1
            flatten.extend(range(*range_))
        else: # 2
            flatten.append(int(x))
    return flatten if format is None else [format % x for x in flatten]


def format_duration(duration, format='standard'):
    if format == 'short':
        units = ['d', 'h', 'm', 's']
    elif format == 'long':
        units = [' days', ' hours', ' minutes', ' seconds']
    else:
        units = [' day', ' hr', ' min', ' sec']
    values = [int(duration//86400), int(duration%86400//3600), int(duration%3600//60), duration%60]
    for K in range(len(values)): # values[K] would be the first non-zero value
        if values[K] > 0:
            break
    formatted = ((('%d' if k<len(values)-1 else '%.3f') % values[k]) + units[k] for k in range(len(values)) if k >= K)
    return ' '.join(formatted)


class ParallelCaller(object):
    def __init__(self):
        self.ps = []

    def check_call(self, cmd, **kwargs):
        '''Asynchronous check_call (return immediately).'''
        splited = isinstance(cmd, six.string_types) and not ('shell' in kwargs and kwargs['shell'])
        if splited:
            cmd = shlex.split(cmd) # Split by space, preserving quoted substrings
        if len(self.ps) == 0:
            self._start_time = time.time()
        print('>> job {0}: {1}'.format(len(self.ps), ' '.join(cmd) if splited else cmd))
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
    # Split 1D file
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
    # Combine output (1D) files
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
