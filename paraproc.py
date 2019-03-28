#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 herrlich10
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

from __future__ import print_function, division, absolute_import, unicode_literals
import sys, shlex, time
import subprocess, multiprocessing, ctypes, time, uuid
import numpy as np

__author__ = 'herrlich10 <herrlich10@gmail.com>'
__version__ = '0.1.5'

# The following are copied from six
# =================================
if sys.version_info[0] == 3:
    string_types = (str,)
else:
    string_types = (basestring,)

def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper
# =================================


def cmd_for_exec(cmd, cmd_kws):
    '''
    Format cmd appropriately for execution according to whether shell=True.

    Split the cmd string into a list, if not shell=True.
    Join the cmd list into a string, if shell=True.
    Do nothing to callable.
    '''
    if not callable(cmd):
        if 'shell' in cmd_kws and cmd_kws['shell']: # cmd string is required
            if not isinstance(cmd, string_types):
                cmd = ' '.join(cmd)
        else: # cmd list is required
            if isinstance(cmd, string_types):
                cmd = shlex.split(cmd) # Split by space, preserving quoted substrings
    return cmd


def cmd_for_disp(cmd):
    '''
    Format cmd for printing.
    '''
    if isinstance(cmd, list):
        return ' '.join(cmd)
    else:
        return cmd


def format_duration(duration, format='standard'):
    '''
    Format duration (in seconds) in a more human friendly way.
    '''
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


class PooledCaller(object):
    '''
    Execute multiple command line programs, as well as python callables, 
    asynchronously and parallelly across a pool of processes.
    '''
    def __init__(self, pool_size=None, verbose=1):
        if pool_size is None:
            self.pool_size = multiprocessing.cpu_count() * 3 // 4
        else:
            self.pool_size = pool_size
        self.verbose = verbose
        self.ps = []
        self.cmd_queue = []
        self._n_cmds = 0 # Accumulated counter for generating cmd idx
        self._pid2job = {}
        self._return_codes = []
        self._log = []
 
    def check_call(self, cmd, *args, **kwargs):
        '''
        Asynchronous check_call (queued execution, return immediately).
        See subprocess.Popen() for more information about the arguments.

        Multiple commands can be separated with ";" and executed sequentially 
        within a single subprocess in linux/mac, only if shell=True.
        
        Python callable can also be executed in parallel via multiprocessing.
        Note that only the return code of the child process will be retrieved
        later when calling wait(), not the actual return value of the callable.
        So the result of the computation needs to be saved in a file.

        Parameters
        ----------
        cmd : list, str, or callable
            Computation in command line programs is handled with subprocess.
            Computation in python callable is handled with multiprocessing.
        shell : bool
            If provided, must be a keyword argument.
            If shell is True, the command will be executed through the shell.
        *args, **kwargs : 
            If cmd is a callable, *args and **kwargs are passed to the callable as its arguments.
            If cmd is a list or str, **kwargs are passed to subprocess.Popen().
        '''
        cmd = cmd_for_exec(cmd, kwargs)
        self.cmd_queue.append((self._n_cmds, cmd, args, kwargs))
        self._n_cmds += 1

    def dispatch(self):
        # If there are free slot and more jobs
        while len(self.ps) < self.pool_size and len(self.cmd_queue) > 0:
            idx, cmd, args, kwargs = self.cmd_queue.pop(0)
            job = {'idx': idx, 'cmd': cmd_for_disp(cmd)}
            if self.verbose:
                print('>> job {0}: {1}'.format(idx, job['cmd']))
            if callable(cmd):
                p = multiprocessing.Process(target=cmd, args=args, kwargs=kwargs)
                p.start()
            else:
                p = subprocess.Popen(cmd, **kwargs)
            self.ps.append(p)
            job['pid'] = p.pid
            job['uuid'] = uuid.uuid4().hex[:6]
            job['start'] = time.time()
            self._pid2job[p.pid] = job
            self._log.append(job)

    def wait(self, pool_size=None):
        '''
        Wait for all jobs in the queue to finish.
        
        Returns
        -------
        codes : list
            The return code of the child process for each job.
        '''
        if pool_size is not None:
            # Allow temporally adjust pool_size for current batch of jobs
            old_size = self.pool_size
            self.pool_size = pool_size
        self._start_time = time.time()
        while len(self.ps) > 0 or len(self.cmd_queue) > 0:
            # Dispatch jobs if possible
            self.dispatch()
            # Poll workers' state
            for p in self.ps:
                if isinstance(p, subprocess.Popen) and p.poll() is not None: # If the process is terminated
                    job = self._pid2job[p.pid]
                    job['stop'] = time.time()
                    job['returncode'] = p.returncode
                    self._return_codes.append((job['idx'], job['returncode']))
                    self.ps.remove(p)
                elif isinstance(p, multiprocessing.Process) and not p.is_alive(): # If the process is terminated
                    job = self._pid2job[p.pid]
                    job['stop'] = time.time()
                    job['returncode'] = p.exitcode # This is different...
                    self._return_codes.append((job['idx'], job['returncode']))
                    self.ps.remove(p)
            time.sleep(0.1)
        codes = [code for idx, code in sorted(self._return_codes)]
        duration = time.time() - self._start_time
        print('>> All {0} jobs done in {1}.'.format(self._n_cmds, format_duration(duration)))
        if np.any(codes):
            print('returncode: {0}', codes)
        else:
            print('all returncodes are 0.')
        self._n_cmds = 0
        self._pid2job = {}
        self._return_codes = []
        if pool_size is not None:
            self.pool_size = old_size
        return codes

    def all_successful(self):
        return not np.any([job['returncode'] for job in self._log])

    def batches(self, total, batch_size=None):
        if batch_size is None:
            batch_size = int(np.ceil(total / self.pool_size / 10))
        return (range(k, min(k+batch_size, total)) for k in range(0, total, batch_size))

    def __call__(self, job_generator):
        # This is similar to the joblib.Parallel signature
        # e.g. pc(pc.check_call(compute_depth, *args) for ids in pc.batches(len(depths)))
        n_batches = 0
        for _ in job_generator:
            n_batches += 1
        print('>> Start with a total of {0} jobs...'.format(n_batches))
        self.wait()


class ArrayWrapper(type):
    '''
    This is the metaclass for classes that wrap an np.ndarray and delegate 
    non-reimplemented operators (among other magic functions) to the wrapped array.
    '''
    def __init__(cls, name, bases, dct):
        def make_descriptor(name):
            return property(lambda self: getattr(self.arr, name))

        type.__init__(cls, name, bases, dct)
        ignore = 'class mro new init setattr getattr getattribute'
        ignore = set('__{0}__'.format(name) for name in ignore.split())
        for name in dir(np.ndarray):
            if name.startswith('__'):
                if name not in ignore and name not in dct:
                    setattr(cls, name, make_descriptor(name))


@add_metaclass(ArrayWrapper) # Compatibility code from six
class SharedMemoryArray(object):
    '''
    This class can be used as a usual np.ndarray, but its data buffer
    is allocated in shared memory (under Cached Files in memory monitor), 
    and can be passed across processes without any data copy/duplication, 
    even when write access happens (which is lock-synchronized).

    The idea is to allocate memory using multiprocessing.Array, and  
    access it from current or another process via a numpy.ndarray view, 
    without actually copying the data.
    So it is both convenient and efficient when used with multiprocessing.

    This implementation also demonstrates the power of composition + metaclass,
    as opposed to the canonical multiple inheritance.
    '''
    def __init__(self, dtype, shape, initializer=None, lock=True):
        self.dtype = np.dtype(dtype)
        self.shape = shape
        if initializer is None:
            # Preallocate memory using multiprocessing is the preferred usage
            self.shared_arr = multiprocessing.Array(self.dtype2ctypes[self.dtype], int(np.prod(self.shape)), lock=lock)
        else:
            self.shared_arr = multiprocessing.Array(self.dtype2ctypes[self.dtype], initializer, lock=lock)
        if not lock:
            self.arr = np.frombuffer(self.shared_arr, dtype=self.dtype).reshape(self.shape)
        else:
            self.arr = np.frombuffer(self.shared_arr.get_obj(), dtype=self.dtype).reshape(self.shape)
 
    @classmethod
    def zeros(cls, shape, dtype=float, lock=True):
        '''
        Return a new array of given shape and dtype, filled with zeros.

        This is the preferred usage, which avoids holding two copies of the
        potentially very large data simultaneously in the memory.
        '''
        return cls(dtype, shape, lock=lock)

    @classmethod
    def from_array(cls, arr, lock=True):
        '''
        Initialize a new shared-memory array with an existing array.
        '''
        # return cls(arr.dtype, arr.shape, arr.ravel(), lock=lock) # Slow and memory inefficient, why?
        a = cls.zeros(arr.shape, dtype=arr.dtype, lock=lock)
        a[:] = arr # This is a more efficient way of initialization
        return a

    def __getattr__(self, attr):
        if attr in self._SHARED_ARR_ATTRIBUTES:
            return getattr(self.shared_arr, attr)
        else:
            return getattr(self.arr, attr)

    def __dir__(self):
        return list(self.__dict__.keys()) + self._SHARED_ARR_ATTRIBUTES + dir(self.arr)

    _SHARED_ARR_ATTRIBUTES = ['acquire', 'release', 'get_lock']

    # At present, only numerical dtypes are supported.
    dtype2ctypes = {
        bool: ctypes.c_bool,
        int: ctypes.c_long,
        float: ctypes.c_double,
        np.dtype('bool'): ctypes.c_bool,
        np.dtype('int64'): ctypes.c_long,
        np.dtype('int32'): ctypes.c_int,
        np.dtype('int16'): ctypes.c_short,
        np.dtype('int8'): ctypes.c_byte,
        np.dtype('uint64'): ctypes.c_ulong,
        np.dtype('uint32'): ctypes.c_uint,
        np.dtype('uint16'): ctypes.c_ushort,
        np.dtype('uint8'): ctypes.c_ubyte,
        np.dtype('float64'): ctypes.c_double,
        np.dtype('float32'): ctypes.c_float,
        }
        
