#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
from mripy import utils
import time, ctypes
import numpy as np


class test_utils(unittest.TestCase):
    @unittest.skip('just for understanding whether and how it works')
    def test_SharedMemoryArray_memory(self):
        '''
        Test shared-memory versus copy-on-write
        '''
        def slow_operation(k, x):
            if isinstance(x, utils.SharedMemoryArray):
                x.arr[:,:100] += 1 # With copy-on-write, the difference becomes apparent when there is write access
                res = np.mean(x.arr) # Read access
            time.sleep(10)
            print(f'#{k}: mean = {res}')
        shape = (1000000,1000) # About 8 GB
        a = utils.SharedMemoryArray.zeros(shape) # This will be allocated in the Cached Files
        # The following two assignments are very different!
        # a.arr = np.random.rand(*shape)
        a.arr[:] = np.random.rand(*shape)
        pc = utils.PooledCaller()
        for k in range(pc.pool_size):
            pc.check_call(slow_operation, k, a)
        pc.wait()

    @unittest.skip('temporary')
    def test_SharedMemoryArray_array(self):
        '''
        Test the life saving container interface
        '''
        def slow_operation(k, x):
            while True:
                if x.acquire():
                    x[:,:100] += 1 # Write access
                    res = np.mean(x) # Read access
                    x.release()
                    break
            time.sleep(10)
            print(f'#{k}: mean = {res}')
        shape = (1000000,1000) # About 8 GB
        a = utils.SharedMemoryArray.zeros(shape)
        a[:] = np.random.rand(*shape)
        pc = utils.PooledCaller()
        for k in range(pc.pool_size):
            pc.check_call(slow_operation, k, a)
        pc.wait()

    def test_SharedMemoryArray_CoW(self):
        '''
        Test the speed of CoW if read-only 
        '''
        def slow_operation(k, x):
            res = np.mean(x) # Read access
            print(f'#{k}: mean = {res}')
        shape = (1000000,1000) # About 8 GB
        a = np.random.rand(*shape) # >> All 12 jobs done in 9.832 sec.
        # a = utils.SharedMemoryArray.zeros(shape) # >> All 12 jobs done in 8.784 sec.
        # a[:] = np.random.rand(*shape)
        pc = utils.PooledCaller()
        for k in range(pc.pool_size):
            pc.check_call(slow_operation, k, a)
        pc.wait()


if __name__ == '__main__':
    unittest.main()
