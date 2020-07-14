#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import unittest
import copy
import numpy as np
from mripy import timecourse


class test_Attributes(unittest.TestCase):
    def setUp(self):
        self.attr = timecourse.Attributes(shape=[6,4])
        self.attr.add('tid', range(6), axis=0)
        self.attr.add('eye', [1, 1, -1, -1], axis=1)
    
    def test_copy(self):
        attr1 = copy.copy(self.attr)
        self.assertNotEqual(id(attr1.attributes), id(self.attr.attributes))
        self.assertNotEqual(id(attr1.attributes['tid']), id(self.attr.attributes['tid']))
        self.assertNotEqual(id(attr1.eye), id(self.attr.eye))
        self.assertTrue(np.all(attr1.tid==self.attr.tid))

    def test_pick(self):
        attr1 = self.attr.pick(slice(0,3), axis=0)
        self.assertNotEqual(id(attr1), id(self.attr))
        self.assertNotEqual(id(attr1.attributes['tid']), id(self.attr.attributes['tid']))
        self.assertNotEqual(id(attr1.eye), id(self.attr.eye))
        self.assertTrue(np.all(attr1.tid==[0, 1, 2]))
        self.assertTrue(np.all(attr1.eye==self.attr.eye))
        attr2 = self.attr.pick([slice(3,None), [True, False, True, False]], axis=[0, 1])
        self.assertTrue(np.all(attr2.tid==[3, 4, 5]))
        self.assertTrue(np.all(attr2.eye==[1, -1]))


class test_Epochs_Attributes(unittest.TestCase):
    def setUp(self):
        self.epochs = timecourse.Epochs.from_array(np.random.rand(6,4,5), TR=2)
        self.epochs.add_event_attr('tid', range(6))
        self.epochs.add_feature_attr('eye', [1, 1, -1, -1])
    
    def test_copy(self):
        epochs1 = self.epochs.copy()
        self.assertEqual(id(epochs1.data), id(self.epochs.data))
        self.assertNotEqual(id(epochs1.attr), id(self.epochs.attr))
        self.assertNotEqual(id(epochs1.attr.eye), id(self.epochs.attr.eye))
    
    def test_pick(self):
        epochs1 = self.epochs[::2]
        self.assertNotEqual(id(epochs1.attr), id(self.epochs.attr))
        self.assertNotEqual(id(epochs1.attr.attributes['eye']), id(self.epochs.attr.attributes['eye']))
        self.assertNotEqual(id(epochs1.attr.tid), id(self.epochs.attr.tid))
        self.assertTrue(np.all(epochs1.attr.tid==[0, 2, 4]))
        self.assertTrue(np.all(epochs1.attr.eye==self.epochs.attr.eye))
        self.assertTrue(np.all(epochs1.data==self.epochs.data[::2]))


# unittest.main(argv=['ignored', '-v'], exit=False) # 'ignored' is required, '-v' is verbose

if __name__ == '__main__':
    unittest.main()
