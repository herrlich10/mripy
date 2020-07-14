#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
from os import sys, path
from timeit import default_timer
import inspect

# Add mripy to Python path
sys.path.append(path.realpath(path.join(__file__, '../../..')))
from mripy import utils


def get_log_printer(fname):
    log_file = open(fname, 'a+') # Will be closed when the script ends
    def log_print(x):
        print(x, file=log_file) # Print to log file
        print(x) # Print to stdout
    return log_print, print


class ScriptTimer(object):
    def __init__(self):
        self.start_time = default_timer()
        self.stopped = False
        parent_frame = inspect.stack()[1][0]
        self.script = parent_frame.f_locals['__file__']

    def stop(self, message=None):
        '''
        message : str
            The message may contain two placeholders:
            - {0} for name of calling script
            - {1} for formatted script execution time
        '''
        if message is None:
            message = '>> {0} is done in {1}'
        self.stopped = True
        duration = default_timer() - self.start_time
        formatted = utils.format_duration(duration)
        print(message.format(path.basename(self.script), formatted))

    def __del__(self):
        if not self.stopped:
            self.stop()
