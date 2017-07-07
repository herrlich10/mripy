#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, glob

if __name__ == '__main__':
    file_glob = sys.argv[1]
    selected = (int(k)-1 for k in sys.argv[2:])
    files = sorted(glob.glob(file_glob))
    print(' '.join([files[k] for k in selected]))
