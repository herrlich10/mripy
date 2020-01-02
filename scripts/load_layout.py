#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, subprocess, re
from collections import OrderedDict

if __name__ == '__main__':
    afni_script = sys.argv[1]
    with open(afni_script) as f:
        ### A single -com "..." is too long
        # lines = f.readlines()
        # com = '; '.join([line.strip() for line in lines if not line.startswith('//')]);
        # subprocess.call('afni -com "CLOSE_WINDOW A.axialimage; \
        #     CLOSE_WINDOW A.sagittalimage; CLOSE_WINDOW A.coronalimage" \
        #     -com "{0}"'.format(com), shell=True)

        ### A single -com "..." can still get too long
        # coms = OrderedDict()
        # for line in f.readlines():
        #     if line.startswith('//'):
        #         continue
        #     match = re.match('\S+ ([A-Z])', line)
        #     if match:
        #         cid = match.group(1)
        #         coms.setdefault(cid, []).append(line.strip())
        # for cid in coms:
        #     coms[cid] = '; '.join(coms[cid])
        # com = ' '.join(['-com "{0}"'.format(c) for c in coms.values()])
        # subprocess.call('afni -com "CLOSE_WINDOW A.axialimage; \
        #     CLOSE_WINDOW A.sagittalimage; CLOSE_WINDOW A.coronalimage" \
        #     {0}'.format(com), shell=True)

        lines = f.readlines()
        com = ' '.join(['-com "{0}"'.format(line.strip()) for line in lines if not line.startswith('//')]);
        subprocess.call('afni -com "CLOSE_WINDOW A.axialimage; \
            CLOSE_WINDOW A.sagittalimage; CLOSE_WINDOW A.coronalimage" \
            {0}'.format(com), shell=True)
