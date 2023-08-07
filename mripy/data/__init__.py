#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import path
import json


def load_atlas(name):
    fname = f"{path.dirname(__file__)}/atlas/{name}.json"
    with open(fname) as f:
        return json.load(f)


if __name__ == '__main__':
    pass
