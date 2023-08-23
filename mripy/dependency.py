#!/usr/bin/env python
# -*- coding: utf-8 -*-


def has(dependency_name, install_command='pip', raise_error=True):
    try:
        __import__(dependency_name)
        return True
    except ModuleNotFoundError as err:
        print(f"`{dependency_name}` is required but not installed.")
        
        if install_command == 'pip':
            print(f"You can install it using: `pip install {dependency_name}`")
        elif install_command:
            print(f"You can install it using: `{install_command}`")
        if raise_error:
            raise err
        else:
            return False