#!/usr/bin/python3
"""Defines utility functions used across the package."""

import types


def inherit_docs(cls):
    """
    Defines a function decorator for inheriting doc-strings from parent if
    they do not exist.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls
