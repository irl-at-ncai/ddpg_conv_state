#!/usr/bin/python3
"""Defines utility functions used across the package."""

import types
import matplotlib.pyplot as plt
import numpy as np


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

# @todo: Task for Hassaan to fix this! Sorry Hassaan :D
def plot_learning(scores, filename, x=None, window=5):
    n = len(scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(n)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)