#!/usr/bin/env python3
"""
Defines the OUActionNoise class.
"""

import numpy as np


class OUActionNoise(object):
    """
    Implements of the Ornstein-Uhlenbeck process for introducing action noise.

    Parameters
    ----------
    mean: Float
        Noise mean
    sigma: Float
        Noise variance
    theta: Float
        Model weight
    t_diff: Float
        Time difference
    x_init: Float
        Initial value
    """
    def __init__(self, mean, sigma=0.15, theta=0.2, t_diff=1e-2, x_init=None):
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.t_diff = t_diff
        self.x_init = x_init
        self.x_prev = \
            self.x_init \
            if self.x_init is not None \
            else np.zeros_like(self.mean)

    def __call__(self):
        """ Calls the object as a function to return the noise."""
        x_new = \
            self.x_prev + \
            self.theta * (self.mean - self.x_prev) * self.t_diff + \
            self.sigma * np.sqrt(self.t_diff) * \
            np.random.normal(size=self.mean.shape)
        self.x_prev = x_new
        return x_new
