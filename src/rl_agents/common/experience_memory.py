#!/usr/bin/env python3
"""Defines the experience memory for observations and observation sequences"""

from collections import namedtuple
import numpy as np
from numpy_ringbuffer import RingBuffer


TRANSITION_ELEMS = ['s', 'a', 'r', 's_next', 'done', 'eps']
Transition = \
    namedtuple('Transition', TRANSITION_ELEMS)


class BuffeInitialSizeUnfilledError(Exception):
    """
    Exception raised when the buffer is not filled upto required initial
    size.
    """


class ExperienceMemory:
    """
    Base class for defining experience replay memory

    Parameters
    ----------
    init_size: int
        Initial size of the buffer to be filled
    max_size: int
        Max buffer size
    prioritized: bool
        Uses prioritized sampling if True
    """
    def __init__(
            self,
            state_inputs,
            init_size=int(5e4),
            max_size=int(1e5),
            prioritized=False):
        self.init_size = init_size
        self.max_size = max_size
        self.prioritized = prioritized
        self.size = 0
        self.state_inputs = state_inputs
        self.buffers = {}
        for elem in TRANSITION_ELEMS:
            if elem in ['s', 's_next']:
                self.buffers[elem] = {}
                for key in state_inputs.keys():
                    self.buffers[elem][key] = \
                        RingBuffer(
                            capacity=max_size,
                            dtype=(np.float32, state_inputs[key]))
            elif elem is 'a':
                self.buffers[elem] = \
                    RingBuffer(
                        capacity=max_size,
                        dtype=(np.float32, state_inputs['actions']))
            else:
                self.buffers[elem] = \
                    RingBuffer(
                        capacity=max_size,
                        dtype=np.float32)

    def add(self, transition):
        """
        Adds an episode transition to the buffer

        Parameters
        ----------
        transition: Transition
            A single transition step in the episode
        """
        t_dict = transition._asdict()
        if self.size < self.max_size:
            self.size += 1

        for elem in TRANSITION_ELEMS:
            if elem in ['s', 's_next']:
                for key, value in t_dict[elem].items():
                    self.buffers[elem][key].append(value)
            else:
                self.buffers[elem].append(t_dict[elem])

    def sample(self, sample_size=32):
        """
        Samples the data of given size from the buffer

        Parameters
        ----------
        sample_size: int
            Size of the sampled data
        """
        if self.size < self.init_size:
            raise BuffeInitialSizeUnfilledError(
                '''Experience memory sampled before required initial size
                is reached. {}/{}'''.format(self.size, self.init_size))
        if not self.prioritized:
            batches = {}
            indices = np.random.choice(self.size, sample_size)
            for elem in TRANSITION_ELEMS:
                if elem in ['s', 's_next']:
                    batches[elem] = {}
                    for key in self.state_inputs.keys():
                        batches[elem][key] = self.buffers[elem][key][indices]
                else:
                    batches[elem] = self.buffers[elem][indices]
            return batches
        else:
            raise NotImplementedError()


class ExperienceTrajMemory(ExperienceMemory):
    """
    Class for defining experience replay memory for sequence of data

    Parameters
    ----------
    init_size: int
        Initial size of the buffer to be filled
    max_size: int
        Max buffer size
    prioritized: bool
        Uses prioritized sampling if True
    """
    def __init__(
            self,
            init_size=int(5e4),
            max_size=int(5e4),
            prioritized=False):
        super(ExperienceTrajMemory, self).__init__(
            self, init_size, max_size, prioritized)

    # pylint: disable=arguments-differ
    def sample(self, sample_size=32, trace_length=8):
        """
        Samples the data of given size from the buffer

        Todo: Update sampling for trajectories according to new ring buffer
            implementation

        Parameters
        ----------
        sample_size: int
            Size of the sampled data
        trace_length: int
            Size of the sampled sequence
        """
        if self.size < self.init_size:
            raise BuffeInitialSizeUnfilledError(
                '''Experience memory sampled before required initial size
                is reached. {}/{}'''.format(self.size, self.init_size))

        if not self.prioritized:
            traced_exp = []

        # get indices for all elements that have at least length = trace_length
        # within an episode
        while True:
            # idx = random.choice(range(self.size))
            # last = idx+trace_length-1
            # if last < self.size and \
            #         self.buffer[idx].eps == self.buffer[last].eps:
            #     traced_exp.append(self.buffer[idx:last+1])
            if len(traced_exp) == sample_size:
                break
            return traced_exp
