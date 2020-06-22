#!/usr/bin/env python3
"""Defines the base class with common functionality across agents"""

import importlib
import os
import rospy
import tensorflow as tf
from agent_ros_wrapper import AgentRosWrapper
tf.compat.v1.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


class AgentBase(object):
    """
    The base class for agents
    """

    def __init__(self, agent_name):
        self.name = agent_name
        gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=gpu_options, log_device_placement=True))
        self.env_wrapper = \
            AgentRosWrapper(rospy.get_param("ros_gym/environment_name"))

    @classmethod
    def get_agent(cls, agent_name):
        """Returns an agent of given name

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        ------
        AgentBase
            An reinforcement learning agent of base type AgentBase()
        """
        module = \
            importlib.import_module('rl_agents.{}.agent'.format(agent_name))
        return module.Agent(agent_name)
