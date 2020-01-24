#!/usr/bin/env python3
"""
Defines the base actor_critic deep network for training a single robot state model
"""

import os
from operator import itemgetter
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import relu, tanh, softmax
from tensorflow.keras.initializers import random_uniform
from tensorflow.keras.regularizers import l2
from rl_agents.common.utils import inherit_docs
from rl_agents.common.state_preprocessors import ImagePreprocessor, \
    StateConcatenator
import rospy


class PreprocessHandler():
    """
    Maps the state from environment to the output state required by
    the model.

    Parameters
    ----------
    input_shapes_env: dict
        Map from env observation keys to their shapes
    cfg: str
        Config name of the model
    """
    def __init__(self, input_shapes_env, cfg='actor_critic_2'):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(script_dir + '/cfg/' + cfg + '.yaml') as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.network_inputs = self.config['network_inputs']

        # update the shapes to what we want
        self.input_shapes_env = input_shapes_env
        self.input_shapes = {}
        for target_key, keys in self.network_inputs.items():
            shapes = [input_shapes_env[key] for key in keys]
            self.input_shapes[target_key] = tuple(map(sum, zip(*shapes)))
        self.setup_input_preprocessors()

        rospy.loginfo("Network takes the following inputs:")
        for idx, (key, value) in \
                enumerate(self.input_shapes.items()):
            rospy.loginfo("{}) {} ({}) (shape = {})".format(
                idx, key, self.network_inputs[key], value))

    def setup_input_preprocessors(self):
        """
        No preprocessing required in this case.
        """
        pass

    def process(self, state, sess):
        """
        Performs pre-processing operations on the state
        """
        processed = state
        for key, value in self.network_inputs.items():
            processed[key] = processed[key][np.newaxis, ...]
        return processed


class ActorCriticBase(tf.keras.Model):
    """
    An actor-critic model for DDPG algorithm used that takes
    image, robot state, and action inputs to determine
    a Q-value for the actions.

    Parameters
    ----------
    input_shapes: tuple
        Shapes of the input robot state that this deep qn deals with.
        Example input_shapes={
                'robot_state': (4,)}
    scope: str
        Tensorflow variable scope
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            input_shapes,
            scope='actor_critic_2',
            cfg='actor_critic_2'):
        with tf.compat.v1.name_scope(scope):
            super(ActorCriticBase, self).__init__()

            script_dir = os.path.dirname(os.path.realpath(__file__))
            with open(script_dir + '/cfg/' + cfg + '.yaml') as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)
            self.input_shapes = input_shapes

            if any(
                    not isinstance(v, tuple)
                    for k, v in self.input_shapes.items()):
                rospy.logerr(
                    'Model input shapes {} must be tuples.'
                    .format(self.input_shapes))

            dense_layer_sizes = self.config['dense_layers']['sizes']

            # state input to dense layer connection
            self.dense1 = \
                self.make_dense_layer(
                    dense_layer_sizes['d1'], self.input_shapes['robot_state'])

    # pylint: disable=arguments-differ
    def call(self, inputs):
        d_1 = self.dense1(inputs['robot_state'])
        d_1 = BatchNormalization()(d_1)
        d_1 = relu(d_1)
        return d_1

    # pylint: disable=R0201
    def make_dense_layer(self, units, input_shape=None):
        """
        Makes a dense layer based on the number of layer units and enforces
        an input shape if required

        Parameters
        ----------
        units : int
            Size of the layer
        input_shape: tuple
            Shape of the input to the layer
        """
        noise = 1.0 / np.sqrt(units)
        if input_shape is None:
            return \
                Dense(
                    units=units,
                    kernel_initializer=random_uniform(-noise, noise),
                    bias_initializer=random_uniform(-noise, noise))
        else:
            return \
                Dense(
                    units=units,
                    input_shape=input_shape,
                    kernel_initializer=random_uniform(-noise, noise),
                    bias_initializer=random_uniform(-noise, noise))


class CriticModel(ActorCriticBase):
    """
    A critic model for DDPG algorithm.

    Parameters
    ----------
    input_shapes: tuple
        Shapes of the input robot state that this deep qn deals with.
        Example input_shapes={
                'robot_state': (4,),
                'actions': (4,)}
    scope: str
        Tensorflow variable scope
    cfg: str
        Config file of the model
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            input_shapes,
            scope='critic_model_2',
            cfg='actor_critic_2'):
        with tf.compat.v1.name_scope(scope):
            super(CriticModel, self).__init__(
                input_shapes,
                scope,
                cfg)

            dense_layer_sizes = self.config['dense_layers']['sizes']
            output_layer_params = self.config['output_params']

            # final dense layer for features
            self.dense2_features = \
                self.make_dense_layer(dense_layer_sizes['d2'])
            # final dense layer for actions input
            self.dense2_actions = \
                self.make_dense_layer(
                    dense_layer_sizes['d2'],
                    input_shape=input_shapes['actions'])
            # returns the q-value for the action taken
            self.dense3 = \
                Dense(
                    units=1,
                    kernel_initializer=random_uniform(
                        minval=-output_layer_params['noise'],
                        maxval=output_layer_params['noise']),
                    bias_initializer=random_uniform(
                        -output_layer_params['noise'],
                        output_layer_params['noise']),
                    kernel_regularizer=l2(output_layer_params['l2']))

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Performs the forward pass of the critic network
        """

        d_1 = super(CriticModel, self).call(inputs)

        # layer 2 from features
        d_2_features = self.dense2_features(d_1)
        d_2_features = BatchNormalization()(d_2_features)

        # get action input from actor and create another layer in parallel to
        # d2_features
        d_2_actions = self.dense2_actions(inputs['actions'])
        d_2 = relu(tf.add(d_2_features, d_2_actions))

        # layer 6
        return self.dense3(d_2)


@inherit_docs
class ActorModel(ActorCriticBase):
    """
    An actor model for DDPG algorithm.

    Parameters
    ----------
    input_shapes: tuple
        Shapes of the input robot state that this deep qn deals with.
        Example input_shapes={
                'robot_state': (4,)}
    scope: str
        Tensorflow variable scope
    cfg: str
        Config file of the model
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            action_bound,
            input_shapes,
            actions_output_shape=(2,),
            scope='actor_model_2',
            cfg='actor_critic_2'):
        with tf.compat.v1.name_scope(scope):
            super(ActorModel, self).__init__(
                input_shapes,
                scope,
                cfg)

            dense_layer_sizes = self.config['dense_layers']['sizes']
            output_layer_params = self.config['output_params']

            # action cut off
            self.action_bound = action_bound

            # final dense layer for features
            self.dense2_features = \
                self.make_dense_layer(dense_layer_sizes['d2'])

            # returns the q-value for the action taken
            self.dense3 = \
                Dense(
                    units=actions_output_shape[0],
                    kernel_initializer=random_uniform(
                        minval=-output_layer_params['noise'],
                        maxval=output_layer_params['noise']),
                    bias_initializer=random_uniform(
                        -output_layer_params['noise'],
                        output_layer_params['noise']),
                    kernel_regularizer=l2(output_layer_params['l2']))

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Performs the forward pass of the actor network
        """

        d_1 = super(ActorModel, self).call(inputs)

        # layer 5 from features
        d_2_features = self.dense2_features(d_1)
        d_2_features = BatchNormalization()(d_2_features)
        d_2_features = relu(d_2_features)

        d_3 = self.dense6(d_3_features)
        d_3 = softmax(d_3)
        # layer 6
        return tf.math.multiply(d_3, self.action_bound)
