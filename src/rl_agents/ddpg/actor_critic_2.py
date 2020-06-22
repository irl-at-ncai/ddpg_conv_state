#!/usr/bin/env python3
"""
Defines the base deep model that is similar between the actor and critic
"""

import os
from operator import itemgetter
import yaml
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import Flatten
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.activations import relu, tanh
from tensorflow.compat.v1.keras.initializers import random_uniform
from tensorflow.compat.v1.keras.regularizers import l2
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
        print(input_shapes_env)
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
        Sets up the preprocessors that convert/reshape inputs from
        environment to the required shapes for the network.
        """
        self.preprocessors = {}
        for key, value in self.network_inputs.items():
            print(key)
            # define a preprocess for robot state
            if key == 'robot_state':
                self.preprocessors[key] = \
                    StateConcatenator(output_name=key)

            # define a preprocessor for image input
            if key == 'image':
                self.preprocessors[key] = \
                    ImagePreprocessor(
                        input_shape=self.input_shapes_env[value[0]],
                        output_shape=tuple(self.config['input_image_shape']))
                self.input_shapes[key] = self.preprocessors[key].output_shape

            # define a preprocessor for image depth input
            if key == 'image_depth':
                self.preprocessors[key] = \
                    ImagePreprocessor(
                        input_shape=self.input_shapes_env[value[0]],
                        output_shape=tuple(
                            self.config['input_depth_image_shape']))
                self.input_shapes[key] = self.preprocessors[key].output_shape
        if len(self.preprocessors) != len(self.network_inputs):
            rospy.logwarn("Preprocessors not defined for all inputs.")

    def process(self, state, sess):
        """
        Performs pre-processing operations on the state
        """
        # update the states to what we want in agent

        processed = {}
        for key, value in self.network_inputs.items():
            processed[key] = \
                self.preprocessors[key]. \
                process(itemgetter(*value)(state), sess)
            processed[key] = processed[key][np.newaxis, ...]
        return processed


class ActorCriticBase(tf.keras.Model):
    """
    An actor-critic model for DDPG algorithm used that takes
    robot state, and action inputs to determine
    a Q-value for the actions.

    Parameters
    ----------
    input_shapes: tuple
        Shape of the robot state that this deep qn deals with.
        Example input_shapes={
                'robot_state': (2,)}
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
            # first input layer
            self.dense1 = \
                self.make_dense_layer(
                    dense_layer_sizes['d1'], self.input_shapes['robot_state'])

            # output layer
            self.dense2 = \
                self.make_dense_layer(dense_layer_sizes['d2'])

    # pylint: disable=arguments-differ
    def call(self, inputs):
        # layer 1 with image input
        d_1 = self.dense1(inputs['robot_state'])
        d_1 = BatchNormalization()(d_1)
        d_1 = relu(d_1)

        d_2 = self.dense2(d_1)
        d_2 = BatchNormalization()(d_2)
        d_2 = relu(d_2)

        return d_2

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
        Shape of the input robot state that this deep qn deals with.
        Example input_shapes={
                'robot_state': (2,),
                'actions': (2,)}
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
            self.dense3_features = \
                self.make_dense_layer(dense_layer_sizes['d3'])
            # final dense layer for actions input
            self.dense3_actions = \
                self.make_dense_layer(
                    dense_layer_sizes['d3'],
                    input_shape=input_shapes['actions'])
            # returns the q-value for the action taken
            self.dense4 = \
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

        d_2 = super(CriticModel, self).call(inputs)

        # layer 3 from features
        d_3_features = self.dense3_features(d_2)
        d_3_features = BatchNormalization()(d_3_features)

        # get action input from actor
        d_3_actions = self.dense3_actions(inputs['actions'])
        d_3 = relu(tf.add(d_3_features, d_3_actions))

        # layer 6
        return self.dense4(d_3)


@inherit_docs
class ActorModel(ActorCriticBase):
    """
    An actor model for DDPG algorithm.

    Parameters
    ----------
    input_shapes: tuple
        Shapes of the input image and robot state that this deep qn deals with.
        Example input_shapes={
                'image': (80, 60, 3),
                'robot_state': (13,)}
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
            actions_output_shape=(4,),
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
            self.dense3_features = \
                self.make_dense_layer(dense_layer_sizes['d3'])

            # returns the q-value for the action taken
            self.dense4 = \
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

        d_2 = super(ActorModel, self).call(inputs)

        # layer 3 from features
        d_3_features = self.dense3_features(d_2)
        d_3_features = BatchNormalization()(d_3_features)
        d_3_features = relu(d_3_features)

        d_4 = self.dense4(d_3_features)
        return tf.math.multiply(d_4, self.action_bound)
