#!/usr/bin/env python3
"""
Defines the base deep model that is similar between the actor and critic
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
from tensorflow.keras.activations import relu, tanh
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
    def __init__(self, input_shapes_env, cfg='actor_critic_1'):
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
    image, robot state, and action inputs to determine
    a Q-value for the actions.

    Parameters
    ----------
    input_shapes: tuple
        Shapes of the input image and robot state that this deep qn deals with.
        Example input_shapes={
                'image': (80, 60, 3),
                'robot_state': (13,)}
    scope: str
        Tensorflow variable scope
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            input_shapes,
            scope='actor_critic_1',
            cfg='actor_critic_1'):
        with tf.name_scope(scope):
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

            conv_layer_sizes = self.config['conv_layers']['sizes']
            conv_kernels = self.config['conv_layers']['kernels']
            dense_layer_sizes = self.config['dense_layers']['sizes']

            # first three layers for image input
            self.conv1 = \
                self.make_conv2d_layer(
                    conv_layer_sizes['c1'],
                    conv_kernels['c1'],
                    input_shape=self.input_shapes['image'])
            self.conv2 = \
                self.make_conv2d_layer(
                    conv_layer_sizes['c2'], conv_kernels['c2'])
            self.conv3 = \
                self.make_conv2d_layer(
                    conv_layer_sizes['c3'], conv_kernels['c3'])
            # convolution to dense layer connection
            self.dense4_image = self.make_dense_layer(dense_layer_sizes['d4'])
            # state input to dense layer connection
            self.dense4_state = \
                self.make_dense_layer(
                    dense_layer_sizes['d4'], self.input_shapes['robot_state'])

    # pylint: disable=arguments-differ
    def call(self, inputs):
        # layer 1 with image input
        c_1 = self.conv1(inputs['image'])
        c_1 = BatchNormalization()(c_1)
        c_1 = relu(c_1)

        # layer 2
        c_2 = self.conv2(c_1)
        c_2 = BatchNormalization()(c_2)
        c_2 = relu(c_2)

        # layer 3
        c_3 = self.conv3(c_2)
        c_3 = BatchNormalization()(c_3)
        c_3 = relu(c_3)
        c_3 = Flatten()(c_3)

        # connect layer 3 output to d4_image layer
        d_4_image = self.dense4_image(c_3)
        d_4_image = BatchNormalization()(d_4_image)

        # connect state input to d4_image layer
        d_4_state = self.dense4_state(inputs['robot_state'])
        d_4_state = BatchNormalization()(d_4_state)

        # add the outputs of two layers and activate
        d_4 = tf.add(d_4_image, d_4_state)
        d_4 = relu(d_4)

        return d_4

    # pylint: disable=R0201
    def make_conv2d_layer(self, filters, kernel_size, input_shape=None):
        """
        Makes a convolutional layer based on filter and kernel size and
        enforces an input shape if required

        Parameters
        ----------
        filters : int
            Filter size
        kernel_size: int
            Kernel size
        input_shape: tuple
            Shape of the input to the layer
        """
        noise = 1.0 / np.sqrt(filters * (kernel_size + 1))
        if input_shape is None:
            return Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                kernel_initializer=random_uniform(
                    minval=-noise, maxval=noise),
                bias_initializer=random_uniform(
                    minval=-noise, maxval=noise))
        else:
            return Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                input_shape=input_shape,
                kernel_initializer=random_uniform(
                    minval=-noise, maxval=noise),
                bias_initializer=random_uniform(
                    minval=-noise, maxval=noise))

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
        Shapes of the input image and robot state that this deep qn deals with.
        Example input_shapes={
                'image': (80, 60, 3),
                'robot_state': (13,),
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
            scope='critic_model_1',
            cfg='actor_critic_1'):
        with tf.name_scope(scope):
            super(CriticModel, self).__init__(
                input_shapes,
                scope,
                cfg)

            dense_layer_sizes = self.config['dense_layers']['sizes']
            output_layer_params = self.config['output_params']

            # final dense layer for features
            self.dense5_features = \
                self.make_dense_layer(dense_layer_sizes['d5'])
            # final dense layer for actions input
            self.dense5_actions = \
                self.make_dense_layer(
                    dense_layer_sizes['d5'],
                    input_shape=input_shapes['actions'])
            # returns the q-value for the action taken
            self.dense6 = \
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

        d_4 = super(CriticModel, self).call(inputs)

        # layer 5 from features
        d_5_features = self.dense5_features(d_4)
        d_5_features = BatchNormalization()(d_5_features)

        # get action input from actor and create another layer in parallel to
        # d5_features
        d_5_actions = self.dense5_actions(inputs['actions'])
        d_5_actions = relu(tf.add(d_5_features, d_5_actions))

        # layer 6
        return self.dense6(d_5_actions)


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
            scope='actor_model_1',
            cfg='actor_critic_1'):
        with tf.name_scope(scope):
            super(ActorModel, self).__init__(
                input_shapes,
                scope,
                cfg)

            dense_layer_sizes = self.config['dense_layers']['sizes']
            output_layer_params = self.config['output_params']

            # action cut off
            self.action_bound = action_bound

            # final dense layer for features
            self.dense5_features = \
                self.make_dense_layer(dense_layer_sizes['d5'])

            # returns the q-value for the action taken
            self.dense6 = \
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

        d_4 = super(ActorModel, self).call(inputs)

        # layer 5 from features
        d_5_features = self.dense5_features(d_4)
        d_5_features = BatchNormalization()(d_5_features)
        d_5_features = relu(d_5_features)

        d_6 = self.dense6(d_5_features)
        d_6 = tanh(d_6)
        # layer 6
        return tf.math.multiply(d_6, self.action_bound)
