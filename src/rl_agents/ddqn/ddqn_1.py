#!/usr/bin/env python3
"""
Defines the base deep model that is similar between the actor and critic
"""

import os
import yaml
import rospy
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, \
    Flatten, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import RandomUniform as random_uniform
from tensorflow.keras.regularizers import l2
from rl_agents.common.utils import inherit_docs

tf.disable_v2_behavior()


class DDQNBase(tf.keras.Model):
    """
    A critic model for DDPG algorithm used that takes
    image, robot state, and action inputs to determine
    a Q-value for the actions.

    Parameters
    ----------
    image_input_shape : tuple
        Shape of the input image
    robot_state_input_shape: tuple
        Shape of the input robot state
    actions_input_shape: tuple
        Shape of the input actions
    scope: str
        Tensorflow variable scope
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            image_input_shape=(80, 60, 1),
            robot_state_input_shape=(13,),
            scope='ddqn_1',
            cfg='ddqn_model_1'):
        with tf.name_scope(scope):
            super(DDQNBase, self).__init__()

            if any(
                    not isinstance(x, tuple)
                    for x in [robot_state_input_shape, image_input_shape]):
                rospy.logerr('Model input shapes must be a tuple.')

            script_dir = os.path.dirname(os.path.realpath(__file__))

            with open(script_dir + '/cfg/' + cfg + '.yaml') as config_file:
                self.config = yaml.load(config_file, Loader=yaml.FullLoader)

            conv_layer_sizes = self.config['conv_layers']['sizes']
            conv_kernels = self.config['conv_layers']['kernels']
            dense_layer_sizes = self.config['dense_layers']['sizes']

            # first three layers for image input
            self.conv1 = self.make_conv2d_layer(
                conv_layer_sizes['c1'],
                conv_kernels['c1'],
                input_shape=image_input_shape)
            self.conv2 = self.make_conv2d_layer(
                conv_layer_sizes['c2'],
                conv_kernels['c2'])
            self.conv3 = self.make_conv2d_layer(
                conv_layer_sizes['c3'],
                conv_kernels['c3'])

            # convolution to dense layer connection
            self.dense4_image = self.make_dense_layer(
                dense_layer_sizes['d4'])

            # state input to dense layer connection
            self.dense4_state = self.make_dense_layer(
                dense_layer_sizes['d4'],
                robot_state_input_shape)

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """Creating hidden layers architecture"""
        # layer 1 with image input
        c_1 = self.conv1(inputs.image)
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
        d_4_state = self.dense4_state(inputs.robot_state)
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
                kernel_initializer=random_uniform(minval=-noise, maxval=noise),
                bias_initializer=random_uniform(minval=-noise, maxval=noise))
        else:
            return Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                input_shape=input_shape,
                kernel_initializer=random_uniform(minval=-noise, maxval=noise),
                bias_initializer=random_uniform(minval=-noise, maxval=noise))

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


@inherit_docs
class DDQNModel(DDQNBase):
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            image_input_shape=(80, 60, 1),
            robot_state_input_shape=(13,),
            actions_shape=7,
            scope='ddqn_model_1',
            cfg='ddqn_model_1'):
        with tf.name_scope(scope):
            super(DDQNModel, self).__init__(
                image_input_shape,
                robot_state_input_shape,
                scope,
                cfg)

            dense_layer_sizes = self.config['dense_layers']['sizes']
            output_layer_params = self.config['output_params']

            # final dense layer for features
            self.dense5_features = self.make_dense_layer(
                dense_layer_sizes['d5'])

            # returns the q-value for the action taken
            self.dense6 = Dense(
                units=actions_shape,
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

        Parameters
        ----------
        inputs: ActorInputs
            Inputs to the network
        """

        d_4 = super(DDQNModel, self).call(inputs)

        # layer 5 from features
        d_5_features = self.dense5_features(d_4)
        d_5_features = BatchNormalization()(d_5_features)
        d_5_features = relu(d_5_features)
        d_5_features = Dropout(0.3)(d_5_features)

        # layer 6
        return self.dense6(d_5_features)
