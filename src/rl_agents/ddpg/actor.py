#!/usr/bin/env python3
"""Defines the actor for DDPG algorithm"""

import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from rl_agents.ddpg.actor_critic_1 import ActorModel

ActorInputs = \
    namedtuple("ActorInputs", ['image', 'robot_state'])


class Actor(object):
    """
    A actor used for determining the Q-value for actions.

    Parameters
    ----------
    sess : Session
        Tensorflow session
    image_preprocessor: ImagePreprocessor
        The preprocessor for image inputs
    robot_state_input_shape: tuple
        Shape of the input state of robot state
    actions_output_shape: tuple
        Shape of the output actions
    learning_rate: Float
        Learning rate for network
    batch_size: Int
        Batch size
    model: tf.Keras.Model
        The model used by the actor for approximation
    optimizer: tf optimizer
        The optimizer used by the actor for loss minimization
    scope: str
        Tensorflow variable scope
    summaries_dir: str
        Directory of storing summaries for training of actor model
    """
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            sess,
            image_preprocessor,
            robot_state_input_shape,
            actions_output_shape,
            action_bound,
            learning_rate,
            batch_size,
            model=ActorModel,
            optimizer=AdamOptimizer,
            scope='actor',
            summaries_dir='tmp/ddpg/actor',
            gpu="/gpu:0"):
        with tf.name_scope(scope):
            self.sess = sess
            self.image_preprocessor = image_preprocessor
            self.robot_state_input_shape = robot_state_input_shape
            self.actions_output_shape = actions_output_shape
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.optimizer = optimizer
            self.scope = scope
            self.gpu = gpu
            self.model = \
                model(
                    action_bound=action_bound,
                    image_input_shape=self.image_preprocessor.output_shape,
                    robot_state_input_shape=self.robot_state_input_shape,
                    actions_output_shape=self.actions_output_shape,
                    scope=self.scope
                )

            self.checkpoint_file = \
                os.path.join(summaries_dir, scope + '_ddpg.checkpoint')
            # Writes Tensorboard summaries to disk
            self.summary_writer = None
            if summaries_dir:
                summary_dir = \
                    os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = \
                    tf.summary.FileWriter(summary_dir)

            with tf.device(self.gpu):
                # Build the graph
                self.build()

            self.saver = tf.train.Saver()

    def build(self):
        """ Builds the tensorflow model graph """
        with tf.name_scope(self.scope):
            # define inputs
            self.image_input = \
                tf.placeholder(
                    name='image_input',
                    shape=(None, *self.image_preprocessor.output_shape),
                    dtype=tf.float32)
            self.robot_state_input = \
                tf.placeholder(
                    name='robot_state_input',
                    shape=(None, *self.robot_state_input_shape),
                    dtype=tf.float32)
            self.actions_gradients = \
                tf.placeholder(
                    name='actions_gradients',
                    shape=(None, *self.actions_output_shape),
                    dtype=tf.float32)

            # process inputs
            self.actions = \
                self.model(
                    ActorInputs(
                        self.image_input,
                        self.robot_state_input))

            # minimize loss
            self.params = tf.trainable_variables(scope=self.scope)

            # action gradients come from critic
            self.unnormalized_actor_gradients = \
                tf.gradients(
                    self.actions, self.params, -self.actions_gradients)
            self.actor_gradients = \
                list(
                    map(lambda x: tf.math.divide(x, self.batch_size),
                        self.unnormalized_actor_gradients))
            self.optimize = \
                self.optimizer(self.learning_rate) \
                    .apply_gradients(zip(self.actor_gradients, self.params))

    def predict(self, inputs):
        """
        Performs the forward pass for the network to predict action
        """
        with tf.device(self.gpu):
            return \
                self.sess.run(
                    self.actions,
                    feed_dict={
                        self.image_input: inputs.image,
                        self.robot_state_input: inputs.robot_state})

    def train(self, inputs, actions_gradients):
        """
        Performs the back-propagation of gradient for loss minimization
        """
        with tf.device(self.gpu):
            _ = \
                self.sess.run(
                    self.optimize,
                    feed_dict={
                        self.image_input: inputs.image,
                        self.robot_state_input: inputs.robot_state,
                        self.actions_gradients: actions_gradients})

    def save_checkpoint(self):
        """Saves the model checkpoint"""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)