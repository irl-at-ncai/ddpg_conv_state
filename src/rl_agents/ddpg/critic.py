#!/usr/bin/env python3
"""Defines the critic for DDPG algorithm"""

import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.losses import mean_squared_error
from rl_agents.ddpg.actor_critic_1 import CriticModel

CriticInputs = \
    namedtuple("CriticInputs", ['image', 'robot_state', 'actions'])


class Critic(object):
    """
    A critic used for determining the Q-value for actions.

    Parameters
    ----------
    sess : Session
        Tensorflow session
    image_preprocessor: ImagePreprocessor
        The preprocessor for image inputs
    robot_state_input_shape: tuple
        Shape of the input state of the robot state
    actions_input_shape: tuple
        Shape of the input actions
    learning_rate: Float
        Learning rate for network
    model: tf.Keras.Model
        The model used by the critic for approximation
    loss_fn: tf loss function
        The loss function used by the critic
    optimizer: tf optimizer
        The optimizer used by the critic for loss minimization
    scope: str
        Tensorflow variable scope
    summaries_dir: str
        Directory of storing summaries for training of critic model
    """
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            sess,
            image_preprocessor,
            robot_state_input_shape,
            actions_input_shape,
            learning_rate,
            model=CriticModel,
            loss_fn=mean_squared_error,
            optimizer=AdamOptimizer,
            scope='critic',
            summaries_dir='tmp/ddpg/critic',
            gpu="/gpu:0"):
        with tf.name_scope(scope):
            self.sess = sess
            self.image_preprocessor = image_preprocessor
            self.robot_state_input_shape = robot_state_input_shape
            self.actions_input_shape = actions_input_shape
            self.learning_rate = learning_rate
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.scope = scope
            self.gpu = gpu
            self.model = \
                model(
                    image_input_shape=self.image_preprocessor.output_shape,
                    robot_state_input_shape=self.robot_state_input_shape,
                    actions_input_shape=self.actions_input_shape,
                    scope=self.scope
                )

            self.checkpoint_file = \
                os.path.join(summaries_dir, scope + '_ddpg.checkpoint')

            self.summary_writer = None
            with tf.device(self.gpu):
                # Build the graph
                self.build()

            # Writes Tensorboard summaries to disk
            if summaries_dir:
                summary_dir = \
                    os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = \
                    tf.summary.FileWriter(summary_dir)

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
            self.actions_input = \
                tf.placeholder(
                    name='actions_input',
                    shape=(None, *self.actions_input_shape),
                    dtype=tf.float32)

            # define outputs
            self.q_target = \
                tf.placeholder(
                    name='q_target',
                    shape=(None, 1),
                    dtype=tf.float32)

            # process inputs
            self.q_value = \
                self.model(
                    CriticInputs(
                        self.image_input,
                        self.robot_state_input,
                        self.actions_input))

            # minimize loss
            self.loss = self.loss_fn(self.q_target, self.q_value)
            self.params = tf.trainable_variables(scope=self.scope)
            self.optimize = \
                self.optimizer(self.learning_rate).minimize(self.loss)
            self.action_gradients = \
                tf.gradients(self.q_value, self.actions_input)

            # Summaries for Tensorboard
            # self.summaries = tf.summary.merge([
            #    tf.summary.scalar("loss", self.loss)
            # ])

    def predict(self, inputs):
        """
        Performs the forward pass for the network to predict action Q-value
        """
        with tf.device(self.gpu):
            return \
                self.sess.run(
                    self.q_value,
                    feed_dict={
                        self.image_input: inputs.image,
                        self.robot_state_input: inputs.robot_state,
                        self.actions_input: inputs.actions})

    def train(self, inputs, q_target):
        """
        Performs the back-propagation of gradient for loss minimization
        """
        with tf.device(self.gpu):
            if False: # @todo: summaries have errors
                _, summaries, global_step = \
                    self.sess.run(
                        [
                            self.optimize,
                            self.summaries,
                            tf.contrib.framework.get_global_step()],
                        feed_dict={
                            self.image_input: inputs.image,
                            self.robot_state_input: inputs.robot_state,
                            self.actions_input: inputs.actions,
                            self.q_target: q_target})
                self.summary_writer.add_summary(summaries, global_step)
            else:
                _ = \
                    self.sess.run(
                        self.optimize,
                        feed_dict={
                            self.image_input: inputs.image,
                            self.robot_state_input: inputs.robot_state,
                            self.actions_input: inputs.actions,
                            self.q_target: q_target})

    def get_action_gradients(self, inputs):
        """
        Returns the action gradients with respect to Q-values
        """
        with tf.device(self.gpu):
            return \
                self.sess.run(
                    self.action_gradients,
                    feed_dict={
                        self.image_input: inputs.image,
                        self.robot_state_input: inputs.robot_state,
                        self.actions_input: inputs.actions})

    def save_checkpoint(self):
        """Saves the model checkpoint"""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)