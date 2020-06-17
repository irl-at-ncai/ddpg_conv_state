#!/usr/bin/env python3
"""Defines the critic for DDPG algorithm"""

import os
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v1.losses import mean_squared_error
from rl_agents.ddpg.actor_critic_1 import CriticModel


class Critic(object):
    """
    A critic used for determining the Q-value for actions.

    Parameters
    ----------
    sess : Session
        Tensorflow session
    input_shapes: Dict
        A dictionary containing shapes of inputs to critic
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
            input_shapes,
            learning_rate,
            model=CriticModel,
            loss_fn=mean_squared_error,
            optimizer=AdamOptimizer,
            scope='critic',
            summaries_dir='tmp/ddpg/critic',
            gpu="/gpu:0"):
        with tf.compat.v1.name_scope(scope):
            self.sess = sess
            self.input_shapes = input_shapes
            self.learning_rate = learning_rate
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.scope = scope
            self.gpu = gpu
            self.model = \
                model(input_shapes=self.input_shapes, scope=self.scope)

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
                    tf.compat.v1.summary.FileWriter(summary_dir)

            self.saver = tf.compat.v1.train.Saver()

    def build(self):
        """ Builds the tensorflow model graph """
        with tf.compat.v1.name_scope(self.scope):
            # define inputs
            self.input_phs = {}
            for key, value in self.input_shapes.items():
                self.input_phs[key] = \
                    tf.compat.v1.placeholder(
                        name=key,
                        shape=(None, *value),
                        dtype=tf.float32)

            # define outputs
            self.q_target = \
                tf.compat.v1.placeholder(
                    name='q_target',
                    shape=(None, 1),
                    dtype=tf.float32)

            # process inputs
            self.q_value = self.model(self.input_phs)

            # minimize loss
            self.loss = self.loss_fn(self.q_target, self.q_value)
            self.params = tf.compat.v1.trainable_variables(scope=self.scope)
            self.optimize = \
                self.optimizer(self.learning_rate).minimize(self.loss)
            self.action_gradients = \
                tf.gradients(ys=self.q_value, xs=self.input_phs['actions'])

            # Summaries for Tensorboard
            # self.summaries = tf.summary.merge([
            #    tf.summary.scalar("loss", self.loss)
            # ])

    def predict(self, inputs):
        """
        Performs the forward pass for the network to predict action Q-value
        """
        with tf.device(self.gpu):
            feed_dict = {}
            for key, value in self.input_phs.items():
                feed_dict[value] = inputs[key]
            return \
                self.sess.run(
                    self.q_value,
                    feed_dict=feed_dict)

    def train(self, inputs, q_target):
        """
        Performs the back-propagation of gradient for loss minimization
        """
        with tf.device(self.gpu):
            feed_dict = {}
            for key, value in self.input_phs.items():
                feed_dict[value] = inputs[key]
            feed_dict[self.q_target] = q_target
            if False:  # @todo: summaries have errors
                _, summaries, global_step = \
                    self.sess.run(
                        [
                            self.optimize,
                            self.summaries,
                            tf.contrib.framework.get_global_step()],
                        feed_dict=feed_dict)
                self.summary_writer.add_summary(summaries, global_step)
            else:
                _ = \
                    self.sess.run(
                        self.optimize,
                        feed_dict=feed_dict)

    def get_action_gradients(self, inputs):
        """
        Returns the action gradients with respect to Q-values
        """
        with tf.device(self.gpu):
            feed_dict = {}
            for key, value in self.input_phs.items():
                feed_dict[value] = inputs[key]
            return \
                self.sess.run(
                    self.action_gradients,
                    feed_dict=feed_dict)

    def save_checkpoint(self):
        """Saves the model checkpoint"""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)
