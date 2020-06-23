#!/usr/bin/env python3

from collections import namedtuple
import os
from rl_agents.ddqn.ddqn_1 import DDQNModel
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v1.losses import mean_squared_error
import rospkg
tf.disable_v2_behavior()


ModelInputs = \
    namedtuple("ModelInputs", ['image', 'robot_state'])


class DeepQNetwork:
    """ Class for Double Deep Q Network

        Parameters
        ----------
        sess : Session
            Tensorflow session
        image_preprocessor: ImagePreprocessor
            The preprocessor for image inputs
        robot_state_input_shape: tuple
            Shape of the input state of the robot state
        n_actions: int
            Number of actions
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
    # pylint: disable=no-member
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            sess,
            image_preprocessor,
            robot_state_input_shape,
            n_actions,
            learning_rate,
            model=DDQNModel,
            loss_fn=mean_squared_error,
            optimizer=AdamOptimizer,
            scope='ddqn',
            summaries_dir='checkpoints/ddqn/',
            gpu="/gpu:0"):

        with tf.name_scope(scope):
            self.sess = sess
            self.image_preprocessor = image_preprocessor
            self.robot_state_input_shape = robot_state_input_shape
            self.n_actions = n_actions
            self.learning_rate = learning_rate
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.scope = scope
            self.gpu = gpu
            self.model = model(
                image_input_shape=self.image_preprocessor.output_shape,
                robot_state_input_shape=self.robot_state_input_shape,
                actions_shape=self.n_actions,
                scope=self.scope)

            ros_pack = rospkg.RosPack()
            path = ros_pack.get_path('rl_agents')
            path = path + '/src/rl_agents/ddqn'
            summaries_dir = os.path.join(path, summaries_dir)

            self.checkpoint_file = os.path.join(
                summaries_dir, scope)

            # Writes Tensorboard summaries to disk
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, scope)
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir, \
                                                           self.sess.graph_def)

            with tf.device(self.gpu):
                # Build the graph
                self.build()
            self.summaries = tf.summary.scalar('Loss', self.loss)
            self.saver = tf.train.Saver()
            self.counter = 1

    def build(self):

        # pylint: disable=too-many-locals
        # pylint: disable=invalid-name, attribute-defined-outside-init
        """ Function for creating the model graph"""
        with tf.variable_scope(self.scope):
            self.img_input = tf.placeholder(
                dtype=tf.float32,
                shape=(None, *self.image_preprocessor.output_shape),
                name='image_input')

            self.robot_state_input = tf.placeholder(
                dtype=tf.float32,
                shape=(None, *self.robot_state_input_shape),
                name='robot_state_input')

            self.actions = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.n_actions],
                name='actions')

            self.q_target = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.n_actions],
                name='targets')

            self.q_value = self.model(
                ModelInputs(
                    self.img_input,
                    self.robot_state_input))

            self.loss = self.loss_fn(self.q_value, self.q_target)
            self.params = tf.trainable_variables(scope=self.scope)
            self.optimize = \
                self.optimizer(self.learning_rate).minimize(self.loss)

    def save_checkpoint(self):
        """Saves the model checkpoint"""
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        """Loads the model checkpoint"""
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def predict(self, inputs):
        """
        Performs the forward pass for the network to predict action Q-value
        """
        with tf.device(self.gpu):
            return \
                self.sess.run(
                    self.q_value,
                    feed_dict={
                        self.img_input: inputs.image,
                        self.robot_state_input: inputs.robot_state})

    def train(self, inputs, q_target):
        """
        Performs the back-propagation of gradient for loss minimization
        """
        with tf.device(self.gpu):
            if True:  # @todo: summaries have errors

                _, summaries = \
                    self.sess.run(
                        [
                            self.optimize,
                            self.summaries],
                        feed_dict={
                            self.img_input: inputs.image,
                            self.robot_state_input: inputs.robot_state,
                            self.q_target: q_target})
                if self.counter % 10 == 0:
                    print('Writing data to tensorboard')
                    self.summary_writer.add_summary(summaries, self.counter)
                self.counter += 1
            else:

                _ = \
                    self.sess.run(
                        self.optimize,
                        feed_dict={
                            self.img_input: inputs.image,
                            self.robot_state_input: inputs.robot_state,
                            self.q_target: q_target})
