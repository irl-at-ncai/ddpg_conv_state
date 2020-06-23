#!/usr/bin/env python3
"""
Defines a reinforcement learning agent based on deep deterministic
policy gradients.
"""

import importlib
import copy
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.compat.v1.losses import mean_squared_error
from rl_agents.common.agent_base import AgentBase
from rl_agents.common.experience_memory import ExperienceMemory
from rl_agents.common.experience_memory import Transition
from rl_agents.common.ouanoise import OUActionNoise
from rl_agents.ddpg.actor import Actor
from rl_agents.ddpg.critic import Critic
from rl_agents.common.utils import plot_learning
import rospy
from gym.spaces import Discrete, Box


class Agent(AgentBase):
    def get_state_preprocessor(self):
        # get the preprocessor class
        return
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, actor_critic_model)),
                'PreprocessHandler')

    """
    A reinforcement learning agent that uses deep deterministic
    policy gradients (DDPG).

    Parameters
    ----------
    agent_name: str
        Name of the agent that also corresponds to its folder directory
    env: gym.Env
        The underlying gym environment the agent acts on
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, agent_name):
        super(Agent, self).__init__(agent_name=agent_name)

        # get ddpg specific configuration parameters
        # tau parameter for weighted target network update
        self.target_soft_update_weight = \
            rospy.get_param('ddpg/target_soft_update_weight')
        self.lr_critic = rospy.get_param('ddpg/lr_critic')
        self.lr_actor = rospy.get_param('ddpg/lr_actor')
        actor_critic_model_name = \
            rospy.get_param('ddpg/actor_critic_model')

        # get the learning model used for critic
        self.critic_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(
                        self.name, actor_critic_model_name)),
                'CriticModel')

        # get the learning model used for actor
        self.actor_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(
                        self.name, actor_critic_model_name)),
                'ActorModel')

        # define critics
        self.critic = \
            self.make_critic(
                scope='critic', summaries_dir='tmp/ddpg/critic')
        self.target_critic = \
            self.make_critic(
                scope='target_critic', summaries_dir='tmp/ddpg/target_critic')

        # define actors
        self.actor = \
            self.make_actor(
                scope='actor', summaries_dir='tmp/ddpg/actor')
        self.target_actor = \
            self.make_actor(
                scope='target_actor', summaries_dir='tmp/ddpg/target_actor')

        self.update_actor = [
            self.target_actor.params[i].assign(
                tf.multiply(
                    self.actor.params[i],
                    self.target_soft_update_weight) +
                tf.multiply(
                    self.target_actor.params[i],
                    1. - self.target_soft_update_weight))
            for i in range(len(self.target_actor.params))
        ]

        self.update_critic = [
            self.target_critic.params[i].assign(
                tf.multiply(
                    self.critic.params[i],
                    self.target_soft_update_weight) +
                tf.multiply(
                    self.target_critic.params[i],
                    1. - self.target_soft_update_weight))
            for i in range(len(self.target_critic.params))
        ]

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.update_target_network_parameters(first_update=True)
        rospy.loginfo('Agent initialized successfully!')

    def make_critic(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns a critic
        """
        return Critic(
            sess=self.sess,
            input_shapes=self.input_shapes,
            learning_rate=self.lr_critic,
            model=self.critic_model,
            loss_fn=mean_squared_error,
            optimizer=AdamOptimizer,
            scope=scope,
            summaries_dir=summaries_dir,
            device=self.device)

    def make_actor(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns an actor
        """
        return Actor(
            sess=self.sess,
            input_shapes=self.state_input_shapes,
            # output actions from actor are input to critic
            actions_output_shape=self.actions_shape,
            action_bound=action_space.high,
            learning_rate=self.lr_critic,
            batch_size=self.batch_size,
            model=self.actor_model,
            optimizer=AdamOptimizer,
            scope=scope,
            summaries_dir=summaries_dir,
            device=self.device)

    def update_target_network_parameters(self, first_update=False):
        """
        Updates the target networks from main networks with a soft-update
        """
        for _, device in enumerate(self.device):
            with tf.device(device):
                if first_update:
                    old_target_soft_update_weight = \
                        self.target_soft_update_weight
                    self.target_soft_update_weight = 1.0
                    self.target_actor.sess.run(self.update_actor)
                    self.target_critic.sess.run(self.update_critic)
                    self.target_soft_update_weight = \
                        old_target_soft_update_weight
                else:
                    self.target_critic.sess.run(self.update_critic)
                    self.target_actor.sess.run(self.update_actor)

    def choose_action(self, state):
        """ Returns an action based on the current state input. """
        action = self.actor.predict(state) + self.noise()
        return action[0]

    def update_network(self):
        """
        Performs the DDPG update to train the actor and critic networks
        """
        for _, device in enumerate(self.device):
            with tf.device(device):
                samples = \
                    self.exp_memory.sample(self.batch_size)

                # target q-value(new_state) with actor's bounded action forward
                # pass
                target_actions = \
                    self.target_actor.predict(samples['s_next'])

                q_values = \
                    self.target_critic.predict(
                        {**samples['s_next'], "actions": target_actions}
                    )

                target = \
                    np.array([
                        samples['r'][j] + self.discount_factor * q_values[j] *
                        samples['done'][j]
                        for j in range(self.batch_size)
                    ])

                self.critic.train(
                    {**samples['s'], "actions": samples['a']}, target)

                # a = mu(s_i)
                next_actions = self.actor.predict(samples['s'])
                # gradients of Q w.r.t actions
                grads = \
                    self.critic.get_action_gradients(
                        {**samples['s'], "actions": next_actions})

                # why is gradient zero?
                self.actor.train(samples['s'], grads[0])
                self.update_target_network_parameters(first_update=False)

    def save_models(self):
        """ Saves a model from a checkpoint file. """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """ Loads a model from a checkpoint file. """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
