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
from rl_agents.sac.actor import Actor
from rl_agents.sac.critic import Critic
from rl_agents.common.utils import plot_learning
import rospy
from gym.spaces import Discrete, Box


class Agent(AgentBase):
    """
    A reinforcement learning agent that uses soft actor critc (SAC) algorithm.

    Parameters
    ----------
    agent_name: str
        Name of the agent that also corresponds to its folder directory
    env: gym.Env
        The underlying gym environment the agent acts on
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, agent_name, env):
        super(Agent, self).__init__(agent_name=agent_name, env=env)

        # get all configuration parameters
        self.discount_factor = rospy.get_param('sac/discount_factor')
        # The tau parameter for weighted target network update
        self.target_soft_update_weight = \
            rospy.get_param('sac/target_soft_update_weight')
        self.batch_size = rospy.get_param('sac/batch_size')
        self.n_episodes = rospy.get_param('sac/n_episodes')
        self.max_episode_steps = rospy.get_param('sac/max_episode_steps')
        self.lr_critic = rospy.get_param('sac/lr_critic')
        self.lr_actor = rospy.get_param('sac/lr_actor')
        replay_memory_initial_size = \
            rospy.get_param('sac/replay_memory_initial_size')
        replay_memory_max_size = \
            rospy.get_param('sac/replay_memory_max_size')
        actor_critic_model = \
            rospy.get_param('sac/actor_critic_model')

        if replay_memory_initial_size == -1:
            replay_memory_initial_size = self.batch_size

        # get environment space info
        # input to critic and output from actor. Shape is the same for both
        if (isinstance(self.env.action_space, Discrete)):
            actions_input_shape = (self.env.action_space.n,)
        elif (isinstance(self.env.action_space, Box)):
            actions_input_shape = (self.env.action_space.shape[0],)
        self.actions_output_shape = actions_input_shape
        self.input_shapes_env = {}
        for key, obs in self.env.observation_space.spaces.items():
            self.input_shapes_env[key] = obs.shape

        # state input info
        rospy.loginfo(
            "Initializing the network with following observations:")
        for idx, (key, value) in \
                enumerate(env.observation_space.spaces.items()):
            rospy.loginfo(
                "{}) {} (shape = {})".format(idx, key, value.shape))

        # get the learning model used for critic
        self.preprocessor = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, actor_critic_model)),
                'PreprocessHandler')
        self.preprocessor = self.preprocessor(self.input_shapes_env)

        # get the learning model used for critic
        self.critic_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, actor_critic_model)),
                'CriticModel')

        # get the learning model used for actor
        self.actor_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, actor_critic_model)),
                'ActorModel')

        self.actor_input_shapes = self.preprocessor.input_shapes
        self.input_shapes = copy.deepcopy(self.actor_input_shapes)

        # add actions to critic inputs
        self.input_shapes['actions'] = actions_input_shape

        # define experience memory
        self.noise = \
            OUActionNoise(mean=np.zeros(self.input_shapes['actions']))
        self.exp_memory = \
            ExperienceMemory(
                state_inputs=self.input_shapes,
                init_size=replay_memory_initial_size,
                max_size=replay_memory_max_size)

        # define critics
        self.critic = \
            self.make_critic(
                scope='critic', summaries_dir='tmp/sac/critic')
        self.target_critic = \
            self.make_critic(
                scope='target_critic', summaries_dir='tmp/sac/target_critic')

        # define actors
        self.actor = \
            self.make_actor(
                scope='actor', summaries_dir='tmp/sac/actor')
        self.target_actor = \
            self.make_actor(
                scope='target_actor', summaries_dir='tmp/sac/target_actor')

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
        rospy.loginfo('Done creating agent!')

    def start_training(self):
        """ Trains the network """
        score_history = []
        np.random.seed(0)
        for eps in range(self.n_episodes):
            state = self.preprocessor.process(self.env.reset(), self.sess)
            done = False
            score = 0
            for step in range(self.max_episode_steps):
                action = self.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.preprocessor.process(new_state, self.sess)
                self.exp_memory.add(
                    Transition(state, action, reward, new_state, done, eps))
                if self.exp_memory.size >= self.batch_size:
                    self.learn()
                score += reward
                rospy.loginfo(
                    '''Epsiode step#{}: Score = {}'''.format(step, score))
                if done:
                    break
                state = new_state
                # env.render() To be linked with ROS
            score_history.append(score)
            rospy.loginfo(
                '''Episode {} - Score {} - 100 game average {}'''.format(
                    eps, score, np.mean(score_history[-100:])))
            if eps + 1 % 200 == 0:
                self.save_models()
        self.env.close()
        filename = rospy.get_param('sac/plot_file_name')
        plot_learning(score_history, filename, window=100)
        self.save_models()

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
            gpu="/gpu:0")

    def make_actor(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns an actor
        """
        return Actor(
            sess=self.sess,
            input_shapes=self.actor_input_shapes,
            # output actions from actor are input to critic
            actions_output_shape=self.actions_output_shape,
            action_bound=self.env.action_space.high,
            learning_rate=self.lr_critic,
            batch_size=self.batch_size,
            model=self.actor_model,
            optimizer=AdamOptimizer,
            scope=scope,
            summaries_dir=summaries_dir,
            gpu="/gpu:0")

    def update_target_network_parameters(self, first_update=False):
        """
        Updates the target networks from main networks with a soft-update
        """
        for _, gpu in enumerate(["/gpu:0"]):
            with tf.device(gpu):
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

    def learn(self):
        """
        Performs the SAC update to train the actor and critic networks
        """
        for _, gpu in enumerate(["/gpu:0"]):
            with tf.device(gpu):
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

                # why is gradient[0] used?
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
