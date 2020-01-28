#!/usr/bin/env python3
"""
Defines a reinforcement learning agent based on deep deterministic
policy gradients.
"""

import importlib
import copy
import numpy as np
import tensorflow as tf
import gym
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


class Agent(AgentBase):
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
    def __init__(self, agent_name, env):
        super(Agent, self).__init__(agent_name=agent_name, env=env)

        # get all configuration parameters
        self.discount_factor = rospy.get_param('ddpg/discount_factor')
        # The tau parameter for weighted target network update
        self.target_soft_update_weight = \
            rospy.get_param('ddpg/target_soft_update_weight')
        self.batch_size = rospy.get_param('ddpg/batch_size')
        self.n_episodes = rospy.get_param('ddpg/n_episodes')
        self.max_episode_steps = rospy.get_param('ddpg/max_episode_steps')
        self.lr_critic = rospy.get_param('ddpg/lr_critic')
        self.lr_actor = rospy.get_param('ddpg/lr_actor')
        replay_memory_initial_size = \
            rospy.get_param('ddpg/replay_memory_initial_size')
        replay_memory_max_size = \
            rospy.get_param('ddpg/replay_memory_max_size')
        self.model_name = \
            rospy.get_param('ddpg/model')

        if replay_memory_initial_size == -1:
            replay_memory_initial_size = self.batch_size

        # get environment space info
        # input to critic and output from actor. Shape is the same for both
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions_shape = (self.env.action_space.n,)
        else:
            self.actions_shape = (self.env.action_space.shape[0],)
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
        print(self.model_name)
        self.preprocessor = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, self.model_name)),
                'PreprocessHandler')
        self.preprocessor = self.preprocessor(self.input_shapes_env)

        self.actor_input_shapes = self.preprocessor.input_shapes
        self.input_shapes = copy.deepcopy(self.actor_input_shapes)
        # add actions to critic inputs
        self.input_shapes['actions'] = self.actions_shape

        # define experience memory
        self.noise = \
            OUActionNoise(mean=np.zeros(self.input_shapes['actions']))
        self.exp_memory = \
            ExperienceMemory(
                state_inputs=self.input_shapes,
                init_size=replay_memory_initial_size,
                max_size=replay_memory_max_size)

        self.shared_gpu = "/job:localhost/replica:0/task:0/device:GPU:0"
        self.actor_gpu = "/job:localhost/replica:0/task:0/device:GPU:0"
        self.critic_gpu = "/job:localhost/replica:0/task:0/device:GPU:0"

        # define critics
        self.critic = \
            self.make_critic(
                scope='critic',
                summaries_dir='tmp/ddpg/critic')
        self.target_critic = \
            self.make_critic(
                scope='target_critic',
                summaries_dir='tmp/ddpg/target_critic')

        # define actors
        self.actor = \
            self.make_actor(
                scope='actor',
                summaries_dir='tmp/ddpg/actor')

        self.target_actor = \
            self.make_actor(
                scope='target_actor',
                summaries_dir='tmp/ddpg/target_actor')

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

        with tf.device(self.shared_gpu):
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.update_target_network_parameters(first_update=True)
        rospy.loginfo('Done creating agent!')

    def start_training(self):
        """ Trains the network """
        score_history = []
        np.random.seed(0)
        for eps in range(self.n_episodes):
            self.global_step = 0
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
                # rospy.loginfo(
                #     '''Epsiode step#{} - Score = {}'''.format(step, score))
                if done:
                    break
                state = new_state
                # self.env.render()
            score_history.append(score)
            rospy.loginfo(
                '''Episode {} - Score {} - 100 game average {}'''.format(
                    eps, score, np.mean(score_history[-100:])))
            if eps + 1 % 200 == 0:
                self.save_models()
        self.env.close()
        filename = rospy.get_param('ddpg/plot_file_name')
        plot_learning(score_history, filename, window=100)
        self.save_models()

    def learn(self):
        """
        Performs the DDPG update to train the actor and critic networks
        """
        self.global_step += 1
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
                    {**samples['s'], "actions": samples['a']},
                    target,
                    step=self.global_step)

                # a = mu(s_i)
                next_actions = self.actor.predict(samples['s'])
                # gradients of Q w.r.t actions
                grads = \
                    self.critic.get_action_gradients(
                        {**samples['s'], "actions": next_actions})

                # why is gradient zero?
                self.actor.train(samples['s'], grads[0])
                self.update_target_network_parameters(first_update=False)

    def make_critic(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns a critic
        """
        # get the learning model used for critic
        critic_model_fn = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, self.model_name)),
                'CriticModel')

        critic_model = \
            critic_model_fn(
                input_shapes=self.input_shapes,
                scope=scope+'_model',
                gpu=self.critic_gpu
            )
        return Critic(
            sess=self.sess,
            input_shapes=self.input_shapes,
            learning_rate=self.lr_critic,
            model=critic_model,
            loss_fn=mean_squared_error,
            optimizer=AdamOptimizer,
            scope=scope,
            summaries_dir=summaries_dir,
            gpu=self.critic_gpu)

    def make_actor(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns an actor
        """
        # get the learning model used for actor
        actor_model_fn = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.{}'.format(self.name, self.model_name)),
                'ActorModel')

        # initialize actor model
        action_bound = None
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            action_bound = self.env.action_space.high
        actor_model = \
            actor_model_fn(
                action_bound=action_bound,
                input_shapes=self.actor_input_shapes,
                actions_output_shape=self.actions_shape,
                scope=scope+'_model',
                gpu=self.actor_gpu
            )

        return Actor(
            sess=self.sess,
            input_shapes=self.actor_input_shapes,
            # output actions from actor are input to critic
            actions_output_shape=self.actions_shape,
            learning_rate=self.lr_actor,
            batch_size=self.batch_size,
            model=actor_model,
            optimizer=AdamOptimizer,
            scope=scope,
            summaries_dir=summaries_dir,
            gpu=self.actor_gpu)

    def update_target_network_parameters(self, first_update=False):
        """
        Updates the target networks from main networks with a soft-update
        """
        for gpu in [self.shared_gpu]:
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
