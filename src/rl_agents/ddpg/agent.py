#!/usr/bin/env python3
"""
Defines a reinforcement learning agent based on deep deterministic
policy gradients.
"""

import importlib
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.losses import mean_squared_error
from rl_agents.common.agent_base import AgentBase
from rl_agents.common.state_preprocessors import ImagePreprocessor
from rl_agents.common.experience_memory import ExperienceMemory
from rl_agents.common.experience_memory import Transition
from rl_agents.common.ouanoise import OUActionNoise
from rl_agents.ddpg.actor import Actor, ActorInputs
from rl_agents.ddpg.critic import Critic, CriticInputs
from rl_agents.common.utils import plot_learning
import rospy

AgentState = \
    namedtuple("AgentState", ['image', 'robot_state'])


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
        network_input_image_shape = \
            rospy.get_param('ddpg/network_input_image_shape')
        actor_critic_model_version = \
            rospy.get_param('ddpg/actor_critic_model_version')

        if replay_memory_initial_size == -1:
            replay_memory_initial_size = self.batch_size

        # get environment space info
        # input to critic and output from actor. Shape is the same for both
        self.actions_input_shape = (self.env.action_space.shape[0],)
        self.actions_output_shape = self.actions_input_shape
        self.robot_state_input_shape = (
            self.env.observation_space['position'].shape[0] +
            self.env.observation_space['velocity'].shape[0],)
        self.image_input_shape = env.observation_space['front_cam'].shape
        self.image_depth_input_shape = \
            env.observation_space['front_cam_depth'].shape

        # state input info
        rospy.loginfo(
            "Initializing the network with following observations:")
        for idx, (key, value) in \
                enumerate(env.observation_space.spaces.items()):
            rospy.loginfo("{}) {} (shape = {})".format(idx, key, value.shape))

        # define a preprocessor for image input
        self.image_preprocessor = \
            ImagePreprocessor(
                input_shape=self.image_input_shape,
                output_shape=tuple(network_input_image_shape))

        rospy.loginfo(
            "Initialized the image preprocessor with the following "
            """parameters:
            1) input_shape (shape = {})
            2) output_shape (shape = {})
            """.format(
                self.image_preprocessor.input_shape,
                self.image_preprocessor.output_shape))

        # get the learning model used for critic
        self.critic_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.actor_critic_{}'.format(
                        self.name,
                        actor_critic_model_version.replace('v', ''))),
                'CriticModel')

        # get the learning model used for actor
        self.actor_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.actor_critic_{}'.format(
                        self.name,
                        actor_critic_model_version.replace('v', ''))),
                'ActorModel')

        # define experience memory
        self.noise = OUActionNoise(mean=np.zeros(self.actions_input_shape))
        self.exp_memory = \
            ExperienceMemory(
                init_size=replay_memory_initial_size,
                max_size=replay_memory_max_size)

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

        self.sess.run(tf.global_variables_initializer())
        self.update_target_network_parameters(first_update=True)
        rospy.loginfo('Done creating agent!')

    def preprocess_state(self, state):
        """
        Performs pre-processing operations on the state
        """
        # resize images
        agent_state = []
        image = \
            self.image_preprocessor.process(self.sess, state['front_cam'])
        robot_state = \
            np.concatenate((state['position'], state['velocity']))
        agent_state = AgentState(image=image, robot_state=robot_state)

        return agent_state

    def get_actor_inputs(self, agent_state):
        """
        Generates actor inputs based on single input state
        """
        # add new axes for single input
        return ActorInputs(
            robot_state=agent_state.robot_state[np.newaxis, ...],
            image=agent_state.image[np.newaxis, ...])

    def start_training(self):
        """ Trains the network """
        score_history = []
        np.random.seed(0)
        for eps in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            score = 0
            for step in range(self.max_episode_steps):
                actor_inputs = self.get_actor_inputs(state)
                action = self.choose_action(actor_inputs)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.preprocess_state(new_state)
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
        filename = rospy.get_param('ddpg/plot_file_name')
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
            image_preprocessor=self.image_preprocessor,
            robot_state_input_shape=self.robot_state_input_shape,
            actions_input_shape=self.actions_input_shape,
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
            image_preprocessor=self.image_preprocessor,
            robot_state_input_shape=self.robot_state_input_shape,
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

    def choose_action(self, inputs):
        """ Returns an action based on the current inputs. """
        action = self.actor.predict(inputs) + self.noise()
        return action[0]

    def learn(self):
        """
        Performs the DDPG update to train the actor and critic networks
        """
        for _, gpu in enumerate(["/gpu:0"]):
            with tf.device(gpu):
                samples = \
                    self.exp_memory.sample(self.batch_size)
                states_batch, actions_batch, rewards_batch, \
                    next_states_batch, done_batch, _ = \
                    map(np.array, zip(*samples))

                next_images, next_robot_states = \
                    map(np.array, zip(*next_states_batch))

                # target q-value(new_state) with actor's bounded action forward
                # pass
                target_actions = \
                    self.target_actor.predict(
                        ActorInputs(
                            image=next_images, robot_state=next_robot_states)
                    )

                q_values = \
                    self.target_critic.predict(
                        CriticInputs(
                            image=next_images,
                            robot_state=next_robot_states,
                            actions=target_actions))

                target = \
                    np.array([
                        rewards_batch[j] + self.discount_factor * q_values[j] *
                        done_batch[j]
                        for j in range(self.batch_size)
                    ])

                images, robot_states = \
                    map(np.array, zip(*states_batch))
                self.critic.train(
                    CriticInputs(
                        image=images,
                        robot_state=robot_states,
                        actions=actions_batch),
                    target)

                # a = mu(s_i)
                actor_inputs = \
                    ActorInputs(
                        image=images,
                        robot_state=robot_states
                    )
                next_actions = self.actor.predict(actor_inputs)

                # gradients of Q w.r.t actions
                grads = \
                    self.critic.get_action_gradients(
                        CriticInputs(
                            image=images,
                            robot_state=robot_states,
                            actions=next_actions))

                self.actor.train(actor_inputs, grads[0])
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
