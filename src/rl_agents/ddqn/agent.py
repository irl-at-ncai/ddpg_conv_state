#!/usr/bin/env python3
"""
Defines a reinforcement learning agent based on deep deterministic
policy gradients.
"""

import importlib
from collections import namedtuple
import numpy as np
import rospy
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
from rl_agents.common.agent_base import AgentBase
from rl_agents.common.state_preprocessors import ImagePreprocessor
from rl_agents.common.experience_memory import ExperienceMemory
from rl_agents.common.experience_memory import Transition
from rl_agents.ddqn.model import DeepQNetwork, ModelInputs
from rl_agents.common.utils import plot_learning


tf.disable_v2_behavior()

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
        self.discount_factor = rospy.get_param('ddqn/discount_factor')
        # The tau parameter for weighted target network update
        self.target_soft_update_weight = \
            rospy.get_param('ddqn/target_soft_update_weight')
        self.batch_size = rospy.get_param('ddqn/batch_size')
        self.n_episodes = rospy.get_param('ddqn/n_episodes')
        self.max_episode_steps = rospy.get_param('ddqn/max_episode_steps')

        self.learning_rate = rospy.get_param('ddqn/lr')
        self.epsilon = rospy.get_param('ddqn/epsilon_starting')
        self.epsilon_decay = rospy.get_param('ddqn/epsilon_decay')
        self.epsilon_end = rospy.get_param('ddqn/epsilon_ending')
        self.n_actions = rospy.get_param('ddqn/n_actions')

        replay_memory_initial_size = \
            rospy.get_param('ddqn/replay_memory_initial_size')
        replay_memory_max_size = \
            rospy.get_param('ddqn/replay_memory_max_size')
        network_input_image_shape = \
            rospy.get_param('ddqn/network_input_image_shape')
        ddqn_model_version = \
            rospy.get_param('ddqn/ddqn_model_version')

        if replay_memory_initial_size == -1:
            replay_memory_initial_size = self.batch_size

        # get environment space info
        # input to critic and output from actor. Shape is the same for both

        self.robot_state_input_shape = (
            self.env.observation_space['position'].shape[0] +
            self.env.observation_space['velocity'].shape[0],)
        # self.image_input_shape = env.observation_space['front_cam'].shape
        self.image_input_shape = env.observation_space['front_cam_depth'].shape

        # state input info
        rospy.loginfo(
            """Initializing the network with following inputs:
            1) robot_state_input_shape (shape = {})
            2) image_input_shape (shape = {})
            3) n_actions (shape = {})
            """.format(
                self.robot_state_input_shape,
                self.image_input_shape,
                self.n_actions))

        # define a preprocessor for image input
        self.image_preprocessor = \
            ImagePreprocessor(
                input_shape=self.image_input_shape,
                output_shape=tuple(network_input_image_shape))

        rospy.loginfo(
            """Initialized the image preprocessor with the following
                parameters:
            1) input_shape (shape = {})
            2) output_shape (shape = {})
            """.format(
                self.image_preprocessor.input_shape,
                self.image_preprocessor.output_shape))

        # get the learning model used for ddqn
        self.ddqn_model = \
            getattr(
                importlib.import_module(
                    'rl_agents.{}.ddqn_{}'.format(
                        self.name,
                        ddqn_model_version.replace('v', ''))),
                'DDQNModel')

        # define experience memory
        self.exp_memory = \
            ExperienceMemory(
                init_size=replay_memory_initial_size,
                max_size=replay_memory_max_size)

        # define ddqn models
        self.original_model = self.make_model(
            scope='ddqn_original_model',
            summaries_dir='checkpoints/ddqn/original_model')
        self.target_model = self.make_model(
            scope='ddqn_target_model',
            summaries_dir='checkpoints/ddqn/target_model')

        self.update_model = [
            self.target_model.params[i].assign(
                tf.multiply(
                    self.original_model.params[i],
                    self.target_soft_update_weight) +
                tf.multiply(
                    self.target_model.params[i],
                    1. - self.target_soft_update_weight))
            for i in range(len(self.target_model.params))
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
            self.image_preprocessor.process(self.sess, state['front_cam_depth'])
        robot_state = \
            np.concatenate((state['position'], state['velocity']))
        agent_state = AgentState(image=image, robot_state=robot_state)

        return agent_state

    def get_model_inputs(self, agent_state):
        """
        Generates actor inputs based on single input state
        """
        # add new axes for single input
        return ModelInputs(
            robot_state=agent_state.robot_state[np.newaxis, ...],
            image=agent_state.image[np.newaxis, ...])

    def start_training(self):
        """ Trains the network """
        score_history = []
        np.random.seed(0)
        for eps in range(self.n_episodes):
            state = self.env.reset()
            while state is None:
                print('State is none, resetting')
                state = self.env.reset()

            state = self.preprocess_state(state)
            done = False
            score = 0
            i = 0
            for _ in range(self.max_episode_steps):
                actor_inputs = self.get_model_inputs(state)
                action = self.choose_action(actor_inputs)
                new_state, reward, done, _ = self.env.step(action)
                if new_state is None:
                    print('New stat was nose, new episode starting')
                    break
                new_state = self.preprocess_state(new_state)
                self.exp_memory.add(
                    Transition(state, action, reward, new_state, done, eps))
                if self.exp_memory.size >= self.batch_size:
                    self.learn()
                score += reward
                print('score', score)
                i+=1
                if done:
                    break
                state = new_state
                # env.render() To be linked with ROS
            score_history.append(score)
            print('''Episode {} - Score {} - 100 game average {} -
                    Steps taken {}'''.format(
                        eps, score, np.mean(score_history[-100:]), i))

            if eps + 1 % 50 == 0:
                self.save_models()
        self.env.close()
        filename = rospy.get_param('ddqn/plot_file_name')
        plot_learning(score_history, filename, window=100)
        self.save_models()

    def make_model(
            self,
            scope,
            summaries_dir):
        """
        Initializes and returns an actor
        """
        return DeepQNetwork(
            sess=self.sess,
            image_preprocessor=self.image_preprocessor,
            robot_state_input_shape=self.robot_state_input_shape,
            n_actions=self.n_actions,
            learning_rate=self.learning_rate,
            model=self.ddqn_model,
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
                    self.target_model.sess.run(self.update_model)
                    self.target_soft_update_weight = \
                        old_target_soft_update_weight
                else:
                    self.target_model.sess.run(self.update_model)

    def choose_action(self, inputs):
        """ Returns an action based on the current inputs. """

        rand = np.random.random()
        self.epsilon = self.epsilon_decay*self.epsilon
        self.epsilon = max(self.epsilon, self.epsilon_end)
        if rand < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = self.original_model.predict(inputs)
            action = np.argmax(action)

        return action

    def learn(self):
        """
        Performs the ddqn update to train the actor and critic networks
        """
        for _, gpu in enumerate(["/gpu:0"]):
            with tf.device(gpu):
                samples = self.exp_memory.sample(self.batch_size)

                states, actions, rewards, next_states,\
                    done, _ = map(np.array, zip(*samples))

                current_images, current_robot_states = map(np.array,
                                                           zip(*states))
                current_qs_list = self.original_model.predict(
                    ModelInputs(
                        image=current_images,
                        robot_state=current_robot_states))

                next_images, next_robot_states = map(np.array,
                                                     zip(*next_states))

                future_qs_list = self.target_model.predict(
                    ModelInputs(
                        image=next_images,
                        robot_state=next_robot_states))

                training_qs = []

                for index in range(self.batch_size):

                    if not done[index]:
                        future_action = np.argmax(current_qs_list[index])
                        future_q = future_qs_list[index]
                        new_q = rewards[index] + self.discount_factor * \
                            future_q[future_action]
                    else:
                        new_q = rewards[index]

                    current_qs = current_qs_list[index]
                    current_qs[actions[index]] = new_q

                    training_qs.append(current_qs)

                self.original_model.train(
                    ModelInputs(image=current_images,
                                robot_state=current_robot_states),
                    q_target=training_qs)

                self.update_target_network_parameters(first_update=False)

    def save_models(self):
        """ Saves a model from a checkpoint file. """
        self.original_model.save_checkpoint()
        self.target_model.save_checkpoint()

    def load_models(self):
        """ Loads a model from a checkpoint file. """
        self.original_model.load_checkpoint()
        self.target_model.load_checkpoint()
