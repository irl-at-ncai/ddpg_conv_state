#!/usr/bin/env python3
"""Defines the base class with common functionality across agents"""

import importlib
import os
import rospy
import tensorflow as tf
from agent_ros_wrapper import AgentRosWrapper
tf.compat.v1.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'


class AgentBase(object):
    """
    The base class for agents
    """

    @classmethod
    def get_agent(cls, agent_name):
        """Returns an agent of given name

        Parameters
        ----------
        agent_name : str
            Name of the agent

        Returns
        ------
        AgentBase
            An reinforcement learning agent of base type AgentBase()
        """
        module = \
            importlib.import_module('rl_agents.{}.agent'.format(agent_name))
        return module.Agent(agent_name)

    def __init__(
        self, agent_name, use_gpu=True, init_noise=True, init_exp_memory=True):
        # initialize the base
        self.name = agent_name
        gpu_options = tf.compat.v1.GPUOptions(visible_device_list="0")
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                gpu_options=gpu_options, log_device_placement=True))
        self.env_wrapper = \
            AgentRosWrapper(rospy.get_param("ros_gym/environment_name"))

        # initialize base training parameters
        self.discount_factor = rospy.get_param('agent_base/discount_factor')
        self.batch_size = rospy.get_param('agent_base/batch_size')
        self.n_episodes = rospy.get_param('agent_base/n_episodes')
        self.max_episode_steps = rospy.get_param('agent_base/max_episode_steps')
        self.replay_memory_initial_size = \
            rospy.get_param('ddpg/replay_memory_initial_size')
        self.replay_memory_max_size = \
            rospy.get_param('ddpg/replay_memory_max_size')
        self.lr = rospy.get_param('ddpg/lr')
        if self.replay_memory_initial_size == -1:
            self.replay_memory_initial_size = self.batch_size

        # get observation and action space info from the ros wrapper env_info
        # service call
        spaces = self.env_wrapper.info_env()
        observation_space, action_space = \
            self.env_wrapper.space_from_ros(
                spaces.observation_space, spaces.action_space)

        # setup shapes for all input/output actions
        self.actions_shape = space_to_shape(action_space)

        # setup shapes for all input observations
        self.input_shapes_env = {}
        for key, obs in observation_space.items():
            self.input_shapes_env[key] = space_to_shape(obs)

        # state input info
        rospy.loginfo(
            "Initializing the network with following observations: ")
        for idx, (key, value) in enumerate(observation_space.items()):
            rospy.loginfo(
                "{}) {} (shape = {})".format(idx, key, value.shape))
        rospy.loginfo(
            "And actions of following shape: {} ".format(self.actions_shape))

        # get input preprocessor from child agent
        preprocessor_cls = self.get_state_preprocessor()

        # convert the shapes from environment to shapes that are needed as
        # network niput
        self.preprocessor = preprocessor_cls(self.input_shapes_env)
        # state_input_shapes = only observations
        # input_shapes = observations + actions
        self.state_input_shapes = self.preprocessor.input_shapes # observations
        self.input_shapes = copy.deepcopy(self.state_input_shapes)
        self.input_shapes['actions'] = actions_shape # actions

        # define noise if required
        if init_noise:
            self.noise = \
                OUActionNoise(mean=np.zeros(self.input_shapes['actions']))

        # define experience memory if required
        if init_exp_memory:
            self.exp_memory = \
                ExperienceMemory(
                    state_inputs=self.input_shapes,
                    init_size=self.replay_memory_initial_size,
                    max_size=self.replay_memory_max_size)

        # define gpu device if available
        if use_gpu:
            self.device = ['/gpu:0']

    def get_state_preprocessor(self):
        """
        Returns the associated preprocessor for received observations states
        to network input.
        """
        raise NotImplementedError()

    @classmethod
    def space_to_shape(space):
        if (isinstance(space, Discrete)):
            return (space.n,)
        elif (isinstance(space, Box)):
            if space.shape[1] == 1:
                return (space.shape[0],)
            else:
                return space.shape

    def start_training(self):
        """ Main training loop of the network """
        score_history = []
        np.random.seed(0)
        for eps in range(self.n_episodes):
            # get first state and preprocess it for the network
            state = self.preprocessor.process(
                self.env_wrapper.reset_env(), self.sess)
            done = False
            score = 0
            for step in range(self.max_episode_steps):
                action = self.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.preprocessor.process(new_state, self.sess)
                self.exp_memory.add(
                    Transition(state, action, reward, new_state, done, eps))
                if self.exp_memory.size >= self.batch_size:
                    self.update_network()
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

    def update_network(self):
        """ Main network update step for a given agent """
        raise NotImplementedError()