"""
Defines the AgentRosWrapper class.
"""

import importlib
import rospy
from ros_gym.task_envs.task_env_map import TASK_ENV_ROS_MAP


class AgentRosWrapper():
    """
    Provides the functionality to agents to interact with a running gym
    environment through ros.
    """
    def __init__(self, env_name):
        env_file = 'ros_gym.task_envs.' + env_name[:-3] + "." + env_name
        self.ros_wrapper = \
            getattr(
                importlib.import_module(env_file), TASK_ENV_ROS_MAP[env_name])

        self._check_service_ready('/env_info')
        self.env_info_srv = \
            rospy.ServiceProxy('/env_info', self.ros_wrapper.info_srv_type())

        self._check_service_ready('/env_reset')
        self.env_reset_srv = \
            rospy.ServiceProxy('/env_reset', self.ros_wrapper.reset_srv_type())

        self._check_service_ready('/env_step')
        self.env_step_srv = \
            rospy.ServiceProxy('/env_step', self.ros_wrapper.step_srv_type())

    def _check_service_ready(self, name, timeout=5.0):
        try:
            rospy.wait_for_service(name, timeout)
        except (rospy.ServiceException, rospy.ROSException):
            rospy.logerr("Service {} unavailable.".format(name))

    def info_env(self):
        """Wrapper for environment info service call."""
        return self.env_info_srv()

    def reset_env(self):
        """Wrapper for environment reset service call."""
        return self.env_reset_srv().state

    def step_env(self):
        """Wrapper for environment step service call."""
        return self.env_step_srv(self.ros_wrapper.action_type())

    def state_from_ros(self, state_ros):
        """Wrapper for converting state from ros state msg."""
        return self.ros_wrapper.state_from_ros(state_ros)

    def space_from_ros(self, observation_space_ros, action_space_ros):
        """Wrapper for converting space from ros space msg."""
        return \
            self.ros_wrapper.space_from_ros(
                observation_space_ros, action_space_ros)

    def action_to_ros(self, action_env):
        """Wrapper for converting action to a ros action msg."""
        return self.ros_wrapper.action_to_ros(action_env)
