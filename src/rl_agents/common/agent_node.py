#!/usr/bin/env python3
""" Initializes a ros node for training using a given agent. """

import rospy
from agent_base import AgentBase


if __name__ == '__main__':
    rospy.init_node(
        'agent_node', anonymous=True, log_level=rospy.INFO)
    # pylint: disable=invalid-name
    agent_name = rospy.get_param('~agent')
    agent = AgentBase.get_agent(agent_name)
    if agent is None:
        rospy.logfatal("No agent found for training.")
    rospy.loginfo('Using agent of type: {}'.format(agent.name))
    agent.start_training()
