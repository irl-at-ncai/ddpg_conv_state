#!/usr/bin/env python3
"""Tests the usage of rl_agents.ddpg.agent module"""

from rl_agents.common.agent_base import AgentBase


agent = AgentBase.get_agent('ddpg', 'any_env')
