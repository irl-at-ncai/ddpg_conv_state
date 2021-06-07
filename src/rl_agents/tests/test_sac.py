#!/usr/bin/env python3
"""Tests the usage of rl_agents.sac.agent module"""

from rl_agents.common.agent_base import AgentBase


agent = AgentBase.get_agent('sac', 'any_env')
