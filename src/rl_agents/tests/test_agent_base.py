#!/usr/bin/env python3
"""Tests the usage of AgentBase class"""

from rl_agents.common.agent_base import AgentBase


AgentBase.get_agent('ddpg', 'any_env')
