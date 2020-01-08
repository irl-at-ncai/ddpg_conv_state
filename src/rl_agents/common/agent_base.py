#!/usr/bin/env python3

class AgentBase(object):
    def __init__(self, agent_name, env):
    	self.name = agent_name
    	self.env = env

    @staticmethod
    def get_agent(agent_name, env):
    	module = 'rl_agents.{}.agent'.format(agent_name)
    	exec('import {}'.format(module))
    	return exec('{}.Agent(env)'.format(module))


