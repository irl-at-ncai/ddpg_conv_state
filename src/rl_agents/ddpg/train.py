#!/usr/bin/env python3

import sys
import time
from agent import Agent
import gym
import numpy as np
from utils import plot_learning
from gym import wrappers
# ROS packages required
import rospy
import rospkg
import rosparam
# import training environment
sys.path.insert(1, '/home/hmi/Projects/catkin_ws/ParrotDrone-RL-Experiments/src/parrot_ardrone_rl/scripts/parrot_gym')
from make_gym_env import GymMake
from roscore_handler import Roscore

if __name__ == '__main__':
    roscore = Roscore()
    roscore.run()
    time.sleep(1)
    rospy.init_node('parrotdrone_test', anonymous=True, log_level=rospy.WARN)

    # Init Gym ENV
    task_env = 'ParrotDroneGoto-v0'
    env = GymMake(task_env)
    rospy.loginfo("Gym environment done")
    #env = gym.make('Pendulum-v0')

    # Set the logging system
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('parrot_ardrone_rl')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")

    obs_dims = [[13],[360,640,3]]
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=obs_dims, tau=0.001,
                env=env, batch_size=64, layer1_size=200, layer2_size=200)

    score_history = []
    np.random.seed(0)
    nepisodes = 1000
    for ep in range(nepisodes):
        obs = env.reset()
        done = False
        score = 0
        for i in range(500):
        #while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score +=reward
            obs = new_state
            #env.render() To be linked with ROS
        score_history.append(score)
        print('episode', i, 'score %.2f' % score,'100 game average %.2f' % np.mean(score_history[-100:]))
        if i+1 % 200 == 0:
            agent.save_models()
    env.close()
    roscore.terminate()
    filename = 'ParrotDrone.png'
    plot_learning(score_history, filename, window=100)
    agent.save_models()