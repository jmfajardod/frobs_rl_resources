#!/bin/python3

import re
import gym
import rospy
import sys
from frobs_rl.common import ros_gazebo, ros_urdf, ros_spawn, ros_node
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper

from kobuki_maze_rl.task_env import kobuki_empty
from kobuki_maze_rl.task_env import kobuki_dynamic_v3
from kobuki_maze_rl.task_env import kobuki_maze

from tf.transformations import quaternion_from_euler
import numpy as np

# Checker
from frobs_rl.models.utils import check_env

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()

    # Launch Gazebo 
    ros_gazebo.launch_Gazebo(paused=False,gui=True, pub_clock_frequency=100)

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('kobuki_maze_test')

    # Launch env
    # env = gym.make('KobukiEmptyEnv-v0')
    # env = gym.make('KobukiDynamicEnv-v3')
    env = gym.make('KobukiMazeEnv-v0')

    # Check env
    check_env(env)

    # obs = env.reset()
    # for ii in range(10000):
    #     action = env.action_space.sample()
    #     obs, _, dones, info = env.step(action)
    #     if dones:
    #         obs = env.reset()

    # env.close()
    # sys.exit()