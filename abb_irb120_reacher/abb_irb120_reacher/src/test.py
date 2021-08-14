#!/bin/python3

from abb_irb120_reacher.task_env.irb120_reacher import abb_irb120_moveit
import gym
import rospy
import sys
from gym_gazebo_sb3.common import ros_gazebo
from gym_gazebo_sb3.common.ros_node import ROS_Kill_All_processes
from gym_gazebo_sb3.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from gym_gazebo_sb3.wrappers.TimeLimitWrapper import TimeLimitWrapper
from gym_gazebo_sb3.wrappers.NormalizeObservWrapper import NormalizeObservWrapper

# SB3 Checker
from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':


    rospy.logwarn("Start")

    rospy.init_node('test_irb120_reacher')

    env = gym.make('ABBIRB120ReacherEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=100)

    env.reset()

    check_env(env, warn=True)
    
    env.close()
    sys.exit()