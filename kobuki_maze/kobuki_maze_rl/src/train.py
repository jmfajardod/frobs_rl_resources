#!/bin/python3

from kobuki_maze_rl.task_env import kobuki_empty
from kobuki_maze_rl.task_env import kobuki_dynamic_v3
from kobuki_maze_rl.task_env import kobuki_maze
import gym
import rospy
import rospkg
import sys
from frobs_rl.common import ros_gazebo, ros_node
from frobs_rl.common.ros_node import ROS_Kill_All_processes
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper

# Models
from frobs_rl.models.td3 import TD3
from frobs_rl.models.sac import SAC

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ROS_Kill_All_processes()

    # Launch Gazebo 
    ros_gazebo.Launch_Gazebo(paused=True, gui=False)

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('kobuki_maze_train')

    # Launch the task environment
    # env = gym.make('KobukiEmptyEnv-v0')
    # env = gym.make('KobukiDynamicEnv-v3')
    env = gym.make('KobukiMazeEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=15000)
    env.reset()

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("kobuki_maze_rl")

    
    #-- TD3
    save_path = pkg_path + "/models/td3/"
    log_path = pkg_path + "/logs/td3/"
    model = TD3(env, save_path, log_path, config_file_pkg="kobuki_maze_rl", config_filename="td3.yaml")

    #-- SAC
    # save_path = pkg_path + "/models/dynamic/sac/"
    # log_path = pkg_path + "/logs/dynamic/sac/"
    # model = SAC(env, save_path, log_path, config_file_pkg="kobuki_maze_rl", config_filename="sac.yaml")


    model.train()
    model.save_model()
    model.close_env()

    sys.exit()