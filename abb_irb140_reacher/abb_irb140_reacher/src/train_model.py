#!/bin/python3


# from abb_irb140_reacher.robot_env.abb_irb140_servo import ABBIRB140Servo
# from abb_irb140_reacher.task_env.abb_irb140_reacher import ABBIRB140ReacherEnv

from abb_irb140_reacher.task_env.abb_irb140_vel_reacher import ABBIRB140VelReacherEnv

import gym
import rospy
import rospkg
import sys
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_node
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper

# Models
from frobs_rl.models.td3 import TD3
from frobs_rl.models.sac import SAC

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()

    # Launch Gazebo 
    ros_gazebo.launch_Gazebo(paused=True, gui=False, custom_world_pkg='abb_irb140_reacher', custom_world_name='no_collision.world')

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('train_irb140_reacher')

    # Launch the task environment
    env = gym.make('ABBIRB140VelReacherEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=10000)
    env.reset()

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("abb_irb140_reacher")

    #-- TD3
    save_path = pkg_path + "/models/td3/"
    log_path = pkg_path + "/logs/td3/"
    model = TD3(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="td3.yaml")

    #-- SAC
    # save_path = pkg_path + "/models/static_reacher/sac/"
    # log_path = pkg_path + "/logs/static_reacher/sac/"
    # model = SAC(env, save_path, log_path, config_file_pkg="abb_irb120_reacher", config_filename="sac.yaml")

    model.train()
    model.save_model()
    model.close_env()

    sys.exit()