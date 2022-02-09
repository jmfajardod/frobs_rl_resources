#!/bin/python3
# import sys,os
# print(os.path.dirname(sys.path[0]))
# print(os.path.join(os.path.dirname(sys.path[0]),'bar'))
# sys.path.append("/home/lucasamos/FYP")

from numpy.lib.npyio import save
from abb_irb140_reacher.task_env.irb140_reacher import abb_irb140_moveit
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
from frobs_rl.models.ddpg import DDPG
from frobs_rl.models.td3 import TD3
from frobs_rl.models.sac import SAC
from frobs_rl.models.ppo import PPO
from frobs_rl.models.dqn import DQN

if __name__ == '__main__':

    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()

    # Launch Gazebo 
    ros_gazebo.launch_Gazebo(paused=True, gui=False)

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('train_irb140_reacher')

    # Launch the task environment
    env = gym.make('ABBIRB140ReacherEnv-v0')

    #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=100)
    env.reset()

    #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("abb_irb140_reacher")

    #-- DDPG
    # save_path = pkg_path + "/models/static_reacher/ddpg/"
    # log_path = pkg_path + "/logs/static_reacher/ddpg/"
    # model = DDPG(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="ddpg.yaml")
    
    #-- TD3
    # save_path = pkg_path + "/models/td3/"
    # log_path = pkg_path + "/logs/td3/"
    save_path = pkg_path + "/models/static_reacher/td3/"
    log_path = pkg_path + "/logs/aux/td3/"
    model = TD3(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="td3.yaml")

    #-- SAC
    # save_path = pkg_path + "/models/static_reacher/sac/"
    # log_path = pkg_path + "/logs/static_reacher/sac/"
    # model = SAC(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="sac.yaml")

    #--- PPO
    # save_path = pkg_path + "/models/static_reacher/ppo/"
    # log_path = pkg_path + "/logs/static_reacher/ppo/"
    # model = PPO(env, save_path, log_path, config_file_pkg="abb_irb140_reacher", config_filename="ppo.yaml")


    model.train()
    model.save_model()
    model.close_env()

    sys.exit()