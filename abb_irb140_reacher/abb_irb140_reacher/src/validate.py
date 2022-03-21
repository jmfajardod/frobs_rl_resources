#!/usr/bin/python3

import sys
import os
from abb_irb140_reacher.task_env.irb140_reacher_pos import ABBIRB140ReacherEnv

from frobs_rl.common import ros_gazebo, ros_node
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper
import gym
import rospy
import rospkg

from rospy.numpy_msg import numpy_msg
import numpy as np
from geometry_msgs.msg import Twist

# Import TD3 algorithm
from frobs_rl.models.td3 import TD3

import rosbag
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def joint_action(angulos):
    joint_points= JointTrajectoryPoint()   
    joint_points.positions=angulos
    joint_ctrl.points.append(joint_points)
    joint_points.time_from_start = rospy.Duration(1)

def scale_obs_joints(array):
    # Set low and max to unnormalize
    low_scale_jts  = np.array([ -3.14159, -1.5708, -4.01426, -3.49066, -2.0071, -6.98132])
    high_scale_jts = np.array([ 3.14159, 1.91986, 0.872665, 3.49066, 2.0071, 6.98132])
    
    return low_scale_jts + (0.5 * (array + 1.0) * (high_scale_jts -  low_scale_jts))

def scale_obs_goal(array):
    # Set low and max to unnormalize
    low_scale_jts  = np.array([ -1.0, -1.0, 0.0])
    high_scale_jts = np.array([  1.0, 1.0, 2.0])
    
    return low_scale_jts + (0.5 * (array + 1.0) * (high_scale_jts -  low_scale_jts))

if __name__ == '__main__':
    # Kill all processes related to previous runs
    print("Started")
    ros_node.ros_kill_all_processes()

    # Launch Gazebo
    ros_gazebo.launch_Gazebo(paused=True, gui=True)

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('irb140_maze_train')

    # Launch the task environment
    env = gym.make('ABBIRB140ReacherPosEnv-v0')

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

    save_path = pkg_path + "/models/reacher_pos/td3/"

    #-- TD3 trained
    model = TD3.load_trained(save_path + "td3_model_455000_steps")

    # Set episodes to validate
    episodes = 100
    epi_count = 0

    rospy.logwarn("/////////////------------------------////////////////////")
    rospy.logwarn("Episode: " + str(epi_count))

    obs = env.reset()

    # Create the folder of validate
    if not os.path.exists(pkg_path + "/validate"):
        os.mkdir(pkg_path + "/validate")

    # Create the msg to save the trajectory
    joint_ctrl=JointTrajectory()
    joint_ctrl.joint_names.append("joint_1");joint_ctrl.joint_names.append("joint_2");joint_ctrl.joint_names.append("joint_3")
    joint_ctrl.joint_names.append("joint_4");joint_ctrl.joint_names.append("joint_5");joint_ctrl.joint_names.append("joint_6")
    
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    joint_ctrl.header=h

    # Scale the observation
    scaled_obs_jts = scale_obs_joints(obs[6:])
    print("Joint positions: ", scaled_obs_jts)
    joint_action(scaled_obs_jts)

    scaled_obs_goal = scale_obs_goal(obs[3:6])
    print("->Goal: ", scaled_obs_goal)

    counter_steps = 0

    while epi_count < episodes:
        
        action, _states = model.predict(obs, deterministic=True)

        obs, _, dones, info = env.step(action)
        counter_steps += 1

        scaled_obs_jts = scale_obs_joints(obs[6:])
        print("Joint positions: ", scaled_obs_jts)
        joint_action(scaled_obs_jts)

        if dones:

            # Save data
            validate_save_path = pkg_path + "/validate/test_" +str(epi_count)
            if not os.path.exists(validate_save_path):
                os.mkdir(validate_save_path)
                print("Directory ", validate_save_path, " Created ")

            validate_save_path = validate_save_path + "/"
            np.savetxt(validate_save_path + 'goal_episode_' + str(epi_count) + '.txt', scaled_obs_goal, delimiter=',', fmt='%1.4f')
            bag = rosbag.Bag(validate_save_path + 'test_'+str(epi_count)+'.bag', 'w')
            bag.write('/joint_path_command', joint_ctrl)
            bag.close()
            if counter_steps<100:
                np.savetxt(validate_save_path + 'valid.txt', np.array([1.0]), delimiter=',', fmt='%1.4f')
            
            rospy.logwarn("/////////////------------------------////////////////////")
            epi_count += 1
            rospy.logwarn("Episode: " + str(epi_count))
            obs = env.reset()
            counter_steps = 0
            joint_ctrl=JointTrajectory()
            joint_ctrl.joint_names.append("joint_1");joint_ctrl.joint_names.append("joint_2");joint_ctrl.joint_names.append("joint_3")
            joint_ctrl.joint_names.append("joint_4");joint_ctrl.joint_names.append("joint_5");joint_ctrl.joint_names.append("joint_6")

            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            joint_ctrl.header=h

            scaled_obs_jts = scale_obs_joints(obs[6:])
            print("Joint positions: ", scaled_obs_jts)
            joint_action(scaled_obs_jts)

            scaled_obs_goal = scale_obs_goal(obs[3:6])
            print("->Goal: ", scaled_obs_goal)
    
    print("bag finalizado")

    env.close()
    sys.exit()