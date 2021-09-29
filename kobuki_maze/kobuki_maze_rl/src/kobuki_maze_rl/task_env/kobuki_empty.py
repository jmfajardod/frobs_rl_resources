#!/bin/python3

import gym
from gym import spaces
from gym.envs.registration import register
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_controllers
from frobs_rl.common import ros_node
from frobs_rl.common import ros_launch
from frobs_rl.common import ros_params
from frobs_rl.common import ros_urdf
from frobs_rl.common import ros_spawn
from kobuki_maze_rl.robot_env import kobuki_lidar_env
import rospy
import tf

import numpy as np
import scipy.spatial
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from visualization_msgs.msg import Marker 

register(
        id='KobukiEmptyEnv-v0',
        entry_point='kobuki_maze_rl.task_env.kobuki_empty:KobukiEmptyEnv',
        max_episode_steps=100000000000,
    )

class KobukiEmptyEnv(kobuki_lidar_env.KobukiLIDAREnv):
    """
    Kobuki Empty Env, specific methods.
    """

    def __init__(self):
        """
        Task is to move the robot from the start position to the goal position, avoiding collisions.
        """
        rospy.loginfo("Starting Kobuki Empty Env")

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("kobuki_maze_rl", "dynamic_obj_task.yaml", ns="/")
        self.get_params()

        """
        Define the action and observation space.
        """
        action_space_low  = np.array([-0.3, -0.3]) 
        action_space_high = np.array([ 0.3,  0.3])
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)
        rospy.logwarn("Action Space->>>" + str(self.action_space))

        pose_low   = np.array([-6.0, -8.0, - np.pi])
        pose_high  = np.array([ 26.0, 7.0, np.pi])
        goal_low   = np.array([-6.0, -8.0])
        goal_high  = np.array([26.0, 7.0])
        dist_low   = np.array([-33.0, -16.0])
        dist_high  = np.array([ 33.0,  16.0])
        vec_ang_low  = np.array([0.0 , -np.pi])
        vec_ang_high = np.array([37.0,  np.pi])
        lidar_low  = 0.0 * np.ones(self.lidar_samples)
        lidar_high = 30.0 * np.ones(self.lidar_samples)

        # With LIDAR
        # observ_space_low  = np.concatenate((pose_low,  goal_low,  dist_low,  lidar_low ))
        # observ_space_high = np.concatenate((pose_high, goal_high, dist_high, lidar_high))
        # Without LIDAR
        observ_space_low  = vec_ang_low  #np.concatenate((dist_low, vec_ang_low))
        observ_space_high = vec_ang_high #np.concatenate((dist_high, vec_ang_high))
        self.observation_space = spaces.Box(low=observ_space_low, high=observ_space_high, dtype=np.float32)
        rospy.logwarn("Observation Space->>>" + str(self.observation_space))
        rospy.logwarn("Observation Space Low  ->>>" + str(self.observation_space.low))
        rospy.logwarn("Observation Space High ->>>" + str(self.observation_space.high))

        # Spaces for initial robot position and goal position (for reset)
        self.robot_pos_space = spaces.Box(low=np.array([-5.3, -7.0, -np.pi]), high=np.array([ 25.0, 6.3, np.pi]), dtype=np.float32)
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)

        """
        Init super class.
        """
        super(KobukiEmptyEnv, self).__init__()

        #--- Make Marker msg for publishing
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id="odom"
        self.goal_marker.header.stamp = rospy.Time.now()
        self.goal_marker.ns = "goal_shapes"
        self.goal_marker.id = 0
        self.goal_marker.type = Marker.SPHERE
        self.goal_marker.action = Marker.ADD

        self.goal_marker.pose.position.x = 0.0
        self.goal_marker.pose.position.y = 0.0
        self.goal_marker.pose.position.z = 0.0
        self.goal_marker.pose.orientation.x = 0.0
        self.goal_marker.pose.orientation.y = 0.0
        self.goal_marker.pose.orientation.z = 0.0
        self.goal_marker.pose.orientation.w = 1.0

        self.goal_marker.scale.x = 0.3
        self.goal_marker.scale.y = 0.3
        self.goal_marker.scale.z = 0.3

        self.goal_marker.color.r = 0.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 1.0
        self.goal_marker.color.a = 1.0

        self.pub_marker = rospy.Publisher("goal_point",Marker,queue_size=10)

        self.tf_br = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of Kobuki Maze Env")

    #-------------------------------------------------------#
    #   Custom available methods for the KobukiEmptyEnv      #

    def _set_episode_init_params(self):
        """
        Function to set some parameters, like the position of the robot, at the beginning of each episode.
        """ 
        # Set the initial position of the robot
        
        init_robot_pos = self.robot_pos_space.sample()
        rospy.logwarn("Initial Robot Position->>>" + "X: " + str(init_robot_pos[0]) + " Y: " + str(init_robot_pos[1]) + " Theta: " + str(init_robot_pos[2]))
        quat = quaternion_from_euler(0.0, 0.0, init_robot_pos[2])
        ros_gazebo.gazebo_set_model_state("kobuki_robot", pos_x=init_robot_pos[0], pos_y=init_robot_pos[1], pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])

        # Set goal position
        self.goal_pos = self.goal_space.sample()
        #self.goal_pos = np.array([25.0, -4.0])

        self.tf_br.sendTransform((self.goal_pos[0], self.goal_pos[1], 0.0),
                        quaternion_from_euler(0, 0, 0),
                        rospy.Time.now()+rospy.Duration(3.0),
                        "goal_frame",
                        "odom")
        

        self.goal_marker.pose.position.x = self.goal_pos[0]
        self.goal_marker.pose.position.y = self.goal_pos[1]
        self.goal_marker.color.r = 0.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 1.0
        self.goal_marker.color.a = 1.0

        self.pub_marker.publish(self.goal_marker)
        return True

    def _send_action(self, action):
        """
        Send linear and angular velocities to the robot.
        """
        rospy.logwarn("Action====>>>>>>>"+str(action))
        self.send_vel(action[0], action[1])
        return True
        

    def _get_observation(self):
        """
        The observation is comprised of the current pose, the goal pose and the lidar data.
        """
        
        # Get the current pose
        pos = np.append( self.get_robot_pos(), self.get_robot_ori() )

        trans = np.array([0.0,0.0,0.0])
        rot = np.array([0.0,0.0,0.0,0.0])
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/base_footprint','/goal_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transformation was not found")

        dist_rob_goal= trans[0:2]
        mag_vec_dist = np.sqrt(np.power(trans[0],2) + np.power(trans[1],2))
        ang_rob_goal = np.arctan2(trans[1], trans[0])

        # Get the lidar data
        lidar_ranges = self.get_lidar_ranges()

        obs = np.concatenate((
            #dist_rob_goal,   # Distance between robot and goal in robot frame
            mag_vec_dist,    # Magnitude of the vector between robot and goal
            ang_rob_goal     # Angle between robot and goal
            #lidar_ranges     # Lidar ranges data
            ),
            axis=None
        )

        #rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(pos) + "  " + str(dist_rob_goal) +  "  " + str(mag_vec_dist) +  "  " + str(ang_rob_goal))
        rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

        return obs.copy()

    def _check_if_done(self):
        """
        Function to check if the episode is done.
        
        If the episode has a success condition then set done as:
            self.info['is_success'] = 1.0
        """
        # Get current robot position
        robot_pos = np.append(self.get_robot_pos(), np.array([0.0]) )

        # Get the goal position
        goal_pos = np.append(self.goal_pos, np.array([0.0]) )

        #--- Function used to calculate 
        done = self.calculate_if_done(goal_pos, robot_pos)
        if done:
            rospy.logdebug("Reached a Desired Position!")
            self.info['is_success']  = 1.0
            self.goal_marker.color.r = 0.0
            self.goal_marker.color.g = 1.0
            self.goal_marker.color.b = 0.0
            self.goal_marker.color.a = 1.0

        self.tf_br.sendTransform((self.goal_pos[0], self.goal_pos[1], 0.0),
                        quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "goal_frame",
                        "odom")

        self.pub_marker.publish(self.goal_marker)

        return done

    def _get_reward(self):
        """
        Function to get the reward from the environment.
        """
        
        # Get current robot position
        robot_pos = np.append(self.get_robot_pos(), np.array([0.0]) )

        # Get the goal position
        goal_pos = np.append(self.goal_pos, np.array([0.0]) )

        # Basic reward function
        reward = self.step_reward

        # Check if done
        done = self.calculate_if_done(goal_pos, robot_pos)
        if done:
            reward += self.reached_goal_reward
        else:
            dist2goal = scipy.spatial.distance.euclidean(robot_pos, goal_pos)
            reward   -= self.mult_dist_reward*dist2goal

        rospy.logwarn("REWARD====>>>>>>>"+str(reward))

        return reward
    

    #-------------------------------------------------------#
    #  Internal methods for the KobukiEmptyEnv               #

    def get_params(self):
        """
        Get configuration parameters
        """

        self.n_actions = rospy.get_param('/kobuki_maze/action_space')
        self.n_observations = rospy.get_param('/kobuki_maze/observation_space')
        self.lidar_samples = rospy.get_param('/kobuki_maze/lidar_samples')

        self.tol_goal_pos = rospy.get_param('/kobuki_maze/tolerance_goal_pos')

        #--- Get reward parameters
        self.reached_goal_reward = rospy.get_param('/kobuki_maze/reached_goal_reward')
        self.step_reward = rospy.get_param('/kobuki_maze/step_reward')
        self.mult_dist_reward = rospy.get_param('/kobuki_maze/multiplier_dist_reward')
        self.collision_reward = rospy.get_param('/kobuki_maze/collision_reward')
        self.collision_distance = rospy.get_param('/kobuki_maze/collision_distance_threshold')

        self.goal_pos = np.array([1.0, 0.0])

        #--- Get Gazebo physics parameters
        if rospy.has_param('/kobuki_maze/time_step'):
            self.t_step = rospy.get_param('/kobuki_maze/time_step')
            ros_gazebo.gazebo_set_time_step(self.t_step)

        if rospy.has_param('/kobuki_maze/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/kobuki_maze/update_rate_multiplier')
            ros_gazebo.gazebo_set_max_update_rate(self.max_update_rate)

    def calculate_if_done(self, goal, current_pos):
        """
        It calculated whether it has finished or not
        """
        # check if the end-effector located within a threshold to the goal
        distance_2_goal = scipy.spatial.distance.euclidean(current_pos, goal)
        done = False

        if distance_2_goal<=self.tol_goal_pos:
            done = True
        
        return done