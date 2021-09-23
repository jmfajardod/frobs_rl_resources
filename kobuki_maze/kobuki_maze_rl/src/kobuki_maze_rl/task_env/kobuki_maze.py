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
        id='KobukiMazeEnv-v0',
        entry_point='kobuki_maze_rl.task_env.kobuki_maze:KobukiMazeEnv',
        max_episode_steps=100000000000,
    )

class KobukiMazeEnv(kobuki_lidar_env.KobukiLIDAREnv):
    """
    Kobuki Maze Env, specific methods.
    """

    def __init__(self):
        """
        Task is to move the robot from the start position to the goal position, avoiding collisions.
        """
        rospy.loginfo("Starting Kobuki Maze Env")

        """
        Load YAML param file
        """
        ros_params.ROS_Load_YAML_from_pkg("kobuki_maze_rl", "maze_task.yaml", ns="/")
        self.get_params()

        """
        Define the action and observation space.
        """
        action_space_low  = np.array([-1.0, -1.0]) # linear velocity, angular velocity
        action_space_high = np.array([ 1.0,  1.0])
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)
        rospy.logwarn("Action Space->>>" + str(self.action_space))

        vec_ang_low  = np.array([0.0 , -np.pi])
        vec_ang_high = np.array([75.0,  np.pi])
        pos_low   = np.array([-10.5, -13.5, -np.pi])
        pos_high  = np.array([  5.5,  13.5,  np.pi])
        goal_low  = np.array([-10.5, -13.5])
        goal_high = np.array([  5.5,  13.5])
        lidar_low  = 0.0 * np.ones(self.lidar_samples)
        lidar_high = 30.0 * np.ones(self.lidar_samples)

        # With LIDAR
        observ_space_low  = np.concatenate((vec_ang_low,  pos_low,  goal_low,  lidar_low))
        observ_space_high = np.concatenate((vec_ang_high, pos_high, goal_high, lidar_high))
        # observ_space_low  = np.concatenate((vec_ang_low,  lidar_low))
        # observ_space_high = np.concatenate((vec_ang_high, lidar_high))
        self.observation_space = spaces.Box(low=observ_space_low, high=observ_space_high, dtype=np.float32)
        rospy.logwarn("Observation Space->>>" + str(self.observation_space))
        rospy.logwarn("Observation Space Low  ->>>" + str(self.observation_space.low))
        rospy.logwarn("Observation Space High ->>>" + str(self.observation_space.high))

        # Spaces for initial robot position and goal position (for reset)
        self.angle_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

        self.mag_vec_dist = 1.0
        self.ang_rob_goal = 0.0

        """
        Init super class.
        """
        super(KobukiMazeEnv, self).__init__()

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

        # Spawn the maze
        MazeObj = ros_urdf.URDF_parse_from_pkg("kobuki_maze_rl", "model.sdf", folder="/worlds/Kobuki_maze_smallv2")
        ros_gazebo.Gazebo_spawn_sdf_string(MazeObj, model_name="maze", pos_x= 3.7, pos_y= -4.3)

        # Spawn the boxes
        UnitBox = ros_urdf.URDF_parse_from_pkg("kobuki_maze_rl", "model.sdf", folder="/worlds/Box")
        ros_gazebo.Gazebo_spawn_sdf_string(UnitBox, model_name="box1", pos_x= -2.5, pos_y=-5.0)
        ros_gazebo.Gazebo_spawn_sdf_string(UnitBox, model_name="box2", pos_x= -6.5, pos_y=-5.0)

        self.init_box_traj = True
        self.box_traj_dir  = 1
        self.box_traj_num_steps = 200
        self.box_traj_count = 0
        self.box1_step = 0.01
        self.box2_step = 0.01

        self.boxes_vel_space = spaces.Box(low=0.0, high=0.02, shape=(2,), dtype=np.float32)

        # Set the possible trajectories for the robot
        self.list_trajs = []
        traj1 = np.array([  [-8.5,  0.0], # 1
                            [-8.5, -2.0], # 2
                            [-1.0, -7.0], # 3
                            [-1.0,-11.5], # 4
                            [ 3.0,-11.5], # 5
                            [ 3.0, -6.0], # 6
                            [ 3.0,  0.0]])# 7
        self.list_trajs.append(traj1)

        traj2 = np.array([  [-8.5,  0.0], # 1
                            [-8.5, -2.0], # 2
                            [-1.0, -7.0], # 3
                            [-1.0,-11.5], # 4
                            [-6.0, -9.5], # 5
                            [-8.5, -9.5], # 6
                            [-8.5,-11.5]])# 7
        self.list_trajs.append(traj2)
        
        traj3 = np.array([  [-8.5,  0.0], # 1
                            [-8.5, -2.0], # 2
                            [-8.5, -3.0], # 3
                            [-8.5, -4.0], # 4
                            [-8.5, -5.0], # 5
                            [-8.5, -6.0], # 6
                            [-8.5, -7.0]])# 7
        self.list_trajs.append(traj3)

        traj4 = np.array([  [-8.5,  0.0], # 1
                            [-8.5, -2.0], # 2
                            [ 0.0, -4.0], # 3
                            [ 0.0, -3.5], # 4
                            [ 0.0, -3.0], # 5
                            [ 0.0, -2.5], # 6
                            [ 0.0, -2.0]])# 7
        self.list_trajs.append(traj4)

        self.curren_traj = self.list_trajs[0]
        self.traj_space = spaces.Discrete(4)
    
        self.current_traj = self.list_trajs[self.traj_space.sample()]
        self.curren_traj_len = self.current_traj.shape[0]
        self.curren_traj_counter = 0

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of Kobuki Maze Env")

    #-------------------------------------------------------#
    #   Custom available methods for the KobukiMazeEnv      #

    def _set_episode_init_params(self):
        """
        Function to set some parameters, like the position of the robot, at the beginning of each episode.
        """ 
        # Set the initial position of the robot
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.Gazebo_set_model_state("kobuki_robot", pos_x=-5.0, pos_y=7.5, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])

        # Set boxes initial positions
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.Gazebo_set_model_state( "box1", ref_frame="world", pos_x=-2.5, pos_y=-5.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.Gazebo_set_model_state( "box2", ref_frame="world", pos_x=-6.5, pos_y=-5.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        

        self.init_box_traj = True
        self.box_traj_dir  = 1
        self.box_traj_count = 0

        sample_box_vel = self.boxes_vel_space.sample()
        self.box1_step = sample_box_vel[0]
        self.box2_step = sample_box_vel[1]

        # Set goal position
        num_traj = self.traj_space.sample()
        self.current_traj = self.list_trajs[num_traj]
        rospy.logwarn("Trajectory #:  " + str(num_traj))
        self.curren_traj_len = self.current_traj.shape[0]
        rospy.logwarn("Trajectory len:  " + str(self.curren_traj_len))
        self.curren_traj_counter = 0
        self.goal_pos = self.current_traj[self.curren_traj_counter]

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

        # Get the goal position
        goal_pos = self.goal_pos


        trans = np.array([0.0,0.0,0.0])
        rot = np.array([0.0,0.0,0.0,0.0])
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/base_footprint','/goal_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transformation was not found")

        dist_rob_goal= trans[0:2]
        self.mag_vec_dist = np.sqrt(np.power(trans[0],2) + np.power(trans[1],2))
        self.ang_rob_goal = np.arctan2(trans[1], trans[0])

        # Get the lidar data
        lidar_ranges = self.get_lidar_ranges()

        obs = np.concatenate((
            self.mag_vec_dist,     # Magnitude of the vector between robot and goal
            self.ang_rob_goal,     # Angle between robot and goal
            pos,                   # Current pose
            goal_pos,              # Goal position
            lidar_ranges           # Lidar ranges data
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+ str(self.mag_vec_dist) +  "  " + str(self.ang_rob_goal))
        rospy.logwarn("LIDAR====>>>>>>>"+ str(np.min(lidar_ranges)) )
        # rospy.logwarn("OBSERVATIONS====>>>>>>>"+ str(self.mag_vec_dist) +  "  " + str(self.ang_rob_goal) + "  " +
                                                    # str(pos) + "  " + str(goal_pos))
        #rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

        return obs.copy()

    def _check_if_done(self):
        """
        Function to check if the episode is done.
        
        If the episode has a success condition then set done as:
            self.info['is_success'] = 1.0
        """

        # Set marker color
        self.goal_marker.color.r = 0.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 1.0
        self.goal_marker.color.a = 1.0

        # Get current robot position
        robot_pos = np.append(self.get_robot_pos(), np.array([0.0]) )

        # Get the goal position
        goal_pos = np.append(self.goal_pos, np.array([0.0]) )

        #--- Function used to calculate 
        done = self.calculate_if_done(goal_pos, robot_pos)
        reached_midway = False
        if done:
            rospy.logdebug("Reached a Desired Position!")
            self.curren_traj_counter += 1
            if self.curren_traj_counter >= self.curren_traj_len:
                self.info['is_success']  = 1.0
                rospy.logwarn("SUCCESS")
                self.goal_marker.color.r = 0.0
                self.goal_marker.color.g = 1.0
                self.goal_marker.color.b = 0.0
                self.goal_marker.color.a = 1.0
            else:
                rospy.logwarn("Reached Midway point")
                done = False
                reached_midway = True
                self.goal_marker.color.r = 1.0
                self.goal_marker.color.g = 0.0
                self.goal_marker.color.b = 1.0
                self.goal_marker.color.a = 1.0

        # Send goal transform
        self.tf_br.sendTransform((self.goal_pos[0], self.goal_pos[1], 0.0),
                        quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "goal_frame",
                        "odom")

        # Publish goal marker
        self.pub_marker.publish(self.goal_marker)

        # If reached midway point then update goal
        if reached_midway:
            self.goal_pos = self.current_traj[self.curren_traj_counter]
            self.goal_marker.pose.position.x = self.goal_pos[0]
            self.goal_marker.pose.position.y = self.goal_pos[1]

        # Set boxes new position
        self.box_traj_count += 1
        if self.init_box_traj:
            if self.box_traj_count > self.box_traj_num_steps/2:
                self.init_box_traj = False
                self.box_traj_dir *= -1 
                self.box_traj_count = 0
        else:
            if self.box_traj_count > self.box_traj_num_steps:
                self.box_traj_dir *= -1 
                self.box_traj_count = 0

        ros_gazebo.Gazebo_set_model_state("box1", ref_frame="box1", pos_x=self.box_traj_dir*self.box1_step, sleep_time=0.0)
        ros_gazebo.Gazebo_set_model_state("box2", ref_frame="box2", pos_x=self.box_traj_dir*self.box2_step, sleep_time=0.0)

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
            if self.curren_traj_counter >= self.curren_traj_len:
                reward += self.reached_goal_reward
            else:
                reward += self.reached_midway_reward
        else:
            reward   -= self.mult_dist_reward*self.mag_vec_dist
            
            abs_angle = np.abs(self.ang_rob_goal)
            angle_diff = np.minimum(abs_angle, np.pi - abs_angle)
            rospy.logwarn("ANGLE DIFF: " + str(angle_diff))
            reward   -= 0.5*angle_diff

            remaining_midway = (self.curren_traj_len-self.curren_traj_counter)
            rospy.logwarn("Remaining midway points: " + str(remaining_midway))
            reward  += self.not_reached_midway_reward*remaining_midway
            

        # Check if the robot is in collision
        lidar_ranges = self.get_lidar_ranges()
        min_range = np.min(lidar_ranges)
        if np.any(min_range < self.collision_distance):
            rospy.logwarn("Collision Detected!")
            reward += self.collision_reward

        rospy.logwarn("REWARD====>>>>>>>"+str(reward))

        return reward
    

    #-------------------------------------------------------#
    #  Internal methods for the KobukiMazeEnv               #

    def get_params(self):
        """
        Get configuration parameters
        """

        self.n_actions = rospy.get_param('/kobuki_maze/action_space')
        self.n_observations = rospy.get_param('/kobuki_maze/observation_space')
        self.lidar_samples = rospy.get_param('/kobuki_maze/lidar_samples')

        self.tol_goal_pos = rospy.get_param('/kobuki_maze/tolerance_goal_pos')

        #--- Get reward parameters
        self.reached_goal_reward       = rospy.get_param('/kobuki_maze/reached_goal_reward')
        self.reached_midway_reward     = rospy.get_param('/kobuki_maze/reached_midway_reward')
        self.not_reached_midway_reward = rospy.get_param('/kobuki_maze/not_reached_midway_reward')
        self.step_reward               = rospy.get_param('/kobuki_maze/step_reward')
        self.mult_dist_reward          = rospy.get_param('/kobuki_maze/multiplier_dist_reward')
        self.collision_reward          = rospy.get_param('/kobuki_maze/collision_reward')
        self.collision_distance        = rospy.get_param('/kobuki_maze/collision_distance_threshold')

        self.goal_pos = np.array([1.0, 0.0])

        #--- Get Gazebo physics parameters
        if rospy.has_param('/kobuki_maze/time_step'):
            self.t_step = rospy.get_param('/kobuki_maze/time_step')
            ros_gazebo.Gazebo_set_time_step(self.t_step)

        if rospy.has_param('/kobuki_maze/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/kobuki_maze/update_rate_multiplier')
            ros_gazebo.Gazebo_set_max_update_rate(self.max_update_rate)

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