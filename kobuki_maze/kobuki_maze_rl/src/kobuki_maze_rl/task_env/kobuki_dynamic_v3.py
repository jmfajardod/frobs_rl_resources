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
        id='KobukiDynamicEnv-v3',
        entry_point='kobuki_maze_rl.task_env.kobuki_dynamic_v3:KobukiDynamicEnv',
        max_episode_steps=100000000000,
    )

class KobukiDynamicEnv(kobuki_lidar_env.KobukiLIDAREnv):
    """
    Kobuki Dynamic Env, specific methods.
    """

    def __init__(self):
        """
        Task is to move the robot from the start position to the goal position, avoiding collisions.
        """
        rospy.loginfo("Starting Kobuki Dynamic Env")

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("kobuki_maze_rl", "dynamic_obj_task.yaml", ns="/")
        self.get_params()

        """
        Define the action and observation space.
        """
        action_space_low  = np.array([-1.0, -1.0]) # linear velocity, angular velocity
        action_space_high = np.array([ 1.0,  1.0])
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)
        rospy.logwarn("Action Space->>>" + str(self.action_space))

        goal_low   = np.array([-25.0, -25.0])
        goal_high  = np.array([ 25.0,  25.0])
        vec_ang_low  = np.array([0.0 , -np.pi])
        vec_ang_high = np.array([71.0,  np.pi])
        lidar_low  = 0.0 * np.ones(self.lidar_samples)
        lidar_high = 30.0 * np.ones(self.lidar_samples)

        # With LIDAR
        observ_space_low  = np.concatenate((vec_ang_low,  lidar_low))
        observ_space_high = np.concatenate((vec_ang_high, lidar_high))
        self.observation_space = spaces.Box(low=observ_space_low, high=observ_space_high, dtype=np.float32)
        rospy.logwarn("Observation Space->>>" + str(self.observation_space))
        rospy.logwarn("Observation Space Low  ->>>" + str(self.observation_space.low))
        rospy.logwarn("Observation Space High ->>>" + str(self.observation_space.high))

        # Spaces for initial robot position and goal position (for reset)
        self.angle_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        self.goal_space  = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)

        self.mag_vec_dist = 1.0
        self.ang_rob_goal = 0.0

        """
        Init super class.
        """
        super(KobukiDynamicEnv, self).__init__()

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

        # Spawn the objects
        UnitBox = ros_urdf.urdf_parse_from_pkg("kobuki_maze_rl", "model.sdf", folder="/worlds/Box")
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box1", pos_x=-13.0, pos_y= 13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box2", pos_x=  0.0, pos_y= 13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box3", pos_x= 13.0, pos_y= 13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box4", pos_x= 13.0, pos_y=  0.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box5", pos_x= 13.0, pos_y=-13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box6", pos_x=  0.0, pos_y=-13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box7", pos_x=-13.0, pos_y=-13.0)
        ros_gazebo.gazebo_spawn_sdf_string(UnitBox, model_name="box8", pos_x=-13.0, pos_y=  0.0)

        self.init_box_traj = True
        self.box_traj_dir  = 1
        self.box_traj_num_steps = 2000
        self.box_traj_count = 0
        self.box1_step = 0.01
        self.box2_step = 0.01
        self.box3_step = 0.01
        self.box4_step = 0.01
        self.box5_step = 0.01
        self.box6_step = 0.01
        self.box7_step = 0.01
        self.box8_step = 0.01

        self.boxes_vel_space = spaces.Box(low=0.0, high=0.02, shape=(8,), dtype=np.float32)


        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of Kobuki Dynamic Env")

    #-------------------------------------------------------#
    #   Custom available methods for the KobukiDynamicEnv      #

    def _set_episode_init_params(self):
        """
        Function to set some parameters, like the position of the robot, at the beginning of each episode.
        """ 
        # Set the initial position of the robot
        
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state("kobuki_robot", pos_x=0.0, pos_y=0.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])

        # Set boxes initial positions
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box1", ref_frame="world", pos_x=-13.0, pos_y= 13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box2", ref_frame="world", pos_x=  0.0, pos_y= 13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box3", ref_frame="world", pos_x= 13.0, pos_y= 13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box4", ref_frame="world", pos_x= 13.0, pos_y=  0.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box5", ref_frame="world", pos_x= 13.0, pos_y=-13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box6", ref_frame="world", pos_x=  0.0, pos_y=-13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box7", ref_frame="world", pos_x=-13.0, pos_y=-13.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
        quat = quaternion_from_euler(0.0, 0.0, self.angle_space.sample())
        ros_gazebo.gazebo_set_model_state( "box8", ref_frame="world", pos_x=-13.0, pos_y=  0.0, pos_z=0.0,
                                            ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])

        self.init_box_traj = True
        self.box_traj_dir  = 1
        self.box_traj_count = 0

        sample_box_vel = self.boxes_vel_space.sample()
        self.box1_step = sample_box_vel[0]
        self.box2_step = sample_box_vel[1]
        self.box3_step = sample_box_vel[2]
        self.box4_step = sample_box_vel[3]
        self.box5_step = sample_box_vel[4]
        self.box6_step = sample_box_vel[5]
        self.box7_step = sample_box_vel[6]
        self.box8_step = sample_box_vel[7]

        # Set goal position
        self.goal_pos = self.goal_space.sample()

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
        self.mag_vec_dist = np.sqrt(np.power(trans[0],2) + np.power(trans[1],2))
        self.ang_rob_goal = np.arctan2(trans[1], trans[0])

        # Get the lidar data
        lidar_ranges = self.get_lidar_ranges()

        obs = np.concatenate((
            #dist_rob_goal,   # Distance between robot and goal in robot frame
            self.mag_vec_dist,     # Magnitude of the vector between robot and goal
            self.ang_rob_goal,     # Angle between robot and goal
            lidar_ranges      # Lidar ranges data
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+ str(self.mag_vec_dist) +  "  " + str(self.ang_rob_goal))
        #rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

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

        # Send goal transform
        self.tf_br.sendTransform((self.goal_pos[0], self.goal_pos[1], 0.0),
                        quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "goal_frame",
                        "odom")

        # Publish goal marker
        self.pub_marker.publish(self.goal_marker)

        # Set boxes new position
        self.box_traj_count += 1
        if self.init_box_traj:
            if self.box_traj_count > self.box_traj_num_steps/2:
                self.init_box_traj = False
                self.box_traj_dir *= -1 
                self.box_traj_count = 0
        else:
            if self.box_traj_count > self.box_traj_num_steps/2:
                self.box_traj_dir *= -1 
                self.box_traj_count = 0

        ros_gazebo.gazebo_set_model_state("box1", ref_frame="box1", pos_x=self.box_traj_dir*self.box1_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box2", ref_frame="box2", pos_x=self.box_traj_dir*self.box2_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box3", ref_frame="box3", pos_x=self.box_traj_dir*self.box3_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box4", ref_frame="box4", pos_x=self.box_traj_dir*self.box4_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box5", ref_frame="box5", pos_x=self.box_traj_dir*self.box5_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box6", ref_frame="box6", pos_x=self.box_traj_dir*self.box6_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box7", ref_frame="box7", pos_x=self.box_traj_dir*self.box7_step, sleep_time=0.0)
        ros_gazebo.gazebo_set_model_state("box8", ref_frame="box8", pos_x=self.box_traj_dir*self.box8_step, sleep_time=0.0)

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
            reward   -= self.mult_dist_reward*self.mag_vec_dist 
            abs_angle = np.abs(self.ang_rob_goal)
            angle_diff = np.minimum(abs_angle, np.pi - abs_angle)
            rospy.logwarn("ANGLE DIFF: " + str(angle_diff))
            reward   -= 0.5*angle_diff
            

        # Check if the robot is in collision
        lidar_ranges = self.get_lidar_ranges()
        min_range = np.min(lidar_ranges)
        if np.any(min_range < self.collision_distance):
            rospy.logwarn("Collision Detected!")
            reward += self.collision_reward

        rospy.logwarn("REWARD====>>>>>>>"+str(reward))

        return reward
    

    #-------------------------------------------------------#
    #  Internal methods for the KobukiDynamicEnv               #

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