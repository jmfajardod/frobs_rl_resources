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
        action_space_low  = np.array([-0.3, -0.3]) 
        action_space_high = np.array([ 0.3,  0.3])
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)
        rospy.logwarn("Action Space->>>" + str(self.action_space))

        pose_low   = np.array([-6.0, -8.0, - np.pi])
        pose_high  = np.array([ 26.0, 7.0, np.pi])
        dist_low   = np.array([-32.0, -15.0])
        dist_high  = np.array([ 32.0,  15.0])
        lidar_low  = 0.0 * np.ones(self.lidar_samples)
        lidar_high = 30.0 * np.ones(self.lidar_samples)
        goal_low   = np.array([-6.0, -8.0])
        goal_high  = np.array([26.0, 7.0])

        observ_space_low  = np.concatenate((pose_low,  dist_low,  lidar_low,  goal_low))
        observ_space_high = np.concatenate((pose_high, dist_high, lidar_high, goal_high))
        self.observation_space = spaces.Box(low=observ_space_low, high=observ_space_high, dtype=np.float32)
        rospy.logwarn("Observation Space->>>" + str(self.observation_space))

        # Spaces for initial robot position and goal position (for reset)
        self.robot_pos_space = spaces.Box(low=np.array([-5.3, -7.0, -np.pi]), high=np.array([ 25.0, 6.3, np.pi]), dtype=np.float32)
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)

        """
        Define subscribers or publishers as needed.
        """
        # self.pub1 = rospy.Publisher('/robot/controller_manager/command', JointState, queue_size=1)
        # self.sub1 = rospy.Subscriber('/robot/joint_states', JointState, self.callback1)

        # Load kobuki maze
        ObjectMaze = ros_urdf.URDF_parse_from_pkg("kobuki_maze_rl", "model.sdf", folder="/worlds/Kobuki_maze")

        # Spawn the maze in Gazebo
        ros_gazebo.Gazebo_spawn_sdf_string(ObjectMaze, model_name="kobuki_maze", pos_x=7.5, pos_y=2.0, pos_z=0.0)

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
        done_rob_init = False
        while not done_rob_init:
            init_robot_pos = self.robot_pos_space.sample()
            rospy.logwarn("Initial Robot Position->>>" + "X: " + str(init_robot_pos[0]) + " Y: " + str(init_robot_pos[1]) + " Theta: " + str(init_robot_pos[2]))
            quat = quaternion_from_euler(0.0, 0.0, init_robot_pos[2])
            ros_gazebo.Gazebo_set_model_state("kobuki_robot", pos_x=init_robot_pos[0], pos_y=init_robot_pos[1], pos_z=2.6,
                                                ori_x=quat[0], ori_y=quat[1], ori_z=quat[2], ori_w=quat[3])
            rospy.sleep(3.0)
            _,check_pos,_,_ = ros_gazebo.Gazebo_get_model_state("kobuki_robot")
            robot_euler = euler_from_quaternion([check_pos.orientation.x, check_pos.orientation.y, check_pos.orientation.z, check_pos.orientation.w])
            if check_pos.position.z < 0.5 and robot_euler[0] < 0.1 and robot_euler[1] < 0.1:
                rospy.logwarn("Valid robot init pos")
                done_rob_init = True

        # Set goal position
        self.goal_pos = self.goal_space.sample()
        #self.goal_pos = np.array([25.0, -4.0])

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

        # Get the goal pose
        goal = self.goal_pos
        goal_dist = np.array([goal[0]-pos[0], goal[1]-pos[1]])

        # Get the lidar data
        lidar_ranges = self.get_lidar_ranges()

        obs = np.concatenate((
            pos,             # Current robot pose obtained from Odom
            goal,            # Goal position   
            goal_dist,       # Distance in X and Y to Goal
            lidar_ranges     # Lidar ranges data
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(pos) + "  " + str(goal_dist) +  "  " + str(goal))
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
        self.reached_goal_reward = rospy.get_param('/kobuki_maze/reached_goal_reward')
        self.step_reward = rospy.get_param('/kobuki_maze/step_reward')
        self.mult_dist_reward = rospy.get_param('/kobuki_maze/multiplier_dist_reward')
        self.collision_reward = rospy.get_param('/kobuki_maze/collision_reward')
        self.collision_distance = rospy.get_param('/kobuki_maze/collision_distance_threshold')

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