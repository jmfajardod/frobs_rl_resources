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
from frobs_rl.envs import robot_BasicEnv
import rospy

import rostopic
from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import numpy as np

register(
        id='KobukiLIDAREnv-v0',
        entry_point='kobuki_maze_rl.robot_env.kobuki_lidar_env:KobukiLIDAREnv',
        max_episode_steps=100000000000,
    )

class KobukiLIDAREnv(robot_BasicEnv.RobotBasicEnv):
    """
    Kobuki LIDAR Env general envionment methods.
    """

    def __init__(self):
        """
        The kobuki robot environment constructor.

        @actuators: The linear and angular velocity of the robot.
        @sensors: The robots LIDAR sensor.
        """
        rospy.loginfo("Starting Kobuki LIDAR Env")

        """
        If spawning the robot using the given spawner then set the corresponding environment variables.
        """
        spawn_robot=True
        model_name_in_gazebo="kobuki_robot"
        namespace="/"
        pkg_name="kobuki_maze_rl"
        urdf_file="kobuki_lidar.urdf.xacro"
        urdf_folder="/urdf"
        controller_file=None
        controller_list=[]
        urdf_xacro_args=None
        rob_state_publisher_max_freq=30
        model_pos_x=0.0; model_pos_y=0.0; model_pos_z=0.0 
        model_ori_x=0.0; model_ori_y=0.0; model_ori_z=0.0; model_ori_w=0.0

        """
        Set the reset mode of gazebo at the beginning of each episode: 1 is "reset_world", 2 is "reset_simulation". Default is 1.
        """
        reset_mode=2
        
        """
        Set the step mode of Gazebo. 1 is "using ROS services", 2 is "using step function of Gazebo".
        """
        step_mode=1

        """
        Init the parent class with the corresponding variables.
        """
        super(KobukiLIDAREnv, self).__init__( spawn_robot=spawn_robot, model_name_in_gazebo=model_name_in_gazebo, namespace=namespace, pkg_name=pkg_name, 
                    urdf_file=urdf_file, urdf_folder=urdf_folder, controller_file=controller_file, controller_list=controller_list, 
                    urdf_xacro_args=urdf_xacro_args, rob_state_publisher_max_freq= rob_state_publisher_max_freq, rob_st_term=False,
                    model_pos_x=model_pos_x, model_pos_y=model_pos_y, model_pos_z=model_pos_z, 
                    model_ori_x=model_ori_x, model_ori_y=model_ori_y, model_ori_z=model_ori_z, model_ori_w=model_ori_w,
                    reset_mode=reset_mode, step_mode=step_mode)

        """
        Define publisher or subscribers as needed.
        """
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.scan = LaserScan()
        self.scan_topic = '/scan'
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        self.odom = Odometry()
        self.odom_topic = '/odom'
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        """
        If using the __check_subs_and_pubs_connection method, then un-comment the lines below.
        """
        ros_gazebo.Gazebo_unpause_physics()
        self._check_subs_and_pubs_connection()
        ros_gazebo.Gazebo_pause_physics()

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of Kobuki LIDAR Env")

    #------------------------------------------#
    #   Custom methods for the KobukiLIDAREnv  #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the Gazebo and ROS connections are ready
        """
        print( rostopic.get_topic_type(self.scan_topic, blocking=True))
        self.scan = rospy.wait_for_message(self.scan_topic, LaserScan, timeout=10)
        self.scan = rospy.wait_for_message(self.scan_topic, LaserScan, timeout=10)
        self.scan = rospy.wait_for_message(self.scan_topic, LaserScan, timeout=10)
        print( rostopic.get_topic_type(self.odom_topic, blocking=True))
        return True

    #--------------------------------------------------#
    #  Methods created for the Kobuki Lidar Env Robot  #

    def lidar_callback(self, scan):
        """
        LIDAR LaserScan Callback
        """
        self.scan = scan

    def get_lidar_ranges(self) -> np.ndarray:
        """
        Return the LIDAR ranges, util for the observation space.
        """
        ranges = np.array(self.scan.ranges)
        ranges = np.nan_to_num(ranges, posinf=30.0, neginf=30.0) 
        return ranges

    def odom_callback(self, odom):
        """
        Odometry Callback
        """
        self.odom = odom
    
    def get_robot_pos(self) -> np.ndarray:
        """
        Return the robots position.
        """
        return np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y])

    def get_robot_ori(self) -> float:
        """
        Return the robots orientation, defined as the yaw angle.
        """
        quat = self.odom.pose.pose.orientation
        euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        return euler[2]

    def send_vel(self, linear, angular):
        """
        Send the linear and angular velocity to the robot.
        """
        vel_msg = Twist()
        vel_msg.linear.x = linear
        vel_msg.angular.z = angular
        self.vel_pub.publish(vel_msg)