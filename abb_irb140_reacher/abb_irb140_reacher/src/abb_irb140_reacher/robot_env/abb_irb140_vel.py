#!/bin/python3

from gym import spaces
from gym.envs.registration import register
from frobs_rl.envs import robot_BasicEnv
import rospy
import rostopic
import tf

#- Uncomment the library modules as neeeed
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_controllers
from frobs_rl.common import ros_node
from frobs_rl.common import ros_launch
from frobs_rl.common import ros_params
# from frobs_rl.common import ros_urdf
# from frobs_rl.common import ros_spawn

import numpy as np

from sensor_msgs.msg import JointState
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import Int8
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

"""
Although it is best to register only the task environment, one can also register the
robot environment.
"""
register(
        id='ABBIRB140Env-v0',
        entry_point='abb_irb140_reacher.robot_env.abb_irb140_vel:ABBIRB140Vel',
        max_episode_steps=10000,
    )

class ABBIRB140Vel(robot_BasicEnv.RobotBasicEnv):
    """
    Superclass for all ABB IRB140 environments.
    """

    def __init__(self):
        """
        Initializes a new ABBIRB140Env environment.

        Sensor Topic List:
        * /joint_states : JointState received for the joints of the robot

        Actuators Topic List:
        * MoveIt! Servo: MoveIt! Servo is used to send the joint velocities to the robot.
        """
        rospy.loginfo("Starting ABBIRB140Vel Env")
        ros_gazebo.gazebo_unpause_physics()

        """
        If launching Gazebo with the env then set the corresponding environment variables.
        """
        launch_gazebo=False
        gazebo_init_paused=True
        gazebo_use_gui=True
        gazebo_recording=False 
        gazebo_freq=100
        gazebo_max_freq=None
        gazebo_timestep=None
        
        """
        If launching Gazebo with a custom world then set the corresponding environment variables.
        """
        world_path=None
        world_pkg=None
        world_filename=None
        
        """
        If spawning the robot using the given spawner then set the corresponding environment variables.
        """
        spawn_robot=True
        model_name_in_gazebo="robot1"
        namespace="/"
        pkg_name="abb_irb140"
        urdf_folder="/urdf"
        urdf_xacro_args=None
        urdf_file="irb140_vel.urdf.xacro"
        
        controller_file="irb140_vel_controller.yaml"
        controller_list=["joint_state_controller","arm140_group_controller"]
        
        rob_state_publisher_max_freq= None
        model_pos_x=0.0; model_pos_y=0.0; model_pos_z=0.0 
        model_ori_x=0.0; model_ori_y=0.0; model_ori_z=0.0; model_ori_w=1.0
        
        """
        Set if the controllers in "controller_list" will be reset at the beginning of each episode, default is False.
        """
        reset_controllers=True

        """
        Set the reset mode of gazebo at the beginning of each episode: 1 is "reset_world", 2 is "reset_simulation". Default is 1.
        """
        reset_mode=1
        
        """
        Set the step mode of Gazebo. 1 is "using ROS services", 2 is "using step function of Gazebo". Default is 1.
        If using the step mode 2 then set the number of steps of Gazebo to take in each episode. Default is 1.
        """
        step_mode=1

        """
        Init the parent class with the corresponding variables.
        """
        ros_gazebo.gazebo_unpause_physics()

        super(ABBIRB140Vel, self).__init__(   launch_gazebo=launch_gazebo, gazebo_init_paused=gazebo_init_paused, 
                    gazebo_use_gui=gazebo_use_gui, gazebo_recording=gazebo_recording, gazebo_freq=gazebo_freq, world_path=world_path, 
                    world_pkg=world_pkg, world_filename=world_filename, gazebo_max_freq=gazebo_max_freq, gazebo_timestep=gazebo_timestep,
                    spawn_robot=spawn_robot, model_name_in_gazebo=model_name_in_gazebo, namespace=namespace, pkg_name=pkg_name, 
                    urdf_file=urdf_file, urdf_folder=urdf_folder, controller_file=controller_file, controller_list=controller_list, 
                    urdf_xacro_args=urdf_xacro_args, rob_state_publisher_max_freq= rob_state_publisher_max_freq,
                    model_pos_x=model_pos_x, model_pos_y=model_pos_y, model_pos_z=model_pos_z, 
                    model_ori_x=model_ori_x, model_ori_y=model_ori_y, model_ori_z=model_ori_z, model_ori_w=model_ori_w,
                    reset_controllers=reset_controllers, reset_mode=reset_mode, step_mode=step_mode)

        """
        Define publisher or subscribers as needed.
        """

        self.joint_state_topic = "/joint_states"
        self.joint_names = [ "joint_1",
                            "joint_2",
                            "joint_3",
                            "joint_4",
                            "joint_5",
                            "joint_6"]

        # Subscriber to get the joint states of the robot
        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()
        self.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # TF listener for the end effector
        self.tf_br = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Publisher to send the joint velocities to the robot
        self.vel_cmd_pub = rospy.Publisher("/arm140_group_controller/command", Float64MultiArray, queue_size=5)

        """
        If using the __check_subs_and_pubs_connection method, then un-comment the lines below.
        """
        self._check_subs_and_pubs_connection()
        rospy.logwarn("Publishers and subscribers are connected!")
        
        #- Load the JointTrajectoryController from the yaml file
        # ros_params.ros_load_yaml_from_pkg(pkg_name, controller_file, ns=namespace)
        ros_gazebo.gazebo_unpause_physics()
        ros_controllers.load_controller_srv("arm140_controller")
        rospy.logwarn("JointTrajectoryController loaded!")

        # Create publisher for the JointTrajectoryController
        self.trajectory_pub = rospy.Publisher("/arm140_controller/command", JointTrajectory, queue_size=5)
        rospy.sleep(3.0)

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of ABBIRB140Vel Env")
        ros_gazebo.gazebo_pause_physics()

    #------------------------------------------#
    #   Custom methods for the CustomRobotEnv  #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the Gazebo and ROS connections are ready
        """
        print(rostopic.get_topic_type(self.joint_state_topic, blocking=True))
        return True

    #--------------------------------------------#
    #  Methods created for the Abb IRB120 Robot  #

    def joint_state_callback(self, joint_state):
        """
        Function to get the joint state of the robot.
        """
        self.joint_state = joint_state

    def get_joints(self):
        return self.joint_state

    def get_joint_names(self):
        return self.joint_names

    def get_ee_pos(self):

        trans = None
        for ii in range(10):
            try:
                (trans,rot) = self.tf_listener.lookupTransform('/world','/ee_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Transformation was not found")

            if trans is not None:
                break

        if trans is None:
            return [0.0, 0.0, 0.0]

        return trans

    def send_vel_cmd(self, vel_command):

        vel_cmd = Float64MultiArray()
        vel_cmd.data = vel_command

        self.vel_cmd_pub.publish(vel_cmd)

    def send_traj_pos_cmd(self, traj_pos_command):

        joint_traj = JointTrajectory()
        joint_traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = traj_pos_command
        point.time_from_start = rospy.Duration(15.0)

        joint_traj.points.append(point)

        self.trajectory_pub.publish(joint_traj)



