#!/bin/python3

from gym import spaces
from gym.envs.registration import register
from frobs_rl.envs import robot_BasicEnv
import rospy
import rostopic
import tf

#- Uncomment the library modules as neeeed
from frobs_rl.common import ros_gazebo
# from frobs_rl.common import ros_controllers
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

"""
Although it is best to register only the task environment, one can also register the
robot environment.
"""
register(
        id='ABBIRB140Env-v0',
        entry_point='abb_irb140_reacher.robot_env.abb_irb140_servo:ABBIRB140Servo',
        max_episode_steps=10000,
    )

class ABBIRB140Servo(robot_BasicEnv.RobotBasicEnv):
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
        rospy.loginfo("Starting ABBIRB140Servo Env")
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
        urdf_file="irb140.urdf.xacro"
        
        controller_file="irb140_pos_controller.yaml"
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
        super(ABBIRB140Servo, self).__init__(   launch_gazebo=launch_gazebo, gazebo_init_paused=gazebo_init_paused, 
                    gazebo_use_gui=gazebo_use_gui, gazebo_recording=gazebo_recording, gazebo_freq=gazebo_freq, world_path=world_path, 
                    world_pkg=world_pkg, world_filename=world_filename, gazebo_max_freq=gazebo_max_freq, gazebo_timestep=gazebo_timestep,
                    spawn_robot=spawn_robot, model_name_in_gazebo=model_name_in_gazebo, namespace=namespace, pkg_name=pkg_name, 
                    urdf_file=urdf_file, urdf_folder=urdf_folder, controller_file=controller_file, controller_list=controller_list, 
                    urdf_xacro_args=urdf_xacro_args, rob_state_publisher_max_freq= rob_state_publisher_max_freq,
                    model_pos_x=model_pos_x, model_pos_y=model_pos_y, model_pos_z=model_pos_z, 
                    model_ori_x=model_ori_x, model_ori_y=model_ori_y, model_ori_z=model_ori_z, model_ori_w=model_ori_w,
                    reset_controllers=reset_controllers, reset_mode=reset_mode, step_mode=step_mode)

        ros_gazebo.gazebo_unpause_physics()
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

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()
        self.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        ros_launch.ros_launch_from_pkg("abb_irb140_moveit", "planning_context.launch", args=["load_robot_description:=False"])
        
        self.servo_node_name = "servo_server"
        servo_params_ns = "optional_parameter_namespace"
        rospy.set_param(self.servo_node_name+"/parameter_ns", servo_params_ns)
        ros_params.ros_load_yaml_from_pkg("abb_irb140_reacher", "abb_servo_config.yaml", ns=self.servo_node_name+"/"+servo_params_ns)
        rospy.sleep(3.0)
        ros_node.ros_node_from_pkg("moveit_servo", "servo_server", name=self.servo_node_name, output="screen", launch_new_term=True)

        self.delta_joint_pub = rospy.Publisher(self.servo_node_name+'/delta_joint_cmds', JointJog, queue_size=5)
        self.delta_twist_pub = rospy.Publisher(self.servo_node_name+'/delta_twist_cmds', TwistStamped, queue_size=5)

        self.collision_sub = rospy.Subscriber(self.servo_node_name+'/internal/collision_velocity_scale', Float64, self.collision_callback)
        self.collision_data = Float64()

        self.worst_time_sub = rospy.Subscriber(self.servo_node_name+'/internal/worst_case_stop_time', Float64, self.worst_time_callback)
        self.worst_time_data = Float64()
        self.worst_time_data.data = 1.0

        self.servo_status_sub = rospy.Subscriber(self.servo_node_name+'/status', Int8, self.servo_status_callback)
        self.servo_status_data = Int8()
        self.worst_time_data.data = 0

        self.tf_br = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        self.pos_cmd_pub = rospy.Publisher("/arm140_group_controller/command", Float64MultiArray, queue_size=5)

        """
        If using the __check_subs_and_pubs_connection method, then un-comment the lines below.
        """
        self._check_subs_and_pubs_connection()

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of ABBIRB140Servo Env")
        ros_gazebo.gazebo_pause_physics()

    #------------------------------------------#
    #   Custom methods for the CustomRobotEnv  #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the Gazebo and ROS connections are ready
        """
        print(rostopic.get_topic_type(self.joint_state_topic, blocking=True))
        print(rostopic.get_topic_type('/'+self.servo_node_name+'/internal/collision_velocity_scale', blocking=True))
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

    def worst_time_callback(self, worst_time):
        """
        Function to get the worst case time of the robot in the servo server.
        """
        self.worst_time_data = worst_time

    def get_worst_time(self):
        return self.worst_time_data.data

    def reset_worst_time(self):
        """
        Function to reset the worst case time of the robot in the servo server.
        """
        self.worst_time_data.data = 1.0

    def servo_status_callback(self, servo_status):
        """
        Function to get the servo status of the robot in the servo server.
        """
        # rospy.logwarn("In Servo status callback")
        self.servo_status_data = servo_status

    def get_servo_status(self):
        return self.servo_status_data.data

    def reset_servo_status(self):
        """
        Function to reset the servo status of the robot in the servo server.
        """
        self.servo_status_data.data = 0

    def collision_callback(self, vel_scale):
        """
        Function to get the velocity scaling factor of the robot in the servo server.
        """
        #rospy.logwarn("In Joint state callback")
        self.collision_data = vel_scale

    def in_collision(self):
        """
        Function to check if the robot is in collision.
        """
        if self.collision_data.data < 1.0:
            return True
        else:
            return False

    def get_vel_scaling(self):
        """
        Function to get the velocity scaling factor of the robot in the servo server.
        """
        return self.collision_data.data
    

    def send_joint_vel_cmd(self, joint_vel_cmd):
        """
        Function to send the joint command to the robot.

        :param joint_cmd: The joint commands to be sent to the robot.
        :type joint_cmd: np.array
        """

        if joint_vel_cmd.shape != (6,):
            rospy.logwarn("Joint command is not of the correct shape. Returning...")
            return False

        joint_cmd = JointJog()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.header.frame_id = "world"
        joint_cmd.joint_names = self.joint_names
        joint_cmd.velocities = joint_vel_cmd.tolist()
        # joint_cmd.displacements = [0,0,0,0,0,0]
        joint_cmd.duration = 0.0

        self.delta_joint_pub.publish(joint_cmd)
        return True

    def send_joint_pos_cmd(self, joint_pos_cmd):
        """
        Function to send the joint command to the robot.

        :param joint_cmd: The joint commands to be sent to the robot.
        :type joint_cmd: np.array
        """

        if joint_pos_cmd.shape != (6,):
            rospy.logwarn("Joint command is not of the correct shape. Returning...")
            return False

        joint_cmd = JointJog()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.header.frame_id = "world"
        joint_cmd.joint_names = self.joint_names
        # joint_cmd.velocities = [0,0,0,0,0,0]
        joint_cmd.displacements = joint_pos_cmd.tolist()
        joint_cmd.duration = 0.0

        self.delta_joint_pub.publish(joint_cmd)
        return True

    def send_twist_cmd(self, linear_vel, angular_vel):
        """
        Function to send the end-effector velocities command to the robot.

        :param linear_vel: The linear velocity of the end-effector.
        :type linear_vel: np.array
        :param angular_vel: The angular velocity of the end-effector.
        :type angular_vel: np.array
        """

        twist_cmd = TwistStamped()
        twist_cmd.header.stamp = rospy.Time.now()
        twist_cmd.header.frame_id = "world"
        twist_cmd.twist.linear.x = linear_vel[0]
        twist_cmd.twist.linear.y = linear_vel[1]
        twist_cmd.twist.linear.z = linear_vel[2]
        twist_cmd.twist.angular.x = angular_vel[0]
        twist_cmd.twist.angular.y = angular_vel[1]
        twist_cmd.twist.angular.z = angular_vel[2]

        self.delta_twist_pub.publish(twist_cmd)
        return True

    def reset_twist_cmd(self):
        """
        Function to set all zero velocities.
        """
        self.send_twist_cmd(np.array([0,0,0]),np.array([0,0,0]))

    def reset_joint_vel_cmd(self):
        """
        Function to set all zero joint commands.
        """
        self.send_joint_vel_cmd(np.array([0,0,0,0,0,0]))

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

    def send_abs_pos_cmd(self, pos_command):

        pos_cmd = Float64MultiArray()
        pos_cmd.data = pos_command

        self.pos_cmd_pub.publish(pos_cmd)



