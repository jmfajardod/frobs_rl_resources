#!/bin/python3

import gym
from gym import utils
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

import rosnode
import rostopic
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

import moveit_commander

import numpy as np
import sys
import os
import time

register(
        id='ABBIRB120MoveItEnv-v0',
        entry_point='abb_irb120_reacher.robot_env.abb_irb120_moveit:ABBIRB120MoveItEnv',
        max_episode_steps=10000,
    )

class ABBIRB120MoveItEnv(robot_BasicEnv.RobotBasicEnv):
    """
    Superclass for all ABB IRB120 environments.
    """

    def __init__(self):
        """
        Initializes a new ABBIRB120Env environment.

        Sensor Topic List:
        * /joint_states : JointState received for the joints of the robot

        Actuators Topic List:
        * MoveIt! : MoveIt! action server is used to send the joint positions to the robot.
        """
        rospy.loginfo("Starting ABBIRB120MoveIt Env")
        ros_gazebo.gazebo_unpause_physics()

        """
        Robot model and controllers parameters
        """
        self.model_name_in_gazebo="robot1"
        self.namespace="/robot1"
        pkg_name="abb_irb120"
        urdf_file="irb120.urdf.xacro" 
        urdf_xacro_args=['use_pos_ctrls:=true']
        model_pos_x=0.0; model_pos_y=0.0; model_pos_z=0.0 
        controller_file="irb120_pos_controller.yaml"
        self.controller_list=["joint_state_controller","arm120_controller"]

        """
        Use parameters to allow multiple robots in the environment.
        """
        if rospy.has_param('/ABB_IRB120_Reacher/current_robot_num'):
            num_robot = int(rospy.get_param("/ABB_IRB120_Reacher/current_robot_num") + 1)
            self.model_name_in_gazebo = "robot" + str(num_robot)
            self.namespace = "/robot" + str(num_robot)
            
        else:
            num_robot = 0
            self.model_name_in_gazebo = "robot" + str(num_robot)
            self.namespace = "/"
        
        rospy.set_param('/ABB_IRB120_Reacher/current_robot_num', num_robot)
        model_pos_x = int(num_robot / 4) * 2.0
        model_pos_y = int(num_robot % 4) * 2.0
        model_pos_z = 0.0

        """
        Set if the controllers in "controller_list" will be reset at the beginning of each episode, default is False.
        """
        reset_controllers=False

        """
        Set the reset mode of gazebo at the beginning of each episode: 1 is "reset_world", 2 is "reset_simulation". Default is 1.
        """
        reset_mode=1
        
        """
        Set the step mode of Gazebo. 1 is "using ros services", 2 is "using step function of gazebo". Default is 1.
        If using the step mode 2 then set the number of steps of Gazebo to take in each episode. Default is 1.
        """
        step_mode=1

        """
        Init the parent class with the corresponding variables.
        """
        super(ABBIRB120MoveItEnv, self).__init__(   launch_gazebo=False, spawn_robot=True, 
                    model_name_in_gazebo=self.model_name_in_gazebo, namespace=self.namespace, pkg_name=pkg_name, 
                    urdf_file=urdf_file, controller_file=controller_file, controller_list=self.controller_list, urdf_xacro_args=urdf_xacro_args,
                    model_pos_x=model_pos_x, model_pos_y=model_pos_y, model_pos_z=model_pos_z, 
                    reset_controllers=reset_controllers, reset_mode=reset_mode, step_mode=step_mode)

        """
        Define publisher or subscribers as needed.
        """
        if self.namespace is not None and self.namespace != '/':
            self.joint_state_topic = self.namespace + "/joint_states"
        else:
            self.joint_state_topic = "/joint_states"

        self.joint_names = [ "joint_1",
                            "joint_2",
                            "joint_3",
                            "joint_4",
                            "joint_5",
                            "joint_6"]

        self.joint_state_sub = rospy.Subscriber(self.joint_state_topic, JointState, self.joint_state_callback)
        self.joint_state = JointState()

        """
        Init MoveIt
        """
        
        ros_launch.ros_launch_from_pkg("abb_irb120_reacher","moveit_init.launch", args=["namespace:="+str(self.namespace)])
        rospy.wait_for_service("/move_group/trajectory_execution/set_parameters")
        print(rostopic.get_topic_type("/planning_scene", blocking=True))
        print(rostopic.get_topic_type("/move_group/status", blocking=True))

        """
        If using the _check_subs_and_pubs_connection method, then un-comment the lines below.
        """
        self._check_subs_and_pubs_connection()

        #--- Start MoveIt Object
        self.move_abb_object = MoveABB()

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of ABBIRB120MoveIt Env")
        ros_gazebo.gazebo_pause_physics()

    #------------------------------------------#
    #   Custom methods for the CustomRobotEnv  #

    def _check_subs_and_pubs_connection(self):
        """
        Function to check if the gazebo and ros connections are ready
        """
        self._check_joint_states_ready()
        return True
    
    
    def _check_joint_states_ready(self):
        """
        Function to check if the joint states are received
        """
        ros_gazebo.gazebo_unpause_physics()
        print( rostopic.get_topic_type(self.joint_state_topic, blocking=True))
        rospy.logdebug("Current "+ self.joint_state_topic +" READY=>" + str(self.joint_state))
            
        return True

    #--------------------------------------------#
    #  Methods created for the Abb IRB120 Robot  #

    def joint_state_callback(self, joint_state):
        """
        Function to get the joint state of the robot.
        """
        #rospy.logwarn("In Joint state callback")
        self.joint_state = joint_state

    def get_joints(self):
        return self.joint_state

    def get_joint_names(self):
        return self.joint_names

    def create_action(self, position, orientation):
        """
        position = [x,y,z]
        orientation= [x,y,z,w]
        """

        gripper_target = np.array(position)
        gripper_rotation = np.array(orientation)
        action = np.concatenate([gripper_target, gripper_rotation])

        return action

    def set_trajectory_ee(self, action):
        """
        Sets the Pose of the EndEffector based on the action variable.
        The action variable contains the position and orientation of the EndEffector.
        """
        # Set up a trajectory message to publish.
        ee_target = Pose()

        # Standard orientation  
        ee_target.orientation.x = -0.707
        ee_target.orientation.y =  0.000
        ee_target.orientation.z =  0.707
        ee_target.orientation.w =  0.001

        ee_target.position.x = action[0]
        ee_target.position.y = action[1]
        ee_target.position.z = action[2]

        result = self.move_abb_object.ee_traj(ee_target)
        return result

    def set_trajectory_joints(self, q_positions):
        """
        Set a joint position target for the joints.
        """
        result = self.move_abb_object.joint_traj(q_positions)

        return result

    def create_joints_dict(self, joints_positions):
        """
        Based on the Order of the positions, they will be assigned to its joint name
        names_in_order:
            joint_1
            joint_2
            joint_3
            joint_4
            joint_5
            joint_6
        """

        assert len(joints_positions) == len(self.joint_names), "Wrong number of joints, there should be "+str(len(self.joint_names))
        joints_dict = dict(zip(self.joint_names, joints_positions))

        return joints_dict

    def get_ee_pose(self):
        """
        Returns geometry_msgs/PoseStamped
        """
        ros_gazebo.gazebo_unpause_physics()
        gripper_pose = self.move_abb_object.ee_pose()
        ros_gazebo.gazebo_pause_physics()
        return gripper_pose

    def get_ee_rpy(self):
        """
        Returns a list of 3 elements defining the [roll, pitch, yaw] of the end-effector.
        """
        gripper_rpy = self.move_abb_object.ee_rpy()
        return gripper_rpy

    def get_joint_angles(self):
        ros_gazebo.gazebo_unpause_physics()
        joint_angles = self.move_abb_object.joint_angles()
        ros_gazebo.gazebo_pause_physics()
        return joint_angles

    def check_goal(self, goal):
        """
        Check if the goal is reachable
        * goal is a list with 3 elements, XYZ positions of the EE
        """
        result = self.move_abb_object.is_goal_reachable(goal)
        return result
    
    def get_randomJointVals(self):
        return self.move_abb_object.group.get_random_joint_values()

    def get_randomPose(self):
        return self.move_abb_object.group.get_random_pose()

# Class to move ABB robot through MoveIt

class MoveABB(object):

    def __init__(self):
        rospy.logwarn("===== In MoveABB")

        #--- Init MoveIt commander
        moveit_commander.roscpp_initialize(sys.argv)

        #--- Instantiate a RobotCommander object
        self.robot = moveit_commander.RobotCommander()

        #--- Instantiate a PlanningSceneInterface object
        self.scene = moveit_commander.PlanningSceneInterface()

        #---Instantiate a MoveGroupCommander object.
        self.group_name = "irb120_arm"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        rospy.logwarn("===== Out MoveABB")

    def ee_traj(self, pose):
        self.group.set_pose_target(pose)
        result = self.execute_trajectory()
        return result

    def joint_traj(self, positions_array):

        self.group_variable_values = self.group.get_current_joint_values()
        self.group_variable_values[0] = float(positions_array[0])
        self.group_variable_values[1] = float(positions_array[1])
        self.group_variable_values[2] = float(positions_array[2])
        self.group_variable_values[3] = float(positions_array[3])
        self.group_variable_values[4] = float(positions_array[4])
        self.group_variable_values[5] = float(positions_array[5])

        #rospy.logwarn("Joint command")
        #rospy.logwarn(self.group_variable_values)
        
        self.group.set_joint_value_target(self.group_variable_values)
        
        result = self.execute_trajectory()

        return result

    def execute_trajectory(self):
        """
        Assuming that the trajecties has been set to the self objects appropriately
        Make a plan to the destination in Homogeneous Space(x,y,z,yaw,pitch,roll)
        and returns the result of execution
        """
        self.plan = self.group.plan()
        
        result = self.group.go(wait=True)
        # result = selg.group.execute(self.plan, wait=True)
        
        self.group.stop()
        
        return result

    def ee_pose(self):
        gripper_pose = self.group.get_current_pose()
        return gripper_pose

    def ee_rpy(self):
        gripper_rpy = self.group.get_current_rpy()
        return gripper_rpy

    def joint_angles(self):
        current_jt_vals = self.group.get_current_joint_values ()
        return current_jt_vals

    def is_goal_reachable(self, goal):
        """
        Check if the goal is reachable
        * goal is the desired XYZ of the EE 
        """

        if isinstance(goal, type(np.array([0]))):
            goal = goal.tolist()

        goal[0] = float(goal[0])
        goal[1] = float(goal[1])
        goal[2] = float(goal[2])

        self.group.set_position_target(goal)
        plan = self.group.plan()        
        result = plan[0]
        self.group.clear_pose_targets()

        return result
