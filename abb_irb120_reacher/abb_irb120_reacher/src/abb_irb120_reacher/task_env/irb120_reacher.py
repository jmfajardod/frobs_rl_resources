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
from abb_irb120_reacher.robot_env import abb_irb120_moveit
import rospy
import rostopic

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest, SetLinkStateResponse

import numpy as np
import scipy.spatial

register(
        id='ABBIRB120ReacherEnv-v0',
        entry_point='abb_irb120_reacher.task_env.irb120_reacher:ABBIRB120ReacherEnv',
        max_episode_steps=10000
    )

class ABBIRB120ReacherEnv(abb_irb120_moveit.ABBIRB120MoveItEnv):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self):
        """
        Describe the task.
        """
        rospy.logwarn("Starting ABBIRB120ReacherEnv Task Env")

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("abb_irb120_reacher", "reacher_task.yaml", ns="/")
        self.get_params()

        """
        Define the action and observation space.
        """
        #--- Define the ACTION SPACE
        # Define a continuous space using BOX and defining its limits
        self.action_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values), dtype=np.float32)

        #--- Define the OBSERVATION SPACE
        #- Define the maximum and minimum positions allowed for the EE
        observations_high_ee_pos_range = np.array(np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]]))
        observations_low_ee_pos_range  = np.array(np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]]))

        observations_high_goal_pos_range = np.array(np.array([self.position_goal_max["x"], self.position_goal_max["y"], self.position_goal_max["z"]]))
        observations_low_goal_pos_range  = np.array(np.array([self.position_goal_min["x"], self.position_goal_min["y"], self.position_goal_min["z"]]))

        observations_high_vec_EE_GOAL = np.array([1.0, 1.0, 1.0])
        observations_low_vec_EE_GOAL  = np.array([-1.0, -1.0, -1.0])

        #- Define the maximum and minimum distance to the GOAL
        #observations_high_dist = np.array([self.max_distance])
        #observations_low_dist = np.array([0.0])

        #--- Concatenate the observation space limits for positions and distance to goal
        #- With Goal pos, EE pos and joint angles, 
        #high = np.concatenate([observations_high_goal_pos_range,observations_high_ee_pos_range, self.max_joint_values])
        #low  = np.concatenate([observations_low_goal_pos_range,observations_low_ee_pos_range, self.min_joint_values,  ])

        #- With Goal pos and joint angles
        #high = np.concatenate([observations_high_goal_pos_range, self.max_joint_values, ])
        #low  = np.concatenate([observations_low_goal_pos_range, self.min_joint_values, ])

        #- With Vector from EE to goal, Goal pos and joint angles
        high = np.concatenate([observations_high_vec_EE_GOAL, observations_high_goal_pos_range, self.max_joint_values, ])
        low  = np.concatenate([observations_low_vec_EE_GOAL,  observations_low_goal_pos_range,  self.min_joint_values, ])

        #--- Observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) 

        #-- Action space for sampling
        self.goal_space = spaces.Box(low=observations_low_goal_pos_range, high=observations_high_goal_pos_range, dtype=np.float32)

        """
        Define subscribers or publishers as needed.
        """

        #--- Make Marker msg for publishing
        self.goal_marker = Marker()
        self.goal_marker.header.frame_id="world"
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

        self.goal_marker.scale.x = 0.1
        self.goal_marker.scale.y = 0.1
        self.goal_marker.scale.z = 0.1

        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        self.pub_marker = rospy.Publisher("goal_point",Marker,queue_size=10)
        self.goal_subs  = rospy.Subscriber("goal_pos", Point, self.goal_callback)
        if self.training:
            ros_node.ros_node_from_pkg("abb_irb120_reacher", "pos_publisher.py", name="pos_publisher", ns="/")
            rospy.wait_for_service("set_init_point")
            self.set_init_goal_client = rospy.ServiceProxy("set_init_point", SetLinkState)

        """
        Init super class.
        """
        super(ABBIRB120ReacherEnv, self).__init__()

        """
        Finished __init__ method
        """
        rospy.logwarn("Finished Init of ABBIRB120ReacherEnv Task Env")

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #

    def _set_episode_init_params(self):
        """
        Sets the Robot in its init pose
        The Simulation will be unpaused for this purpose.
        """
        

        self.init_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = self.set_trajectory_joints(self.init_pos)
        if not result:
            rospy.logwarn("Homing is failed....")

        #--- If training set random goal
        if self.training:
            self.init_pos = self.get_randomJointVals()
            init_goal_vector = self.get_randomValidGoal()
            self.goal = init_goal_vector
            init_goal_msg = SetLinkStateRequest()
            init_goal_msg.link_state.pose.position.x = init_goal_vector[0]
            init_goal_msg.link_state.pose.position.y = init_goal_vector[1]
            init_goal_msg.link_state.pose.position.z = init_goal_vector[2]

            self.set_init_goal_client.call(init_goal_msg)
            rospy.logwarn("Desired goal--->" + str(self.goal))

        #--- Make Marker msg for publishing
        self.goal_marker.pose.position.x = self.goal[0]
        self.goal_marker.pose.position.y = self.goal[1]
        self.goal_marker.pose.position.z = self.goal[2]

        self.goal_marker.color.r = 1.0
        self.goal_marker.color.g = 0.0
        self.goal_marker.color.b = 0.0
        self.goal_marker.color.a = 1.0

        self.goal_marker.lifetime = rospy.Duration(secs=30)
        
        self.pub_marker.publish(self.goal_marker)

        rospy.logwarn("Initializing with values" + str(self.init_pos))
        result = self.set_trajectory_joints(self.init_pos)
        self.joint_angles = self.init_pos
        if not result:
            rospy.logwarn("Initialisation is failed....")

    def _send_action(self, action):
        """
        The action are the joint positions
        TODO Check what to do if movement result is False
        """
        rospy.logwarn("=== Action: {}".format(action))

        #--- Make actions as deltas
        action = self.joint_values + action
        action = np.clip(action, self.min_joint_values, self.max_joint_values)

        self.movement_result = self.set_trajectory_joints(action)
        if not self.movement_result:
            rospy.logwarn("Movement_result failed with the action of : " + str(action))
        

    def _get_observation(self):
        """
        It returns the position of the EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered
        TODO Check if observations are enough
        """

        #--- Get Current Joint values
        self.joint_values = self.get_joint_angles()

        #--- Get current goal
        current_goal = self.goal

        #--- Get EE position
        ee_pos_v = self.get_ee_pose() # Get a geometry_msgs/PoseStamped msg
        self.ee_pos = np.array([ee_pos_v.pose.position.x, ee_pos_v.pose.position.y, ee_pos_v.pose.position.z])

        #--- Vector to goal
        vec_EE_GOAL = current_goal - self.ee_pos
        vec_EE_GOAL = vec_EE_GOAL / np.linalg.norm(vec_EE_GOAL)

        obs = np.concatenate((
            vec_EE_GOAL,             # Vector from EE to Goal
            current_goal,               # Position of Goal
            #self.ee_pos,            # Current position of EE
            self.joint_values        # Current joint angles
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

        #--- UNCOMMENT TO RETURN A DICT WITH OBS AND GOAL
        # return {
        #    'observation':   obs.copy(),
        #    'desired_goal':  self.goal.copy(),
        #}

        return obs.copy()

    def _get_reward(self):
        """
        Given a success of the execution of the action
        Calculate the reward: binary => 1 for success, 0 for failure
        TODO give reward if current distance is lower than previous
        """

        #--- Get current EE pos 
        #current_pos = observations['observation'][:3] # If using DICT
        current_pos = self.ee_pos # If using ARRAY

        #- Init reward
        reward = 0

        #- Check if the EE reached the goal
        done = False
        done = self.calculate_if_done(self.movement_result, self.goal, current_pos)
        if done:
            if self.pos_dynamic is False:
                rospy.logwarn("SUCCESS Reached a Desired Position!")
                self.info['is_success'] = 1.0

            #- Success reward
            reward += self.reached_goal_reward

            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 0.0
            self.goal_marker.color.g = 1.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)
            

        else:
            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 1.0
            self.goal_marker.color.g = 0.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)

            #- Distance from EE to Goal reward
            dist2goal = scipy.spatial.distance.euclidean(current_pos, self.goal)
            reward += - self.mult_dist_reward*dist2goal 

            #- Constant reward
            reward += self.step_reward

        self.pub_marker.publish(self.goal_marker)

        #- Check if joints are in limits
        joint_angles = np.array(self.joint_values)
        min_joint_values = np.array(self.min_joint_values)
        max_joint_values = np.array(self.max_joint_values)
        in_limits = np.any(joint_angles<=(min_joint_values+0.0001)) or np.any(joint_angles>=(max_joint_values-0.0001))
        reward += in_limits*self.joint_limits_reward

        rospy.logwarn(">>>REWARD>>>"+str(reward))

        return reward
    
    def _check_if_done(self):
        """
        Check if the EE is close enough to the goal
        """

        #--- Get current EE based on the observation
        #current_pos = observations['observation'][:3] # If using DICT
        current_pos = self.ee_pos # If using ARRAY

        #--- Function used to calculate 
        done = self.calculate_if_done(self.movement_result, self.goal, current_pos)
        if done:
            rospy.logdebug("Reached a Desired Position!")

        #--- If the position is dynamic the episode is never done
        if self.pos_dynamic is True:
            done = False

        return done

    #-------------------------------------------------------#
    #  Internal methods for the ABBIRB120ReacherEnv         #

    def get_params(self):
        """
        get configuration parameters
        """
        
        self.sim_time = rospy.get_time()
        self.n_actions = rospy.get_param('/irb120/n_actions')
        self.n_observations = rospy.get_param('/irb120/n_observations')

        #--- Get parameter associated with ACTION SPACE

        self.min_joint_values = rospy.get_param('/irb120/min_joint_pos')
        self.max_joint_values = rospy.get_param('/irb120/max_joint_pos')

        assert len(self.min_joint_values) == self.n_actions , "The min joint values do not have the same size as n_actions"
        assert len(self.max_joint_values) == self.n_actions , "The max joint values do not have the same size as n_actions"

        #--- Get parameter associated with OBSERVATION SPACE

        self.position_ee_max = rospy.get_param('/irb120/position_ee_max')
        self.position_ee_min = rospy.get_param('/irb120/position_ee_min')
        self.position_goal_max = rospy.get_param('/irb120/position_goal_max')
        self.position_goal_min = rospy.get_param('/irb120/position_goal_min')
        self.max_distance = rospy.get_param('/irb120/max_distance')

        #--- Get parameter asociated to goal tolerance
        self.tol_goal_ee = rospy.get_param('/irb120/tolerance_goal_pos')
        self.training = rospy.get_param('/irb120/training')
        self.pos_dynamic = rospy.get_param('/irb120/pos_dynamic')
        rospy.logwarn("Dynamic position:  " + str(self.pos_dynamic))

        #--- Get reward parameters
        self.reached_goal_reward = rospy.get_param('/irb120/reached_goal_reward')
        self.step_reward = rospy.get_param('/irb120/step_reward')
        self.mult_dist_reward = rospy.get_param('/irb120/multiplier_dist_reward')
        self.joint_limits_reward = rospy.get_param('/irb120/joint_limits_reward')

        #--- Get Gazebo physics parameters
        if rospy.has_param('/irb120/time_step'):
            self.t_step = rospy.get_param('/irb120/time_step')
            ros_gazebo.gazebo_set_time_step(self.t_step)

        if rospy.has_param('/irb120/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/irb120/update_rate_multiplier')
            ros_gazebo.gazebo_set_max_update_rate(self.max_update_rate)


    def get_elapsed_time(self):
        """
        Returns the elapsed time since the last check
        Useful to calculate rewards based on time
        """
        current_time = rospy.get_time()
        dt = self.sim_time - current_time
        self.sim_time = current_time
        return dt

    def test_goalPose(self, goal):
        """
        Function used to check if the defined goal is reachable
        """
        rospy.logwarn("Goal to check: " + str(goal))
        result = self.check_goal(goal)
        if result == False:
            rospy.logwarn( "The goal is not reachable")
        
        return result

    def get_randomValidGoal(self):
        is_valid = False
        while is_valid is False:
            goal = self.goal_space.sample()
            is_valid = self.test_goalPose(goal)
        
        return goal

    def calculate_if_done(self, movement_result, goal, current_pos):
        """
        It calculated whether it has finished or not
        """
        done = False

        # If the previous movement was succesful
        if movement_result:
            rospy.logdebug("Movement was succesful")
        
        else:
            rospy.logwarn("Movement not succesful")

        # check if the end-effector located within a threshold to the goal
        distance_2_goal = scipy.spatial.distance.euclidean(current_pos, goal)

        if distance_2_goal<=self.tol_goal_ee:
            done = True
        
        return done

    def goal_callback(self, data):
        """
        Callback to the topic used to send goals
        """
        self.goal = np.array([data.x, data.y, data.z])

        #--- Publish goal marker
        self.goal_marker.pose.position.x = self.goal[0]
        self.goal_marker.pose.position.y = self.goal[1]
        self.goal_marker.pose.position.z = self.goal[2]
        self.goal_marker.lifetime = rospy.Duration(secs=1)
        self.pub_marker.publish(self.goal_marker)