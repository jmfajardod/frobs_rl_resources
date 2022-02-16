#!/bin/python3

from gym import spaces
from gym.envs.registration import register
from abb_irb140_reacher.robot_env import abb_irb140_servo
import rospy

#- Uncomment the library modules as neeeed
from frobs_rl.common import ros_gazebo
from frobs_rl.common import ros_controllers
from frobs_rl.common import ros_node
# from frobs_rl.common import ros_launch
from frobs_rl.common import ros_params
# from frobs_rl.common import ros_urdf
# from frobs_rl.common import ros_spawn

import numpy as np
import scipy.spatial

from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest
from geometry_msgs.msg import Point

register(
        id='ABBIRB140ReacherEnv-v0',
        entry_point='abb_irb140_reacher.task_env.abb_irb140_reacher:ABBIRB140ReacherEnv',
        max_episode_steps=10000,
    )

class ABBIRB140ReacherEnv(abb_irb140_servo.ABBIRB140Servo):
    """
    Custom Task Env, use this env to implement a task using the robot defined in the CustomRobotEnv
    """

    def __init__(self):
        """
        Describe the task.
        """
        rospy.loginfo("Starting ABBIRB140ReacherEnv Env")

        """
        Load YAML param file
        """
        ros_params.ros_load_yaml_from_pkg("abb_irb140_reacher", "reacher_task.yaml", ns="/")
        self.get_params()

        #--- Define the ACTION SPACE
        # Define a continuous space using BOX and defining its limits
        self.action_space = spaces.Box(low=-1.0*np.array(self.limit_joint_vel), high=np.array(self.limit_joint_vel), dtype=np.float32)

        #--- Define the OBSERVATION SPACE
        #- Define the maximum and minimum positions allowed for the EE
        observations_high_ee_pos_range = np.array(np.array([self.position_ee_max["x"], self.position_ee_max["y"], self.position_ee_max["z"]]))
        observations_low_ee_pos_range  = np.array(np.array([self.position_ee_min["x"], self.position_ee_min["y"], self.position_ee_min["z"]]))

        observations_high_goal_pos_range = np.array(np.array([self.position_goal_max["x"], self.position_goal_max["y"], self.position_goal_max["z"]]))
        observations_low_goal_pos_range  = np.array(np.array([self.position_goal_min["x"], self.position_goal_min["y"], self.position_goal_min["z"]]))

        observations_high_vec_EE_GOAL = np.array([1.0, 1.0, 1.0])
        observations_low_vec_EE_GOAL  = np.array([-1.0, -1.0, -1.0])

        #- With Vector from EE to goal, Goal pos and joint angles
        high = np.concatenate([observations_high_vec_EE_GOAL, observations_high_goal_pos_range, self.max_joint_values, -1.0*np.array(self.limit_joint_vel), ])
        low  = np.concatenate([observations_low_vec_EE_GOAL,  observations_low_goal_pos_range,  self.min_joint_values,  np.array(self.limit_joint_vel), ])

        #--- Observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) 

        rospy.logwarn(self.action_space)
        rospy.logwarn(self.observation_space)


        #-- Action space for sampling
        self.joint_pos_space = spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values), dtype=np.float32)
        self.goal_space = spaces.Box(low=observations_low_goal_pos_range, high=observations_high_goal_pos_range, dtype=np.float32)

        """
        Define subscribers or publishers as needed.
        """
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

        rospy.logwarn("Publisher and subscribers initialized")

        #- If training then launch goal position publisher
        if self.training:
            ros_node.ros_node_from_pkg("abb_irb140_reacher", "pos_publisher.py", name="pos_publisher", ns="/", launch_new_term=True)
            rospy.wait_for_service("set_init_point")
            self.set_init_goal_client = rospy.ServiceProxy("set_init_point", SetLinkState)

        """
        Init super class.
        """
        super(ABBIRB140ReacherEnv, self).__init__()

        """
        Finished __init__ method
        """
        rospy.loginfo("Finished Init of ABBIRB140Reacher Env")

    #-------------------------------------------------------#
    #   Custom available methods for the CustomTaskEnv      #

    def _set_episode_init_params(self):
        """
        Function to set some parameters, like the position of the robot, at the beginning of each episode.
        """

        # Reset controllers
        ros_node.ros_kill_node(self.servo_node_name) # Kill servo node
        ros_controllers.reset_controllers_srv(["arm140_group_controller"])

        #- Reset robot position
        initial_joint_pos =  self.joint_pos_space.sample().tolist() # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rospy.loginfo("Initial joint position: {}".format(initial_joint_pos))

        current_joint_pos = self.get_joints().position
        counter = 0
        while np.any(np.abs(np.array(initial_joint_pos)-np.array(current_joint_pos))>0.01):
            ros_controllers.reset_controllers_srv(["arm140_group_controller"])
            rospy.sleep(3.0)
            self.send_abs_pos_cmd(initial_joint_pos)
            rospy.sleep(3.0)
            current_joint_pos = self.get_joints().position
            counter += 1
            rospy.logwarn("Waiting for robot to reach initial position, counter: "+str(counter))
            if counter > 100:
                rospy.logerr("Failed to set initial joint position")
                break

        #- Launch new servo node
        ros_node.ros_node_from_pkg("moveit_servo", "servo_server", name=self.servo_node_name, output="screen", launch_new_term=True)
        self.prev_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        #--- If training set random goal
        if self.training:
            init_goal_vector = self.goal_space.sample()
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

    def _send_action(self, action):
        """
        The action are the joint velocities.
        """
        self.prev_action = action
        rospy.logwarn("=== Action: {}".format(action))

        #--- Send the action to the robot
        self.send_joint_vel_cmd(action)
        

    def _get_observation(self):
        """
        It returns the position of the EndEffector as observation.
        And the distance from the desired point
        Orientation for the moment is not considered
        TODO Check if observations are enough
        """
        #--- Get Current Joint values
        self.joint_values = self.get_joints().position

        #--- Get current goal
        current_goal = self.goal

        #--- Get EE position
        self.ee_pos = self.get_ee_pos()

        #--- Vector to goal
        vec_EE_GOAL = current_goal - self.ee_pos
        vec_EE_GOAL = vec_EE_GOAL / np.linalg.norm(vec_EE_GOAL)

        obs = np.concatenate((
            vec_EE_GOAL,             # Vector from EE to Goal
            current_goal,            # Position of Goal
            #self.ee_pos,            # Current position of EE
            self.joint_values,       # Current joint angles
            self.prev_action         # Previous joint velocities
            ),
            axis=None
        )

        rospy.logwarn("OBSERVATIONS====>>>>>>>"+str(obs))

        return obs.copy()


    def _get_reward(self):
        """
        Function to get the reward from the environment.
        """
        current_pos = self.ee_pos

        #- Init reward
        reward = 0

        close_to_goal = self.ee_close_to_goal(self.goal, current_pos)

        if close_to_goal:
            reward += self.reached_goal_reward
            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 0.0
            self.goal_marker.color.g = 1.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)
        else:
            #- Distance from EE to Goal reward
            dist2goal = scipy.spatial.distance.euclidean(current_pos, self.goal)
            reward -= self.mult_dist_reward*dist2goal 
            # Publish goal_marker 
            self.goal_marker.header.stamp = rospy.Time.now()
            self.goal_marker.color.r = 1.0
            self.goal_marker.color.g = 0.0
            self.goal_marker.lifetime = rospy.Duration(secs=5)

        self.pub_marker.publish(self.goal_marker)

        # If in self-collision 
        if self.in_collision():
            reward -= self.joint_limits_reward
            rospy.logwarn("In collision")

        #- If joint limits are violated¿
        joint_angles = np.array(self.get_joints().position)
        min_joint_values = np.array(self.min_joint_values)
        max_joint_values = np.array(self.max_joint_values)
        in_limits = np.any(joint_angles<=(min_joint_values+0.01)) or np.any(joint_angles>=(max_joint_values-0.01))
        reward -= in_limits*self.joint_limits_reward
        if in_limits:
            rospy.logwarn("Joint limits violated")

        rospy.logwarn(">>>REWARD>>>"+str(reward))

        return reward

    
    def _check_if_done(self):
        """
        Function to check if the episode is done.
        
        The task only terminates when enough steps are done or when a collision is detected.
        """
        # servo_status = self.get_servo_status()
        # if servo_status == 4 or self.get_vel_scaling() <= 0.0001:
        #     rospy.logwarn("Collision detected, episode terminated")
        #     return True

        return False
    

    #---------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------------------------------#
    #  Internal methods for the ABBIRB120ReacherEnv         #

    def get_params(self):
        """
        get configuration parameters
        """
        
        self.sim_time = rospy.get_time()
        self.n_actions = rospy.get_param('/irb140/n_actions')
        self.n_observations = rospy.get_param('/irb140/n_observations')

        #--- Get parameter associated with ACTION SPACE

        self.limit_joint_vel = rospy.get_param('/irb140/limit_joint_vel')

        assert len(self.limit_joint_vel) == self.n_actions , "The limit joint velocities values do not have the same size as n_actions"

        #--- Get parameter associated with OBSERVATION SPACE

        self.min_joint_values = rospy.get_param('/irb140/min_joint_pos')
        self.max_joint_values = rospy.get_param('/irb140/max_joint_pos')

        self.position_ee_max = rospy.get_param('/irb140/position_ee_max')
        self.position_ee_min = rospy.get_param('/irb140/position_ee_min')

        self.position_goal_max = rospy.get_param('/irb140/position_goal_max')
        self.position_goal_min = rospy.get_param('/irb140/position_goal_min')

        self.max_distance = rospy.get_param('/irb140/max_distance')

        #--- Get parameter asociated to goal tolerance
        self.tol_goal_ee = rospy.get_param('/irb140/tolerance_goal_pos')
        self.training = rospy.get_param('/irb140/training')
        self.pos_dynamic = rospy.get_param('/irb140/pos_dynamic')
        rospy.logwarn("Dynamic position:  " + str(self.pos_dynamic))

        #--- Get reward parameters
        self.reached_goal_reward = rospy.get_param('/irb140/reached_goal_reward')
        self.step_reward = rospy.get_param('/irb140/step_reward')
        self.mult_dist_reward = rospy.get_param('/irb140/multiplier_dist_reward')
        self.joint_limits_reward = rospy.get_param('/irb140/joint_limits_reward')

        #--- Get Gazebo physics parameters
        if rospy.has_param('/irb140/time_step'):
            self.t_step = rospy.get_param('/irb140/time_step')
            ros_gazebo.gazebo_set_time_step(self.t_step)

        if rospy.has_param('/irb140/update_rate_multiplier'):
            self.max_update_rate = rospy.get_param('/irb140/update_rate_multiplier')
            ros_gazebo.gazebo_set_max_update_rate(self.max_update_rate)

    def ee_close_to_goal(self, goal, current_pos):
        """
        It calculated whether it has finished or not
        """
        done = False

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
