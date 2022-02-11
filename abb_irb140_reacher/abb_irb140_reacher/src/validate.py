import sys
from abb_irb140_reacher.task_env.irb140_reacher import abb_irb140_moveit
from frobs_rl.common import ros_gazebo, ros_node
from frobs_rl.wrappers.NormalizeActionWrapper import NormalizeActionWrapper
from frobs_rl.wrappers.TimeLimitWrapper import TimeLimitWrapper
from frobs_rl.wrappers.NormalizeObservWrapper import NormalizeObservWrapper
import gym
import rospy
import rospkg

from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Twist

# Import TD3 algorithm
from frobs_rl.models.td3 import TD3

if __name__ == '__main__':
    # Kill all processes related to previous runs
    ros_node.ros_kill_all_processes()

    # Launch Gazebo
    ros_gazebo.launch_Gazebo(paused=True, gui=True)

    # Start node
    rospy.logwarn("Start")
    rospy.init_node('irb140_maze_train')

    # Launch the task environment
    env = gym.make('ABBIRB140ReacherEnv-v0')

     #--- Normalize action space
    env = NormalizeActionWrapper(env)

    #--- Normalize observation space
    env = NormalizeObservWrapper(env)

    #--- Set max steps
    env = TimeLimitWrapper(env, max_steps=100)
    env.reset()

     #--- Set the save and log path
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("abb_irb140_reacher")
    save_path = pkg_path + "/models/static_reacher/td3/"

    #-- TD3 trained
    model = TD3.load_trained(save_path + "td3_model")


    obs = env.reset()
    episodes = 10
    epi_count = 0

    pub = rospy.Publisher("accion_topic", numpy_msg(Twist), queue_size=10)
    joints=Twist()

    joints.linear.x = obs[6]
    joints.linear.y = obs[7]
    joints.linear.z = obs[8]
    
    joints.angular.x = obs[9]
    joints.angular.y = obs[10]
    joints.angular.z = obs[11]
    pub.publish(joints)

    while epi_count < episodes:
        action, _states = model.predict(obs, deterministic=True)
        joints.linear.x = action[0]
        joints.linear.y = action[1]
        joints.linear.z = action[2]
        
        joints.angular.x = action[3]
        joints.angular.y = action[4]
        joints.angular.z = action[5]
        
        pub.publish(joints)

        obs, _, dones, info = env.step(action)
        if dones:
            epi_count += 1
            rospy.logwarn("Episode: " + str(epi_count))
            obs = env.reset()
            joints.linear.x = -1
            joints.linear.y = -1
            joints.linear.z = -1
            
            joints.angular.x = -1
            joints.angular.y = -1
            joints.angular.z = -1
            pub.publish(joints)

            joints.linear.x = obs[6]
            joints.linear.y = obs[7]
            joints.linear.z = obs[8]
            
            joints.angular.x = obs[9]
            joints.angular.y = obs[10]
            joints.angular.z = obs[11]
            pub.publish(joints)

    env.close()
    sys.exit()