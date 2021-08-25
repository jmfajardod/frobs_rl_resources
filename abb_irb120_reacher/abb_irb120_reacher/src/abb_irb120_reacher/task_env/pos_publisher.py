#!/bin/python3

import rospy
import numpy as np
import scipy as sp

from geometry_msgs.msg import Point
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest, SetLinkStateResponse

class PosPublisher:

    def __init__(self):

        self.get_params()
        
        self.pub = rospy.Publisher('/goal_pos', Point, queue_size=10)
        
        self.init_point = Point()
        self.init_point.x = 0.0; self.init_point.y = 0.0; self.init_point.z = 0.0

        self.point = Point()
        self.point.x = 0.0; self.point.y = 0.0; self.point.z = 0.0

        srv_init_pos = rospy.Service('set_init_point', SetLinkState, self.set_new_pos)

        self.spin()

    def get_params(self):

        self.time = rospy.Time.now()

        self.pos_dynamic = rospy.get_param('irb120/pos_dynamic')
        if self.pos_dynamic:
            self.x_max_amplitude = rospy.get_param('irb120/pos_dynamic_params/x_max_amplitude')
            self.y_max_amplitude = rospy.get_param('irb120/pos_dynamic_params/y_max_amplitude')
            self.z_max_amplitude = rospy.get_param('irb120/pos_dynamic_params/z_max_amplitude')

            self.x_max_freq = rospy.get_param('irb120/pos_dynamic_params/x_max_freq')
            self.y_max_freq = rospy.get_param('irb120/pos_dynamic_params/y_max_freq')
            self.z_max_freq = rospy.get_param('irb120/pos_dynamic_params/z_max_freq')

            self.x_amplitude = np.random.uniform(low= 0.0, high=self.x_max_amplitude)
            self.y_amplitude = np.random.uniform(low= 0.0, high=self.y_max_amplitude)
            self.z_amplitude = np.random.uniform(low= 0.0, high=self.z_max_amplitude)

            self.x_freq = np.random.uniform(low= 0.0, high=self.x_max_freq)
            self.y_freq = np.random.uniform(low= 0.0, high=self.y_max_freq)
            self.z_freq = np.random.uniform(low= 0.0, high=self.z_max_freq)


    def set_new_pos(self, link_state):

        self.init_point.x = link_state.link_state.pose.position.x
        self.init_point.y = link_state.link_state.pose.position.y
        self.init_point.z = link_state.link_state.pose.position.z

        self.time = rospy.Time.now()

        if self.pos_dynamic:
            self.x_amplitude = np.random.uniform(low= 0.0, high=self.x_max_amplitude)
            self.y_amplitude = np.random.uniform(low= 0.0, high=self.y_max_amplitude)
            self.z_amplitude = np.random.uniform(low= 0.0, high=self.z_max_amplitude)

            self.x_freq = np.random.uniform(low= 0.0, high=self.x_max_freq)
            self.y_freq = np.random.uniform(low= 0.0, high=self.y_max_freq)
            self.z_freq = np.random.uniform(low= 0.0, high=self.z_max_freq)

        resp = SetLinkStateResponse()
        resp.success = True
        resp.status_message = "New position set"

        return resp

    def spin(self):

        r = rospy.Rate(30)

        while not rospy.is_shutdown():
            
            current_time = rospy.Time.now()

            if self.pos_dynamic:
                self.point.x = self.init_point.x + self.x_amplitude * np.sin(2 * np.pi * self.x_freq * (current_time - self.time).to_sec())
                self.point.y = self.init_point.y + self.y_amplitude * np.sin(2 * np.pi * self.y_freq * (current_time - self.time).to_sec())
                self.point.z = self.init_point.z + self.z_amplitude * np.sin(2 * np.pi * self.z_freq * (current_time - self.time).to_sec())
            else:
                self.point.x = self.init_point.x
                self.point.y = self.init_point.y
                self.point.z = self.init_point.z

            self.pub.publish(self.point)

            r.sleep()


if __name__ == "__main__":
    rospy.init_node('pos_publisher', anonymous=True)
    pos_pub = PosPublisher()





