# abb_irb140
ROS implementation (URDF and setups) for simulation of 6DOF robot IRB140 from ABB using ros effort_controllers for each joint.

**Author:** Sebastian Realpe Rua, mainly for use in the robotics lab of Universidad Nacional de Colombia

**With the help from:**

https://github.com/FreddyMartinez/abb_irb140_support as a good starting point, helped to reduce initial setup work.

# Installation
Run the following commands, currently only for ros-kinetic in ubuntu 16.04

```
cd ~/catkin_ws/src/
git clone https://github.com/GitSRealpe/abb_irb140.git
cd ..
catkin_make
```

# Running
There are two options or environments to simulate the robot, RViz and Gazebo
## RViz
Simply run `roslaunch abb_irb140 irb140_rviz.launch`
## Gazebo
Simply run `roslaunch abb_irb140 irb140_gazebo.launch`
