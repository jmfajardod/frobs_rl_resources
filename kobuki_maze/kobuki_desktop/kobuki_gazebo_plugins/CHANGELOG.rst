=========
Changelog
=========

0.5.6 (2017-04-01)
------------------
* update at the end of the event loop so the bumper sensor can be caught

0.5.3 (2017-02-23)
------------------
* add_dependencies with catkin_EXPORTED_TARGETS, not xyz_gencpp targets which might not be there

0.5.1 (2016-09-20)
------------------
* bugfix for ros logging.

0.4.2 (2015-03-02)
------------------
* kobuki_gazebo_plugins: Resolve IMU sensor using fully scoped name
  This ensures uniqueness among multiple models in which the sensor has the
  same name. In this case, our specific motivation is the Kobuki Gazebo
  plugin being spawned multiple times using kobuki_gazebo.urdf.xacro in the
  package kobuki_description.
* Contributors: Scott Livingston

0.4.1 (2014-09-19)
------------------
* kobuki_gazebo_plugins: makes bump detection more reliable
* remove duplicated spinonce
* renamed updater to updates and loader to loads. reorder headers
* tf_prefix added. base_prefix added
* Contributors: Jihoon Lee, Marcus Liebhardt

0.4.0 (2014-08-11)
------------------
* cherry-picking `#30 <https://github.com/yujinrobot/kobuki_desktop/issues/30>`_
* trivial update.
* removes email addresses from authors
* replace deprecated shared_dynamic_cast (fixes `#25 <https://github.com/yujinrobot/kobuki_desktop/issues/25>`_)
* Contributors: Daniel Stonier, Marcus Liebhardt, Nikolaus Demmel, Samir Benmendil

0.3.1 (2013-10-14)
------------------
* fixes gazebo header paths (refs `#22 <https://github.com/yujinrobot/kobuki_desktop/issues/22>`_)

0.3.0 (2013-08-30)
------------------
* fixes bumper & cliff event publishing
* adds reset odometry and fixes imu and odom data processing
* adds IMU data processing
* adds bugtracker and repo info to package.xml

0.2.0 (2013-07-11)
------------------
* ROS Hydro beta release.
* Adds catkinized kobuki_qtestsuite
