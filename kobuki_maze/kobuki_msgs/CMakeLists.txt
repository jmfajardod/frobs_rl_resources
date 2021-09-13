cmake_minimum_required(VERSION 2.8.3)
project(kobuki_msgs)
find_package(catkin REQUIRED COMPONENTS std_msgs actionlib_msgs message_generation)

add_message_files(DIRECTORY msg
                      FILES BumperEvent.msg
                            CliffEvent.msg
                            DigitalOutput.msg
                            ExternalPower.msg
                            Led.msg
                            PowerSystemEvent.msg
                            SensorState.msg
                            VersionInfo.msg
                            ControllerInfo.msg
                            ButtonEvent.msg
                            DigitalInputEvent.msg
                            DockInfraRed.msg
                            KeyboardInput.msg
                            MotorPower.msg
                            RobotStateEvent.msg
                            Sound.msg
                            ScanAngle.msg
                            WheelDropEvent.msg
                 )

add_action_files(DIRECTORY action
                     FILES AutoDocking.action
                )

generate_messages(DEPENDENCIES std_msgs actionlib_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs actionlib_msgs)
