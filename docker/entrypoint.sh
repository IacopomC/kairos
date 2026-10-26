#!/bin/bash
# Sources ROS 2 and the colcon install space, then runs whatever was asked for.
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
exec "$@"
