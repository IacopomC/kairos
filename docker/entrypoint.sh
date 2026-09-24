#!/bin/bash
# Sources ROS and the catkin devel space, then runs whatever was asked for.
# catkin's setup.bash inspects "$@" when sourced, so the arguments are cleared
# first and restored afterwards; otherwise a leading --help or -c breaks it and
# LD_LIBRARY_PATH is never set (libkimera_pgmo.so then fails to load).
args=("$@")
set --
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
set -- "${args[@]}"
exec "$@"
