# Source this INSIDE the kairos container to activate the kairos run environment.
#   source /ws/src/kairos/docker/kenv.sh
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
# kairos_python runs from the source tree; the compiled _kairos_bindings.so lives
# alongside it (see setup_runtime.sh).
export PYTHONPATH=/ws/src/kairos/python/src:${PYTHONPATH:-}
