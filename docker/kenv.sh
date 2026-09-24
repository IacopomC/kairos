# Source this INSIDE the kairos container to activate the kairos run environment.
#   source /ws/src/kairos/docker/kenv.sh
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
# kairos_python is exposed via PYTHONPATH (its setup.py ships no package metadata);
# the compiled _kairos_bindings.so lives alongside it (see setup_runtime.sh step 5).
export PYTHONPATH=/ws/src/kairos/python/src:${PYTHONPATH:-}
# `kairos` as a command, as in the self-contained image.
kairos() { python3 -m kairos_python "$@"; }
export -f kairos
