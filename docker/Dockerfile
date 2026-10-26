# Kairos on ROS 2 Jazzy, self-contained.
#
# Built from this repository: the other repositories of the workspace are
# fetched from their public remotes (kairos.repos) and the workspace is
# compiled, so the resulting image runs `kairos run` with no mounts beyond your
# data and output directories. This is the image to build if you want to use
# Kairos; Dockerfile.dev is the mounted-source variant for working on the C++.
#
#   cd kairos
#   docker build -t kairos -f docker/Dockerfile .      # or docker/build_image.sh
#
# Run:
#   docker run --rm -v ~/datasets:/data -v ~/out:/out kairos \
#       kairos run /data/tbd/scene_01 -c tbd -l ade20k_outdoor -o /out/run01 \
#       --no-publish --no-enable-lcd
ARG BASE_IMAGE=osrf/ros:jazzy-desktop
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Build tooling and Hydra's build dependencies that rosdep does not resolve.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-venv build-essential git ros-dev-tools \
        nlohmann-json3-dev libgoogle-glog-dev libcli11-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# This repository is the build context; every other repository of the workspace
# is fetched from its public remote at the version kairos.repos pins.
COPY . /ws/src/kairos
RUN vcs import /ws/src < /ws/src/kairos/kairos.repos

RUN apt-get update && rosdep update && \
    rosdep install --from-paths /ws/src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# GTSAM is built with TBB, and its parallel solve races on the dense
# deformation graphs Kairos produces; serialize it (see the script).
RUN python3 /ws/src/kairos/docker/serialize_gtsam_solve.py /ws/src

# Parallelism is capped so the build fits in about 16 GB of RAM.
RUN source /opt/ros/jazzy/setup.bash && cd /ws && \
    MAKEFLAGS="-j4" colcon build --parallel-workers 3 --packages-up-to kairos hydra_ros \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DCONFIG_UTILS_ENABLE_ROS=OFF --no-warn-unused-cli && \
    rm -rf /ws/build /ws/log

# Python: Spark-DSG and Hydra's bindings, then Kairos's. Dependencies are listed
# explicitly and numpy is held at the system version, which pip cannot replace.
# hydra_python imports spark_dataset_interfaces at import time, so it is
# installed too, although Kairos reads scenes with its own loader.
RUN source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && \
    printf 'numpy==%s\n' "$(python3 -c 'import numpy; print(numpy.__version__)')" \
      > /tmp/constraints.txt && \
    pip3 install --break-system-packages --no-cache-dir -c /tmp/constraints.txt /ws/src/spark_dsg && \
    pip3 install --break-system-packages --no-cache-dir -c /tmp/constraints.txt \
        click distinctipy imageio networkx pandas pyqtgraph pyyaml rosbags scipy \
        sortedcontainers tqdm && \
    pip3 install --break-system-packages --no-cache-dir --no-deps \
        "spark_dataset_interfaces @ git+https://github.com/MIT-SPARK/spark_dataset_interfaces" && \
    pip3 install --break-system-packages --no-cache-dir --no-deps /ws/src/hydra && \
    CMAKE_BUILD_PARALLEL_LEVEL=4 \
      pip3 install --break-system-packages --no-cache-dir --no-deps /ws/src/kairos

# The Rerun renderer, in its own environment: Rerun needs numpy 2, the system
# Python keeps numpy 1. `kairos-visualize` runs it.
RUN python3 -m venv /opt/kairos_viz && \
    /opt/kairos_viz/bin/pip install --no-cache-dir "rerun-sdk==0.34.1" \
      /ws/src/kairos/kairos_visualizer && \
    ln -s /opt/kairos_viz/bin/kairos-visualize /usr/local/bin/kairos-visualize

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    printf '#!/bin/bash\nexec python3 -m kairos_python "$@"\n' > /usr/local/bin/kairos && \
    chmod +x /usr/local/bin/kairos

WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
