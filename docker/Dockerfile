# Kairos, self-contained.
#
# Built from this repository: the other repositories of the workspace are
# fetched from their public remotes (install/kairos.rosinstall) and the
# workspace is compiled, so the resulting image runs `kairos run` with no mounts
# beyond your data and output directories. This is the image to build if you
# want to use Kairos; Dockerfile.dev is the mounted-source variant for working
# on the C++.
#
#   cd kairos
#   docker build -t kairos -f docker/Dockerfile .      # or docker/build_image.sh
#
# Run:
#   docker run --rm -v ~/datasets:/data -v ~/out:/out kairos \
#       kairos run /data/tbd/scene_01 -c tbd -l ade20k_outdoor -o /out/run01 \
#       --no-publish --no-enable-lcd
FROM ros:noetic-perception-focal

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Newer CMake (config_utilities needs >= 3.22; focal ships 3.16) from Kitware,
# build tooling, nlohmann_json (spark_dsg), and GTSAM (borglab PPA, the version
# Hydra targets on Noetic).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates wget gnupg \
    && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
        > /etc/apt/sources.list.d/kitware.list \
    && add-apt-repository -y ppa:borglab/gtsam-release-4.1 \
    && apt-get update && apt-get install -y --no-install-recommends \
        cmake build-essential git \
        python3-pip python3-catkin-tools python3-rosdep python3-vcstool python3-dev \
        libeigen3-dev libgoogle-glog-dev libgflags-dev \
        libboost-all-dev nlohmann-json3-dev libzmq3-dev \
        libgtsam-dev libgtsam-unstable-dev \
        ros-noetic-rviz ros-noetic-dynamic-reconfigure \
        ros-noetic-tf2-eigen ros-noetic-tf2-geometry-msgs \
        ros-noetic-rosbag ros-noetic-interactive-markers \
        ros-noetic-image-transport ros-noetic-cv-bridge ros-noetic-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

# Focal's apt pybind11 is 2.4.x and lacks py::module_ and
# pybind11/stl/filesystem.h, which the bindings use.
RUN pip3 install --no-cache-dir "pybind11>=2.11"

# This repository is the build context; every other repository of the workspace
# is fetched from its public remote at the version install/kairos.rosinstall
# pins.
COPY . /ws/src/kairos
RUN vcs import /ws/src < /ws/src/kairos/install/kairos.rosinstall

# GTSAM here is built with TBB, whose parallel multifrontal elimination races on
# the dense deformation-graph structure many loop closures produce. Serializing
# that one solve costs nothing measurable: it takes a few ms. TBB_NUM_THREADS
# does not constrain GTSAM's arena; only tbb::global_control does.
RUN f=/ws/src/Kimera-RPGO/src/RobustSolver.cpp && \
    if [ -f "$f" ] && ! grep -q global_control "$f"; then \
      sed -i '1i #include <tbb/global_control.h>' "$f" && \
      sed -i 's|^\(void RobustSolver::optimize().*{\)|\1\n  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);|' "$f"; \
    fi

# Release is required; a Debug build is too slow to run a full scene.
# BUILD_TESTING=OFF skips the vendored dependencies' own test suites.
# -j is capped because the parallel C++ build otherwise gets OOM-killed.
RUN source /opt/ros/noetic/setup.bash && cd /ws && \
    catkin init >/dev/null && \
    catkin config --extend /opt/ros/noetic \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      -DCONFIG_UTILS_BUILD_DEMOS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >/dev/null && \
    rosdep update >/dev/null 2>&1 || true && \
    rosdep install --from-paths /ws/src --ignore-src -r -y || true && \
    catkin build --no-status --summarize -j"$(( $(nproc) > 6 ? 6 : $(nproc) ))" hydra kairos

# Python runtime. spark_dsg is installed from the local fork because the PyPI
# spark_dsg[viz] extra pulls open3d >= 0.17, which has no cp38 wheel. numpy is
# pinned to a version matplotlib accepts; it stays ABI-compatible with the
# apt-built libraries.
RUN source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash && \
    pip3 install --no-cache-dir /ws/src/spark-dsg-kairos && \
    pip3 install --no-cache-dir --no-deps \
        imageio networkx scipy tqdm distinctipy click pyqtgraph "numpy==1.24.4" && \
    pip3 install --no-cache-dir --no-deps /ws/src/hydra-kairos

# The Kairos bindings are built directly rather than pip-installed: the package
# declares no metadata and its default relative HYDRA_FLOW_DIR does not survive
# pip's temporary copy.
RUN source /opt/ros/noetic/setup.bash && source /ws/devel/setup.bash && \
    cmake -S /ws/src/kairos/python -B /tmp/kairos_pybuild \
      -DHYDRA_FLOW_DIR=/ws/src/hydra-kairos \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    cmake --build /tmp/kairos_pybuild -j"$(nproc)" && \
    cp /tmp/kairos_pybuild/_kairos_bindings*.so /ws/src/kairos/python/src/kairos_python/ && \
    rm -rf /tmp/kairos_pybuild

# `kairos` resolves without sourcing anything: the entrypoint sources ROS and
# the catkin devel space, which is also what puts libkimera_pgmo.so on the
# library path.
ENV PYTHONPATH=/ws/src/kairos/python/src
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    printf '#!/bin/bash\nexec python3 -m kairos_python "$@"\n' > /usr/local/bin/kairos && \
    chmod +x /usr/local/bin/kairos

WORKDIR /ws
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
