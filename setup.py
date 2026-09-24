"""Setup script for the kairos python package (builds _kairos_bindings).

Mirrors hydra_python's CMake-extension build. Build the catkin workspace first
(so libhydra/libkairos and their CMake configs are in devel/) and source it,
then `pip install .` from this directory.
"""

import multiprocessing
import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", extra_cmake_flags=None):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.extra_cmake_flags = list(extra_cmake_flags or [])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "RelWithDebInfo"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # Tolerate third-party deps that declare an ancient cmake_minimum.
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ]
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [x for x in os.environ["CMAKE_ARGS"].split(" ") if x]
        cmake_args += ext.extra_cmake_flags

        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    ext_modules=[CMakeExtension("kairos_python._kairos_bindings", sourcedir="python")],
    cmdclass={"build_ext": CMakeBuild},
)
