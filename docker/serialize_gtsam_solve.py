#!/usr/bin/env python3
"""Serialize Kimera-PGMO's pose-graph solve.

GTSAM is built with TBB, and its parallel multifrontal elimination races on the
dense deformation graph that many loop closures produce, crashing inside
OptimizeClique. This limits TBB to one thread for the duration of
KimeraRpgoOptimizer::update(), the ROS 2 counterpart of the patch the ROS 1
build applied to Kimera-RPGO's RobustSolver::optimize(). The solve takes
milliseconds, so the cost is not measurable. TBB_NUM_THREADS=1 does not work;
only tbb::global_control constrains GTSAM's arena.

Kimera-PGMO is upstream MIT-SPARK code, so this patches a checkout in place;
re-run it after a fresh checkout. Running it again is a no-op.

    serialize_gtsam_solve.py <workspace src/>
"""
import pathlib
import sys

SOURCE = "kimera_pgmo/kimera_pgmo/src/optimizer/kimera_rpgo_optimizer.cpp"
SIGNATURE_END = "const std::set<size_t>& temp_known_inliers) {\n"
GUARD = "  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);\n"


def main():
    path = pathlib.Path(sys.argv[1]) / SOURCE
    source = path.read_text()
    if "tbb::global_control" in source:
        print(f"{path}: already serialized")
        return

    if source.count(SIGNATURE_END) != 1:
        sys.exit(f"{path}: KimeraRpgoOptimizer::update signature changed, patch by hand")

    source = "#include <tbb/global_control.h>\n" + source
    source = source.replace(SIGNATURE_END, SIGNATURE_END + GUARD)
    path.write_text(source)
    print(f"{path}: serialized the pose-graph solve")


if __name__ == "__main__":
    main()
