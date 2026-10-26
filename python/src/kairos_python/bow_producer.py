"""Offline DBoW producer for the Python pipeline.

The ROS pipeline gets BoW input from Kimera-VIO via the bow_subscriber. The
offline (Python) pipeline has no such producer, which leaves agent-layer LCD
empty — every detect call returns 0 valid matches because the agent
descriptors have no dbow_ids/dbow_values populated.

This module produces those messages from the dataset's RGB stream by driving
a C++-side BowProducer (wraps cv::ORB + DBoW3) bound through the
_kairos_bindings module:

  RGB  ->  BowProducer.compute_and_push(rgb, robot_id, pose_id)
                              |
                              v
        cv::ORB  ->  DBoW3::Vocabulary.transform  ->  PipelineQueues.bow_queue

The pose_id stamped on each BowQuery must match the agent NodeSymbol key set
by the C++ frontend at graph_builder.cpp:674. That key comes from
PoseGraphFromOdom::update (pose_graph_from_odom.cpp:95-130), which counts only
keyframes that pass min_pose_separation / min_time_separation_s. Since the
Python data_loader has no view into that count, we mirror the tracker's
acceptance logic in `KeyframeTracker` so the pose_id we emit matches.
"""

import logging
import pathlib

import numpy as np

from hydra_python.data_callbacks import register_data_callback


class KeyframeTracker:
    """Mirrors hydra::PoseGraphFromOdom::update.

    Returns the agent keys that the C++ tracker will assign for the input
    frame, paired with the color image we want BoW'd for that key. A frame
    that the tracker would drop returns an empty list (and does NOT advance
    prev_pose_, matching the C++ early-return).
    """

    def __init__(
        self,
        min_pose_separation=0.5,
        rotation_separation_weight=0.1,
        min_time_separation_s=0.2,
    ):
        self.min_pose_separation = float(min_pose_separation)
        self.rotation_separation_weight = float(rotation_separation_weight)
        self.min_time_separation_s = float(min_time_separation_s)
        self._num_received = 0
        self._prev_stamp_ns = None
        self._prev_world_T_body = None
        self._prev_color = None

    def update(self, timestamp_ns, world_T_body, color):
        """Return list of (pose_id, color) for any new agent keys.

        On the first accepted frame the tracker emits two nodes (the cached
        previous pose at key 0 and the current at key 1) — addNode is called
        twice in pose_graph_from_odom.cpp:84-85 and only the second is "new"
        for subsequent frames thanks to the
        `if (node.key < agents.numNodes()) continue;` skip in
        graph_builder.cpp:664. We push BoW for both so the first agent isn't
        permanently descriptor-less.
        """
        emissions = []

        if self._num_received > 0:
            diff_s = (timestamp_ns - self._prev_stamp_ns) * 1e-9
            if (
                self.min_time_separation_s
                and diff_s < self.min_time_separation_s
            ):
                return emissions

            pose_diff = np.linalg.inv(world_T_body) @ self._prev_world_T_body
            translation_diff = np.linalg.norm(pose_diff[:3, 3])
            rotation_diff = np.linalg.norm(pose_diff[:3, :3])  # Frobenius
            total_diff = (
                translation_diff
                + self.rotation_separation_weight * rotation_diff
            )
            if (
                self.min_pose_separation
                and total_diff < self.min_pose_separation
            ):
                return emissions

            new_key = self._num_received
            if self._num_received == 1:
                emissions.append((0, self._prev_color))
            emissions.append((new_key, color))

        self._prev_stamp_ns = timestamp_ns
        self._prev_world_T_body = world_T_body
        self._prev_color = color
        self._num_received += 1
        return emissions


@register_data_callback("bow_producer")
class BowProducerCallback:
    """data_callback that runs the offline DBoW producer per frame.

    Hooked into DataLoader.run before pipeline.step, so the BoW for frame N
    is queued before the C++ tracker registers agent N. assignBowVectors is
    called at the end of updatePoseGraph in the same step and matches by
    pose_id.
    """

    def __init__(
        self,
        vocab_path,
        robot_id=0,
        n_features=1000,
        min_pose_separation=0.5,
        rotation_separation_weight=0.1,
        min_time_separation_s=0.2,
        color_is_rgb=True,
    ):
        # Local import keeps the tracker importable without the C++ bindings
        # (e.g. during tracker-only unit tests).
        from kairos_python._kairos_bindings import BowProducer

        vocab_path = pathlib.Path(vocab_path).expanduser().absolute()
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found at {vocab_path}. Run "
                "scripts/fetch_orb_vocab.sh to download it."
            )

        self._tracker = KeyframeTracker(
            min_pose_separation=min_pose_separation,
            rotation_separation_weight=rotation_separation_weight,
            min_time_separation_s=min_time_separation_s,
        )
        self._producer = BowProducer(
            str(vocab_path),
            n_features=int(n_features),
            color_is_rgb=bool(color_is_rgb),
        )
        self._robot_id = int(robot_id)
        self._n_emitted = 0

    def __call__(self, packet):
        emissions = self._tracker.update(
            int(packet.timestamp), packet.world_T_body, packet.color
        )
        if not emissions:
            return
        for pose_id, color in emissions:
            self._producer.compute_and_push(
                np.ascontiguousarray(color), self._robot_id, int(pose_id)
            )
            self._n_emitted += 1
            if self._n_emitted <= 5 or self._n_emitted % 50 == 0:
                logging.info(
                    "[bow_producer] pushed BoW for pose_id=%d (total=%d)",
                    pose_id,
                    self._n_emitted,
                )
