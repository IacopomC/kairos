/* -----------------------------------------------------------------------------
 * Offline bag-of-words producer for agent-level loop closure.
 *
 * Without ROS there is no visual odometry front end to supply bag-of-words
 * vectors, so agent-level loop closure gets nothing. BowProducer computes them
 * from an RGB image (ORB features + a DBoW3 vocabulary) and pushes a BowQuery
 * onto Hydra's bow_queue, where the frontend attaches it to the agent node.
 * -------------------------------------------------------------------------- */
#pragma once

#include <pybind11/pybind11.h>

#include <DBoW3.h>
#include <opencv2/features2d.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace kairos::python {

class BowProducer {
 public:
  BowProducer(const std::string& vocab_path,
              int n_features = 1000,
              bool color_is_rgb = true);

  // Run ORB on `rgb`, transform to a BowVector via the loaded vocabulary,
  // and push a BowQuery(robot_id, pose_id, ...) onto bow_queue. Throws if
  // bow_queue is not allocated (frontend.lcd_use_bow_vectors must be true).
  void compute_and_push(const pybind11::buffer& rgb,
                        uint32_t robot_id,
                        uint32_t pose_id);

 private:
  cv::Ptr<cv::ORB> orb_;
  DBoW3::Vocabulary voc_;
  bool color_is_rgb_;
};

//! Push a precomputed bag-of-words vector onto bow_queue
void pushBowQuery(uint32_t robot_id,
                  uint32_t pose_id,
                  const std::vector<uint32_t>& word_ids,
                  const std::vector<float>& word_values);

namespace bow_producer {
void addBindings(pybind11::module_& m);
}  // namespace bow_producer

}  // namespace kairos::python
