// RT-DETR TensorRT detector (plan native-g0-g1 step 9, spec C§5, C§7). Opening refuses an engine whose manifest
// identity differs from this binary, this host or the profile; inference runs on one non-default stream.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "motionman/types.hpp"

namespace motionman::gpu {

// Thrown when an engine must not be loaded; what() lists every mismatch.
class IdentityError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct DetectorPaths {
  std::string profile_json;  // configs/native/<profile>.json
  std::string model_dir;     // cache dir with manifest.json, engine_manifest.<precision>.json and the engine
  std::string precision = "fp32";
};

class TrtDetector {
 public:
  static std::unique_ptr<TrtDetector> open(const DetectorPaths& paths);
  ~TrtDetector();
  TrtDetector(const TrtDetector&) = delete;
  TrtDetector& operator=(const TrtDetector&) = delete;

  // One preprocessed image (RGB float NCHW, input_width × input_height) whose source was src_w × src_h. Returns every
  // engine output (300 queries) in source pixels; thresholding is the caller's.
  std::vector<Detection> infer(const float* input_nchw, int src_w, int src_h);

  int input_width() const;
  int input_height() const;
  const std::vector<std::string>& class_names() const;
  const std::string& engine_sha256() const;
  const std::string& gpu_name() const;
  const std::string& driver() const;
  static std::string tensorrt_version();  // the TensorRT this binary was compiled against, e.g. "11.2.1.2"

 private:
  struct Impl;
  explicit TrtDetector(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace motionman::gpu
