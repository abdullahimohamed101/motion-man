// Native ByteTrack (plan native-g2 step 3, spec C§8, ADR-006).
// The algorithm is ByteTrack (Zhang et al., MIT; native/third_party/NOTICE.md). Its behaviour matches the Python
// reference, Ultralytics 8.4.196 BYTETracker with bytetrack.yaml, which the replay fixtures in
// tests/fixtures/tracker/ check. No Ultralytics code is copied. Reference behaviour this port keeps:
//   - box conversion, IoU and score fusion in float32, the Kalman filter in double;
//   - tracks started after the first frame stay unconfirmed until matched on the next frame;
//   - unconfirmed tracks are not predicted; a lost track's height velocity is zeroed before predicting;
//   - a track removed for staleness stays in the lost pool for one more frame;
//   - IDs come from this tracker's own allocator, starting at 1 (no shared counter, INV-1, ADR-004).
// One instance owns one stream's state; frames must be fed in order (INV-1).
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <vector>

#include "motionman/track/kalman.hpp"

namespace motionman::track {

// agnostic: as the reference (any detection may update any track). aware: a detection only matches a track of
// its own class, and duplicate suppression only compares tracks of one class.
enum class ClassPolicy { Agnostic, Aware };

struct Config {
  float track_high_thresh = 0.25F;
  float track_low_thresh = 0.1F;
  float new_track_thresh = 0.25F;
  int track_buffer = 30;
  float match_thresh = 0.8F;
  bool fuse_score = true;
  ClassPolicy class_policy = ClassPolicy::Aware;
};

struct Detection {
  float x1, y1, x2, y2, score;
  int class_id;
};

// One tracked object this frame: its filtered box, its ID, and the detection that updated it.
struct TrackOutput {
  std::array<float, 4> xyxy;
  std::int64_t track_id;
  float score;
  int class_id;
  int detection_index;
};

class ByteTracker {
 public:
  explicit ByteTracker(Config config);
  ~ByteTracker();
  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;
  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;

  // Feed one frame's detections (every candidate; the tracker splits them by score) and get the confirmed tracks.
  std::vector<TrackOutput> update(std::span<const Detection> detections);
  // True when an unconfirmed track is pending. The legacy backend records no objects for a frame with no output
  // while this holds (ADR-006).
  bool has_unconfirmed() const;
  const Config& config() const { return config_; }

  struct Track;

 private:
  Config config_;
  int frame_id_ = 0;
  std::int64_t next_id_ = 1;
  std::vector<std::shared_ptr<Track>> tracked_;
  std::vector<std::shared_ptr<Track>> lost_;
  std::deque<std::int64_t> removed_ids_;  // recent removals, at most 1000, as the reference keeps
};

}  // namespace motionman::track
