// Native writers for tracks.jsonl and timings.jsonl, and the trail buffer (plan native-g2 step 4, spec C§8, A§4).
// Lines are byte-identical to what the Python pipeline writes (records.frame_record + json.dumps; the timings row
// shape), checked against tests/fixtures/native_records.json, which Python itself generates.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "motionman/track/bytetrack.hpp"

namespace motionman::records {

inline constexpr const char* kSchemaVersion = "1.0";
inline constexpr double kOverrunTolerancePx = 0.5;

struct RecordError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct RecordObject {
  std::optional<std::int64_t> track_id;  // nullopt is recorded as null (INV-2)
  int class_id = 0;
  std::string class_name;  // printable ASCII only (COCO names); anything else is refused
  double confidence = 0;
  std::array<double, 4> bbox_xyxy{};
};

// repr(round(v, digits)) as Python prints it: correctly rounded, ties to even, shortest form ("12.0", "0.12").
std::string py_round(double v, int digits);

// Python's _checked_box: finite, at most 0.5 px outside the image (then clipped), positive extent.
std::optional<std::array<double, 4>> checked_box(const std::array<double, 4>& box, int width, int height);

// One tracks.jsonl line (no newline). Throws RecordError for a box the Python writer would refuse.
std::string frame_record_line(std::string_view run_id, std::int64_t frame_index, double fps, int width, int height,
                              std::span<const RecordObject> objects);

// One timings.jsonl line: {"source_frame_index": n, "<stage>_ms": round(v, 3), ...} in the given order.
std::string timing_line(std::int64_t frame_index, std::span<const std::pair<std::string, double>> stages_ms);

struct FrameObjects {
  std::vector<RecordObject> objects;
  int dropped_no_extent = 0;  // tracker boxes that clip to nothing inside the image (counted, never written)
};

// What a frame records (ADR-006, as Ultralytics 8.4.196 track mode):
//   - tracker output non-empty: its tracks in order, boxes clamped to the image in float32, with IDs;
//   - empty and an unconfirmed track pending: nothing;
//   - empty otherwise: the admitted detections with score >= conf, in detection order, with null IDs.
FrameObjects frame_objects(std::span<const track::TrackOutput> outputs, bool has_unconfirmed,
                           std::span<const track::Detection> detections, float conf, int width, int height,
                           const std::vector<std::string>& class_names);

// render.Trails: up to max_points centres per ID, dropped after max_age frames unseen (INV-4). Null IDs never
// get a trail.
class Trails {
 public:
  explicit Trails(std::size_t max_points = 30, int max_age = 60) : max_points_(max_points), max_age_(max_age) {}
  void update(std::int64_t frame_index, std::span<const RecordObject> objects);
  std::vector<std::pair<int, int>> points(std::int64_t track_id) const;
  std::size_t size() const { return points_.size(); }

 private:
  std::size_t max_points_;
  int max_age_;
  std::map<std::int64_t, std::deque<std::pair<int, int>>> points_;
  std::map<std::int64_t, std::int64_t> last_seen_;
};

}  // namespace motionman::records
