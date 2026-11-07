#include "motionman/records.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>

#include "motionman/source_time.hpp"

namespace motionman::records {
namespace {

// A JSON string as json.dumps writes it, for the ASCII the engine emits (run IDs, COCO class names).
std::string py_string(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    const auto u = static_cast<unsigned char>(c);
    if (u < 0x20 || u > 0x7e) throw RecordError("record strings must be printable ASCII");
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string box_json(const std::array<double, 4>& b) {
  return "[" + py_round(b[0], 2) + ", " + py_round(b[1], 2) + ", " + py_round(b[2], 2) + ", " + py_round(b[3], 2) +
         "]";
}

// torch's clamp_(0, limit) on a float32 coordinate.
float clamp_f32(float v, float limit) { return std::min(std::max(v, 0.0F), limit); }

}  // namespace

std::string py_round(double v, int digits) {
  if (!std::isfinite(v)) throw RecordError("cannot write a non-finite number");
  if (std::fabs(v) >= 1e15) throw RecordError("number out of range for the record writer");
  // printf rounds the exact binary value; in the default mode ties go to even, as Python's round() does.
  const int saved = std::fegetround();
  std::fesetround(FE_TONEAREST);
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, v);
  std::fesetround(saved);
  std::string s = buf;
  // Python prints the shortest form of the rounded value: strip trailing zeros but keep one decimal.
  if (s.find('.') != std::string::npos) {
    while (s.back() == '0') s.pop_back();
    if (s.back() == '.') s += '0';
  } else {
    s += ".0";
  }
  return s;
}

std::optional<std::array<double, 4>> checked_box(const std::array<double, 4>& box, int width, int height) {
  auto [x1, y1, x2, y2] = box;
  if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) return std::nullopt;
  const double tol = kOverrunTolerancePx;
  const auto w = static_cast<double>(width);
  const auto h = static_cast<double>(height);
  if (x1 < -tol || y1 < -tol || x2 > w + tol || y2 > h + tol) return std::nullopt;
  // max(x1, 0.0) and min(x2, w) keep the first argument on ties, as Python's do (so -0.0 stays -0.0).
  x1 = std::max(x1, 0.0);
  y1 = std::max(y1, 0.0);
  x2 = std::min(x2, w);
  y2 = std::min(y2, h);
  if (x2 <= x1 || y2 <= y1) return std::nullopt;
  return std::array<double, 4>{x1, y1, x2, y2};
}

std::string frame_record_line(std::string_view run_id, std::int64_t frame_index, double fps, int width, int height,
                              std::span<const RecordObject> objects) {
  const std::string index = std::to_string(frame_index);
  std::string objs;
  for (std::size_t n = 0; n < objects.size(); ++n) {
    const RecordObject& o = objects[n];
    const auto box = checked_box(o.bbox_xyxy, width, height);
    if (!box) throw RecordError("bbox_xyxy lies outside the source image or has no positive extent");
    objs += (n ? ", " : "") + std::string("{\"observation_id\": \"") + index + ":" + std::to_string(n) +
            "\", \"track_id\": " + (o.track_id ? std::to_string(*o.track_id) : "null") +
            ", \"class_id\": " + std::to_string(o.class_id) + ", \"class_name\": " + py_string(o.class_name) +
            ", \"confidence\": " + py_round(o.confidence, 4) + ", \"bbox_xyxy\": " + box_json(*box) +
            ", \"mask_ref\": null}";
  }
  return "{\"schema_version\": \"" + std::string(kSchemaVersion) + "\", \"run_id\": " + py_string(run_id) +
         ", \"source_frame_index\": " + index + ", \"source_t_us\": " + std::to_string(source_t_us(frame_index, fps)) +
         ", \"sample_index\": " + index + ", \"frame_status\": \"INFERRED\", \"source_width\": " +
         std::to_string(width) + ", \"source_height\": " + std::to_string(height) + ", \"objects\": [" + objs + "]}";
}

std::string timing_line(std::int64_t frame_index, std::span<const std::pair<std::string, double>> stages_ms) {
  std::string out = "{\"source_frame_index\": " + std::to_string(frame_index);
  for (const auto& [name, ms] : stages_ms) out += ", " + py_string(name) + ": " + py_round(ms, 3);
  return out + "}";
}

FrameObjects frame_objects(std::span<const track::TrackOutput> outputs, bool has_unconfirmed,
                           std::span<const track::Detection> detections, float conf, int width, int height,
                           const std::vector<std::string>& class_names) {
  const auto w = static_cast<float>(width);
  const auto h = static_cast<float>(height);
  FrameObjects out;
  auto add = [&](std::optional<std::int64_t> id, int cls, float score, float x1, float y1, float x2, float y2) {
    x1 = clamp_f32(x1, w);
    y1 = clamp_f32(y1, h);
    x2 = clamp_f32(x2, w);
    y2 = clamp_f32(y2, h);
    if (!(x2 > x1 && y2 > y1)) {  // nothing left inside the image: the Python writer would refuse it
      ++out.dropped_no_extent;
      return;
    }
    if (cls < 0 || static_cast<std::size_t>(cls) >= class_names.size()) {
      throw RecordError("class id " + std::to_string(cls) + " is outside the model's class list");
    }
    out.objects.push_back({id, cls, class_names[cls], score, {x1, y1, x2, y2}});
  };
  if (!outputs.empty()) {
    for (const auto& o : outputs) add(o.track_id, o.class_id, o.score, o.xyxy[0], o.xyxy[1], o.xyxy[2], o.xyxy[3]);
  } else if (!has_unconfirmed) {
    for (const auto& d : detections) {
      if (d.score > conf) add(std::nullopt, d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);  // strict, as Ultralytics
    }
  }
  return out;
}

void Trails::update(std::int64_t frame_index, std::span<const RecordObject> objects) {
  for (const auto& o : objects) {
    if (!o.track_id) continue;  // no ID, no trail (spec §6, INV-2)
    const auto& b = o.bbox_xyxy;
    auto& pts = points_[*o.track_id];
    // Python's round() on the centre: nearest integer, ties to even.
    pts.emplace_back(static_cast<int>(std::nearbyint((b[0] + b[2]) / 2)),
                     static_cast<int>(std::nearbyint((b[1] + b[3]) / 2)));
    while (pts.size() > max_points_) pts.pop_front();
    last_seen_[*o.track_id] = frame_index;
  }
  for (auto it = last_seen_.begin(); it != last_seen_.end();) {
    if (frame_index - it->second >= max_age_) {
      points_.erase(it->first);
      it = last_seen_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<std::pair<int, int>> Trails::points(std::int64_t track_id) const {
  const auto it = points_.find(track_id);
  if (it == points_.end()) return {};
  return {it->second.begin(), it->second.end()};
}

}  // namespace motionman::records
