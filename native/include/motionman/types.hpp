// Plain values shared by every native stage (spec C§7, C§8). Portable: no CUDA or OpenCV headers.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace motionman {

// Identity of one decoded frame, carried unchanged from decode to record writing.
struct FrameKey {
  std::uint32_t stream_id = 0;
  std::int64_t source_frame_index = 0;  // contiguous from 0 within a stream (record schema 1.0)
  std::int64_t source_t_us = 0;         // source time from index and FPS, never wall clock
};

// One detector output in source-image pixels, before tracking.
struct Detection {
  std::int32_t class_id = 0;
  float confidence = 0.0F;
  std::array<float, 4> bbox_xyxy{};  // x1, y1, x2, y2
};

// One tracked object as written to tracks.jsonl. A null track ID is kept, never invented (INV-2).
struct TrackedObject {
  Detection detection;
  std::optional<std::int64_t> track_id;
  std::string class_name;
};

// A box is valid when it is finite, has positive size and lies inside a width x height frame.
bool valid_box(const std::array<float, 4>& b, std::int32_t width, std::int32_t height);

}  // namespace motionman

#include "motionman/detail/types_inl.hpp"
