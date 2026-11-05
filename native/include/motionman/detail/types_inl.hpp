// Inline definitions for motionman/types.hpp.
#pragma once

#include <cmath>

namespace motionman {

inline bool valid_box(const std::array<float, 4>& b, std::int32_t width, std::int32_t height) {
  for (float v : b) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  const auto w = static_cast<float>(width);
  const auto h = static_cast<float>(height);
  return 0.0F <= b[0] && b[0] < b[2] && b[2] <= w && 0.0F <= b[1] && b[1] < b[3] && b[3] <= h;
}

}  // namespace motionman
