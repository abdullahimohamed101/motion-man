// Source time for constant-frame-rate input (spec C§8, A§4). Must equal Python's records.source_t_us bit for bit:
// round(index * 1e6 / fps) on an IEEE double, ties to even. tests/fixtures/source_time.json pins the contract.
#pragma once

#include <cfenv>
#include <cmath>
#include <cstdint>

namespace motionman {

inline std::int64_t source_t_us(std::int64_t frame_index, double fps) {
  // Python computes int * int exactly, then converts to double once for the division; do the same.
  const double us = static_cast<double>(frame_index * 1'000'000) / fps;
  // nearbyint rounds in the current mode; FE_TONEAREST is ties-to-even, matching Python's round().
  const int saved = std::fegetround();
  std::fesetround(FE_TONEAREST);
  const double r = std::nearbyint(us);
  std::fesetround(saved);
  return static_cast<std::int64_t>(r);
}

}  // namespace motionman
