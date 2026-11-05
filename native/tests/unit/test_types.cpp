// Shared native types: defaults and box validity (plan native-g0-g1 step 1).
#include <cmath>
#include <limits>

#include "../check.hpp"
#include "motionman/types.hpp"

using motionman::Detection;
using motionman::FrameKey;
using motionman::TrackedObject;
using motionman::valid_box;

int main() {
  FrameKey key;
  MM_CHECK(key.stream_id == 0 && key.source_frame_index == 0 && key.source_t_us == 0);

  TrackedObject obj;
  MM_CHECK(!obj.track_id.has_value());  // null until the tracker assigns one (INV-2)

  MM_CHECK(valid_box({0, 0, 1920, 1080}, 1920, 1080));          // the whole frame
  MM_CHECK(valid_box({10.5F, 5, 20, 40}, 64, 48));
  MM_CHECK(!valid_box({10, 5, 10, 40}, 64, 48));                // zero width
  MM_CHECK(!valid_box({20, 5, 10, 40}, 64, 48));                // reversed
  MM_CHECK(!valid_box({-1, 5, 10, 40}, 64, 48));                // outside on the left
  MM_CHECK(!valid_box({0, 5, 65, 40}, 64, 48));                 // outside on the right
  MM_CHECK(!valid_box({0, std::nanf(""), 10, 40}, 64, 48));     // non-finite
  MM_CHECK(!valid_box({0, 0, std::numeric_limits<float>::infinity(), 40}, 64, 48));

  Detection d;
  MM_CHECK(d.confidence == 0.0F && d.bbox_xyxy[2] == 0.0F);
  return motionman::test::failures == 0 ? 0 : 1;
}
