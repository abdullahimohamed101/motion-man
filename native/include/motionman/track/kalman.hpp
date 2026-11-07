// Constant-velocity Kalman filter on (cx, cy, aspect, height) for ByteTrack (plan native-g2 step 3).
// Same model and noise weights as the reference's KalmanFilterXYAH (position 1/20, velocity 1/160 of height),
// computed in double. A 4x4 solve uses partial pivoting like LAPACK's gesv; results agree with numpy to
// rounding, which the replay tolerance (1e-3 px) covers.
#pragma once

#include <array>

namespace motionman::track {

using Vec4 = std::array<double, 4>;
using Vec8 = std::array<double, 8>;
using Mat8 = std::array<std::array<double, 8>, 8>;

struct KalmanState {
  Vec8 mean{};
  Mat8 covariance{};
};

KalmanState kalman_initiate(const Vec4& xyah);
// Predict one frame. A track that is not in the Tracked state has its height velocity zeroed first.
void kalman_predict(KalmanState& s, bool tracked);
void kalman_update(KalmanState& s, const Vec4& xyah);

}  // namespace motionman::track
