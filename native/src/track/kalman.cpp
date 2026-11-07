#include "motionman/track/kalman.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace motionman::track {
namespace {

constexpr double kPos = 1.0 / 20;
constexpr double kVel = 1.0 / 160;

double sq(double v) { return v * v; }

}  // namespace

KalmanState kalman_initiate(const Vec4& z) {
  KalmanState s;
  for (int i = 0; i < 4; ++i) s.mean[i] = z[i];
  const double h = z[3];
  const std::array<double, 8> std_dev = {2 * kPos * h, 2 * kPos * h, 0.01, 2 * kPos * h,
                                         10 * kVel * h, 10 * kVel * h, 1e-5, 10 * kVel * h};
  for (int i = 0; i < 8; ++i) s.covariance[i][i] = sq(std_dev[i]);
  return s;
}

void kalman_predict(KalmanState& s, bool tracked) {
  if (!tracked) s.mean[7] = 0;
  const double h = s.mean[3];
  const std::array<double, 8> std_dev = {kPos * h, kPos * h, 0.01, kPos * h, kVel * h, kVel * h, 1e-5, kVel * h};
  // mean = F mean, with F = [I I; 0 I].
  for (int i = 0; i < 4; ++i) s.mean[i] += s.mean[4 + i];
  // P = F P F^T + Q. F P adds row i+4 to row i; (F P) F^T adds column j+4 to column j.
  Mat8& p = s.covariance;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 8; ++j) p[i][j] += p[i + 4][j];
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) p[i][j] += p[i][j + 4];
  }
  for (int i = 0; i < 8; ++i) p[i][i] += sq(std_dev[i]);
}

void kalman_update(KalmanState& s, const Vec4& z) {
  const double h = s.mean[3];
  const std::array<double, 4> std_dev = {kPos * h, kPos * h, 0.1, kPos * h};
  // Innovation covariance S = H P H^T + R (the top-left 4x4 block).
  double sm[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) sm[i][j] = s.covariance[i][j];
    sm[i][i] += sq(std_dev[i]);
  }
  // Kalman gain K (8x4) solves S K^T = (P H^T)^T, i.e. S X = B with B = P[:, :4]^T (4x8): LU with partial pivoting.
  double a[4][4];
  double b[4][8];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) a[i][j] = sm[i][j];
    for (int j = 0; j < 8; ++j) b[i][j] = s.covariance[j][i];
  }
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r) {
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    }
    if (a[piv][c] == 0.0) throw std::runtime_error("kalman_update: singular innovation covariance");
    if (piv != c) {
      std::swap(a[piv], a[c]);
      std::swap(b[piv], b[c]);
    }
    for (int r = c + 1; r < 4; ++r) {
      const double f = a[r][c] / a[c][c];
      for (int k = c; k < 4; ++k) a[r][k] -= f * a[c][k];
      for (int k = 0; k < 8; ++k) b[r][k] -= f * b[c][k];
    }
  }
  double x[4][8];  // x = S^-1 B = K^T
  for (int r = 3; r >= 0; --r) {
    for (int k = 0; k < 8; ++k) {
      double acc = b[r][k];
      for (int c = r + 1; c < 4; ++c) acc -= a[r][c] * x[c][k];
      x[r][k] = acc / a[r][r];
    }
  }
  double innovation[4];
  for (int i = 0; i < 4; ++i) innovation[i] = z[i] - s.mean[i];
  for (int i = 0; i < 8; ++i) {
    double acc = 0;
    for (int k = 0; k < 4; ++k) acc += innovation[k] * x[k][i];
    s.mean[i] += acc;
  }
  // P -= K S K^T
  double ks[8][4];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      double acc = 0;
      for (int k = 0; k < 4; ++k) acc += x[k][i] * sm[k][j];
      ks[i][j] = acc;
    }
  }
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      double acc = 0;
      for (int k = 0; k < 4; ++k) acc += ks[i][k] * x[k][j];
      s.covariance[i][j] -= acc;
    }
  }
}

}  // namespace motionman::track
