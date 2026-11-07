// Port of Pillow's Resample.c BILINEAR path for 8-bit images (see preprocess.hpp). Names follow Pillow so the two
// can be compared line by line: precompute_coeffs, normalize_coeffs_8bpc, the horizontal and vertical passes, clip8.
#include "motionman/preprocess.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace motionman {
namespace {

constexpr int kPrecisionBits = 32 - 8 - 2;

double bilinear_filter(double x) {
  if (x < 0.0) x = -x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

struct Coeffs {
  int ksize = 0;
  std::vector<int> bounds;      // per output pixel: first input index, count
  std::vector<std::int32_t> k;  // per output pixel: ksize fixed-point weights
};

Coeffs precompute_coeffs(int in_size, int out_size) {
  constexpr double support_base = 1.0;  // BILINEAR filter support
  const float in0 = 0.0f;
  const float in1 = static_cast<float>(in_size);
  const double scale = static_cast<double>(in1 - in0) / out_size;
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = support_base * filterscale;
  Coeffs c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.bounds.resize(static_cast<std::size_t>(out_size) * 2);
  c.k.assign(static_cast<std::size_t>(out_size) * c.ksize, 0);
  std::vector<double> kk(static_cast<std::size_t>(c.ksize));
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);  // C truncation, as in Pillow
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double w = bilinear_filter((x + xmin - center + 0.5) * ss);
      kk[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) kk[x] /= ww;
    }
    for (int x = 0; x < xmax; ++x) {  // normalize_coeffs_8bpc
      const double v = kk[x] * (1 << kPrecisionBits);
      c.k[static_cast<std::size_t>(xx) * c.ksize + x] = static_cast<std::int32_t>(v < 0 ? -0.5 + v : 0.5 + v);
    }
    c.bounds[xx * 2] = xmin;
    c.bounds[xx * 2 + 1] = xmax;
  }
  return c;
}

std::uint8_t clip8(std::int32_t in) {
  if (in >= (1 << kPrecisionBits << 8)) return 255;
  if (in <= 0) return 0;
  return static_cast<std::uint8_t>(in >> kPrecisionBits);
}

}  // namespace

std::vector<std::uint8_t> resize_bilinear_pillow(const Bgr8View& src, int out_w, int out_h) {
  if (!src.data || src.width <= 0 || src.height <= 0 || out_w <= 0 || out_h <= 0 ||
      src.stride < static_cast<std::size_t>(src.width) * 3) {
    throw std::invalid_argument("resize_bilinear_pillow: bad image or size");
  }
  const int in_w = src.width;
  const int in_h = src.height;
  // Pillow returns a copy when the size is unchanged, and skips a pass whose dimension is unchanged.
  std::vector<std::uint8_t> cur(static_cast<std::size_t>(in_w) * in_h * 3);
  for (int y = 0; y < in_h; ++y) {
    std::memcpy(&cur[static_cast<std::size_t>(y) * in_w * 3], src.data + y * src.stride, static_cast<std::size_t>(in_w) * 3);
  }
  int cur_w = in_w;
  int cur_h = in_h;
  // Pillow 13.0.0 ImagingResampleInner: "If height is being scaled down more than twice the amount the width is,
  // run the vertical pass first". The order changes the intermediate uint8 rounding, so it must match.
  const bool horizontal_first = !((in_h - out_h) > 0 && (in_h - out_h) > (in_w - out_w) * 2);
  const auto horizontal = [&] {
    if (out_w == cur_w) return;
    const Coeffs c = precompute_coeffs(cur_w, out_w);
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(out_w) * cur_h * 3);
    for (int y = 0; y < cur_h; ++y) {
      const std::uint8_t* row = &cur[static_cast<std::size_t>(y) * cur_w * 3];
      for (int xx = 0; xx < out_w; ++xx) {
        const int xmin = c.bounds[xx * 2];
        const int xmax = c.bounds[xx * 2 + 1];
        const std::int32_t* k = &c.k[static_cast<std::size_t>(xx) * c.ksize];
        std::int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
        for (int x = 0; x < xmax; ++x) {
          const std::uint8_t* p = row + (x + xmin) * 3;
          s0 += p[0] * k[x];
          s1 += p[1] * k[x];
          s2 += p[2] * k[x];
        }
        std::uint8_t* o = &tmp[(static_cast<std::size_t>(y) * out_w + xx) * 3];
        o[0] = clip8(s0);
        o[1] = clip8(s1);
        o[2] = clip8(s2);
      }
    }
    cur.swap(tmp);
    cur_w = out_w;
  };
  const auto vertical = [&] {
    if (out_h == cur_h) return;
    const Coeffs c = precompute_coeffs(cur_h, out_h);
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(cur_w) * out_h * 3);
    for (int yy = 0; yy < out_h; ++yy) {
      const int ymin = c.bounds[yy * 2];
      const int ymax = c.bounds[yy * 2 + 1];
      const std::int32_t* k = &c.k[static_cast<std::size_t>(yy) * c.ksize];
      for (int x = 0; x < cur_w * 3; ++x) {
        std::int32_t s = 1 << (kPrecisionBits - 1);
        for (int y = 0; y < ymax; ++y) s += cur[static_cast<std::size_t>(y + ymin) * cur_w * 3 + x] * k[y];
        tmp[static_cast<std::size_t>(yy) * cur_w * 3 + x] = clip8(s);
      }
    }
    cur.swap(tmp);
    cur_h = out_h;
  };
  if (horizontal_first) {
    horizontal();
    vertical();
  } else {
    vertical();
    horizontal();
  }
  return cur;
}

void bgr_to_rgb_nchw_unit(const std::uint8_t* bgr, int width, int height, float* dst) {
  const std::size_t plane = static_cast<std::size_t>(width) * height;
  for (std::size_t i = 0; i < plane; ++i) {
    for (int c = 0; c < 3; ++c) dst[c * plane + i] = static_cast<float>(bgr[i * 3 + (2 - c)]) / 255.0f;
  }
}

void preprocess_rtdetr(const Bgr8View& src, int out_w, int out_h, float* dst) {
  const auto resized = resize_bilinear_pillow(src, out_w, out_h);
  bgr_to_rgb_nchw_unit(resized.data(), out_w, out_h, dst);
}

}  // namespace motionman
