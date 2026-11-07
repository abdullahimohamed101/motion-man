// RT-DETR preprocessing (spec C§5, C§6; contract in configs/native/rtdetr_r18_640.json "preprocessing").
// The upstream validation transform is torchvision Resize on a PIL image, i.e. Pillow's Image.resize(BILINEAR):
// an antialiased, separable triangle filter in 22-bit fixed point, horizontal pass first, rounding to uint8 after
// each pass. This is a portable port of that algorithm (Pillow src/libImaging/Resample.c), so the C++ engine sees
// the same pixels as the PyTorch reference. No OpenCV: callers pass raw interleaved 8-bit BGR (OpenCV's layout).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace motionman {

struct Bgr8View {
  const std::uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  std::size_t stride = 0;  // bytes per row
};

// Pillow-exact BILINEAR resize of a 3-channel 8-bit image; returns tightly packed rows (out_w * 3 bytes each).
std::vector<std::uint8_t> resize_bilinear_pillow(const Bgr8View& src, int out_w, int out_h);

// BGR uint8 HWC (tightly packed) → RGB float32 NCHW (batch 1), value / 255.0f, exactly as upstream ConvertPILImage.
void bgr_to_rgb_nchw_unit(const std::uint8_t* bgr, int width, int height, float* dst);

// The full upstream transform: resize to out_w × out_h, then layout and scaling. dst holds 3 * out_w * out_h floats.
void preprocess_rtdetr(const Bgr8View& src, int out_w, int out_h, float* dst);

}  // namespace motionman
