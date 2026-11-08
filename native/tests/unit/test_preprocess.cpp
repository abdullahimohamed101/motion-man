// Portable checks of the Pillow-exact resize and the layout conversion (plan native-g0-g1 step 7). The PC's golden
// test compares against the real Python reference; these pin hand-computed cases on any host.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../check.hpp"
#include "motionman/preprocess.hpp"

int main() {
  using motionman::Bgr8View;

  // 4×1 → 2×1: scale 2, support 2; output 0 has weights 3/7, 3/7, 1/7 over inputs 0..2 (Pillow's triangle filter).
  std::vector<std::uint8_t> row = {0, 0, 0, 100, 100, 100, 200, 200, 200, 250, 250, 250};
  const auto down = motionman::resize_bilinear_pillow({row.data(), 4, 1, 12}, 2, 1);
  MM_CHECK(down.size() == 6);
  MM_CHECK(down[0] == 71);   // (0*3 + 100*3 + 200*1) / 7 = 71.4
  MM_CHECK(down[3] == 207);  // (100*1 + 200*3 + 250*3) / 7 = 207.1; Pillow 13 gives 71, 207

  // Unchanged size is a copy, including a padded stride.
  std::vector<std::uint8_t> padded = {1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 99, 99};
  const auto same = motionman::resize_bilinear_pillow({padded.data(), 2, 2, 8}, 2, 2);
  MM_CHECK((same == std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));

  // A flat image stays flat at any scale (weights sum to one in fixed point).
  std::vector<std::uint8_t> flat(37 * 23 * 3, 77);
  for (const auto v : motionman::resize_bilinear_pillow({flat.data(), 37, 23, 37 * 3}, 64, 64)) MM_CHECK(v == 77);
  for (const auto v : motionman::resize_bilinear_pillow({flat.data(), 37, 23, 37 * 3}, 5, 3)) MM_CHECK(v == 77);

  // Layout: BGR HWC → RGB planes, value / 255.
  const std::vector<std::uint8_t> px = {10, 20, 30, 255, 0, 128};  // two pixels, BGR
  std::vector<float> t(6);
  motionman::bgr_to_rgb_nchw_unit(px.data(), 2, 1, t.data());
  MM_CHECK(t[0] == 30.0f / 255.0f && t[1] == 128.0f / 255.0f);  // R plane
  MM_CHECK(t[2] == 20.0f / 255.0f && t[3] == 0.0f);             // G plane
  MM_CHECK(t[4] == 10.0f / 255.0f && t[5] == 1.0f);             // B plane

  std::printf("preprocess unit checks done\n");
  return motionman::test::failures == 0 ? 0 : 1;
}
