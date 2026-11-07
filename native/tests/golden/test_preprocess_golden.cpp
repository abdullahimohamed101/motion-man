// C++ preprocessing vs the Python reference tensors (plan native-g0-g1 step 7, NG-7, spec C§6 gates).
// Usage: test_preprocess_golden <golden PNG dir> <golden_tensors dir>. Each <name>.npy was written by
// tools/model_export/reference.py from <name>.png with the upstream transform; this test rebuilds the tensor from
// the same PNG (decoded by OpenCV as BGR) and compares. Same-size images take the layout gate (max error 1e-6);
// resized images the resize gate (MAE ≤ 1/255, p99 ≤ 3/255).
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include "../check.hpp"
#include "motionman/preprocess.hpp"

namespace fs = std::filesystem;

namespace {

// Minimal .npy reader: little-endian float32, C order, any shape.
bool read_npy_f32(const fs::path& path, std::vector<long>& shape, std::vector<float>& data) {
  std::ifstream in(path, std::ios::binary);
  char magic[6];
  if (!in.read(magic, 6) || std::string(magic, 6) != "\x93NUMPY") return false;
  unsigned char ver[2];
  in.read(reinterpret_cast<char*>(ver), 2);
  std::uint32_t hlen = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    hlen = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  }
  std::string header(hlen, '\0');
  in.read(header.data(), hlen);
  if (header.find("'descr': '<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
    return false;
  std::smatch m;
  if (!std::regex_search(header, m, std::regex("'shape': \\(([^)]*)\\)"))) return false;
  shape.clear();
  std::size_t n = 1;
  const std::string dims = m[1];
  for (std::sregex_iterator it(dims.begin(), dims.end(), std::regex("[0-9]+")), end; it != end; ++it) {
    shape.push_back(std::stol(it->str()));
    n *= static_cast<std::size_t>(shape.back());
  }
  data.resize(n);
  return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: test_preprocess_golden <golden png dir> <golden_tensors dir>\n");
    return 2;
  }
  const fs::path png_dir = argv[1];
  const fs::path npy_dir = argv[2];
  int images = 0;
  for (const auto& entry : fs::directory_iterator(npy_dir)) {
    if (entry.path().extension() != ".npy") continue;
    const std::string name = entry.path().stem().string();
    std::vector<long> shape;
    std::vector<float> ref;
    MM_CHECK(read_npy_f32(entry.path(), shape, ref));
    MM_CHECK(shape.size() == 4 && shape[0] == 1 && shape[1] == 3);
    if (shape.size() != 4) continue;
    const int out_h = static_cast<int>(shape[2]);
    const int out_w = static_cast<int>(shape[3]);
    const cv::Mat img = cv::imread((png_dir / (name + ".png")).string(), cv::IMREAD_COLOR);
    MM_CHECK(!img.empty() && img.type() == CV_8UC3);
    if (img.empty()) continue;
    ++images;

    std::vector<float> got(ref.size());
    motionman::preprocess_rtdetr({img.data, img.cols, img.rows, img.step[0]}, out_w, out_h, got.data());
    std::vector<double> err(ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) err[i] = std::fabs(static_cast<double>(got[i]) - ref[i]);
    double sum = 0.0;
    double max_err = 0.0;
    for (const double e : err) {
      sum += e;
      max_err = std::max(max_err, e);
    }
    const double mae = sum / static_cast<double>(err.size());
    std::vector<double> sorted = err;
    const std::size_t k99 = static_cast<std::size_t>(0.99 * static_cast<double>(sorted.size() - 1));
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(k99), sorted.end());
    const double p99 = sorted[k99];
    const long differ = std::count_if(err.begin(), err.end(), [](double e) { return e > 0.5 / 255.0; });

    const bool resized = img.cols != out_w || img.rows != out_h;
    std::printf("%-22s %4dx%-4d -> %dx%d %-6s max %.3g  mae %.3g  p99 %.3g  pixels differing %ld\n", name.c_str(),
                img.cols, img.rows, out_w, out_h, resized ? "resize" : "layout", max_err, mae, p99, differ);
    if (resized) {
      MM_CHECK(mae <= 1.0 / 255.0);
      MM_CHECK(p99 <= 3.0 / 255.0);
    } else {
      MM_CHECK(max_err <= 1e-6);
    }
  }
  MM_CHECK(images >= 11);  // every golden image has a tensor
  std::printf("%d images\n", images);
  return motionman::test::failures == 0 ? 0 : 1;
}
