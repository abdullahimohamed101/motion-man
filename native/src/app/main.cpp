// motionman-native: the native engine's command line (spec C§3). G1 provides `infer`: preprocess golden images in
// C++, run the TensorRT engine, and write every detection as JSON in the shape of r0.json for tools/model_export/
// parity.py. G2 adds `run` (app/run.hpp): one video into a staging folder for Python to validate and publish.
// Exit codes: 0 ok, 1 error, 2 usage, 3 engine refused (identity mismatch), 130 interrupted (run only).
#include <opencv2/imgcodecs.hpp>

#include <NvInferVersion.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "app/run.hpp"
#include "gpu/trt_detector.hpp"
#include "motionman/json.hpp"
#include "motionman/preprocess.hpp"

namespace fs = std::filesystem;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: motionman-native infer --profile <configs/native/x.json> --model-dir <dir> --images <dir> "
               "--out <file.json> [--precision fp32] [--name <run name>]\n"
               "       motionman-native run --runspec <runspec.json> --run-id <id> --source <video> --out <staging dir>"
               " --profile <configs/native/x.json> --model-dir <dir>\n");
  return 2;
}

std::string num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.9g", v);
  return buf;
}

int infer(const std::map<std::string, std::string>& a) {
  for (const char* k : {"--profile", "--model-dir", "--images", "--out"}) {
    if (!a.count(k)) return usage();
  }
  motionman::gpu::DetectorPaths paths{a.at("--profile"), a.at("--model-dir"),
                                      a.count("--precision") ? a.at("--precision") : "fp32"};
  const std::string name = a.count("--name") ? a.at("--name") : "trt_" + paths.precision;
  auto det = motionman::gpu::TrtDetector::open(paths);

  std::vector<fs::path> pngs;
  for (const auto& e : fs::directory_iterator(a.at("--images"))) {
    if (e.path().extension() == ".png") pngs.push_back(e.path());
  }
  std::sort(pngs.begin(), pngs.end());
  if (pngs.empty()) throw std::runtime_error("no .png images in " + a.at("--images"));

  const int w = det->input_width();
  const int h = det->input_height();
  std::vector<float> input(static_cast<std::size_t>(3) * w * h);
  std::string images;
  for (const auto& p : pngs) {
    const cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
    if (img.empty() || img.type() != CV_8UC3) throw std::runtime_error("cannot decode " + p.string());
    motionman::preprocess_rtdetr({img.data, img.cols, img.rows, img.step[0]}, w, h, input.data());
    const auto dets = det->infer(input.data(), img.cols, img.rows);
    int non_finite = 0;
    int above = 0;
    std::string list;
    for (const auto& d : dets) {
      const bool finite = std::isfinite(d.confidence) && std::all_of(d.bbox_xyxy.begin(), d.bbox_xyxy.end(),
                                                                       [](float v) { return std::isfinite(v); });
      non_finite += finite ? 0 : 1;
      above += d.confidence >= 0.5F ? 1 : 0;
      const auto& names = det->class_names();
      const std::string cls = d.class_id >= 0 && d.class_id < static_cast<int>(names.size()) ? names[d.class_id] : "?";
      list += (list.empty() ? "" : ",\n") + std::string("    {\"label\": ") + std::to_string(d.class_id) +
              ", \"class_name\": " + motionman::json::quote(cls) + ", \"score\": " + num(d.confidence) +
              ", \"box_xyxy\": [" + num(d.bbox_xyxy[0]) + ", " + num(d.bbox_xyxy[1]) + ", " + num(d.bbox_xyxy[2]) +
              ", " + num(d.bbox_xyxy[3]) + "]}";
    }
    images += (images.empty() ? "" : ",\n") + std::string("  ") + motionman::json::quote(p.stem().string()) +
              ": {\"width\": " + std::to_string(img.cols) + ", \"height\": " + std::to_string(img.rows) +
              ", \"non_finite\": " + std::to_string(non_finite) + ", \"detections\": [\n" + list + "]}";
    std::printf("%s: %dx%d, %d detections >= 0.5, non-finite %d\n", p.stem().c_str(), img.cols, img.rows, above,
                non_finite);
  }
  std::ofstream out(a.at("--out"));
  out << "{\"schema_version\": \"1.0\", \"name\": " << motionman::json::quote(name)
      << ", \"precision\": " << motionman::json::quote(paths.precision)
      << ", \"engine_sha256\": " << motionman::json::quote(det->engine_sha256()) << ", \"runtime\": \"TensorRT "
      << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
      << "\", \"inputs\": \"C++ preprocessing of the golden PNGs\", \"images\": {\n"
      << images << "}}\n";
  if (!out) throw std::runtime_error("cannot write " + a.at("--out"));
  std::printf("wrote %s (%zu images)\n", a.at("--out").c_str(), pngs.size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc >= 2 ? argv[1] : "";
  if ((cmd != "infer" && cmd != "run") || argc % 2 != 0) return usage();
  std::map<std::string, std::string> args;
  for (int i = 2; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
  if (cmd == "run") return motionman::app::run(args);  // handles its own errors and always reports a status
  try {
    return infer(args);
  } catch (const motionman::gpu::IdentityError& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
