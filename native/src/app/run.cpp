#include "app/run.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "app/annotate.hpp"
#include "gpu/trt_detector.hpp"
#include "motionman/json.hpp"
#include "motionman/preprocess.hpp"
#include "motionman/records.hpp"
#include "motionman/sha256.hpp"
#include "motionman/track/bytetrack.hpp"

namespace fs = std::filesystem;
namespace j = motionman::json;

namespace motionman::app {
namespace {

constexpr const char* kReceiptSchema = "1.0";
constexpr const char* kTracks = "tracks.jsonl";
constexpr const char* kVideo = "annotated.mp4";
constexpr const char* kTimings = "timings.jsonl";
constexpr const char* kReceipt = "native_receipt.json";
constexpr double kTimestampToleranceFrames = 0.5;  // video_io.TIMESTAMP_TOLERANCE_FRAMES
constexpr double kProgressIntervalS = 1.0;

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int sig) { g_stop = sig; }

// The runspec, the source or the engine does not match what this run was asked to do (exit 3, nothing processed).
struct Refused : std::runtime_error {
  using std::runtime_error::runtime_error;
};

using Clock = std::chrono::steady_clock;
double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string utc_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm g{};
  gmtime_r(&t, &g);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &g);
  return buf;
}

std::string hostname() {
  char buf[256] = {};
  return gethostname(buf, sizeof buf - 1) == 0 ? std::string(buf) : std::string("unknown");
}

std::string opt_str(const std::string& s) { return s.empty() ? "null" : j::quote(s); }
std::string opt_int(const std::optional<std::int64_t>& v) { return v ? std::to_string(*v) : "null"; }

// One JSON status record per line on stdout (ADR-006); human logs go to stderr.
void status_line(const std::string& body) {
  std::fputs(body.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

// --- Runspec: the settings native needs, read from the exact bytes Python wrote -------------------------------------

struct Spec {
  int width = 0;
  int height = 0;
  double fps = 0;
  std::string input_id;
  std::optional<std::int64_t> container_frames;
  std::optional<std::int64_t> stop;  // frame_range[1]: the user's requested end (exclusive), or to the end
  float conf = 0;
  track::Config tracker;
  std::string class_policy;
  std::string codec;
  std::size_t trail_points = 0;
  int trail_expiry = 0;
  std::optional<std::int64_t> frame_limit;  // guardrails: exceeding one is an error, never a silent truncation
  std::optional<double> max_duration_s;
  std::string backend_name, precision, profile, engine_sha256, gpu;
  std::vector<std::pair<int, std::string>> class_names;  // empty when the runspec records none
};

std::int64_t integer(const j::Value& v, const char* what) {
  const double d = v.num();
  if (d != std::floor(d) || std::fabs(d) > 9e15) throw Refused(std::string("runspec: ") + what + " is not an integer");
  return static_cast<std::int64_t>(d);
}

std::optional<std::int64_t> opt_integer(const j::Value* v, const char* what) {
  if (!v || v->kind == j::Value::Kind::Null) return std::nullopt;
  return integer(*v, what);
}

Spec read_spec(const std::string& path) {
  try {
    const j::Value doc = j::parse_file(path);
    const j::Value& s = doc.at("settings");
    Spec spec;
    const j::Value& in = s.at("input");
    spec.width = static_cast<int>(integer(in.at("width"), "input.width"));
    spec.height = static_cast<int>(integer(in.at("height"), "input.height"));
    spec.fps = in.at("fps").num();
    spec.input_id = in.at("input_id").str();
    spec.container_frames = opt_integer(in.find("frame_count_reported_by_container"), "input.frame_count");
    const j::Value& range = in.at("frame_range");
    if (range.kind != j::Value::Kind::Array || range.items.size() != 2 || integer(range.items[0], "frame_range") != 0) {
      throw Refused("runspec: frame_range must be [0, stop]; a native run starts at frame 0");
    }
    spec.stop = opt_integer(&range.items[1], "frame_range stop");
    if (spec.width <= 0 || spec.height <= 0 || !(spec.fps > 0) || !std::isfinite(spec.fps)) {
      throw Refused("runspec: input width, height and fps must be positive");
    }

    const j::Value& b = s.at("backend");
    spec.backend_name = b.at("name").str();
    spec.precision = b.at("precision").str();
    spec.profile = b.at("profile").str();
    spec.engine_sha256 = b.at("engine_sha256").str();
    if (const j::Value* g = b.find("gpu"); g && g->kind == j::Value::Kind::String) spec.gpu = g->str();
    if (spec.backend_name != "tensorrt") throw Refused("runspec backend is '" + spec.backend_name + "', not tensorrt");
    if (s.at("inference").at("precision").str() != spec.precision) {
      throw Refused("runspec: inference.precision differs from backend.precision");
    }
    spec.conf = static_cast<float>(s.at("inference").at("conf").num());

    const j::Value& t = s.at("tracking");
    const j::Value& c = t.at("config");
    if (t.at("type").str() != "bytetrack" || c.at("tracker_type").str() != "bytetrack") {
      throw Refused("runspec: native runs support only the bytetrack tracker");
    }
    spec.tracker.track_high_thresh = static_cast<float>(c.at("track_high_thresh").num());
    spec.tracker.track_low_thresh = static_cast<float>(c.at("track_low_thresh").num());
    spec.tracker.new_track_thresh = static_cast<float>(c.at("new_track_thresh").num());
    spec.tracker.track_buffer = static_cast<int>(integer(c.at("track_buffer"), "track_buffer"));
    spec.tracker.match_thresh = static_cast<float>(c.at("match_thresh").num());
    if (c.at("fuse_score").kind != j::Value::Kind::Bool) throw Refused("runspec: fuse_score must be true or false");
    spec.tracker.fuse_score = c.at("fuse_score").boolean;
    spec.class_policy = t.at("class_policy").str();
    if (spec.class_policy == "aware") {
      spec.tracker.class_policy = track::ClassPolicy::Aware;
    } else if (spec.class_policy == "agnostic") {
      spec.tracker.class_policy = track::ClassPolicy::Agnostic;
    } else {
      throw Refused("runspec: class_policy must be 'aware' or 'agnostic', not '" + spec.class_policy + "'");
    }

    const j::Value& r = s.at("rendering");
    if (r.at("codec").kind != j::Value::Kind::String || r.at("codec").str().size() != 4) {
      throw Refused("runspec: rendering.codec must name a four-character codec");
    }
    spec.codec = r.at("codec").str();
    spec.trail_points = static_cast<std::size_t>(integer(r.at("trail_points"), "trail_points"));
    spec.trail_expiry = static_cast<int>(integer(r.at("trail_expiry_frames"), "trail_expiry_frames"));

    const j::Value& lim = s.at("limits");
    spec.frame_limit = opt_integer(lim.find("max_frames"), "limits.max_frames");
    if (const j::Value* d = lim.find("max_duration_s"); d && d->kind != j::Value::Kind::Null) {
      spec.max_duration_s = d->num();
    }

    if (const j::Value* names = doc.find("class_names"); names && names->kind == j::Value::Kind::Object) {
      for (std::size_t k = 0; k < names->keys.size(); ++k) {
        spec.class_names.emplace_back(std::stoi(names->keys[k]), names->items[k].str());
      }
    }
    return spec;
  } catch (const Refused&) {
    throw;
  } catch (const std::exception& e) {
    throw Refused(std::string("runspec ") + path + ": " + e.what());
  }
}

// --- Outputs ---------------------------------------------------------------------------------------------------------

struct FileCloser {
  void operator()(std::FILE* f) const { std::fclose(f); }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File create_exclusive(const fs::path& p) {
  File f(std::fopen(p.c_str(), "wx"));  // never overwrite (INV-5)
  if (!f) throw std::runtime_error("cannot create " + p.string());
  return f;
}

void put_line(const File& f, const std::string& line, const char* what) {
  if (std::fputs(line.c_str(), f.get()) == EOF || std::fputc('\n', f.get()) == EOF) {
    throw std::runtime_error(std::string("write failed for ") + what);
  }
}

void close_checked(File& f, const char* what) {
  if (std::fclose(f.release()) != 0) throw std::runtime_error(std::string("closing ") + what + " failed");
}

struct Receipt {
  std::string status = "error";
  std::string error;
  std::string run_id, runspec_sha256, source_sha256, engine_sha256, profile, precision, gpu, driver, codec;
  std::string video_backend, class_policy, started_at, finished_at;
  std::int64_t decoded = 0, processed = 0, records = 0, timing_rows = 0, admitted = 0, recorded_objects = 0;
  std::int64_t dropped_no_extent = 0, max_track_id = 0;
  double model_load_s = 0, wall_s = 0;
  std::array<std::pair<std::string, double>, 6> stage_total_ms{{{"decode_ms", 0},
                                                                 {"preprocess_ms", 0},
                                                                 {"infer_ms", 0},
                                                                 {"track_ms", 0},
                                                                 {"render_ms", 0},
                                                                 {"write_ms", 0}}};

  std::string json() const {
    std::string stages;
    for (const auto& [name, v] : stage_total_ms) {
      stages += (stages.empty() ? "" : ", ") + j::quote(name) + ": " + records::py_round(v, 3);
    }
    return "{\"schema_version\": \"" + std::string(kReceiptSchema) + "\", \"status\": " + j::quote(status) +
           ", \"error\": " + opt_str(error) + ", \"run_id\": " + opt_str(run_id) +
           ", \"runspec_sha256\": " + opt_str(runspec_sha256) + ", \"source_sha256\": " + opt_str(source_sha256) +
           ",\n \"frames\": {\"decoded\": " + std::to_string(decoded) + ", \"processed\": " +
           std::to_string(processed) + ", \"records\": " + std::to_string(records) + ", \"timing_rows\": " +
           std::to_string(timing_rows) + ", \"video_written\": " + std::to_string(processed) + "}" +
           ",\n \"objects\": {\"admitted_detections\": " + std::to_string(admitted) + ", \"recorded\": " +
           std::to_string(recorded_objects) + ", \"dropped_no_extent\": " + std::to_string(dropped_no_extent) +
           ", \"max_track_id\": " + std::to_string(max_track_id) + ", \"class_policy\": " + opt_str(class_policy) +
           "}" + ",\n \"engine\": {\"runtime\": \"tensorrt\", \"tensorrt\": " +
           j::quote(gpu::TrtDetector::tensorrt_version()) + ", \"profile\": " + opt_str(profile) +
           ", \"precision\": " + opt_str(precision) + ", \"engine_sha256\": " + opt_str(engine_sha256) +
           ", \"gpu\": " + opt_str(gpu) + ", \"driver\": " + opt_str(driver) + "}" +
           ",\n \"video\": {\"codec\": " + opt_str(codec) + ", \"decode_backend\": " + opt_str(video_backend) + "}" +
           ",\n \"timing\": {\"model_load_s\": " + records::py_round(model_load_s, 3) +
           ", \"wall_s\": " + records::py_round(wall_s, 3) + ", \"stage_total_ms\": {" + stages + "}}" +
           ",\n \"host\": " + j::quote(hostname()) + ", \"started_at\": " + opt_str(started_at) +
           ", \"finished_at\": " + opt_str(finished_at) + "}\n";
  }
};

void write_receipt(const fs::path& out, const Receipt& rc) {
  const fs::path tmp = out / (std::string(kReceipt) + ".tmp");
  {
    File f(std::fopen(tmp.c_str(), "w"));
    if (!f || std::fputs(rc.json().c_str(), f.get()) == EOF) throw std::runtime_error("cannot write " + tmp.string());
    close_checked(f, kReceipt);
  }
  fs::rename(tmp, out / kReceipt);  // the receipt appears whole or not at all
}

// --- The run ---------------------------------------------------------------------------------------------------------

void check_identity(const Spec& spec, const std::string& profile_json, const gpu::TrtDetector& det) {
  std::vector<std::string> problems;
  const std::string profile = j::parse_file(profile_json).at("profile").str();
  if (profile != spec.profile) problems.push_back("profile: runspec '" + spec.profile + "', engine '" + profile + "'");
  if (det.engine_sha256() != spec.engine_sha256) {
    problems.push_back("engine_sha256: runspec " + spec.engine_sha256 + ", loaded " + det.engine_sha256());
  }
  if (!spec.gpu.empty() && spec.gpu != det.gpu_name()) {
    problems.push_back("gpu: runspec '" + spec.gpu + "', this host '" + det.gpu_name() + "'");
  }
  if (!spec.class_names.empty()) {
    const auto& names = det.class_names();
    bool same = spec.class_names.size() == names.size();
    for (const auto& [k, v] : spec.class_names) {
      same = same && k >= 0 && static_cast<std::size_t>(k) < names.size() && names[k] == v;
    }
    if (!same) problems.push_back("class_names: the runspec's class list differs from the model's");
  }
  if (!problems.empty()) {
    std::string msg = "refusing run (" + std::to_string(problems.size()) + " mismatches with the runspec):";
    for (const auto& p : problems) msg += "\n  " + p;
    throw Refused(msg);
  }
}

int execute(const std::map<std::string, std::string>& a, const fs::path& out, Receipt& rc) {
  const auto t_start = Clock::now();
  const Spec spec = read_spec(a.at("--runspec"));
  rc.runspec_sha256 = sha256_file(a.at("--runspec"));
  rc.class_policy = spec.class_policy;
  rc.codec = spec.codec;
  rc.profile = spec.profile;
  rc.precision = spec.precision;

  const std::string& source = a.at("--source");
  rc.source_sha256 = sha256_file(source);  // the bytes Python imported, before anything is processed
  if (rc.source_sha256 != spec.input_id) {
    throw Refused("source " + source + " is " + rc.source_sha256 + ", the runspec's input_id is " + spec.input_id);
  }

  const auto t_load = Clock::now();
  auto det = gpu::TrtDetector::open({a.at("--profile"), a.at("--model-dir"), spec.precision});
  check_identity(spec, a.at("--profile"), *det);
  rc.model_load_s = ms_between(t_load, Clock::now()) / 1000;
  rc.engine_sha256 = det->engine_sha256();
  rc.gpu = det->gpu_name();
  rc.driver = det->driver();
  const auto& names = det->class_names();

  cv::VideoCapture cap(source, cv::CAP_FFMPEG);
  if (!cap.isOpened()) throw std::runtime_error("cannot open video " + source);
  rc.video_backend = cap.getBackendName();
  cv::VideoWriter writer((out / kVideo).string(),
                         cv::VideoWriter::fourcc(spec.codec[0], spec.codec[1], spec.codec[2], spec.codec[3]),
                         spec.fps, {spec.width, spec.height});
  if (!writer.isOpened()) throw std::runtime_error("cannot open a video writer with codec " + spec.codec);
  File tracks = create_exclusive(out / kTracks);
  File timings = create_exclusive(out / kTimings);

  track::ByteTracker tracker(spec.tracker);  // one tracker, this run only, frames in order (INV-1)
  records::Trails trails(spec.trail_points, spec.trail_expiry);
  std::vector<float> input(static_cast<std::size_t>(3) * det->input_width() * det->input_height());
  std::vector<track::Detection> admitted;
  admitted.reserve(512);
  std::array<std::pair<std::string, double>, 6> stages = rc.stage_total_ms;
  const auto w = static_cast<float>(spec.width);
  const auto h = static_cast<float>(spec.height);
  cv::Mat frame;

  status_line("{\"event\": \"started\", \"run_id\": " + j::quote(rc.run_id) + ", \"frames_expected\": " +
              opt_int(spec.stop ? spec.stop : spec.container_frames) + ", \"gpu\": " + j::quote(rc.gpu) + "}");
  const auto t_loop = Clock::now();
  auto last_progress = t_loop;
  for (std::int64_t index = 0; !spec.stop || index < *spec.stop; ++index) {
    if (g_stop) break;  // between frames only: every written frame has its record and timing row
    const auto t0 = Clock::now();
    if (!cap.read(frame)) break;
    const auto t1 = Clock::now();
    ++rc.decoded;
    if (spec.frame_limit && index >= *spec.frame_limit) {
      throw std::runtime_error("video exceeds the " + std::to_string(*spec.frame_limit) +
                               "-frame limit; nothing past it was processed");
    }
    if (spec.max_duration_s && static_cast<double>(index) / spec.fps >= *spec.max_duration_s) {
      throw std::runtime_error("video exceeds the duration limit; nothing past it was processed");
    }
    const double drift = std::fabs(cap.get(cv::CAP_PROP_POS_MSEC) - static_cast<double>(index) * 1000.0 / spec.fps) *
                         spec.fps / 1000.0;
    if (drift > kTimestampToleranceFrames) {
      throw std::runtime_error("frame " + std::to_string(index) + " timestamp is " + records::py_round(drift, 2) +
                               " frames from constant fps; variable frame rate is not supported");
    }
    if (frame.cols != spec.width || frame.rows != spec.height || frame.type() != CV_8UC3) {
      throw std::runtime_error("frame " + std::to_string(index) + " is not " + std::to_string(spec.width) + "x" +
                               std::to_string(spec.height) + " BGR as the runspec says");
    }

    preprocess_rtdetr({frame.data, frame.cols, frame.rows, frame.step[0]}, det->input_width(), det->input_height(),
                      input.data());
    const auto t2 = Clock::now();
    const auto dets = det->infer(input.data(), frame.cols, frame.rows);
    const auto t3 = Clock::now();

    // Admission (ADR-006): every detection above track_low_thresh, its box clamped to the image in float32.
    admitted.clear();
    for (const auto& d : dets) {
      if (!(d.confidence > spec.tracker.track_low_thresh)) continue;
      const auto& b = d.bbox_xyxy;
      if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) || !std::isfinite(b[3])) {
        throw std::runtime_error("the detector produced a non-finite box at frame " + std::to_string(index));
      }
      admitted.push_back({std::min(std::max(b[0], 0.0F), w), std::min(std::max(b[1], 0.0F), h),
                          std::min(std::max(b[2], 0.0F), w), std::min(std::max(b[3], 0.0F), h), d.confidence,
                          d.class_id});
    }
    rc.admitted += static_cast<std::int64_t>(admitted.size());
    const auto outputs = tracker.update(admitted);
    const auto fo =
        records::frame_objects(outputs, tracker.has_unconfirmed(), admitted, spec.conf, spec.width, spec.height, names);
    const auto t4 = Clock::now();

    rc.dropped_no_extent += fo.dropped_no_extent;
    rc.recorded_objects += static_cast<std::int64_t>(fo.objects.size());
    for (const auto& o : fo.objects) rc.max_track_id = std::max(rc.max_track_id, o.track_id.value_or(0));
    const std::string line =
        records::frame_record_line(rc.run_id, index, spec.fps, spec.width, spec.height, fo.objects);
    trails.update(index, fo.objects);
    annotate(frame, fo.objects, trails);
    const auto t5 = Clock::now();
    writer.write(frame);
    put_line(tracks, line, kTracks);  // streamed, one record per processed frame (INV-3)
    ++rc.records;
    const auto t6 = Clock::now();

    const Clock::time_point marks[] = {t0, t1, t2, t3, t4, t5, t6};
    for (std::size_t k = 0; k < stages.size(); ++k) {
      stages[k].second = ms_between(marks[k], marks[k + 1]);
      rc.stage_total_ms[k].second += stages[k].second;
    }
    put_line(timings, records::timing_line(index, stages), kTimings);
    ++rc.timing_rows;
    ++rc.processed;
    if (ms_between(last_progress, t6) >= kProgressIntervalS * 1000) {
      last_progress = t6;
      status_line("{\"event\": \"progress\", \"processed\": " + std::to_string(rc.processed) +
                  ", \"elapsed_s\": " + records::py_round(ms_between(t_loop, t6) / 1000, 3) + "}");
    }
  }
  writer.release();
  close_checked(tracks, kTracks);
  close_checked(timings, kTimings);
  rc.wall_s = ms_between(t_start, Clock::now()) / 1000;
  if (g_stop) {
    rc.status = "interrupted";
    rc.error = std::string("stopped by ") + (g_stop == SIGTERM ? "SIGTERM" : "SIGINT") + " after frame " +
               std::to_string(rc.processed);
    return 130;
  }
  if (rc.processed == 0) throw std::runtime_error("no decodable frames in " + source);
  rc.status = "success";
  return 0;
}

int usage() {
  std::fprintf(stderr,
               "usage: motionman-native run --runspec <runspec.json> --run-id <id> --source <video> --out <staging dir>"
               " --profile <configs/native/x.json> --model-dir <dir>\n");
  return 2;
}

}  // namespace

int run(const std::map<std::string, std::string>& a) {
  static const char* const kArgs[] = {"--runspec", "--run-id", "--source", "--out", "--profile", "--model-dir"};
  for (const char* k : kArgs) {
    if (!a.count(k)) return usage();
  }
  if (a.size() != std::size(kArgs)) return usage();

  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // The staging folder exists (Python made it and wrote runspec.json) and holds none of native's outputs (INV-5).
  // Failing here writes nothing, not even a receipt.
  const fs::path out = a.at("--out");
  std::error_code ec;
  if (!fs::is_directory(out, ec)) {
    std::fprintf(stderr, "error: --out %s is not an existing directory\n", out.c_str());
    status_line("{\"event\": \"finished\", \"status\": \"error\", \"exit_code\": 1}");
    return 1;
  }
  for (const char* name : {kTracks, kVideo, kTimings, kReceipt}) {
    if (fs::exists(out / name, ec)) {
      std::fprintf(stderr, "error: %s already exists; native never overwrites outputs\n", (out / name).c_str());
      status_line("{\"event\": \"finished\", \"status\": \"error\", \"exit_code\": 1}");
      return 1;
    }
  }

  Receipt rc;
  rc.run_id = a.at("--run-id");
  rc.started_at = utc_now();
  int code = 1;
  try {
    code = execute(a, out, rc);  // resources are released on every exit from here (INV-6)
  } catch (const gpu::IdentityError& e) {
    rc.status = "refused";
    rc.error = e.what();
    code = 3;
  } catch (const Refused& e) {
    rc.status = "refused";
    rc.error = e.what();
    code = 3;
  } catch (const std::exception& e) {
    rc.status = "error";
    rc.error = e.what();
    code = 1;
  }
  rc.finished_at = utc_now();
  if (!rc.error.empty()) std::fprintf(stderr, "%s: %s\n", rc.status.c_str(), rc.error.c_str());
  try {
    write_receipt(out, rc);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    if (code == 0) {  // no receipt, no success
      code = 1;
      rc.status = "error";
    }
  }
  status_line("{\"event\": \"finished\", \"status\": " + j::quote(rc.status) + ", \"processed\": " +
              std::to_string(rc.processed) + ", \"exit_code\": " + std::to_string(code) + "}");
  return code;
}

}  // namespace motionman::app
