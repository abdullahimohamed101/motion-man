// Native ByteTrack vs the Python reference on the committed replay fixtures (plan native-g2 step 3, N2-4, C§8).
// agnostic mode must reproduce every fixture: same output detections, the same raw IDs (both allocators start at
// 1), equal scores and classes, boxes within 1e-3 px, and the same unconfirmed flag. aware mode is checked on
// its own cases: it must not change a track's class, and on single-class input it must equal agnostic.
// usage: test_tracker_replay <tests/fixtures/tracker>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../check.hpp"
#include "motionman/json.hpp"
#include "motionman/track/bytetrack.hpp"

namespace fs = std::filesystem;
namespace j = motionman::json;
using motionman::track::ByteTracker;
using motionman::track::ClassPolicy;
using motionman::track::Config;
using motionman::track::Detection;
using motionman::track::TrackOutput;

namespace {

constexpr double kBoxTol = 1e-3;

struct Fixture {
  std::string name;
  Config config;
  int instances = 1;
  std::vector<std::vector<Detection>> frames;
  std::vector<std::vector<std::vector<double>>> tracks;  // expected rows per frame
  std::vector<bool> unconfirmed;
};

Fixture load(const fs::path& path) {
  const j::Value doc = j::parse_file(path.string());
  Fixture f;
  f.name = doc.at("scenario").str();
  const j::Value& c = doc.at("config");
  f.config.track_high_thresh = static_cast<float>(c.at("track_high_thresh").num());
  f.config.track_low_thresh = static_cast<float>(c.at("track_low_thresh").num());
  f.config.new_track_thresh = static_cast<float>(c.at("new_track_thresh").num());
  f.config.track_buffer = static_cast<int>(c.at("track_buffer").num());
  f.config.match_thresh = static_cast<float>(c.at("match_thresh").num());
  f.config.fuse_score = c.at("fuse_score").boolean;
  f.config.class_policy = doc.at("class_policy").str() == "agnostic" ? ClassPolicy::Agnostic : ClassPolicy::Aware;
  f.instances = static_cast<int>(doc.at("instances").num());
  for (const j::Value& frame : doc.at("frames").items) {
    std::vector<Detection> dets;
    for (const j::Value& r : frame.at("dets").items) {
      dets.push_back({static_cast<float>(r.items[0].num()), static_cast<float>(r.items[1].num()),
                      static_cast<float>(r.items[2].num()), static_cast<float>(r.items[3].num()),
                      static_cast<float>(r.items[4].num()), static_cast<int>(r.items[5].num())});
    }
    f.frames.push_back(dets);
    std::vector<std::vector<double>> rows;
    for (const j::Value& r : frame.at("tracks").items) {
      std::vector<double> row;
      for (const j::Value& v : r.items) row.push_back(v.num());
      rows.push_back(row);
    }
    f.tracks.push_back(rows);
    f.unconfirmed.push_back(frame.at("unconfirmed").boolean);
  }
  return f;
}

struct Run {
  std::vector<std::vector<TrackOutput>> outputs;
  std::vector<bool> unconfirmed;
};

Run run(const Fixture& f, ClassPolicy policy) {
  Config cfg = f.config;
  cfg.class_policy = policy;
  ByteTracker tracker(cfg);
  Run r;
  for (const auto& dets : f.frames) {
    r.outputs.push_back(tracker.update(dets));
    r.unconfirmed.push_back(tracker.has_unconfirmed());
  }
  return r;
}

bool same_outputs(const std::vector<TrackOutput>& a, const std::vector<TrackOutput>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].track_id != b[i].track_id || a[i].detection_index != b[i].detection_index ||
        a[i].class_id != b[i].class_id || a[i].score != b[i].score || a[i].xyxy != b[i].xyxy) {
      return false;
    }
  }
  return true;
}

// Problems of a native run against the reference's rows, matched by detection index; empty means it agrees.
std::vector<std::string> compare(const Fixture& f, const Run& r) {
  std::vector<std::string> problems;
  double worst = 0;
  for (std::size_t n = 0; n < f.frames.size(); ++n) {
    const std::string at = "frame " + std::to_string(n);
    if (r.unconfirmed[n] != f.unconfirmed[n]) problems.push_back(at + ": unconfirmed flag differs");
    std::map<int, const std::vector<double>*> expected;
    for (const auto& row : f.tracks[n]) expected[static_cast<int>(row[7])] = &row;
    std::map<int, const TrackOutput*> actual;
    for (const auto& o : r.outputs[n]) actual[o.detection_index] = &o;
    std::set<int> ke, ka;
    for (const auto& [k, v] : expected) ke.insert(k);
    for (const auto& [k, v] : actual) ka.insert(k);
    if (ke != ka) {
      problems.push_back(at + ": " + std::to_string(ka.size()) + " outputs on other detections than the " +
                         std::to_string(ke.size()) + " expected");
      continue;
    }
    for (const auto& [idx, row] : expected) {
      const TrackOutput& o = *actual.at(idx);
      const std::string det = at + " det " + std::to_string(idx);
      if (o.track_id != static_cast<std::int64_t>((*row)[4])) {
        problems.push_back(det + ": id " + std::to_string(o.track_id) + ", expected " +
                           std::to_string(static_cast<std::int64_t>((*row)[4])));
      }
      if (static_cast<double>(o.score) != (*row)[5] || o.class_id != static_cast<int>((*row)[6])) {
        problems.push_back(det + ": score or class differs");
      }
      for (int k = 0; k < 4; ++k) worst = std::max(worst, std::fabs(static_cast<double>(o.xyxy[k]) - (*row)[k]));
    }
  }
  if (worst > kBoxTol) problems.push_back("box error " + std::to_string(worst) + " px > 1e-3");
  std::printf("  %-24s %zu frames, max box error %.2e px\n", f.name.c_str(), f.frames.size(), worst);
  return problems;
}

bool any_track_changes_class(const Run& r) {
  std::map<std::int64_t, int> cls;
  for (const auto& frame : r.outputs) {
    for (const auto& o : frame) {
      if (cls.count(o.track_id) && cls[o.track_id] != o.class_id) return true;
      cls[o.track_id] = o.class_id;
    }
  }
  return false;
}

}  // namespace

// One replay file (plan native-g2 step 7, N2-8: real detections dumped by `motionman-native run`): agnostic
// native tracking must reproduce the reference exactly, as for the committed fixtures.
int check_one(const fs::path& path) {
  const Fixture f = load(path);
  const Run r = run(f, f.config.class_policy);
  const auto problems = compare(f, r);
  std::set<std::int64_t> ids;
  std::size_t outputs = 0, dets = 0;
  for (std::size_t n = 0; n < f.frames.size(); ++n) {
    dets += f.frames[n].size();
    outputs += r.outputs[n].size();
    for (const auto& o : r.outputs[n]) ids.insert(o.track_id);
  }
  for (std::size_t i = 0; i < problems.size() && i < 10; ++i) std::fprintf(stderr, "  %s\n", problems[i].c_str());
  std::printf("%s: %zu frames, %zu detections, %zu track outputs, %zu IDs; %zu problems\n", f.name.c_str(),
              f.frames.size(), dets, outputs, ids.size(), problems.size());
  return problems.empty() ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_tracker_replay <fixtures/tracker dir | one replay .json>\n");
    return 2;
  }
  if (fs::is_regular_file(argv[1])) return check_one(argv[1]);
  std::vector<fs::path> paths;
  for (const auto& e : fs::directory_iterator(argv[1])) {
    if (e.path().extension() == ".json") paths.push_back(e.path());
  }
  std::sort(paths.begin(), paths.end());
  MM_CHECK(paths.size() == 15);

  std::map<std::string, Fixture> by_name;
  int agree = 0;
  std::printf("agnostic replay vs the Python reference:\n");
  for (const auto& p : paths) {
    const Fixture f = load(p);
    MM_CHECK(f.config.class_policy == ClassPolicy::Agnostic);
    const Run first = run(f, ClassPolicy::Agnostic);
    for (int k = 1; k < f.instances; ++k) {  // separate trackers on identical input give identical IDs
      const Run other = run(f, ClassPolicy::Agnostic);
      for (std::size_t n = 0; n < f.frames.size(); ++n) MM_CHECK(same_outputs(first.outputs[n], other.outputs[n]));
    }
    const auto problems = compare(f, first);
    for (std::size_t i = 0; i < problems.size() && i < 5; ++i) {
      std::fprintf(stderr, "  %s: %s\n", f.name.c_str(), problems[i].c_str());
    }
    MM_CHECK(problems.empty());
    agree += problems.empty() ? 1 : 0;
    by_name[f.name] = f;
  }
  std::printf("%d/%zu fixtures reproduced\n", agree, paths.size());

  // aware: identical to agnostic wherever classes never meet.
  for (const char* name : {"entry_exit", "reappear_within_buffer", "reappear_after_buffer", "crossing", "overlap",
                           "low_conf_recovery", "low_conf_below_floor", "low_conf_jump", "empty_frames",
                           "assignment_ties",
                           "new_track_threshold", "degenerate_boxes", "two_instances"}) {
    const Fixture& f = by_name.at(name);
    const Run a = run(f, ClassPolicy::Agnostic);
    const Run b = run(f, ClassPolicy::Aware);
    bool same = true;
    for (std::size_t n = 0; n < f.frames.size(); ++n) same = same && same_outputs(a.outputs[n], b.outputs[n]);
    if (!same) std::fprintf(stderr, "  aware differs from agnostic on %s\n", name);
    MM_CHECK(same);
  }

  // aware: a person detection on a car track starts a new person track instead of relabelling the car.
  {
    const Fixture& f = by_name.at("class_conflict");
    const Run agnostic = run(f, ClassPolicy::Agnostic);
    const Run aware = run(f, ClassPolicy::Aware);
    MM_CHECK(any_track_changes_class(agnostic));  // the reference behaviour, recorded in the fixture
    MM_CHECK(!any_track_changes_class(aware));
    MM_CHECK(aware.outputs[5].size() == 1 && aware.outputs[5][0].track_id == 1 && aware.outputs[5][0].class_id == 2);
    MM_CHECK(aware.outputs[10].empty());  // the new person track is unconfirmed on its first frame
    MM_CHECK(aware.outputs[11].size() == 1 && aware.outputs[11][0].track_id == 2 && aware.outputs[11][0].class_id == 0);
  }

  // aware: no track in the mixed crowd ever changes class.
  MM_CHECK(!any_track_changes_class(run(by_name.at("crowd"), ClassPolicy::Aware)));

  // aware: a person and a car on the same box every frame keep two separate tracks.
  {
    ByteTracker t(Config{});
    std::set<std::int64_t> ids;
    for (int n = 0; n < 10; ++n) {
      const float x = 100.0F + 4.0F * static_cast<float>(n);
      const std::vector<Detection> d = {{x, 100, x + 80, 200, 0.9F, 0}, {x, 100, x + 80, 200, 0.85F, 2}};
      for (const auto& o : t.update(d)) ids.insert(o.track_id);
    }
    MM_CHECK(ids.size() == 2);
  }

  // Default config is the documented one: aware, bytetrack.yaml thresholds (ADR-006).
  const Config d{};
  MM_CHECK(d.class_policy == ClassPolicy::Aware && d.track_buffer == 30 && d.match_thresh == 0.8F && d.fuse_score);
  return motionman::test::failures == 0 ? 0 : 1;
}
