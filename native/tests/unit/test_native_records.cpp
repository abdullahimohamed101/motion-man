// Native tracks.jsonl / timings.jsonl lines and trails vs Python's own output (plan native-g2 step 4, N2-5).
// usage: test_native_records <tests/fixtures/native_records.json>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "../check.hpp"
#include "motionman/json.hpp"
#include "motionman/records.hpp"

namespace j = motionman::json;
namespace r = motionman::records;
using motionman::track::Detection;
using motionman::track::TrackOutput;

namespace {

double number(const j::Value& v) {
  // Standard JSON has no NaN: the fixture stores a non-finite coordinate as null.
  return v.kind == j::Value::Kind::Number ? v.num() : std::numeric_limits<double>::quiet_NaN();
}

r::RecordObject object(const j::Value& o) {
  r::RecordObject obj;
  if (o.items[0].kind != j::Value::Kind::Null) obj.track_id = static_cast<std::int64_t>(o.items[0].num());
  obj.class_id = static_cast<int>(o.items[1].num());
  obj.class_name = o.items[2].str();
  obj.confidence = o.items[3].num();
  for (int k = 0; k < 4; ++k) obj.bbox_xyxy[k] = number(o.items[4 + k]);
  return obj;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_native_records <native_records.json>\n");
    return 2;
  }
  const j::Value doc = j::parse_file(argv[1]);
  MM_CHECK(doc.at("schema").str() == "motionman.native_records/1");

  int lines = 0, same = 0, refused = 0;
  for (const j::Value& c : doc.at("records").items) {
    std::vector<r::RecordObject> objs;
    for (const j::Value& o : c.at("objects").items) objs.push_back(object(o));
    const auto& expected = c.at("line");
    std::string got;
    bool threw = false;
    try {
      got = r::frame_record_line(c.at("run_id").str(), static_cast<std::int64_t>(c.at("frame_index").num()),
                                 c.at("fps").num(), static_cast<int>(c.at("width").num()),
                                 static_cast<int>(c.at("height").num()), objs);
    } catch (const r::RecordError&) {
      threw = true;
    }
    ++lines;
    if (expected.kind == j::Value::Kind::Null) {
      refused += threw ? 1 : 0;
      same += threw ? 1 : 0;
      if (!threw) std::fprintf(stderr, "record %d: Python refuses it, native wrote %s\n", lines, got.c_str());
    } else if (!threw && got == expected.str()) {
      ++same;
    } else if (same + 5 > lines) {
      std::fprintf(stderr, "record %d differs:\n  python %s\n  native %s\n", lines, expected.str().c_str(),
                   threw ? "(refused)" : got.c_str());
    }
  }
  std::printf("records: %d/%d lines byte-identical (%d refused by both)\n", same, lines, refused);
  MM_CHECK(lines >= 130 && same == lines && refused == 3);

  int timings = 0, timings_same = 0;
  for (const j::Value& c : doc.at("timings").items) {
    std::vector<std::pair<std::string, double>> stages;
    for (const j::Value& s : c.at("stages").items) stages.emplace_back(s.items[0].str(), s.items[1].num());
    const std::string got = r::timing_line(static_cast<std::int64_t>(c.at("frame_index").num()), stages);
    ++timings;
    if (got == c.at("line").str()) {
      ++timings_same;
    } else {
      std::fprintf(stderr, "timing %d differs:\n  python %s\n  native %s\n", timings, c.at("line").str().c_str(),
                   got.c_str());
    }
  }
  std::printf("timings: %d/%d lines byte-identical\n", timings_same, timings);
  MM_CHECK(timings == 60 && timings_same == timings);

  r::Trails trails;
  int checked = 0, trails_same = 0;
  const auto& frames = doc.at("trails").items;
  for (std::size_t f = 0; f < frames.size(); ++f) {
    std::vector<r::RecordObject> objs;
    for (const j::Value& o : frames[f].at("objects").items) {
      r::RecordObject obj;
      if (o.items[0].kind != j::Value::Kind::Null) obj.track_id = static_cast<std::int64_t>(o.items[0].num());
      for (int k = 0; k < 4; ++k) obj.bbox_xyxy[k] = o.items[1 + k].num();
      objs.push_back(obj);
    }
    trails.update(static_cast<std::int64_t>(f), objs);
    const j::Value* points = frames[f].find("points");
    if (!points) continue;
    ++checked;
    bool ok = true;
    for (std::int64_t id : {1, 2, 3}) {
      const j::Value* want = points->find(std::to_string(id));
      const auto got = trails.points(id);
      const std::size_t n = want ? want->items.size() : 0;
      ok = ok && got.size() == n;
      for (std::size_t k = 0; ok && k < n; ++k) {
        ok = got[k].first == static_cast<int>(want->items[k].items[0].num()) &&
             got[k].second == static_cast<int>(want->items[k].items[1].num());
      }
    }
    if (!ok) std::fprintf(stderr, "trails differ at frame %zu\n", f);
    trails_same += ok ? 1 : 0;
  }
  std::printf("trails: %d/%d checkpoints equal (30-point cap, 60-frame expiry)\n", trails_same, checked);
  MM_CHECK(checked >= 20 && trails_same == checked);
  MM_CHECK(trails.size() == 2);  // ID 3 left at frame 49 and expired at 109; IDs 1 and 2 remain

  // py_round edge cases beyond the fixture.
  MM_CHECK(r::py_round(0.125, 2) == "0.12" && r::py_round(0.375, 2) == "0.38" && r::py_round(-0.0, 2) == "-0.0");
  MM_CHECK(r::py_round(3.0, 4) == "3.0" && r::py_round(0.00005, 4) == "0.0001" && r::py_round(1e-9, 3) == "0.0");

  // The frame-objects rule (ADR-006).
  const std::vector<std::string> names = {"person", "bicycle", "car"};
  const std::vector<Detection> dets = {{10, 10, 50, 90, 0.9F, 0}, {100, 100, 140, 180, 0.25F, 2},
                                       {200, 50, 230, 120, 0.4F, 2}, {-5, 0, 0, 30, 0.8F, 0}};
  {
    const std::vector<TrackOutput> outs = {{{-3.0F, 10.0F, 1285.0F, 90.0F}, 4, 0.9F, 0, 0},
                                           {{1300.0F, 10.0F, 1310.0F, 20.0F}, 5, 0.8F, 2, 2}};
    const auto fo = r::frame_objects(outs, false, dets, 0.25F, 1280, 720, names);
    MM_CHECK(fo.objects.size() == 1 && fo.dropped_no_extent == 1);  // the second clamps to zero width
    MM_CHECK(fo.objects[0].track_id == 4 && fo.objects[0].bbox_xyxy[0] == 0.0 && fo.objects[0].bbox_xyxy[2] == 1280.0);
    MM_CHECK(fo.objects[0].class_name == "person");
  }
  {
    const auto pending = r::frame_objects({}, true, dets, 0.25F, 1280, 720, names);
    MM_CHECK(pending.objects.empty() && pending.dropped_no_extent == 0);  // unconfirmed pending: nothing
    const auto fallback = r::frame_objects({}, false, dets, 0.25F, 1280, 720, names);
    // score > conf only (0.25 is not > 0.25), in detection order, null IDs; the zero-width one is dropped.
    MM_CHECK(fallback.objects.size() == 2 && fallback.dropped_no_extent == 1);
    MM_CHECK(!fallback.objects[0].track_id && fallback.objects[0].confidence == static_cast<double>(0.9F));
    MM_CHECK(fallback.objects[1].class_name == "car" && fallback.objects[1].bbox_xyxy[0] == 200.0);
  }
  bool threw = false;
  try {
    r::frame_objects({}, false, std::vector<Detection>{{1, 1, 5, 5, 0.9F, 7}}, 0.25F, 1280, 720, names);
  } catch (const r::RecordError&) {
    threw = true;
  }
  MM_CHECK(threw);  // an unknown class id is an error, never a made-up name
  return motionman::test::failures == 0 ? 0 : 1;
}
