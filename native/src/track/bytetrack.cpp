// Native ByteTrack; see bytetrack.hpp for the reference behaviour this keeps. Compiled with -ffp-contract=off so
// the float32 steps round exactly as numpy's do.
#include "motionman/track/bytetrack.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "motionman/track/lapjv.hpp"

namespace motionman::track {

enum class State { New, Tracked, Lost, Removed };

struct ByteTracker::Track {
  KalmanState kf;
  bool has_mean = false;
  std::array<float, 4> tlwh{};  // the detection's box, used until the track has a Kalman state
  float score = 0;
  int class_id = 0;
  int detection_index = 0;
  std::int64_t track_id = 0;
  State state = State::New;
  bool is_activated = false;
  int frame_id = 0;
  int start_frame = 0;
  int tracklet_len = 0;
};

namespace {

using TrackPtr = std::shared_ptr<ByteTracker::Track>;
using Track = ByteTracker::Track;
constexpr std::size_t kRemovedBuffer = 1000;

// A detection as the reference's STrack holds it: xywh in float32, then left-top in double cast to float32.
TrackPtr from_detection(const Detection& d, int index) {
  const float cx = (d.x1 + d.x2) / 2;
  const float cy = (d.y1 + d.y2) / 2;
  const float w = d.x2 - d.x1;
  const float h = d.y2 - d.y1;
  auto t = std::make_shared<Track>();
  t->tlwh = {static_cast<float>(static_cast<double>(cx) - static_cast<double>(w) / 2),
             static_cast<float>(static_cast<double>(cy) - static_cast<double>(h) / 2), w, h};
  t->score = d.score;
  t->class_id = d.class_id;
  t->detection_index = index;
  return t;
}

bool valid_box(const Detection& d) { return d.x2 - d.x1 > 0 && d.y2 - d.y1 > 0; }

// The measurement (cx, cy, aspect, height), computed in float32 from the detection box.
Vec4 measurement(const Track& det) {
  const auto& b = det.tlwh;
  const float cx = b[0] + b[2] / 2;
  const float cy = b[1] + b[3] / 2;
  const float a = b[2] / b[3];
  return {cx, cy, a, b[3]};
}

// The box used for IoU: from the Kalman mean (double) if the track has one, else the detection's float32 box.
std::array<float, 4> iou_box(const Track& t) {
  if (!t.has_mean) {
    const auto& b = t.tlwh;
    return {b[0], b[1], b[2] + b[0], b[3] + b[1]};
  }
  const auto& m = t.kf.mean;
  const double w = m[2] * m[3];
  const double h = m[3];
  const double l = m[0] - w / 2;
  const double top = m[1] - h / 2;
  return {static_cast<float>(l), static_cast<float>(top), static_cast<float>(w + l), static_cast<float>(h + top)};
}

// 1 - IoU for every pair, float32, with the reference's order of operations and eps.
std::vector<float> iou_distance(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b) {
  std::vector<float> out(a.size() * b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto p = iou_box(*a[i]);
    const float area_p = (p[2] - p[0]) * (p[3] - p[1]);
    for (std::size_t j = 0; j < b.size(); ++j) {
      const auto q = iou_box(*b[j]);
      const float iw = std::max(std::min(p[2], q[2]) - std::max(p[0], q[0]), 0.0F);
      const float ih = std::max(std::min(p[3], q[3]) - std::max(p[1], q[1]), 0.0F);
      const float inter = iw * ih;
      const float area_q = (q[2] - q[0]) * (q[3] - q[1]);
      const float uni = area_q + area_p - inter;
      out[i * b.size() + j] = 1.0F - inter / (uni + 1e-7F);
    }
  }
  return out;
}

std::vector<float> distances(const std::vector<TrackPtr>& tracks, const std::vector<TrackPtr>& dets, bool fuse,
                             ClassPolicy policy) {
  std::vector<float> d = iou_distance(tracks, dets);
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    for (std::size_t j = 0; j < dets.size(); ++j) {
      float& c = d[i * dets.size() + j];
      if (fuse) c = 1.0F - (1.0F - c) * dets[j]->score;
      if (policy == ClassPolicy::Aware && tracks[i]->class_id != dets[j]->class_id) c = 1.0F;  // never matches
    }
  }
  return d;
}

struct Matching {
  std::vector<std::pair<std::size_t, std::size_t>> matches;
  std::vector<std::size_t> unmatched_a;
  std::vector<std::size_t> unmatched_b;
};

// Ultralytics' linear_assignment with lap: matches in row order, unmatched indices ascending.
Matching assign(const std::vector<float>& cost, std::size_t rows, std::size_t cols, float thresh) {
  Matching m;
  if (rows == 0 || cols == 0) {
    for (std::size_t i = 0; i < rows; ++i) m.unmatched_a.push_back(i);
    for (std::size_t j = 0; j < cols; ++j) m.unmatched_b.push_back(j);
    return m;
  }
  const std::vector<double> c(cost.begin(), cost.end());
  const Assignment a = lapjv_limited(rows, cols, c, static_cast<double>(thresh));
  for (std::size_t i = 0; i < rows; ++i) {
    if (a.row_to_col[i] >= 0) {
      m.matches.emplace_back(i, static_cast<std::size_t>(a.row_to_col[i]));
    } else {
      m.unmatched_a.push_back(i);
    }
  }
  for (std::size_t j = 0; j < cols; ++j) {
    if (a.col_to_row[j] < 0) m.unmatched_b.push_back(j);
  }
  return m;
}

void update_track(Track& t, const Track& det, int frame_id) {
  t.frame_id = frame_id;
  t.tracklet_len += 1;
  kalman_update(t.kf, measurement(det));
  t.state = State::Tracked;
  t.is_activated = true;
  t.score = det.score;
  t.class_id = det.class_id;
  t.detection_index = det.detection_index;
}

void reactivate_track(Track& t, const Track& det, int frame_id) {
  kalman_update(t.kf, measurement(det));
  t.tracklet_len = 0;
  t.state = State::Tracked;
  t.is_activated = true;
  t.frame_id = frame_id;
  t.score = det.score;
  t.class_id = det.class_id;
  t.detection_index = det.detection_index;
}

// Union by track ID; entries of `a` win.
std::vector<TrackPtr> joint(const std::vector<TrackPtr>& a, const std::vector<TrackPtr>& b) {
  std::unordered_set<std::int64_t> ids;
  for (const auto& t : a) ids.insert(t->track_id);
  std::vector<TrackPtr> out = a;
  for (const auto& t : b) {
    if (!ids.count(t->track_id)) out.push_back(t);
  }
  return out;
}

template <class Ids>
std::vector<TrackPtr> without_ids(const std::vector<TrackPtr>& a, const Ids& ids) {
  std::vector<TrackPtr> out;
  for (const auto& t : a) {
    if (std::find(ids.begin(), ids.end(), t->track_id) == ids.end()) out.push_back(t);
  }
  return out;
}

std::vector<std::int64_t> ids_of(const std::vector<TrackPtr>& a) {
  std::vector<std::int64_t> out;
  for (const auto& t : a) out.push_back(t->track_id);
  return out;
}

}  // namespace

ByteTracker::ByteTracker(Config config) : config_(config) {}
ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

bool ByteTracker::has_unconfirmed() const {
  return std::any_of(tracked_.begin(), tracked_.end(), [](const TrackPtr& t) { return !t->is_activated; });
}

std::vector<TrackOutput> ByteTracker::update(std::span<const Detection> detections) {
  ++frame_id_;
  const ClassPolicy policy = config_.class_policy;
  std::vector<TrackPtr> activated, refind, lost, removed;

  std::vector<TrackPtr> dets, dets_low;
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    if (!valid_box(d)) continue;  // h = 0 would make the aspect ratio infinite
    if (d.score >= config_.track_high_thresh) {
      dets.push_back(from_detection(d, static_cast<int>(i)));
    } else if (d.score > config_.track_low_thresh) {
      dets_low.push_back(from_detection(d, static_cast<int>(i)));
    }
  }

  std::vector<TrackPtr> unconfirmed, confirmed;
  for (const auto& t : tracked_) (t->is_activated ? confirmed : unconfirmed).push_back(t);
  const std::vector<TrackPtr> pool = joint(confirmed, lost_);
  for (const auto& t : pool) kalman_predict(t->kf, t->state == State::Tracked);

  // A tracked track is updated (activated); a lost one is re-activated (refind).
  auto apply = [&](const TrackPtr& t, const Track& det) {
    if (t->state == State::Tracked) {
      update_track(*t, det, frame_id_);
      activated.push_back(t);
    } else {
      reactivate_track(*t, det, frame_id_);
      refind.push_back(t);
    }
  };

  // First association: confirmed and lost tracks against high-score detections.
  const Matching first =
      assign(distances(pool, dets, config_.fuse_score, policy), pool.size(), dets.size(), config_.match_thresh);
  for (const auto& [it, id] : first.matches) apply(pool[it], *dets[id]);

  // Second association: still-tracked leftovers against low-score detections, IoU only.
  std::vector<TrackPtr> remaining;
  for (std::size_t i : first.unmatched_a) {
    if (pool[i]->state == State::Tracked) remaining.push_back(pool[i]);
  }
  std::vector<std::size_t> still_unmatched;
  if (!remaining.empty() && !dets_low.empty()) {
    const Matching second =
        assign(distances(remaining, dets_low, false, policy), remaining.size(), dets_low.size(), 0.5F);
    for (const auto& [it, id] : second.matches) apply(remaining[it], *dets_low[id]);
    still_unmatched = second.unmatched_a;
  } else {
    for (std::size_t i = 0; i < remaining.size(); ++i) still_unmatched.push_back(i);
  }
  for (std::size_t i : still_unmatched) {
    remaining[i]->state = State::Lost;
    lost.push_back(remaining[i]);
  }

  // Unconfirmed tracks against the high-score detections left over.
  std::vector<TrackPtr> left;
  for (std::size_t j : first.unmatched_b) left.push_back(dets[j]);
  std::vector<std::size_t> new_candidates;
  if (unconfirmed.empty()) {
    for (std::size_t j = 0; j < left.size(); ++j) new_candidates.push_back(j);
  } else {
    const Matching third =
        assign(distances(unconfirmed, left, config_.fuse_score, policy), unconfirmed.size(), left.size(), 0.7F);
    for (const auto& [it, id] : third.matches) {
      update_track(*unconfirmed[it], *left[id], frame_id_);
      activated.push_back(unconfirmed[it]);
    }
    for (std::size_t i : third.unmatched_a) {
      unconfirmed[i]->state = State::Removed;
      removed.push_back(unconfirmed[i]);
    }
    new_candidates = third.unmatched_b;
  }

  // New tracks. Only the first frame's are confirmed at once.
  for (std::size_t j : new_candidates) {
    Track& t = *left[j];
    if (t.score < config_.new_track_thresh) continue;
    t.track_id = next_id_++;
    t.kf = kalman_initiate(measurement(t));
    t.has_mean = true;
    t.tracklet_len = 0;
    t.state = State::Tracked;
    t.is_activated = frame_id_ == 1;
    t.frame_id = frame_id_;
    t.start_frame = frame_id_;
    activated.push_back(left[j]);
  }

  for (const auto& t : lost_) {
    if (frame_id_ - t->frame_id > config_.track_buffer) {
      t->state = State::Removed;
      removed.push_back(t);
    }
  }

  // End-of-frame bookkeeping, in the reference's order.
  std::vector<TrackPtr> kept;
  for (const auto& t : tracked_) {
    if (t->state == State::Tracked) kept.push_back(t);
  }
  tracked_ = joint(joint(kept, activated), refind);
  lost_ = without_ids(lost_, ids_of(tracked_));
  lost_.insert(lost_.end(), lost.begin(), lost.end());
  lost_ = without_ids(lost_, removed_ids_);  // removals from earlier frames only: this frame's leave next frame

  // Duplicate tracked/lost pairs (1 - IoU < 0.15): keep the one tracked for longer.
  {
    const std::vector<float> d = iou_distance(tracked_, lost_);
    std::unordered_set<std::size_t> drop_a, drop_b;
    for (std::size_t p = 0; p < tracked_.size(); ++p) {
      for (std::size_t q = 0; q < lost_.size(); ++q) {
        if (!(d[p * lost_.size() + q] < 0.15F)) continue;
        if (policy == ClassPolicy::Aware && tracked_[p]->class_id != lost_[q]->class_id) continue;
        const int tp = tracked_[p]->frame_id - tracked_[p]->start_frame;
        const int tq = lost_[q]->frame_id - lost_[q]->start_frame;
        (tp > tq ? drop_b : drop_a).insert(tp > tq ? q : p);
      }
    }
    std::vector<TrackPtr> a, b;
    for (std::size_t p = 0; p < tracked_.size(); ++p) {
      if (!drop_a.count(p)) a.push_back(tracked_[p]);
    }
    for (std::size_t q = 0; q < lost_.size(); ++q) {
      if (!drop_b.count(q)) b.push_back(lost_[q]);
    }
    tracked_ = std::move(a);
    lost_ = std::move(b);
  }
  for (const auto& t : removed) removed_ids_.push_back(t->track_id);
  while (removed_ids_.size() > kRemovedBuffer) removed_ids_.pop_front();

  std::vector<TrackOutput> out;
  for (const auto& t : tracked_) {
    if (!t->is_activated) continue;
    out.push_back({iou_box(*t), t->track_id, t->score, t->class_id, t->detection_index});
  }
  return out;
}

}  // namespace motionman::track
