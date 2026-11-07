#include "app/annotate.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace motionman::app {
namespace {

// BGR, as render.PALETTE and render.NO_ID_COLOR.
const cv::Scalar kPalette[] = {{53, 230, 163}, {238, 211, 34}, {36, 191, 251}, {250, 139, 167}};
const cv::Scalar kNoId{170, 161, 161};
const cv::Scalar kText{5, 17, 11};
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

cv::Scalar color_for(const std::optional<std::int64_t>& id) { return id ? kPalette[*id % 4] : kNoId; }

int py_round_int(double v) { return static_cast<int>(std::nearbyint(v)); }  // round(): nearest, ties to even

void label(cv::Mat& img, const std::string& text, cv::Point origin, const cv::Scalar& color, double scale,
           int thickness) {
  int base = 0;
  const cv::Size size = cv::getTextSize(text, kFont, scale, thickness, &base);
  const int x = origin.x;
  const int y = origin.y;
  const int top = y - size.height - base - 4 >= 0 ? y - size.height - base - 4 : y;  // flip below at the top edge
  cv::rectangle(img, {x, top}, {x + size.width + 4, top + size.height + base + 4}, color, cv::FILLED);
  cv::putText(img, text, {x + 2, top + size.height + 2}, kFont, scale, kText, thickness, cv::LINE_AA);
}

}  // namespace

void annotate(cv::Mat& frame, std::span<const records::RecordObject> objects, const records::Trails& trails) {
  const double scale = std::max(0.4, std::min(frame.rows, frame.cols) / 1000.0);
  const int thickness = std::max(2, py_round_int(scale * 3));
  for (const auto& o : objects) {
    if (!o.track_id) continue;
    const auto pts = trails.points(*o.track_id);
    if (pts.size() < 2) continue;
    std::vector<cv::Point> line;
    line.reserve(pts.size());
    for (const auto& [px, py] : pts) line.emplace_back(px, py);
    cv::polylines(frame, line, false, color_for(o.track_id), thickness, cv::LINE_AA);
  }
  for (const auto& o : objects) {
    const cv::Scalar color = color_for(o.track_id);
    const auto& b = o.bbox_xyxy;
    const cv::Point p1{py_round_int(b[0]), py_round_int(b[1])};
    cv::rectangle(frame, p1, {py_round_int(b[2]), py_round_int(b[3])}, color, thickness);
    char conf[32];
    std::snprintf(conf, sizeof conf, "%.2f", o.confidence);
    const std::string prefix = o.track_id ? "#" + std::to_string(*o.track_id) : "#?";
    label(frame, prefix + " " + o.class_name + " " + conf, p1, color, scale, thickness);
  }
}

}  // namespace motionman::app
