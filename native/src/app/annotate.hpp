// Annotated video drawing (plan native-g2 D8): the rules of motionman/render.py `annotate`, in C++ with OpenCV.
// Same palette per ID, grey for no ID, label "#<id> <class> <conf:.2f>", trails only for objects visible this frame.
// Checked by frame count, not pixels.
#pragma once

#include <opencv2/core.hpp>

#include <span>

#include "motionman/records.hpp"

namespace motionman::app {

// Draws trails, boxes and labels onto `frame` in place (it is not used again after drawing).
void annotate(cv::Mat& frame, std::span<const records::RecordObject> objects, const records::Trails& trails);

}  // namespace motionman::app
