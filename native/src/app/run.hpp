// `motionman-native run` (plan native-g2 step 5, ADR-006, spec C§7): one video through decode, the G1
// preprocessing, TensorRT, admission, native ByteTrack, records, annotated video and timings, then a receipt.
#pragma once

#include <map>
#include <string>

namespace motionman::app {

// Exit codes: 0 success, 1 error, 2 usage, 3 engine or runspec refused, 130 interrupted (SIGINT or SIGTERM).
// The exit code alone is never success: Python checks native_receipt.json and every output (ADR-006).
int run(const std::map<std::string, std::string>& args);

}  // namespace motionman::app
