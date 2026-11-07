// Minimal JSON for the engine's own files: profiles and manifests in, detections out (plan D4: no library download).
// Reads RFC 8259 JSON into a small tree; writes strings with the escapes JSON needs. Not a general-purpose library.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace motionman::json {

struct Value {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<Value> items;       // array elements, or object values
  std::vector<std::string> keys;  // object keys, parallel to items

  // Object member or nullptr. at() throws std::runtime_error naming the missing key.
  const Value* find(std::string_view key) const;
  const Value& at(std::string_view key) const;
  const std::string& str() const;  // throws unless a string
  double num() const;              // throws unless a number
};

Value parse(std::string_view text);  // throws std::runtime_error with the byte offset on malformed input
Value parse_file(const std::string& path);
std::string quote(std::string_view s);  // "..." with JSON escapes

}  // namespace motionman::json
