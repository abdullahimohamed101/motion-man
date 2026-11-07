#include "motionman/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace motionman::json {
namespace {

class Parser {
 public:
  explicit Parser(std::string_view t) : t_(t) {}

  Value document() {
    Value v = value();
    ws();
    if (i_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("json: ") + what + " at byte " + std::to_string(i_));
  }
  void ws() {
    while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\n' || t_[i_] == '\r' || t_[i_] == '\t')) ++i_;
  }
  bool eat(char c) {
    ws();
    if (i_ < t_.size() && t_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail("unexpected character");
  }
  bool literal(std::string_view word) {
    if (t_.substr(i_, word.size()) != word) return false;
    i_ += word.size();
    return true;
  }

  Value value() {
    ws();
    if (i_ >= t_.size()) fail("unexpected end");
    Value v;
    const char c = t_[i_];
    if (c == '{') {
      ++i_;
      v.kind = Value::Kind::Object;
      if (eat('}')) return v;
      do {
        ws();
        v.keys.push_back(string());
        expect(':');
        v.items.push_back(value());
      } while (eat(','));
      expect('}');
    } else if (c == '[') {
      ++i_;
      v.kind = Value::Kind::Array;
      if (eat(']')) return v;
      do v.items.push_back(value());
      while (eat(','));
      expect(']');
    } else if (c == '"') {
      v.kind = Value::Kind::String;
      v.string = string();
    } else if (literal("true")) {
      v.kind = Value::Kind::Bool;
      v.boolean = true;
    } else if (literal("false")) {
      v.kind = Value::Kind::Bool;
    } else if (literal("null")) {
      v.kind = Value::Kind::Null;
    } else {
      v.kind = Value::Kind::Number;
      const std::size_t start = i_;
      while (i_ < t_.size() && std::string_view("+-0123456789.eE").find(t_[i_]) != std::string_view::npos) ++i_;
      if (start == i_) fail("unexpected character");
      const std::string num(t_.substr(start, i_ - start));
      char* end = nullptr;
      v.number = std::strtod(num.c_str(), &end);
      if (end != num.c_str() + num.size()) fail("bad number");
    }
    return v;
  }

  static void utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  unsigned hex4() {
    if (i_ + 4 > t_.size()) fail("bad \\u escape");
    const std::string h(t_.substr(i_, 4));
    char* end = nullptr;
    const unsigned v = static_cast<unsigned>(std::strtoul(h.c_str(), &end, 16));
    if (end != h.c_str() + 4) fail("bad \\u escape");
    i_ += 4;
    return v;
  }
  std::string string() {
    if (i_ >= t_.size() || t_[i_] != '"') fail("expected string");
    ++i_;
    std::string out;
    while (true) {
      if (i_ >= t_.size()) fail("unterminated string");
      const char c = t_[i_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= t_.size()) fail("unterminated escape");
      const char e = t_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && literal("\\u")) {
            const unsigned lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }

  std::string_view t_;
  std::size_t i_ = 0;
};

}  // namespace

const Value* Value::find(std::string_view key) const {
  if (kind != Kind::Object) return nullptr;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] == key) return &items[i];
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (!v) throw std::runtime_error("json: missing key \"" + std::string(key) + "\"");
  return *v;
}

const std::string& Value::str() const {
  if (kind != Kind::String) throw std::runtime_error("json: expected a string");
  return string;
}

double Value::num() const {
  if (kind != Kind::Number) throw std::runtime_error("json: expected a number");
  return number;
}

Value parse(std::string_view text) { return Parser(text).document(); }

Value parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("json: cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

}  // namespace motionman::json
