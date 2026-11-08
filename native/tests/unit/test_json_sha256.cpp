// Portable checks of the engine's JSON reader and SHA-256 (plan native-g0-g1 step 9). Runs on the Mac and the PC.
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "../check.hpp"
#include "motionman/json.hpp"
#include "motionman/sha256.hpp"

namespace {
bool throws(const char* text) {
  try {
    motionman::json::parse(text);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}
}  // namespace

int main(int argc, char** argv) {
  namespace j = motionman::json;
  // FIPS 180-4 test vectors.
  const std::string abc = "abc";
  MM_CHECK(motionman::sha256_hex(reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size()) ==
           "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  MM_CHECK(motionman::sha256_hex(nullptr, 0) ==
           "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  MM_CHECK(motionman::sha256_hex(reinterpret_cast<const std::uint8_t*>(two_blocks.data()), two_blocks.size()) ==
           "sha256:248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

  const auto v = j::parse(R"( {"a": [1, -2.5e3, true, false, null], "s": "x\"y\\z§😀", "o": {}} )");
  MM_CHECK(v.at("a").items.size() == 5);
  MM_CHECK(v.at("a").items[1].num() == -2500.0);
  MM_CHECK(v.at("a").items[2].boolean && !v.at("a").items[3].boolean);
  MM_CHECK(v.at("a").items[4].kind == j::Value::Kind::Null);
  MM_CHECK(v.at("s").str() == "x\"y\\z\xc2\xa7\xf0\x9f\x98\x80");
  MM_CHECK(v.at("o").kind == j::Value::Kind::Object && v.at("o").keys.empty());
  MM_CHECK(v.find("missing") == nullptr);
  MM_CHECK(throws("{\"a\": 1,}") && throws("[1 2]") && throws("\"open") && throws("{} x") && throws(""));
  MM_CHECK(j::quote("a\"b\\c\n") == "\"a\\\"b\\\\c\\n\"");

  if (argc == 2) {  // the real profile parses and its pins are readable
    const auto p = j::parse_file(argv[1]);
    MM_CHECK(p.at("checkpoint").at("sha256").str().rfind("sha256:", 0) == 0);
    MM_CHECK(p.at("parity").at("gates").at("fp32").at("min_iou").num() == 0.99);
  }
  std::printf("json and sha256 checks done\n");
  return motionman::test::failures == 0 ? 0 : 1;
}
