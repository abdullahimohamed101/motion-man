// Source time agrees with Python on every shared case (plan native-g0-g1 step 2, NG-3).
// Reads tests/fixtures/source_time.json, which keeps one case per line so no JSON library is needed.
#include <cstdio>
#include <fstream>
#include <string>

#include "../check.hpp"
#include "motionman/source_time.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_source_time <path to source_time.json>\n");
    return 2;
  }
  std::ifstream in(argv[1]);
  MM_CHECK(in.good());
  int cases = 0;
  std::string line;
  while (std::getline(in, line)) {
    long long index = 0;
    long long expected = 0;
    double fps = 0.0;
    if (std::sscanf(line.c_str(), " {\"index\": %lld, \"fps\": %lf, \"t_us\": %lld}", &index, &fps, &expected) != 3) {
      continue;  // not a case line
    }
    ++cases;
    const auto got = motionman::source_t_us(index, fps);
    if (got != expected) {
      std::fprintf(stderr, "index %lld fps %.17g: got %lld, Python %lld\n", index, fps, static_cast<long long>(got),
                   expected);
    }
    MM_CHECK(got == expected);
  }
  MM_CHECK(cases > 100);  // the reader really saw the fixture
  std::printf("%d cases\n", cases);
  return motionman::test::failures == 0 ? 0 : 1;
}
