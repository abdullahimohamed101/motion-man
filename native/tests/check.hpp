// A minimal check macro for CTest executables (plan D4). A failed check prints its location; main returns non-zero.
#pragma once

#include <cstdio>

namespace motionman::test {
inline int failures = 0;
}

#define MM_CHECK(cond)                                                          \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++motionman::test::failures;                                              \
    }                                                                           \
  } while (0)
