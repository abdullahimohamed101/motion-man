// Linear assignment for ByteTrack association (plan native-g2 step 3, ADR-006).
// Jonker-Volgenant dense solver following the structure of the `lap` package's lapjv (BSD-2-Clause, see
// native/third_party/NOTICE.md), which the Python reference calls through Ultralytics' `linear_assignment`.
// Tie-breaking follows lap's scan order, so equal-cost cases assign the same way; checked against lap's own
// output in tests/fixtures/lapjv.json.
#pragma once

#include <cstddef>
#include <vector>

namespace motionman::track {

struct Assignment {
  std::vector<int> row_to_col;  // -1 = unassigned
  std::vector<int> col_to_row;  // -1 = unassigned
};

// Square dense LAPJV on an n x n row-major cost matrix.
Assignment lapjv_square(std::size_t n, const std::vector<double>& cost);

// lap.lapjv(cost, extend_cost=True, cost_limit=limit): rows x cols matrix extended to (rows+cols)^2 with
// limit/2 outside the real block and 0 in the dummy block; a row or column paired with a dummy is unassigned.
Assignment lapjv_limited(std::size_t rows, std::size_t cols, const std::vector<double>& cost, double limit);

}  // namespace motionman::track
