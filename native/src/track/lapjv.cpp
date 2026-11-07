// Jonker-Volgenant dense assignment (see lapjv.hpp). Phases as in lap's lapjv: column reduction with reduction
// transfer, two rounds of augmenting row reduction, then shortest augmenting paths for the remaining free rows.
#include "motionman/track/lapjv.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace motionman::track {
namespace {

constexpr double kLarge = std::numeric_limits<double>::max();

struct Problem {
  std::size_t n;
  const std::vector<double>& cost;
  double c(std::size_t i, std::size_t j) const { return cost[i * n + j]; }
};

// Column reduction and reduction transfer. Returns the free rows.
std::vector<int> column_reduction(const Problem& p, std::vector<int>& x, std::vector<int>& y, std::vector<double>& v) {
  const std::size_t n = p.n;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = -1;
    v[i] = kLarge;
    y[i] = 0;
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double c = p.c(i, j);
      if (c < v[j]) {
        v[j] = c;
        y[j] = static_cast<int>(i);
      }
    }
  }
  std::vector<bool> unique(n, true);
  std::size_t j = n;
  do {
    --j;
    const int i = y[j];
    if (x[i] < 0) {
      x[i] = static_cast<int>(j);
    } else {
      unique[i] = false;
      y[j] = -1;
    }
  } while (j > 0);
  std::vector<int> free_rows;
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] < 0) {
      free_rows.push_back(static_cast<int>(i));
    } else if (unique[i]) {
      const auto jj = static_cast<std::size_t>(x[i]);
      double min = kLarge;
      for (std::size_t j2 = 0; j2 < n; ++j2) {
        if (j2 == jj) continue;
        const double c = p.c(i, j2) - v[j2];
        if (c < min) min = c;
      }
      v[jj] -= min;
    }
  }
  return free_rows;
}

// Augmenting row reduction. Rewrites free_rows in place and returns how many remain free.
std::size_t augmenting_row_reduction(const Problem& p, std::size_t n_free, std::vector<int>& free_rows,
                                     std::vector<int>& x, std::vector<int>& y, std::vector<double>& v) {
  const std::size_t n = p.n;
  std::size_t current = 0;
  std::size_t new_free = 0;
  std::size_t rr_cnt = 0;
  while (current < n_free) {
    ++rr_cnt;
    const auto free_i = static_cast<std::size_t>(free_rows[current++]);
    std::size_t j1 = 0;
    double v1 = p.c(free_i, 0) - v[0];
    int j2 = -1;
    double v2 = kLarge;
    for (std::size_t j = 1; j < n; ++j) {
      const double c = p.c(free_i, j) - v[j];
      if (c < v2) {
        if (c >= v1) {
          v2 = c;
          j2 = static_cast<int>(j);
        } else {
          v2 = v1;
          v1 = c;
          j2 = static_cast<int>(j1);
          j1 = j;
        }
      }
    }
    int i0 = y[j1];
    const double v1_new = v[j1] - (v2 - v1);
    const bool v1_lowers = v1_new < v[j1];
    if (rr_cnt < current * n) {
      if (v1_lowers) {
        v[j1] = v1_new;
      } else if (i0 >= 0 && j2 >= 0) {
        j1 = static_cast<std::size_t>(j2);
        i0 = y[j1];
      }
      if (i0 >= 0) {
        if (v1_lowers) {
          free_rows[--current] = i0;
        } else {
          free_rows[new_free++] = i0;
        }
      }
    } else if (i0 >= 0) {
      free_rows[new_free++] = i0;
    }
    x[free_i] = static_cast<int>(j1);
    y[j1] = static_cast<int>(free_i);
  }
  return new_free;
}

// Columns cols[lo..hi) get the minimum d; returns the new hi.
std::size_t find_min_columns(std::size_t n, std::size_t lo, const std::vector<double>& d, std::vector<int>& cols) {
  std::size_t hi = lo + 1;
  double mind = d[cols[lo]];
  for (std::size_t k = hi; k < n; ++k) {
    const int j = cols[k];
    if (d[j] <= mind) {
      if (d[j] < mind) {
        hi = lo;
        mind = d[j];
      }
      cols[k] = cols[hi];
      cols[hi++] = j;
    }
  }
  return hi;
}

// Scans the columns in cols[lo..hi); returns a free column reached at the minimum distance, or -1.
int scan_columns(const Problem& p, std::size_t& plo, std::size_t& phi, std::vector<double>& d, std::vector<int>& cols,
                 std::vector<int>& pred, const std::vector<int>& y, const std::vector<double>& v) {
  std::size_t lo = plo;
  std::size_t hi = phi;
  while (lo != hi) {
    int j = cols[lo++];
    const auto i = static_cast<std::size_t>(y[j]);
    const double mind = d[j];
    const double h = p.c(i, j) - v[j] - mind;
    for (std::size_t k = hi; k < p.n; ++k) {
      j = cols[k];
      const double cred = p.c(i, j) - v[j] - h;
      if (cred < d[j]) {
        d[j] = cred;
        pred[j] = static_cast<int>(i);
        if (cred == mind) {
          if (y[j] < 0) return j;
          cols[k] = cols[hi];
          cols[hi++] = j;
        }
      }
    }
  }
  plo = lo;
  phi = hi;
  return -1;
}

// Dijkstra-style shortest augmenting path from start_i; updates the duals of the scanned columns.
int find_path(const Problem& p, int start_i, const std::vector<int>& y, std::vector<double>& v,
              std::vector<int>& pred) {
  const std::size_t n = p.n;
  std::size_t lo = 0;
  std::size_t hi = 0;
  int final_j = -1;
  std::size_t n_ready = 0;
  std::vector<int> cols(n);
  std::vector<double> d(n);
  for (std::size_t j = 0; j < n; ++j) {
    cols[j] = static_cast<int>(j);
    pred[j] = start_i;
    d[j] = p.c(static_cast<std::size_t>(start_i), j) - v[j];
  }
  while (final_j == -1) {
    if (lo == hi) {
      n_ready = lo;
      hi = find_min_columns(n, lo, d, cols);
      for (std::size_t k = lo; k < hi; ++k) {
        const int j = cols[k];
        if (y[j] < 0) final_j = j;
      }
    }
    if (final_j == -1) final_j = scan_columns(p, lo, hi, d, cols, pred, y, v);
  }
  const double mind = d[cols[lo]];
  for (std::size_t k = 0; k < n_ready; ++k) {
    const int j = cols[k];
    v[j] += d[j] - mind;
  }
  return final_j;
}

void augment(const Problem& p, std::size_t n_free, const std::vector<int>& free_rows, std::vector<int>& x,
             std::vector<int>& y, std::vector<double>& v) {
  std::vector<int> pred(p.n);
  for (std::size_t f = 0; f < n_free; ++f) {
    const int free_i = free_rows[f];
    int i = -1;
    std::size_t k = 0;
    int j = find_path(p, free_i, y, v, pred);
    while (i != free_i) {
      i = pred[j];
      y[j] = i;
      std::swap(j, x[i]);
      if (++k > p.n) throw std::logic_error("lapjv: augmenting path longer than n");
    }
  }
}

}  // namespace

Assignment lapjv_square(std::size_t n, const std::vector<double>& cost) {
  if (cost.size() != n * n) throw std::invalid_argument("lapjv: cost matrix is not n x n");
  Assignment a{std::vector<int>(n, -1), std::vector<int>(n, -1)};
  if (n == 0) return a;
  const Problem p{n, cost};
  std::vector<double> v(n);
  std::vector<int> free_rows = column_reduction(p, a.row_to_col, a.col_to_row, v);
  std::size_t n_free = free_rows.size();
  for (int round = 0; n_free > 0 && round < 2; ++round) {
    n_free = augmenting_row_reduction(p, n_free, free_rows, a.row_to_col, a.col_to_row, v);
  }
  if (n_free > 0) augment(p, n_free, free_rows, a.row_to_col, a.col_to_row, v);
  return a;
}

Assignment lapjv_limited(std::size_t rows, std::size_t cols, const std::vector<double>& cost, double limit) {
  if (cost.size() != rows * cols) throw std::invalid_argument("lapjv: cost matrix size does not match its shape");
  const std::size_t n = rows + cols;
  std::vector<double> ext(n * n, limit / 2.0);
  for (std::size_t i = rows; i < n; ++i) {
    for (std::size_t j = cols; j < n; ++j) ext[i * n + j] = 0.0;
  }
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) ext[i * n + j] = cost[i * cols + j];
  }
  const Assignment full = lapjv_square(n, ext);
  Assignment a{std::vector<int>(rows, -1), std::vector<int>(cols, -1)};
  for (std::size_t i = 0; i < rows; ++i) {
    if (full.row_to_col[i] < static_cast<int>(cols)) a.row_to_col[i] = full.row_to_col[i];
  }
  for (std::size_t j = 0; j < cols; ++j) {
    if (full.col_to_row[j] < static_cast<int>(rows)) a.col_to_row[j] = full.col_to_row[j];
  }
  return a;
}

}  // namespace motionman::track
