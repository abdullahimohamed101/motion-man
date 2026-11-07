// Native LAPJV vs the `lap` package on 400 recorded cases, ties included (plan native-g2 step 3).
// usage: test_lapjv <tests/fixtures/lapjv.json>
#include <cstdio>
#include <vector>

#include "../check.hpp"
#include "motionman/json.hpp"
#include "motionman/track/lapjv.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_lapjv <lapjv.json>\n");
    return 2;
  }
  namespace j = motionman::json;
  const j::Value doc = j::parse_file(argv[1]);
  MM_CHECK(doc.at("schema").str() == "motionman.lapjv/1");
  int n_cases = 0;
  int agree = 0;
  for (const j::Value& c : doc.at("cases").items) {
    const auto& rows = c.at("cost").items;
    const std::size_t r = rows.size();
    const std::size_t k = rows.at(0).items.size();
    std::vector<double> cost;
    for (const auto& row : rows) {
      for (const auto& v : row.items) cost.push_back(static_cast<double>(static_cast<float>(v.num())));
    }
    const auto a = motionman::track::lapjv_limited(r, k, cost, c.at("limit").num());
    bool same = a.row_to_col.size() == c.at("x").items.size() && a.col_to_row.size() == c.at("y").items.size();
    for (std::size_t i = 0; same && i < r; ++i) same = a.row_to_col[i] == static_cast<int>(c.at("x").items[i].num());
    for (std::size_t i = 0; same && i < k; ++i) same = a.col_to_row[i] == static_cast<int>(c.at("y").items[i].num());
    if (!same && n_cases - agree < 5) std::fprintf(stderr, "case %d (%zux%zu) differs from lap\n", n_cases, r, k);
    agree += same ? 1 : 0;
    ++n_cases;
  }
  std::printf("lapjv: %d/%d cases match lap\n", agree, n_cases);
  MM_CHECK(n_cases == 400 && agree == n_cases);
  return motionman::test::failures == 0 ? 0 : 1;
}
