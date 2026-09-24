#pragma once

#include <utility>
#include <vector>

namespace edge_tracking::tracking {

struct AssignmentResult {
    std::vector<std::pair<int, int>> matches;  // (row, col), in row order
    std::vector<int> unmatched_rows;           // ascending
    std::vector<int> unmatched_cols;           // ascending
};

// Minimum-cost matching of rows to columns where any pair costing more than `cost_limit`
// is better left unmatched. Same semantics as lap.lapjv(cost, extend_cost=True,
// cost_limit=...) used by Ultralytics: the matrix is extended with dummy rows/columns at
// cost_limit / 2, then solved exactly (Hungarian algorithm). `cost` is rows x cols, row-major.
AssignmentResult linear_assignment(const std::vector<float>& cost, int rows, int cols, double cost_limit);

}  // namespace edge_tracking::tracking
