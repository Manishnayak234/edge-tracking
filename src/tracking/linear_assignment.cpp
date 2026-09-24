#include "linear_assignment.hpp"

#include <cstddef>
#include <limits>

namespace edge_tracking::tracking {

AssignmentResult linear_assignment(const std::vector<float>& cost, int rows, int cols, double cost_limit) {
    AssignmentResult result;
    if (rows == 0 || cols == 0) {
        for (int i = 0; i < rows; ++i) result.unmatched_rows.push_back(i);
        for (int j = 0; j < cols; ++j) result.unmatched_cols.push_back(j);
        return result;
    }

    // Extended square matrix: [cost, limit/2; limit/2, 0]. Leaving row i and column j both
    // unmatched costs limit/2 + limit/2, so a real pair is only used if it costs less.
    const int n = rows + cols;
    auto a = [&](int i, int j) -> double {
        if (i < rows && j < cols) return cost[static_cast<std::size_t>(i) * cols + j];
        if (i >= rows && j >= cols) return 0.0;
        return cost_limit / 2.0;
    };

    // Hungarian algorithm with potentials, O(n^3); 1-based, column 0 is a sentinel.
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);  // p[j]: row assigned to column j
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, inf);
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = a(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> row_to_col(rows, -1);
    std::vector<bool> col_matched(cols, false);
    for (int j = 1; j <= cols; ++j) {
        const int i = p[j] - 1;
        if (i < rows) {
            row_to_col[i] = j - 1;
            col_matched[j - 1] = true;
        }
    }
    for (int i = 0; i < rows; ++i) {
        if (row_to_col[i] >= 0) {
            result.matches.emplace_back(i, row_to_col[i]);
        } else {
            result.unmatched_rows.push_back(i);
        }
    }
    for (int j = 0; j < cols; ++j) {
        if (!col_matched[j]) result.unmatched_cols.push_back(j);
    }
    return result;
}

}  // namespace edge_tracking::tracking
