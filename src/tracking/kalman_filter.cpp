#include "kalman_filter.hpp"

#include <cmath>
#include <utility>

namespace edge_tracking::tracking {

namespace {

using Mat4 = std::array<std::array<double, 4>, 4>;
using Mat84 = std::array<std::array<double, 4>, 8>;

// Inverse of a symmetric positive-definite 4x4 matrix (Gauss-Jordan, partial pivoting).
Mat4 invert(Mat4 a) {
    Mat4 inv{};
    for (int i = 0; i < 4; ++i) inv[i][i] = 1.0;
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;
        }
        std::swap(a[col], a[pivot]);
        std::swap(inv[col], inv[pivot]);
        const double d = a[col][col];
        for (int c = 0; c < 4; ++c) {
            a[col][c] /= d;
            inv[col][c] /= d;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            for (int c = 0; c < 4; ++c) {
                a[r][c] -= f * a[col][c];
                inv[r][c] -= f * inv[col][c];
            }
        }
    }
    return inv;
}

}  // namespace

void KalmanFilterXYAH::initiate(const Vec4& z, Vec8& mean, Mat8& covariance) const {
    const double h = z[3];
    const double std[8] = {2 * kStdWeightPosition * h,  2 * kStdWeightPosition * h,  1e-2,
                           2 * kStdWeightPosition * h,  10 * kStdWeightVelocity * h, 10 * kStdWeightVelocity * h,
                           1e-5,                        10 * kStdWeightVelocity * h};
    for (int i = 0; i < 4; ++i) {
        mean[i] = z[i];
        mean[i + 4] = 0.0;
    }
    covariance = {};
    for (int i = 0; i < 8; ++i) covariance[i][i] = std[i] * std[i];
}

void KalmanFilterXYAH::predict(Vec8& mean, Mat8& covariance) const {
    const double h = mean[3];  // process noise uses the height before the motion step
    const double std[8] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h,
                           kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h};

    // x' = F x, with F = [[I, I], [0, I]]: position += velocity.
    for (int i = 0; i < 4; ++i) mean[i] += mean[i + 4];

    // P' = F P F^T + Q
    Mat8 fp;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) fp[i][j] = covariance[i][j] + (i < 4 ? covariance[i + 4][j] : 0.0);
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) covariance[i][j] = fp[i][j] + (j < 4 ? fp[i][j + 4] : 0.0);
        covariance[i][i] += std[i] * std[i];
    }
}

void KalmanFilterXYAH::update(Vec8& mean, Mat8& covariance, const Vec4& z) const {
    const double h = mean[3];
    const double std[4] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-1, kStdWeightPosition * h};

    // Innovation covariance S = H P H^T + R (H selects the first 4 state entries).
    Mat4 s;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) s[i][j] = covariance[i][j];
        s[i][i] += std[i] * std[i];
    }
    const Mat4 s_inv = invert(s);

    // Kalman gain K = P H^T S^-1 (8x4).
    Mat84 k{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int m = 0; m < 4; ++m) k[i][j] += covariance[i][m] * s_inv[m][j];
        }
    }

    double innovation[4];
    for (int i = 0; i < 4; ++i) innovation[i] = z[i] - mean[i];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) mean[i] += k[i][j] * innovation[j];
    }

    // P' = P - K S K^T
    Mat84 ks{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int m = 0; m < 4; ++m) ks[i][j] += k[i][m] * s[m][j];
        }
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double v = 0.0;
            for (int m = 0; m < 4; ++m) v += ks[i][m] * k[j][m];
            covariance[i][j] -= v;
        }
    }
}

}  // namespace edge_tracking::tracking
