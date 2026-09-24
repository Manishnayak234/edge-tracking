#pragma once

#include <array>

namespace edge_tracking::tracking {

using Vec4 = std::array<double, 4>;
using Vec8 = std::array<double, 8>;
using Mat8 = std::array<std::array<double, 8>, 8>;

// Constant-velocity Kalman filter over (cx, cy, a = w / h, h) and their velocities.
// Port of Ultralytics KalmanFilterXYAH: noise scales with box height, time step 1 frame.
class KalmanFilterXYAH {
public:
    // New track from a first measurement (cx, cy, a, h): zero velocity, large uncertainty.
    void initiate(const Vec4& measurement, Vec8& mean, Mat8& covariance) const;
    // Advance one frame.
    void predict(Vec8& mean, Mat8& covariance) const;
    // Correct with a measurement (cx, cy, a, h).
    void update(Vec8& mean, Mat8& covariance, const Vec4& measurement) const;

private:
    static constexpr double kStdWeightPosition = 1.0 / 20.0;
    static constexpr double kStdWeightVelocity = 1.0 / 160.0;
};

}  // namespace edge_tracking::tracking
