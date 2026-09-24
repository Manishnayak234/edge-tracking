#pragma once

#include "edge_tracking/camera/argus_camera.hpp"
#include "edge_tracking/frame/color_space.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace edge_tracking::camera {

// One NV12 frame that stays in the camera's NVMM buffer, exposed to CUDA as device
// pointers (pitch-linear). Nothing is copied.
//
// Only valid until the next ArgusGpuCamera::read() or stop(): the buffer goes back to the
// camera then. Finish (synchronize) any GPU work that reads it before calling read() again.
struct GpuFrame {
    int width = 0;
    int height = 0;
    uint64_t index = 0;   // sequence number of frames delivered by this camera
    int64_t pts_ns = -1;  // GStreamer presentation timestamp, -1 if unknown
    // Capture time on std::chrono::steady_clock (ns since its epoch), converted from the
    // buffer timestamp; -1 if unknown. Used to measure end-to-end latency.
    int64_t capture_time_ns = -1;

    const uint8_t* y = nullptr;   // device pointer, `height` rows of `y_pitch` bytes
    const uint8_t* uv = nullptr;  // device pointer, `height / 2` rows of `uv_pitch` bytes (interleaved UV)
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
    ColorSpace color_space = ColorSpace::Unknown;  // as reported by the NVMM buffer
};

// Captures from a CSI camera through Argus (nvarguscamerasrc), converts to pitch-linear on
// the VIC engine, and maps each NVMM buffer into CUDA through EGL, so frames go from the
// ISP to CUDA kernels without a CPU copy.
class ArgusGpuCamera {
public:
    explicit ArgusGpuCamera(CameraConfig config);
    ~ArgusGpuCamera();

    ArgusGpuCamera(const ArgusGpuCamera&) = delete;
    ArgusGpuCamera& operator=(const ArgusGpuCamera&) = delete;

    // Also initializes the CUDA primary context on the current device.
    bool start(std::string* error = nullptr);
    void stop();

    // Releases the previous frame, then waits up to `timeout` for the next one.
    // Can be called from any thread (one at a time); it makes the CUDA context current.
    // Returns false on timeout, end of stream or error (see last_error()).
    bool read(GpuFrame& frame, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    const std::string& last_error() const;
    std::string pipeline_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_tracking::camera
