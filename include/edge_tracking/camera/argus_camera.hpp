#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace edge_tracking::camera {

struct CameraConfig {
    int sensor_id = 0;
    int sensor_mode = 4;  // IMX219 mode 4: 1280x720 @ 60 fps (docs/camera_modes.txt)
    int width = 1280;
    int height = 720;
    int fps = 60;
};

// One NV12 frame in CPU memory, tightly packed (no row padding):
// Y plane (width * height bytes) followed by interleaved UV plane (width * height / 2 bytes).
struct Frame {
    int width = 0;
    int height = 0;
    uint64_t index = 0;   // sequence number of frames delivered by this camera
    int64_t pts_ns = -1;  // GStreamer presentation timestamp, -1 if unknown
    std::vector<uint8_t> nv12;

    const uint8_t* y() const { return nv12.data(); }
    const uint8_t* uv() const { return nv12.data() + static_cast<size_t>(width) * height; }
};

// Captures from a CSI camera through Argus (nvarguscamerasrc) and copies frames out of
// NVMM into CPU memory. Baseline implementation: simple and measurable, not zero-copy.
class ArgusCamera {
public:
    explicit ArgusCamera(CameraConfig config);
    ~ArgusCamera();

    ArgusCamera(const ArgusCamera&) = delete;
    ArgusCamera& operator=(const ArgusCamera&) = delete;

    bool start(std::string* error = nullptr);
    void stop();

    // Waits up to `timeout` for the next frame and writes it into `frame`, reusing its
    // buffer. Returns false on timeout, end of stream or pipeline error (see last_error()).
    bool read(Frame& frame, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    const std::string& last_error() const;
    std::string pipeline_description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_tracking::camera
