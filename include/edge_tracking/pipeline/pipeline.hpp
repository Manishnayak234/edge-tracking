#pragma once

#include "edge_tracking/camera/argus_camera.hpp"
#include "edge_tracking/postprocess/yolo_decoder.hpp"
#include "edge_tracking/telemetry/telemetry.hpp"
#include "edge_tracking/tracking/byte_tracker.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace edge_tracking::pipeline {

struct PipelineConfig {
    camera::CameraConfig camera;
    std::string engine_path = "models/yolov8n_fp16.engine";
    postprocess::DecoderConfig decoder = low_threshold_decoder();  // tracking needs scores >= 0.1
    tracking::TrackerConfig tracker;

    // Frames in flight (each slot holds a network input tensor and, if enabled, an image).
    // When all are busy, new camera frames are dropped, which bounds latency.
    int num_slots = 4;
    int sink_queue_capacity = 2;  // results waiting for the sink; extra ones are dropped
    bool keep_image = false;      // copy each frame (NV12) to host memory for sinks that draw it
    // Threads sleep while waiting for the GPU instead of spinning on a CPU core. Applies to the
    // whole process and only if set before CUDA is first used (start() warns otherwise).
    bool blocking_sync = true;

    static postprocess::DecoderConfig low_threshold_decoder() {
        postprocess::DecoderConfig c;
        c.score_threshold = 0.1f;
        return c;
    }
};

// Everything the pipeline produced for one camera frame. Valid only during the callback.
struct FrameResult {
    uint64_t frame_index = 0;       // camera sequence number
    int64_t capture_time_ns = -1;   // steady_clock time of capture, -1 if unknown
    int width = 0;
    int height = 0;
    std::vector<postprocess::Detection> detections;
    std::vector<tracking::Track> tracks;
    const uint8_t* nv12 = nullptr;  // packed NV12 image in host memory if keep_image, else null
};

using ResultCallback = std::function<void(const FrameResult&)>;

// Camera -> detection -> tracking in three threads connected by bounded queues:
//   capture:   camera read + GPU preprocessing into a free slot (releases the camera buffer fast)
//   inference: TensorRT + decode/NMS + ByteTrack, in frame order
//   sink:      the user callback, so a slow consumer never stalls detection
class Pipeline {
public:
    explicit Pipeline(PipelineConfig config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Loads the engine, starts the camera and the threads. `on_result` runs on the sink thread.
    bool start(ResultCallback on_result, std::string* error = nullptr);
    // Stops the threads and the camera. Safe to call more than once.
    void stop();

    // False once stopped or after a fatal error (see error()).
    bool running() const;
    std::string error() const;

    telemetry::Recorder& telemetry();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_tracking::pipeline
