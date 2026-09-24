#include "edge_tracking/pipeline/pipeline.hpp"

#include "bounded_queue.hpp"
#include "edge_tracking/camera/argus_gpu_camera.hpp"
#include "edge_tracking/inference/trt_engine.hpp"
#include "edge_tracking/preprocess/letterbox.hpp"

#include <cuda_runtime.h>
#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <thread>

namespace edge_tracking::pipeline {

namespace {

using Clock = std::chrono::steady_clock;
using telemetry::Counter;
using telemetry::Stage;

constexpr std::chrono::milliseconds kPollTimeout(100);  // how often idle threads check for stop
constexpr std::chrono::milliseconds kFirstFrameTimeout(5000);  // Argus takes 1-2 s to start
constexpr std::chrono::milliseconds kFrameTimeout(1000);

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int64_t steady_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// One frame in flight: its network input and (optionally) a host copy of the image.
struct Slot {
    float* d_tensor = nullptr;    // device, 3 x H x W
    uint8_t* h_nv12 = nullptr;    // pinned host, packed NV12, only if keep_image
    FrameResult result;
};

}  // namespace

struct Pipeline::Impl {
    PipelineConfig config;
    ResultCallback on_result;
    telemetry::Recorder recorder;

    inference::TrtEngine engine;
    std::unique_ptr<postprocess::YoloDecoder> decoder;
    std::unique_ptr<tracking::ByteTracker> tracker;
    std::unique_ptr<camera::ArgusGpuCamera> camera;

    std::vector<Slot> slots;
    float* d_output = nullptr;
    cudaStream_t capture_stream = nullptr;
    cudaStream_t infer_stream = nullptr;

    // Slot indices: free pool -> capture -> infer queue -> inference -> sink queue -> sink -> free pool.
    std::unique_ptr<BoundedQueue<int>> free_slots;
    std::unique_ptr<BoundedQueue<int>> infer_queue;
    std::unique_ptr<BoundedQueue<int>> sink_queue;

    std::atomic<bool> stop_requested{false};
    std::atomic<bool> running{false};
    mutable std::mutex error_mutex;
    std::string error;
    std::thread capture_thread, infer_thread, sink_thread;

    void fail(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error.empty()) error = message;
        }
        stop_requested = true;
    }

    void capture_loop();
    void infer_loop();
    void sink_loop();
    void release_resources();
};

void Pipeline::Impl::capture_loop() {
    camera::GpuFrame frame;
    preprocess::LetterboxParams letterbox;
    const int input_size = static_cast<int>(engine.input().shape[3]);
    const double frame_interval_ns = 1e9 / config.camera.fps;
    int64_t prev_pts = -1;
    bool have_letterbox = false;

    bool first = true;
    while (!stop_requested) {
        if (!camera->read(frame, first ? kFirstFrameTimeout : kFrameTimeout)) {
            fail("camera: " + camera->last_error());
            break;
        }
        const auto t_read = Clock::now();
        first = false;
        recorder.count(Counter::CameraFrames);
        if (prev_pts >= 0 && frame.pts_ns >= 0) {
            const long missing = std::lround((frame.pts_ns - prev_pts) / frame_interval_ns) - 1;
            if (missing > 0) recorder.count(Counter::CameraDropped, missing);
        }
        prev_pts = frame.pts_ns;

        const std::optional<int> slot_index = free_slots->try_pop();
        if (!slot_index) {  // every slot busy downstream: drop this frame, keep latency bounded
            recorder.count(Counter::IngestDropped);
            continue;
        }
        if (!have_letterbox) {
            letterbox = preprocess::compute_letterbox(frame.width, frame.height, input_size, input_size);
            have_letterbox = true;
        }

        Slot& slot = slots[*slot_index];
        cudaError_t status = preprocess::nv12_to_rgb_letterbox(frame.y, frame.y_pitch, frame.uv, frame.uv_pitch,
                                                               frame.color_space, letterbox, preprocess::TensorFormat{},
                                                               slot.d_tensor, capture_stream);
        if (status == cudaSuccess && config.keep_image) {
            const size_t y_size = static_cast<size_t>(frame.width) * frame.height;
            status = cudaMemcpy2DAsync(slot.h_nv12, frame.width, frame.y, frame.y_pitch, frame.width, frame.height,
                                       cudaMemcpyDeviceToHost, capture_stream);
            if (status == cudaSuccess) {
                status = cudaMemcpy2DAsync(slot.h_nv12 + y_size, frame.width, frame.uv, frame.uv_pitch, frame.width,
                                           frame.height / 2, cudaMemcpyDeviceToHost, capture_stream);
            }
        }
        // The camera buffer is released by the next read(): the GPU must be done with it.
        if (status == cudaSuccess) status = cudaStreamSynchronize(capture_stream);
        if (status != cudaSuccess) {
            fail(std::string("capture: ") + cudaGetErrorString(status));
            free_slots->try_push(*slot_index);
            break;
        }

        FrameResult& r = slot.result;
        r.frame_index = frame.index;
        r.capture_time_ns = frame.capture_time_ns;
        r.width = frame.width;
        r.height = frame.height;
        r.nv12 = config.keep_image ? slot.h_nv12 : nullptr;
        r.detections.clear();
        r.tracks.clear();
        recorder.add(Stage::Capture, ms_between(t_read, Clock::now()));
        // Cannot fail: the queue holds as many entries as there are slots.
        infer_queue->try_push(*slot_index);
    }
}

void Pipeline::Impl::infer_loop() {
    preprocess::LetterboxParams letterbox;
    bool have_letterbox = false;
    const int input_size = static_cast<int>(engine.input().shape[3]);

    while (true) {
        const std::optional<int> slot_index = infer_queue->pop(kPollTimeout);
        if (!slot_index) {
            if (stop_requested) break;
            continue;
        }
        Slot& slot = slots[*slot_index];
        FrameResult& r = slot.result;
        if (!have_letterbox) {
            letterbox = preprocess::compute_letterbox(r.width, r.height, input_size, input_size);
            have_letterbox = true;
        }

        const auto t0 = Clock::now();
        if (!engine.enqueue(slot.d_tensor, d_output, infer_stream)) {
            fail("inference: TensorRT enqueue failed");
            free_slots->try_push(*slot_index);
            break;
        }
        try {
            r.detections = decoder->decode(d_output, letterbox, infer_stream);  // synchronizes the stream
        } catch (const std::exception& e) {
            fail(std::string("decode: ") + e.what());
            free_slots->try_push(*slot_index);
            break;
        }
        const auto t1 = Clock::now();
        r.tracks = tracker->update(r.detections);
        const auto t2 = Clock::now();
        recorder.add(Stage::Inference, ms_between(t0, t1));
        recorder.add(Stage::Tracking, ms_between(t1, t2));

        if (!sink_queue->try_push(*slot_index)) {  // sink busy: skip showing this result
            recorder.count(Counter::SinkDropped);
            free_slots->try_push(*slot_index);
        }
    }
}

void Pipeline::Impl::sink_loop() {
    while (true) {
        const std::optional<int> slot_index = sink_queue->pop(kPollTimeout);
        if (!slot_index) {
            if (stop_requested) break;
            continue;
        }
        const FrameResult& r = slots[*slot_index].result;
        if (r.capture_time_ns >= 0) recorder.add(Stage::Latency, (steady_now_ns() - r.capture_time_ns) / 1e6);
        recorder.count(Counter::Results);

        const auto t0 = Clock::now();
        if (on_result) on_result(r);
        recorder.add(Stage::Sink, ms_between(t0, Clock::now()));
        free_slots->try_push(*slot_index);
    }
}

void Pipeline::Impl::release_resources() {
    camera.reset();
    for (Slot& slot : slots) {
        cudaFree(slot.d_tensor);
        cudaFreeHost(slot.h_nv12);
    }
    slots.clear();
    cudaFree(d_output);
    d_output = nullptr;
    if (capture_stream) cudaStreamDestroy(capture_stream);
    if (infer_stream) cudaStreamDestroy(infer_stream);
    capture_stream = infer_stream = nullptr;
}

Pipeline::Pipeline(PipelineConfig config) : impl_(std::make_unique<Impl>()) { impl_->config = std::move(config); }

Pipeline::~Pipeline() { stop(); }

bool Pipeline::start(ResultCallback on_result, std::string* error) {
    Impl& s = *impl_;
    if (s.running) return true;
    auto fail_start = [&](const std::string& message) {
        s.release_resources();
        s.error = message;
        if (error) *error = message;
        return false;
    };

    if (s.config.blocking_sync && cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error; keep running with the existing flags
        std::fprintf(stderr, "pipeline: blocking sync not applied (CUDA already initialized)\n");
    }

    std::string err;
    if (!s.engine.load(s.config.engine_path, &err)) return fail_start("engine: " + err);
    const inference::TensorInfo& in = s.engine.input();
    const inference::TensorInfo& out = s.engine.output();
    if (in.shape.size() != 4 || out.shape.size() != 3) return fail_start("engine is not a YOLOv8-style detector");

    s.decoder = std::make_unique<postprocess::YoloDecoder>(static_cast<int>(out.shape[2]),
                                                           static_cast<int>(out.shape[1]) - 4, s.config.decoder);
    s.tracker = std::make_unique<tracking::ByteTracker>(s.config.tracker);

    if (cudaStreamCreate(&s.capture_stream) != cudaSuccess || cudaStreamCreate(&s.infer_stream) != cudaSuccess ||
        cudaMalloc(&s.d_output, out.elements * sizeof(float)) != cudaSuccess) {
        return fail_start("CUDA allocation failed");
    }
    const size_t image_bytes = static_cast<size_t>(s.config.camera.width) * s.config.camera.height * 3 / 2;
    s.slots.resize(s.config.num_slots);
    for (Slot& slot : s.slots) {
        if (cudaMalloc(&slot.d_tensor, in.elements * sizeof(float)) != cudaSuccess ||
            (s.config.keep_image && cudaMallocHost(&slot.h_nv12, image_bytes) != cudaSuccess)) {
            return fail_start("CUDA allocation failed");
        }
    }

    s.free_slots = std::make_unique<BoundedQueue<int>>(s.slots.size());
    s.infer_queue = std::make_unique<BoundedQueue<int>>(s.slots.size());
    s.sink_queue = std::make_unique<BoundedQueue<int>>(static_cast<size_t>(s.config.sink_queue_capacity));
    for (int i = 0; i < static_cast<int>(s.slots.size()); ++i) s.free_slots->try_push(i);

    s.camera = std::make_unique<camera::ArgusGpuCamera>(s.config.camera);
    if (!s.camera->start(&err)) return fail_start("camera: " + err);

    s.on_result = std::move(on_result);
    s.error.clear();
    s.stop_requested = false;
    s.running = true;
    // Named so per-thread CPU shows up clearly in top -H / htop.
    s.capture_thread = std::thread([&s] {
        pthread_setname_np(pthread_self(), "et-capture");
        s.capture_loop();
    });
    s.infer_thread = std::thread([&s] {
        pthread_setname_np(pthread_self(), "et-infer");
        s.infer_loop();
    });
    s.sink_thread = std::thread([&s] {
        pthread_setname_np(pthread_self(), "et-sink");
        s.sink_loop();
    });
    return true;
}

void Pipeline::stop() {
    Impl& s = *impl_;
    if (!s.running) return;
    s.stop_requested = true;
    // Capture first (it may be waiting up to ~1 s in camera read), then the stages it feeds.
    if (s.capture_thread.joinable()) s.capture_thread.join();
    s.infer_queue->close();
    if (s.infer_thread.joinable()) s.infer_thread.join();
    s.sink_queue->close();
    if (s.sink_thread.joinable()) s.sink_thread.join();
    s.camera->stop();
    s.release_resources();
    s.running = false;
}

bool Pipeline::running() const { return impl_->running && !impl_->stop_requested; }

std::string Pipeline::error() const {
    std::lock_guard<std::mutex> lock(impl_->error_mutex);
    return impl_->error;
}

telemetry::Recorder& Pipeline::telemetry() { return impl_->recorder; }

}  // namespace edge_tracking::pipeline
